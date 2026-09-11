#pragma once

#include "AerialLoader.h"
#include "ForestLoader.h"
#include "TreeLoader.h"
#include "CoordTransformer.h"

#define TRAIN_PATH			"/train"
#define PREDICT_PATH		"/predict"
#define REJECT_PATH		"/rejected"

// ─────────────────────────────────────────────
//  나무별 수종 레이블 부여
//  LiDAR 나무 위치 → 임상도 폴리곤 교차
// ─────────────────────────────────────────────

struct LabeledTree
{
	// LiDAR 원본 정보
	TreePoint tp;
	// Label info
	bool insideForest = false;			// inside a forest polygon?
	std::string speciesCode;				// KOFTR_GROU 값
	std::string speciesName;
	bool inForestMap = false;			// inside a forest polygon?
	double distToEdge = 0.0;				// 폴리곤 경계까지 거리(m) - 신뢰도 지표
	bool reliable = false;					// distToEdge >= minEdgeDist
	// Crown image info
	int cropX = 0;								// crop top-left pixel X
	int cropY = 0;								// crop top-left pixel Y
	int cropSize = 0;							// crop size (pixels, square)
	bool cropValid = false;					// crop within image bounds?
	double emptyRatio = 0.0;			// ratio of empty/nodata pixels (0~1)
	bool imageUsable = false;			// emptyRatio <= threshold
	std::string imagePath;					// saved image path
	// 특징 벡터
	std::vector<float> features;
	// 예측 결과
	std::string predictedCode;
};

// ─────────────────────────────────────────────
//  Labeling options
// ─────────────────────────────────────────────

struct LabelOptions
{
	double minEdgeDist = 5.0;			// min distance to polygon edge for reliable label (m)
	double crownPadding = 1.4;		// crown crop padding multiplier (1.0 = exact crown)
	int minCropPx = 16;						// minimum crop size in pixels
	int maxCropPx = 512;					// maximum crop size in pixels
	bool squareCrop = true;				// force square crop (max of W/H)
	bool saveImages = true;				// save cropped images to disk
	std::string outputDir = "labeled_trees";

	// Empty region detection
	double maxEmptyRatio = 0.1;		// max allowed empty pixel ratio (0~1, default 10%)
	int emptyThreshold = 5;				// pixel brightness <= this is "empty" (0~255)

	// Coordinate systems (for tree->forest CRS transform)
	int aerialCrsEpsg = 0;					// tree CSV EPSG (0 = same as forest)
	int forestCrsEpsg = 0;					// forest SHP EPSG (0 = read from file)
	int treeCrsEpsg = 0;						// aerial EPSG (0 = read from file)
};

class TreeLabeler
{
public:

	static std::vector<LabeledTree> label(const AerialPhoto& aerialImg, const ForestLayer& forest, const TreeData& trees, const LabelOptions& opts = LabelOptions())
	{
		std::cout << "=== Tree Labeling Pipeline ===\n";

		// 1. Setup coordinate transforms
		int aerialEpsg = (opts.aerialCrsEpsg > 0) ? opts.aerialCrsEpsg : aerialImg.epsg;
		int forestEpsg = (opts.forestCrsEpsg > 0) ? opts.forestCrsEpsg : forest.epsg;
		int treeEpsg = (opts.treeCrsEpsg > 0) ? opts.treeCrsEpsg : 0;

		// Tree -> Forest CRS (for polygon Contains test)
		OGRCoordinateTransformation* treeToForestCT = nullptr;

		if ((treeEpsg > 0) && (forestEpsg > 0) && (treeEpsg != forestEpsg))
		{
			treeToForestCT = CoordTransformer::createTransform(treeEpsg, forestEpsg);

			std::cout << "  Tree->Forest CRS: EPSG:" << treeEpsg << " -> EPSG:" << forestEpsg << "\n";
		}

		// Tree -> Aerial CRS (for image crop)
		OGRCoordinateTransformation* treeToAerialCT = nullptr;

		if ((treeEpsg > 0) && (aerialEpsg > 0) && (treeEpsg != aerialEpsg))
		{
			treeToAerialCT = CoordTransformer::createTransform(treeEpsg, aerialEpsg);

			std::cout << "  Tree->Aerial CRS: EPSG:" << treeEpsg << " -> EPSG:" << aerialEpsg << "\n";
		}
		else if ((forestEpsg > 0) && (aerialEpsg > 0) && (forestEpsg != aerialEpsg))
		{
			// Trees are in forest CRS, need to go to aerial CRS
			treeToAerialCT = CoordTransformer::createTransform(forestEpsg, aerialEpsg);

			std::cout << "  Forest->Aerial CRS: EPSG:" << forestEpsg << " -> EPSG:" << aerialEpsg << "\n";
		}

		// Inverse GeoTransform for aerial geo->pixel
		double gt[6] = { aerialImg.originX, aerialImg.pixelW, aerialImg.rotationX, aerialImg.originY, aerialImg.rotationY, aerialImg.pixelH };
		double igt[6];

		if (!CoordTransformer::computeInverseGT(gt, igt))
		{
			throw std::runtime_error("Failed to compute inverse GeoTransform");
		}

		double pixPerMeter = std::abs(igt[1]);

		std::cout << "  Aerial: " << aerialImg.width << "x" << aerialImg.height << " (EPSG:" << aerialEpsg << ", " << std::fixed << std::setprecision(2) << (1.0 / pixPerMeter * 100) << " cm/px)\n";
		std::cout << "  Forest: " << forest.features.size() << " polygons" << " (EPSG:" << forestEpsg << ")\n";
		std::cout << "  Trees:  " << trees.tps.size() << " (EPSG:" << (treeEpsg > 0 ? treeEpsg : forestEpsg) << ")\n";
		std::cout << "  Min edge dist: " << opts.minEdgeDist << " m\n";
		std::cout << std::setprecision(6);

		// 2. Build spatial index (simple bbox pre-filter)
		std::cout << "  Step 1: Labeling trees against forest polygons...\n";

		std::vector<LabeledTree> result;
		result.reserve(trees.tps.size());

		int insideCount = 0;
		int outsideCount = 0;
		
		int reliableCount = 0;
		int tooCloseCount = 0;

		int nearBoundaryCount = 0;

		int cropOk = 0;
		int cropFail = 0;

		std::map<std::string, int> speciesCount;

		int total = (int)trees.tps.size();
		int progressStep = std::max(1, total / 20);

		for (int ti = 0 ; ti < total ; ++ti)
		{
			auto& tp = trees.tps[ti];

			LabeledTree lt;
			
			lt.tp.treeId = tp.treeId;
			lt.tp.x = tp.x;
			lt.tp.y = tp.y;
			lt.tp.height = tp.height;
			lt.tp.crownD = tp.crownD;
			lt.tp.crownArea = tp.crownArea;

			// Transform tree position to forest CRS for Contains test
			double fx = tp.x;
			double fy = tp.y;

			if (treeToForestCT)
			{
				double z = 0;
				treeToForestCT->Transform(1, &fx, &fy, &z);
			}

			OGRPoint pt(fx, fy);

			const ForestFeature* bestFeat = nullptr;
			double bestDist = -1.0;

			// 나무 위치를 포함하는 폴리곤 탐색
			for (auto& ft : forest.features)
			{
				if (!ft.geometry)
				{
					continue;
				}

				// Quick bbox pre-filter
				OGREnvelope env;
				ft.geometry->getEnvelope(&env);

				if ((fx < env.MinX) || (fx > env.MaxX) || (fy < env.MinY) || (fy > env.MaxY))
				{
					continue;
				}

				if (!ft.geometry->Contains(&pt))
				{
					continue;
				}

				// 경계까지 거리 계산 (내부면 양수)
				double dist = 0.0;

				try
				{
					OGRGeometry* boundary = ft.geometry->Boundary();

					if (boundary)
					{
						dist = pt.Distance(boundary);
						OGRGeometryFactory::destroyGeometry(boundary);
					}
				}
				catch (...)
				{
					dist = 0.0;
				}

				if (dist > bestDist)
				{
					bestDist = dist;
					bestFeat = &ft;
				}
			}

			if (bestFeat)
			{
				lt.insideForest = true;
				lt.distToEdge = bestDist;
				lt.reliable = (bestDist >= opts.minEdgeDist);

				auto itCode = bestFeat->attributes.find(SPECIES_CODE);
				lt.speciesCode = (itCode != bestFeat->attributes.end()) ? itCode->second : "";

				auto itName = bestFeat->attributes.find(SPECIES_NAME);
				lt.speciesName = (itName != bestFeat->attributes.end()) ? itName->second : "";

				++insideCount;

				if (lt.reliable)
				{
					++reliableCount;
				}
				else
				{
					++tooCloseCount;
				}

				if (!lt.speciesCode.empty())
				{
					speciesCount[lt.speciesCode]++;
				}
			}
			else
			{
				// Fallback: find nearest polygon within 3m
				// (catches trees on polygon boundaries)

				const double nearThreshold = 3.0;			// meters
				const ForestFeature* nearFeature = nullptr;
				double nearDist = nearThreshold + 1.0;

				for (auto& ft : forest.features)
				{
					if (!ft.geometry)
					{
						continue;
					}

					// Quick bbox pre-filter
					OGREnvelope env;
					ft.geometry->getEnvelope(&env);

					if ((fx < env.MinX - nearThreshold) || (fx > env.MaxX + nearThreshold) || (fy < env.MinY - nearThreshold) || (fy > env.MaxY + nearThreshold))
					{
						continue;
					}

					double dist = pt.Distance(ft.geometry);

					if (dist < nearDist)
					{
						nearDist = dist;
						nearFeature = &ft;
					}
				}

				if (nearFeature && (nearDist <= nearThreshold))
				{
					lt.insideForest = true;
					lt.distToEdge = -nearDist;		// negative = outside but close
					lt.reliable = false;
				
					auto itCode = nearFeature->attributes.find(SPECIES_CODE);
					lt.speciesCode = (itCode != nearFeature->attributes.end()) ? itCode->second : "";

					auto itName = nearFeature->attributes.find(SPECIES_NAME);
					lt.speciesName = (itName != nearFeature->attributes.end()) ? itName->second : "";

					++nearBoundaryCount;

					if (!lt.speciesCode.empty())
					{
						speciesCount[lt.speciesCode]++;
					}
				}
				else
				{
					lt.insideForest =false;
					++outsideCount;
				}
			}

			// 3. Compute crown crop rectangle in aerial image
			double ax = tp.x;
			double ay = tp.y;

			if (treeToAerialCT)
			{
				double z = 0;
				treeToAerialCT->Transform(1, &ax, &ay, &z);
			}

			// Geo -> pixel
			double pxF = igt[0] + igt[1] * ax + igt[2] * ay;
			double pyF = igt[3] + igt[4] * ax + igt[5] * ay;

			// Crown radius in pixels
			double crownR = lt.tp.crownD * 0.5 * opts.crownPadding;
			int crownPxR = (int)std::ceil(crownR * pixPerMeter);
			crownPxR = std::max(crownPxR, opts.minCropPx / 2);

			int cropHalf = std::min(crownPxR, opts.maxCropPx / 2);

			int cx = (int)std::round(pxF);
			int cy = (int)std::round(pyF);

			lt.cropX = cx - cropHalf;
			lt.cropY = cy - cropHalf;
			lt.cropSize = cropHalf * 2;

			// Check bounds
			if ((lt.cropX >= 0) && (lt.cropY >= 0) && 
				(lt.cropX + lt.cropSize <= aerialImg.image.cols) &&
				(lt.cropY + lt.cropSize <= aerialImg.image.rows) &&
				(lt.cropSize >= opts.minCropPx))
			{
				lt.cropValid = true;
				++cropOk;
			}
			else
			{
				lt.cropValid = false;
				++cropFail;
			}

			result.push_back(lt);

			// Progress
			if (((ti + 1) % progressStep == 0) || (ti == total -1))
			{
				std::cout << "    " << (ti + 1) << "/" << total << " (" << (100 * (ti + 1) / total) << "%)\r" << std::flush;
			}
		}

		std::cout << "\n";

		// Cleanup CRS transforms
		if (treeToForestCT)
		{
			OGRCoordinateTransformation::DestroyCT(treeToForestCT);
		}

		if (treeToAerialCT)
		{
			OGRCoordinateTransformation::DestroyCT(treeToAerialCT);
		}

		// 4. Print summary
		std::cout << "\n  === Labeling Summary ===\n";
		std::cout << "    Total trees     : " << total << "\n";
		std::cout << "    Inside forest   : " << (insideCount + nearBoundaryCount) << " (train candidates)\n";
		std::cout << "      Reliable (>=" << opts.minEdgeDist << "m): " << reliableCount << "\n";
		std::cout << "      Near edge     : " << tooCloseCount << "\n";
		std::cout << "      Near boundary : " << nearBoundaryCount << " (on polygon edge, not reliable)\n";
		std::cout << "    Outside forest  : " << outsideCount << " (predict targets)\n";
		std::cout << "    Crop valid      : " << cropOk << "\n";
		std::cout << "    Crop out-of-bounds: " << cropFail << "\n";

		if (!speciesCount.empty())
		{
			std::cout << "    Species distribution:\n";
			std::vector<std::pair<std::string, int>> sorted(speciesCount.begin(), speciesCount.end());
			std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
			
			for (auto& kv : sorted)
			{
				std::cout << "      " << std::setw(12) << std::left << kv.first << " : " << kv.second << "\n";
			}
		}

		// 5. Save crown images
		if (opts.saveImages)
		{
			std::cout << "\n  Step 2: Saving crown images...\n";

			saveCrownImages(result, aerialImg, opts);
		}

		// 6. Save label CSV
		saveLabelCSV(result, opts);
		
		return result;
	}

	// ─── Save crown images organized by category ───
	static void saveCrownImages(std::vector<LabeledTree>& labeledTrees, const AerialPhoto& aerialImg, const LabelOptions& opts)
	{
		// Create directory structure
		// 디렉토리 생성 (없으면)
		_mkdir(opts.outputDir.c_str());
		_mkdir((opts.outputDir + TRAIN_PATH).c_str());
		_mkdir((opts.outputDir + PREDICT_PATH).c_str());
		_mkdir((opts.outputDir + REJECT_PATH).c_str());
	
	// Track created species directories
		std::set<std::string> createdDirs;

		int savedTrain = 0;
		int savedPredict = 0;
		int skipped = 0;
		int rejected = 0;

		int total = (int)labeledTrees.size();
		int progressStep = std::max(1, total / 20);

		for (int i = 0 ; i < total ; ++i)
		{
			auto& lt = labeledTrees[i];

			if (!lt.cropValid)
			{
				++skipped;

				continue;
			}

			// Crop from aerial image

			cv::Rect roi(lt.cropX, lt.cropY, lt.cropSize, lt.cropSize);
			cv::Mat crop = aerialImg.image(roi).clone();
/*
			const int BASE_R = 10;
			const int BASE_SIZE = 128;
			const int MAX_SIZE = 512;

			int outSize = (int)(BASE_SIZE * ((double)(lt.cropSize * 0.5) / BASE_R));
			outSize = std::max(BASE_SIZE, outSize);
			outSize = std::min(MAX_SIZE, outSize);

			// 32의 배수로 정렬 (깔끔한 크기)
			int outImgSize = ((outSize + 31) / 32) * 32;

			cv::Mat crop;
			cv::resize(aerialImg.image(roi), crop, cv::Size(outImgSize, outImgSize));
*/

			// Check empty/nodata pixel ratio
			lt.emptyRatio = computeEmptyRatio(crop, opts.emptyThreshold);
			lt.imageUsable = (lt.emptyRatio <= opts.maxEmptyRatio);

			// Determine save path
			std::string subDir;
			std::string prefix;

			if (!lt.imageUsable)
			{
				// Rejected: too much empty area
				subDir = opts.outputDir + REJECT_PATH;
				prefix = "";

				++rejected;
			}
			else if (lt.insideForest && !lt.speciesCode.empty())
			{
				// Training data: organize by species code
				std::string dirName = lt.speciesCode;
				// Sanitize directory name
				for (char& c : dirName)
				{
					if (c == '/' || c == '\\' || c == ':' || c == ' ')
					{
						c = '_';
					}
				}

				// 임상도 내부 나무 → train 폴더, 수종 코드별 서브폴더
				subDir = opts.outputDir + TRAIN_PATH + "/" + dirName;
				prefix = lt.speciesCode;

				++savedTrain;
			}
			else
			{
				// 임상도 외부 나무 → predict 폴더
				subDir = opts.outputDir + PREDICT_PATH;
				prefix = "unknown";

				++savedPredict;
			}

			_mkdir(subDir.c_str());
			
			std::string fname = subDir + "/"
				+ "tree_"
				+ prefix
				+ "_id" + std::to_string(lt.tp.treeId)
				+ "_cx" + std::to_string(lt.cropX)
				+ "_cy" + std::to_string(lt.cropY)
				+ ".png";

			cv::imwrite(fname, crop);
			lt.imagePath = fname;

			// Progress
			if ((i + 1) % progressStep == 0 || i == total - 1)
			{
				std::cout << "    Saved " << (savedTrain + savedPredict) << "/" << total << " (" << (100 * (i + 1) / total) << "%)\r" << std::flush;
			}
		}

		std::cout << "\n";

		std::cout << "    Train images  : " << savedTrain << " (in " << createdDirs.size() << " species folders)\n";
		std::cout << "    Predict images: " << savedPredict << "\n";
		std::cout << "    Rejected (empty): " << rejected << " (>" << (int)(opts.maxEmptyRatio * 100) << "% empty)\n";
		std::cout << "    Skipped (OOB) : " << skipped << "\n";
		std::cout << "    Output dir    : " << opts.outputDir << "/\n";
	}

	// ─── Compute empty/nodata pixel ratio ───
	// Counts pixels where ALL channels are <= threshold
	static double computeEmptyRatio(const cv::Mat& crop, int threshold)
	{
		if (crop.empty())
		{
			return 1.0;
		}

		int totalPixels = crop.rows * crop.cols;
		int emptyCount = 0;

		for (int y = 0 ; y < crop.rows ; ++y)
		{
			const uint8_t* row = crop.ptr<uint8_t>(y);

			for (int x = 0 ; x < crop.cols ; ++x)
			{
				int idx = x * crop.channels();
				bool empty = true;

				for (int c = 0 ; c < crop.channels() ; ++c)
				{
					if (row[idx + c] > threshold)
					{
						empty = false;

						break;
					}
				}

				if (empty)
				{
					++emptyCount;
				}
			}
		}

		return (double)emptyCount / totalPixels;
	}

	// ─── Save label info CSV ───
	static void saveLabelCSV(const std::vector<LabeledTree>& labeledTrees, const LabelOptions& opts)
	{
		std::string csvPath = opts.outputDir + "/" + "label_info.csv";

		std::ofstream out(csvPath.c_str());

		if (!out.is_open())
		{
			std::cerr << "  WARNING: Failed to create " << csvPath << "\n";

			return;
		}

		out << "TreeID,X,Y,Height,CrownD,CrownArea," 
			<< "InsideForest,SpeciesCode,SpeciesName,"
			<< "DistToEdge,Reliable,"
			<< "CropX,CropY,CropSize,CropValid,"
			<< "EmptyRatio,ImageUsable,ImagePath\n";

		for (auto& lt : labeledTrees)
		{
			out << lt.tp.treeId <<","
				<< std::fixed << std::setprecision(3)
				<< lt.tp.x << "," << lt.tp.y << ","
				<< std::setprecision(1)
				<< lt.tp.height << "," << lt.tp.crownD << "," << lt.tp.crownArea << ","
				<< (lt.insideForest ? "Y" : "N") << ","
				<< lt.speciesCode << ","
				<< lt.speciesName << ","
				<< std::setprecision(2)
				<< lt.distToEdge << ","
				<< (lt.reliable ? "Y" : "N") << ","
				<< lt.cropX << "," << lt.cropY << "," << lt.cropSize << "," << (lt.cropValid ? "Y" : "N") << ","
				<< std::setprecision(3)
				<< lt.emptyRatio << ","
				<< (lt.imageUsable ? "Y" : "N") << ","
				<< lt.imagePath << "\n";
		}

		out.close();

		std::cout << "    Label CSV: " << csvPath << "\n";
	}
};

class TreeCrownCopper
{
public:

	struct CropResult
	{
		cv::Mat patch;				// 수관 이미지 패치 (BGR)
		bool valid = false;
	};

	/**
	  * tree    : 나무 정보
	  * aerial  : AerialPhoto (geoMinX, geoMaxY, resolution 포함)
	  * padRatio: 수관 반경에 추가할 여백 비율 (0.2 = 20%)
	  * outSize : 출력 패치 크기 (픽셀)
      */

	static CropResult crop(const LabeledTree& labeledTree, const AerialPhoto& aerial, const LabelOptions& opts, double padRatio = 0.2, int outSize = 0)
	{
		CropResult res;

		if (aerial.image.empty())
		{
			return res;
		}

		// scaleX/Y: 실제 이미지 픽셀 수 기반 (해상도 자동 반영)
		double scaleX = aerial.image.cols / (aerial.geoMaxX - aerial.geoMinX);
		double scaleY = aerial.image.rows / (aerial.geoMaxY - aerial.geoMinY);

		// 지리 좌표 → 픽셀 좌표
		double geoX = labeledTree.tp.x - aerial.geoMinX;
		double geoY = aerial.geoMaxY - labeledTree.tp.y;

		int cx = (int)(geoX * scaleX);
		int cy = (int)(geoY * scaleY);
		// 수관 반경 (여백 포함)
		double radius = (labeledTree.tp.crownD / 2.0) * (1.0 + padRatio);
		int r = std::max(1, (int)(radius * scaleX));

		int x1 = cx - r;
		int y1 = cy - r;
		int x2 = cx + r;
		int y2 = cy + r;

		// 경계 클램핑
		int W = aerial.image.cols;
		int H = aerial.image.rows;

		if ((x2 < 0) || (y2 < 0) || (x1 >= W) || (y1 >= H))
		{
			return res;
		}

		x1 = std::max(0, x1);
		y1 = std::max(0, y1);
		x2 = std::min(W - 1, x2);
		y2 = std::min(H - 1, y2);

		if ((x2 <= x1) || (y2 <= y1))
		{
			return res;
		}

		cv::Rect roi(x1, y1, x2 - x1, y2 - y1);

		if ((roi.width < 2) || (roi.height < 2))
		{
			return res;
		}

		int outImgSize = outSize;

		if (outSize == 0)
		{
			const int BASE_R = 10;
			const int BASE_SIZE = 128;
			const int MAX_SIZE = 512;

			int outSize = (int)(BASE_SIZE * ((double)r / BASE_R));
			outSize = std::max(BASE_SIZE, outSize);
			outSize = std::min(MAX_SIZE, outSize);
			
			// 32의 배수로 정렬 (깔끔한 크기)
			outImgSize = ((outSize + 31) / 32) * 32;
		}

		cv::Mat patch;
		cv::resize(aerial.image(roi), patch, cv::Size(outImgSize, outImgSize));

		std::string subDir;
		std::string prefix;

		if (!labeledTree.speciesCode.empty())
		{
			// 임상도 내부 나무 → train 폴더, 수종 코드별 서브폴더
			subDir = opts.outputDir + TRAIN_PATH + "/" + labeledTree.speciesCode;
			prefix = labeledTree.speciesCode;
		}
		else
		{
			// 임상도 외부 나무 → predict 폴더
			subDir = opts.outputDir + PREDICT_PATH;
			prefix = "unknown";
		}

		// 디렉토리 생성 (없으면)
		_mkdir(opts.outputDir.c_str());
		_mkdir((opts.outputDir + TRAIN_PATH).c_str());
		_mkdir((opts.outputDir + PREDICT_PATH).c_str());
		_mkdir(subDir.c_str());

		// 파일명: {수종코드}_{treeId}_{cx}_{cy}.png
		std::string fname = subDir + "/"
			+ prefix
			+ "_id" + std::to_string(labeledTree.tp.treeId)
			+ "_cx" + std::to_string(cx)
			+ "_cy" + std::to_string(cy)
			+ ".png";
			
		cv::imwrite(fname, patch);

		res.patch = patch;
		res.valid = true;

		return res;
	}

	static void saveDebugOverview(const std::vector<LabeledTree>& labeledTrees, const AerialPhoto& aerial, const std::string& savePath, double padRatio = 0.2, double overviewScale = 0.05)
	{
		if (aerial.image.empty())
		{
			return;
		}

		double scaleX = aerial.image.cols / (aerial.geoMaxX - aerial.geoMinX);
		double scaleY = aerial.image.rows / (aerial.geoMaxY - aerial.geoMinY);

		cv::Mat overview;
		cv::resize(aerial.image, overview, cv::Size(0, 0), overviewScale, overviewScale);

		for (const auto& lt : labeledTrees)
		{
			if (lt.tp.treeId < 0)
			{
				continue;
			}

			if (((lt.tp.treeId + 5 ) % 10) != 0)
			{
				continue;
			}

			// 픽셀 좌표 계산
			int cx = (int)((lt.tp.x - aerial.geoMinX) * scaleX);
			int cy = (int)((aerial.geoMaxY - lt.tp.y) * scaleY);

			double radius = (lt.tp.crownD / 2.0) * (1.0 + padRatio);
			int r = std::max(1, (int)(radius * scaleX));

			// 축소 비율 적용
			int sx = (int)(cx * overviewScale);
			int sy = (int)(cy * overviewScale);
			int sr = std::max(1, (int)(r * overviewScale));

			// 임상도 내부(학습용) = 초록, 외부(예측용) = 빨강
			cv::Scalar color = lt.inForestMap ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 200);

			if ((sx >= 0) && (sx < overview.cols) && (sy >= 0) && (sy < overview.rows))
			{
				// 원 그리기
				cv::circle(overview, cv::Point(sx, sy), std::max(1, sr), color, 1);
				// treeId 텍스트
				cv::putText(overview, std::to_string(lt.tp.treeId), cv::Point(sx + sr + 1, sy), cv::FONT_HERSHEY_PLAIN, 0.6, color, 1);
			}
		}

		cv::imwrite(savePath, overview);
	}
};
