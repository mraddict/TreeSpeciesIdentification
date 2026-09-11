#pragma once

// ─────────────────────────────────────────────────────────
//  FeatureExtractor.h
//  Crown image feature extraction for species classification
//
//  Input:  LabeledTree results from TreeLabeler
//  Output: Feature vectors (14-dim) for each tree
//
//  Feature vector layout (20 dimensions):
//    [0]  B_mean       - Blue channel mean
//    [1]  G_mean       - Green channel mean
//    [2]  R_mean       - Red channel mean
//    [3]  B_std        - Blue channel stddev
//    [4]  G_std        - Green channel stddev
//    [5]  R_std        - Red channel stddev
//    [6]  H_mean       - Hue mean (HSV)
//    [7]  S_mean       - Saturation mean (HSV)
//    [8]  V_mean       - Value mean (HSV)
//    [9]  Contrast     - Texture contrast (Sobel gradient magnitude)
//    [10] Entropy      - Grayscale histogram entropy
//    [11] LBP_mean     - Local Binary Pattern mean
//    [12] LBP_std      - Local Binary Pattern stddev
//    [13] GLCM_energy  - GLCM energy (angular second moment)
//    [14] GLCM_corr    - GLCM correlation
//    [15] GLCM_homo    - GLCM homogeneity (inverse difference moment)
//    [16] GLCM_dissim  - GLCM dissimilarity
//    [17] Height       - Tree height (m, from LiDAR)
//    [18] CrownD       - Crown diameter (m, from LiDAR)
//    [19] H_CrownD     - Height / CrownD ratio
// 
//  Dependencies:
//    - TreeLabeler.h (LabeledTree)
//    - OpenCV
//
//  Build:
//    g++ -std=c++17 -O2 -o feature_extract main_feature_extract.cpp \
//        $(gdal-config --cflags --libs) \
//        $(pkg-config --cflags --libs opencv4)
// ─────────────────────────────────────────────────────────

#include "TreeLabeler.h"

// ─────────────────────────────────────────────
//  Feature vector for a single tree
// ─────────────────────────────────────────────

struct TreeFeature
{
	long long treeId = -1;
	std::string speciesCode;				// KOFTR_GROU(empty if outside forest)
	std::string speciesName;				// KOFTR_NM
	bool insideForest = false;
	bool reliable = false;					// inside forest + far from edge. 폴리곤 경계에서 충분히 안쪽
	bool imageUsable = false;			// not rejected by empty check
	double distToEdge = 0.0;				// distance to polygon boundary (m)
	std::string imagePath;					// saved crop image path
	std::vector<float> feature;			// feature vector (14-dim)

	// Feature names (static, for CSV header)
	static const std::vector<std::string>& featureNames()
	{
		static std::vector<std::string> names =
		{
			"B_mean", "G_mean", "R_mean",
			"B_std",  "G_std",  "R_std",
			"H_mean", "S_mean", "V_mean",
			"Contrast", "Entropy",
			"LBP_mean", "LBP_std",
			"GLCM_energy", "GLCM_corr", "GLCM_homo", "GLCM_dissim",
			"Height", "CrownD", "H_CrownD"
		};

		return names;
	}
};

// ─────────────────────────────────────────────
//  Extraction result summary
// ─────────────────────────────────────────────

struct ExtractionResult
{
	std::vector<TreeFeature> all;				// all extracted features
	std::vector<TreeFeature*> train;			// reliable + usable (for training)
	std::vector<TreeFeature*> predict;		// outside forest + usable (for prediction)
	int skippedEmpty = 0;							// rejected by empty check
	int skippedNoCrop = 0;						// no valid crop
	int skippedReadFail = 0;						// image read failure
};

class FeatureExtractor
{
public:

	// ─── Extract from saved crop images (reads from disk) ───
	static ExtractionResult extractFromFiles(const std::vector<LabeledTree>& labeledTrees)
	{
		std::cout << "=== Feature Extraction (from saved images) ===\n";

		ExtractionResult result;
		result.all.reserve(labeledTrees.size());

		int total = (int)labeledTrees.size();
		int progressStep = std::max(1, total / 20);

		for (int i = 0 ; i < total ; ++i)
		{
			auto& lt = labeledTrees[i];

			// Skip unusable
			if (!lt.cropValid)
			{
				++result.skippedNoCrop;

				continue;
			}

			if (!lt.imageUsable)
			{
				++result.skippedEmpty;

				continue;
			}

			if (lt.imagePath.empty())
			{
				++result.skippedNoCrop;

				continue;
			}

			// Read crop image
			cv::Mat crop = cv::imread(lt.imagePath, cv::IMREAD_COLOR);

			if (crop.empty())
			{
				++result.skippedReadFail;

				continue;
			}

			// Extract features
			TreeFeature tf;

			tf.treeId = lt.tp.treeId;
			tf.speciesCode = lt.speciesCode;
			tf.speciesName = lt.speciesName;
			tf.insideForest = lt.insideForest;
			tf.reliable = lt.reliable;
			tf.imageUsable = lt.imageUsable;
			tf.distToEdge = lt.distToEdge;
			tf.imagePath = lt.imagePath;
			tf.feature = computeFeature(crop, lt.tp);

			result.all.push_back(std::move(tf));

			if (((i + 1) % progressStep == 0) || (i == total - 1))
			{
				std::cout << "  " << (i + 1) << "/" << total << " (" << (100 * (i + 1) / total) << "%)\r" << std::flush;
			}
		}

		std::cout << "\n";

		// Categorize
		categorize(result);

		printSummary(result);

		return result;
	}

	// ─── Extract directly from aerial image (no disk I/O) ───
	static ExtractionResult extractFromAerialImage(const std::vector<LabeledTree>& labeledTrees, const AerialPhoto& aerialImg)
	{
		std::cout << "=== Feature Extraction (from aerial image) ===\n";

		int total = (int)labeledTrees.size();
		int progressStep = std::max(1, total / 20);

		ExtractionResult result;
		result.all.reserve(total);

		for (int i = 0 ; i < total ; ++i)
		{
			auto& lt = labeledTrees[i];
		
			// Skip unusable
			if (!lt.cropValid)
			{
				++result.skippedNoCrop;

				continue;
			}

			if (!lt.imageUsable)
			{
				++result.skippedEmpty;

				continue;
			}

			// Crop directly from aerial
			cv::Rect roi(lt.cropX, lt.cropY, lt.cropSize, lt.cropSize);

			if ((roi.x < 0) || (roi.y < 0) || (roi.x + roi.width > aerialImg.image.cols) || (roi.y + roi.height > aerialImg.image.rows))
			{
				++result.skippedNoCrop;

				continue;
			}

			cv::Mat crop = aerialImg.image(roi);

			// Extract features
			TreeFeature tf;
			tf.treeId = lt.tp.treeId;
			tf.speciesCode = lt.speciesCode;
			tf.speciesName = lt.speciesName;
			tf.insideForest = lt.insideForest;
			tf.reliable = lt.reliable;
			tf.imageUsable = lt.imageUsable;
			tf.distToEdge = lt.distToEdge;
			tf.imagePath = lt.imagePath;
			tf.feature = computeFeature(crop, lt.tp);

			result.all.push_back(std::move(tf));

			if (((i + 1) % progressStep == 0) || (i == total - 1))
			{
				std::cout << "  " << (i + 1) << "/" << total << " (" << (100 * (i + 1) / total) << "%)\r" << std::flush;
			}
		}

		std::cout << "\n";

		// Categorize
		categorize(result);

		printSummary(result);

		return result;
	}

	// ─── Load features from CSV (resume from step 3) ───
	static ExtractionResult loadCSV(const std::string& path)
	{
		std::cout << "=== Loading features from CSV ===\n";
		std::cout << "  File: " << path << "\n";

		std::ifstream file(path);

		if (!file.is_open())
		{
			throw std::runtime_error("Failed to open feature CSV: " + path);
		}

		ExtractionResult result;
		std::string line;

		// Read header
		if (!std::getline(file, line))
		{
			throw std::runtime_error("Feature CSV is empty");
		}

		// Parse header to find feature column start index
		// Format: TreeID,InsideForest,Reliable,DistToEdge,ImagePath,SpeciesCode,SpeciesName,B_mean,...

		int metaCols = 7;		// columns before feature values

		auto& featNames = TreeFeature::featureNames();
		int nFeats = (int)featNames.size();

		int rowCount = 0;

		while (std::getline(file, line))
		{
			if (line.empty())
			{
				continue;
			}

			// Remove trailing \r
			if (!line.empty() && line.back() == '\r')
			{
				line.pop_back();
			}

			// Split by comma
			std::vector<std::string> cols;
			std::stringstream ss(line);
			std::string cell;

			while (std::getline(ss, cell, ','))
			{
				cols.push_back(cell);
			}

			if ((int)cols.size() < metaCols + nFeats)
			{
				continue;
			}

			TreeFeature tf;

			// Meta columns
			try { tf.treeId = std::stoll(cols[0]); }
			catch (...) { tf.treeId = rowCount; }
			tf.insideForest = (cols[1] == "Y");
			tf.reliable = (cols[2] == "Y");
			try { tf.distToEdge = std::stod(cols[3]); }
			catch (...) { tf.distToEdge = 0.0; }
			tf.imagePath = cols[4];
			tf.speciesCode = cols[5];
			tf.speciesName = cols[6];
			tf.imageUsable = true;

			// Feature columns
			tf.feature.resize(nFeats);
			bool parseOk = true;

			for (int fi = 0; fi < nFeats; ++fi)
			{
				try
				{
					tf.feature[fi] = std::stof(cols[metaCols + fi]);
				}
				catch (...)
				{
					parseOk = false;
					break;
				}
			}

			if (parseOk)
			{
				result.all.push_back(std::move(tf));
				++rowCount;
			}
		}

		file.close();

		// Categorize
		categorize(result);

		std::cout << "  Loaded: " << result.all.size() << " trees\n";
		std::cout << "  Train:  " << result.train.size() << "\n";
		std::cout << "  Predict: " << result.predict.size() << "\n";

		return result;
	}

	// ─── Save features to CSV ───
	static void saveCSV(const ExtractionResult& result, const std::string& outPath)
	{
		std::ofstream out(outPath);

		if (!out.is_open())
		{
			std::cerr << "  WARNING: Failed to create " << outPath << "\n";

			return;
		}

		// Header
		auto& names = TreeFeature::featureNames();

		out << "TreeID,InsideForest,Reliable,DistToEdge,ImagePath,SpeciesCode,SpeciesName";

		for (auto& n : names)
		{
			out << "," << n;
		}

		out << "\n";

		// Data

		for (auto& tf : result.all)
		{
			out << tf.treeId << ","
				<< (tf.insideForest ? "Y" : "N") << ","
				<< (tf.reliable ? "Y" : "N") << ","
				<< std::fixed << std::setprecision(2) << tf.distToEdge << ","
				<< tf.imagePath << ","
				<< tf.speciesCode << ","
				<< tf.speciesName;

			out << std::fixed;

			for (size_t i = 0 ; i < tf.feature.size() ; ++i)
			{
				// LiDAR features (17,18,19) with 2 decimals, others with 4
				if (i >= 17)
				{
					out << "," << std::setprecision(2) << tf.feature[i];
				}
				else
				{
					out << "," << std::setprecision(4) << tf.feature[i];
				}
			}

			out << "\n";
		}

		out.close();

		std::cout << "  Feature CSV saved: " << outPath << " (" << result.all.size() << " rows)\n";
	}

	// ─── Print feature importance summary ───
	static void printFeatureStats(const ExtractionResult& result)
	{
		if (result.train.empty())
		{
			return;
		}

		auto& names = TreeFeature::featureNames();
		int nFeat = (int)names.size();

		std::cout << "\n  [Feature statistics - training data]\n";
		std::cout << "  " << std::left << std::setw(12) << "Feature" << std::right << std::setw(10) << "Mean" << std::setw(10) << "Std" << std::setw(10) << "Min" << std::setw(10) << "Max" << "\n";
		std::cout << "  " << std::string(52, '-') << "\n";

		for (int fi = 0; fi < nFeat; ++fi)
		{
			double sum = 0, sqSum = 0;
			double minV = 1e30, maxV = -1e30;
			int n = 0;

			for (auto* tf : result.train)
			{
				if (fi < (int)tf->feature.size())
				{
					float v = tf->feature[fi];

					sum += v;
					sqSum += v * v;
					minV = std::min(minV, (double)v);
					maxV = std::max(maxV, (double)v);

					++n;
				}
			}

			if (n > 0)
			{
				double mean = sum / n;
				double std = std::sqrt(sqSum / n - mean * mean);
			
				std::cout << "  " << std::left << std::setw(12) << names[fi]
					<< std::right << std::fixed
					<< std::setw(10) << std::setprecision(2) << mean
					<< std::setw(10) << std << std::setprecision(2)
					<< std::setw(10) << minV
					<< std::setw(10) << maxV << "\n";
			}
		}

		// Per-species mean for key features
		if (result.train.size() > 10)
		{
			std::cout << "\n  [Per-species mean (top features)]\n";
		
			std::map<std::string, std::vector<double>> specSums;
			std::map<std::string, int> specCounts;

			for (auto* tf : result.train)
			{
				specSums[tf->speciesCode].resize(nFeat, 0.0);
			
				for (int fi = 0; fi < nFeat && fi < (int)tf->feature.size(); ++fi)
				{
					specSums[tf->speciesCode][fi] += tf->feature[fi];
				}

				specCounts[tf->speciesCode]++;
			}

			// Print G_mean, H_mean, S_mean, Height, CrownD per species
			int keyFeats[] = { 1, 6, 7, 17, 18 };  // G_mean, H_mean, S_mean, Height, CrownD
			std::cout << "  " << std::left << std::setw(12) << "Species" << std::setw(6) << "N";

			for (int ki : keyFeats)
			{
				std::cout << std::setw(10) << names[ki];
			}

			std::cout << "\n";
			std::cout << "  " << std::string(58, '-') << "\n";

			// Sort by count
			std::vector<std::pair<std::string, int>> sorted(specCounts.begin(), specCounts.end());
			std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });

			for (auto& kv : sorted)
			{
				std::cout << "  " << std::left << std::setw(12) << kv.first << std::setw(6) << kv.second;

				for (int ki : keyFeats)
				{
					double mean = specSums[kv.first][ki] / kv.second;
					std::cout << std::setw(10) << std::setprecision(1) << mean;
				}

				std::cout << "\n";
			}
		}

		std::cout << std::setprecision(6);
	}

private:

	// ─── Feature computation (20 dimensions) ───

	static std::vector<float> computeFeature(const cv::Mat& crop, const TreePoint& tp)
	{
		std::vector<float> f;
		f.reserve(20);

		// BGR 통계
		// [0-5] BGR mean & stddev

		cv::Scalar mean;
		cv::Scalar stddev;

		cv::meanStdDev(crop, mean, stddev);

		f.push_back((float)mean[0]);		// B_mean
		f.push_back((float)mean[1]);		// G_mean
		f.push_back((float)mean[2]);		// R_mean
		f.push_back((float)stddev[0]);		// B_std
		f.push_back((float)stddev[1]);		// G_std
		f.push_back((float)stddev[2]);		// R_std

		// [6-8] HSV mean

		cv::Mat hsv;
		cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);

		cv::Scalar hsvMean = cv::mean(hsv);

		f.push_back((float)hsvMean[0]);		// Hue_mean
		f.push_back((float)hsvMean[1]);		// Saturation_mean
		f.push_back((float)hsvMean[2]);		// Value_mean

		// Grayscale for texture features

		cv::Mat gray;
		cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);

		cv::Mat grayF;
		gray.convertTo(grayF, CV_32F);

		// [9] Texture contrast (Sobel gradient magnitude mean)
		// 대조도: 인접 픽셀 차의 제곱 평균

		cv::Mat dx;
		cv::Mat dy;

		cv::Sobel(grayF, dx, CV_32F, 1, 0);
		cv::Sobel(grayF, dy, CV_32F, 0, 1);

		cv::Mat mag2;
		cv::magnitude(dx, dy, mag2);

		cv::Scalar contract = cv::mean(mag2);
		f.push_back((float)contract[0]);				// Contrast

		// [10] Texture entropy (grayscale histogram)
		// 균일성: 히스토그램 엔트로피

		cv::Mat grayU8;
		grayF.convertTo(grayU8, CV_8U);
		int histSize = 32;
		float range[] = { 0 , 256 };
		const float* histRange = { range };

		cv::Mat hist;
		cv::calcHist(&grayU8, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange);

		hist /= (float)(crop.rows * crop.cols);
		
		float entropy = 0.0f;

		for (int i = 0; i < histSize; ++i)
		{
			float p = hist.at<float>(i);

			if (p > 1e-6)
			{
				entropy -= p * std::log2(p);
			}
		}

		f.push_back(entropy);								// Entropy	

		// [11-12] LBP (Local Binary Pattern) - mean & stddev

		cv::Mat lbpImg = computeLBP(gray);
		
		cv::Scalar lbpMean;
		cv::Scalar lbpStd;

		cv::meanStdDev(lbpImg, lbpMean, lbpStd);

		f.push_back((float)lbpMean[0]);				// LBP_mean
		f.push_back((float)lbpStd[0]);					// LBP_std

		// [13-16] GLCM features

		float glcmEnergy;
		float glcmCorr;
		float glcmHomo;
		float glcmDissim;
		computeGLCM(gray, glcmEnergy, glcmCorr, glcmHomo, glcmDissim);

		f.push_back(glcmEnergy);						// GLCM_energy
		f.push_back(glcmCorr);							// GLCM_correlation
		f.push_back(glcmHomo);							// GLCM_homogeneity
		f.push_back(glcmDissim);						// GLCM_dissimilarity

		// [17-19] LiDAR structural features

		f.push_back((float)tp.height);																		// Height
		f.push_back((float)tp.crownD);																	// CrownD
		f.push_back((tp.crownD > 0.0) ? (float)(tp.height / tp.crownD) : 0.0f);		// H/CrownD ratio

		return f;		// 총 14차원
	}

	// ─── LBP computation ───
	// 8-neighbor Local Binary Pattern
	static cv::Mat computeLBP(const cv::Mat& gray)
	{
		int rows = gray.rows;
		int cols = gray.cols;

		// Need at least 3x3 for LBP
		if ((rows < 3) || (cols < 3))
		{
			return cv::Mat::zeros(1, 1, CV_8UC1);
		}

		cv::Mat lbp = cv::Mat::zeros(rows - 2, cols - 2, CV_8UC1);

		for (int y = 1; y < rows - 1; ++y)
		{
			const uint8_t* prevRow = gray.ptr<uint8_t>(y - 1);
			const uint8_t* currRow = gray.ptr<uint8_t>(y);
			const uint8_t* nextRow = gray.ptr<uint8_t>(y + 1);
		
			uint8_t* lbpRow = lbp.ptr<uint8_t>(y - 1);

			for (int x = 1; x < cols - 1; ++x)
			{
				uint8_t center = currRow[x];
				uint8_t code = 0;

				code |= (prevRow[x] >= center) << 0;
				code |= (prevRow[x + 1] >= center) << 1;
				code |= (currRow[x + 1] >= center) << 2;
				code |= (nextRow[x + 1] >= center) << 3;
				code |= (nextRow[x] >= center) << 4;
				code |= (nextRow[x - 1] >= center) << 5;
				code |= (currRow[x - 1] >= center) << 6;
				code |= (prevRow[x - 1] >= center) << 7;
				
				lbpRow[x - 1] = code;
			}
		}

		return lbp;
	}

	// ─── GLCM computation ───
	// Gray-Level Co-occurrence Matrix at distance=1, angle=0 (horizontal)
	static void computeGLCM(const cv::Mat& gray, float& energy, float& correlation, float& homogeneity, float& dissimilarity)
	{
		const int levels = 16;
		cv::Mat quantized;
		gray.convertTo(quantized, CV_8U, (levels - 1) / 255.0);

		// Build symmetric GLCM
		cv::Mat glcm = cv::Mat::zeros(levels, levels, CV_32F);
		int count = 0;

		for (int y = 0 ; y < quantized.rows ; ++y)
		{
			for (int x = 0 ; x < quantized.cols - 1 ; ++x)
			{
				int i = quantized.at<uint8_t>(y, x);
				int j = quantized.at<uint8_t>(y, x + 1);

				if ((i < levels) && (j < levels))
				{
					glcm.at<float>(i, j) += 1.0f;
					glcm.at<float>(j, i) += 1.0f;

					count += 2;
				}
			}
		}

		if (count > 0)
		{
			glcm /= (float)count;
		}

		// Marginal means and stds
		float muI = 0.0f;
		float muJ = 0.0f;
		float sigI = 0.0f;
		float sigJ = 0.0f;

		for (int i = 0 ; i < levels ; ++i)
		{
			float sumRow = 0.0f;
			float sumCol = 0.0f;

			for (int j = 0 ; j < levels ; ++j)
			{
				sumRow += glcm.at<float>(i, j);
				sumCol += glcm.at<float>(j, i);
			}

			muI += i * sumRow;
			muJ += i * sumCol;
		}

		for (int i = 0; i < levels; ++i)
		{
			float sumRow = 0.0f;
			float sumCol = 0.0f;

			for (int j = 0; j < levels; ++j)
			{
				sumRow += glcm.at<float>(i, j);
				sumCol += glcm.at<float>(j, i);
			}

			sigI += (i - muI) * (i - muI) * sumRow;
			sigJ += (i - muJ) * (i - muJ) * sumCol;
		}

		sigI = std::sqrt(sigI);
		sigJ = std::sqrt(sigJ);

		// Compute features
		energy = 0.0f;
		correlation = 0.0f;
		homogeneity = 0.0f;
		dissimilarity = 0.0f;

		for (int i = 0 ; i < levels ; ++i)
		{
			for (int j = 0 ; j < levels ; ++j)
			{
				float p = glcm.at<float>(i, j);

				if (p < 1e-10f)
				{
					continue;
				}

				energy += p* p;
				homogeneity += p / (1.0f + std::abs(i - j));
				dissimilarity += p * std::abs(i - j);

				if ((sigI > 1e-6f) && (sigJ > 1e-6f))
				{
					correlation += p * (i - muI) * (j - muJ) / (sigI * sigJ);
				}
			}
		}
	}


	// ─── Categorize into train/predict ───
	static void categorize(ExtractionResult& result)
	{
		for (auto& tf : result.all)
		{
			if (tf.insideForest && tf.reliable && !tf.speciesCode.empty())
			{
				result.train.push_back(&tf);
			}
			else if (!tf.insideForest)
			{
				result.predict.push_back(&tf);
			}
		}
	}

	// ─── Print summary ───
	static void printSummary(const ExtractionResult& result)
	{
		std::cout << "\n  === Feature Extraction Summary ===\n";
		std::cout << "    Total extracted : " << result.all.size() << "\n";
		std::cout << "    Train (reliable): " << result.train.size() << "\n";
		std::cout << "    Predict         : " << result.predict.size() << "\n";
		std::cout << "    Skipped (empty) : " << result.skippedEmpty << "\n";
		std::cout << "    Skipped (no crop): " << result.skippedNoCrop << "\n";
	
		if (result.skippedReadFail > 0)
		{
			std::cout << "    Skipped (read fail): " << result.skippedReadFail << "\n";
		}

		// Species distribution in training set
		if (!result.train.empty())
		{
			std::map<std::string, int> specCount;
		
			for (auto* tf : result.train)
			{
				specCount[tf->speciesCode]++;
			}

			std::cout << "    Training species: " << specCount.size() << "\n";
			
			std::vector<std::pair<std::string, int>> sorted(specCount.begin(), specCount.end());
			std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
			
			for (auto& kv : sorted)
			{
				std::cout << "      " << std::setw(12) << std::left << kv.first << " : " << kv.second << "\n";
			}
		}
	}
};