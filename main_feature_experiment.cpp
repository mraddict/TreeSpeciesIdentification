
// ----------------------------------------------------------------------------------------------------------------------------
//  main_feature_experiment.cpp
//  Handcrafted(25) vs DINOv2(+LiDAR) feature comparison
//  with leave-one-course-out evaluation.
//
//  Two phases, so the slow part (embedding 230k crops) runs once:
//
//    Phase 1  extract:
//      feature_experiment extract <labeled_input> <out_dir> <dino.onnx>
//        → <out_dir>/features_handcrafted.csv
//        → <out_dir>/features_dino.csv
//
//    Phase 2  evaluate (fast, repeatable):
//      feature_experiment eval <out_dir>
//        → LOCO + random 5-fold for both CSVs, side-by-side comparison
//
//  <labeled_input> is whatever your current pipeline uses to produce the
//  std::vector<LabeledTree> for all 6 courses merged (TreeID offsets applied).
//  Adapt loadAllLabeledTrees() below to your existing loader calls.
// ----------------------------------------------------------------------------------------------------------------------------

#include "TSICommon.h"
#include "OutputLogger.h"
#include "TreeLabeler.h"
#include "FeatureExtractor.h"
#include "Hierarchicalclassifier.h"
#include "CourseCrossValidator.h"
#include "CourseDataSetLoader.h"
#include "OverlayRenderer.h"
#include "TreeEditor.h"

#include <unordered_map>
#include <unordered_set>

// Adapt this to your pipeline 
//  Must return LabeledTree for ALL courses merged, with cropValid/imageUsable/
//  imagePath filled (i.e. crops already saved to disk) and TreeID offsets applied.
static std::vector<LabeledTree> loadAllLabeledTrees(const std::string& input)
{
	// e.g.
	//   auto cfg = CourseDataSetLoader::load(input);
	//   std::vector<LabeledTree> all;
	//   for (auto& course : cfg.courses) { ... TreeLabeler::label(...) ... append }
	//   return all;
	throw std::runtime_error("loadAllLabeledTrees(): plug in your existing loader here (" + input + ")");
}

// Optional: species code → L1 group for group-level accuracy 
//  Return "C" (conifer) / "B" (broadleaf) / "O" (other) by KOFTR_GROU code.
//  If you already have this mapping in ForestCommon / HierarchicalClassifier, call that instead.
static std::string groupOf(const std::string& code)
{
	return SpeciesCodes::group(code);		// "C" / "D" / "E" / "M" / "N"
}

static int runExtract(const std::string& input, const std::string& outDir, const std::string& onnxPath)
{
	return -1;
}

int runExtract(std::vector<LabeledTree>& labeled, const std::string& outDir, const std::string& onnxPath)
{
	TSICommon::mkdirs(outDir);

	std::cout << "Labeled trees: " << labeled.size() << "\n";

	// A) Handcrafted 25-dim (legacy)
	{
		FeatureConfig cfg;
		cfg.mode = FeatureMode::Handcrafted;

		FeatureExtractor::configure(cfg);

		auto feats = FeatureExtractor::extractFromFiles(labeled);
		FeatureExtractor::saveCSV(feats, TSICommon::join(outDir, "features_handcrafted.csv"));
	}

	// B) DINOv2 + LiDAR 8
	{
		FeatureConfig cfg;
		cfg.mode = FeatureMode::Dino;
		cfg.dino.modelPath = onnxPath;
		cfg.dino.batchSize = 64;
		cfg.dino.intraOpThreads = 0;			// ORT default = all cores
		//cfg.dino.useCUDA = true;				// if built against onnxruntime-gpu

		FeatureExtractor::configure(cfg);

		auto feats = FeatureExtractor::extractFromFiles(labeled);
		FeatureExtractor::saveCSV(feats, TSICommon::join(outDir, "features_dino.csv"));
	}

	return 0;
}

int runEval(const std::string& outDir)
{
	CourseCrossValidator::Options opts;
	opts.rf.maxDepth = 20;
	opts.rf.minSampleCount = 5;
	opts.rf.maxTrees = 200;
	opts.rf.activeVarCount = 0;						// sqrt(nFeats): 5 for 25-dim, ~20 for 392-dim
	opts.courseIdDivisor = 100000;				// TreeLoader offset step
	opts.groupOf = &groupOf;
	opts.runRandomKFold = true;
	opts.randomFolds = 5;

	// Handcrafted
	auto fa = FeatureExtractor::loadCSV(TSICommon::join(outDir, "features_handcrafted.csv"));
	CourseCrossValidator::Report ra = CourseCrossValidator::run(fa, opts);
	CourseCrossValidator::print(ra, "Handcrafted (17 image + 8 LiDAR)");

	// Dino
	auto fb = FeatureExtractor::loadCSV(TSICommon::join(outDir, "features_dino.csv"));
	CourseCrossValidator::Report rb = CourseCrossValidator::run(fb, opts);
	CourseCrossValidator::print(rb, "DINOv2 + 8 LiDAR");

	CourseCrossValidator::compare(ra, "Handcrafted", rb, "Dino");

	return 0;
}

int runSweep(const std::string& outDir)
{
	auto feats = FeatureExtractor::loadCSV(TSICommon::join(outDir, "features_dino.csv"));

	CourseCrossValidator::Options base;
	base.rf.maxDepth = 20;
	base.rf.minSampleCount = 5;
	base.rf.maxTrees = 200;
	base.rf.activeVarCount = 0;						// sqrt(nFeats): 5 for 25-dim, ~20 for 392-dim
	base.courseIdDivisor = 100000;				// TreeLoader offset step
	base.groupOf = &groupOf;
	base.runRandomKFold = true;

	std::vector<std::pair<std::string, CourseCrossValidator::Options>> variants;

	// 0. baseline (same as eval)
	variants.push_back({ "A baseline", base });

	// 1. drop Mixed Forest
	{
		auto opts = base;
		opts.excludeCodes = { "77" };
		variants.push_back({ "B - 77", opts });
	}

	// 2. class priors
	{
		auto otps = base;
		otps.excludeCodes = { "77" };
		otps.classPriors = true;
		variants.push_back({ "C -77 priors", otps });
	}

	// 3. PCA 64
	{
		auto otps = base;
		otps.excludeCodes = { "77" };
		otps.pcaDims = 64;
		variants.push_back({ "D -77 pca64", otps });
	}

	// 4. PCA 64 + priors
	{
		auto otps = base;
		otps.excludeCodes = { "77" };
		otps.pcaDims = 64;
		otps.classPriors = true;
		variants.push_back({ "E -77 pca64 priors", otps });
	}

	// 5. logistic regression on PCA 64
	{
		auto otps = base;
		otps.excludeCodes = { "77" };
		otps.pcaDims = 64;
		otps.classifier = CourseCrossValidator::Classifier::Logistic;
		variants.push_back({ "F -77 pca64 logistic", otps });
	}

	// 6. GROUP target (C/B) the ceiling for "asset group + confidence"
	{
		auto otps = base;
		otps.target = CourseCrossValidator::Target::Group;
		otps.excludeCodes = { "77" };
		variants.push_back({ "G group -77", otps });
	}

	// 7. GROUP target + PCA 64 + priors
	{
		auto otps = base;
		otps.target = CourseCrossValidator::Target::Group;
		otps.excludeCodes = { "77" };
		otps.pcaDims = 64;
		otps.classPriors = true;
		variants.push_back({ "H group -77 pca64 priors", otps });
	}

	std::vector<CourseCrossValidator::Report> reports;

	for (auto& v : variants)
	{
		std::cout << "\n############ " << v.first << " ############\n";

		CourseCrossValidator::Report r = CourseCrossValidator::run(feats, v.second);
		r.config = v.first + "  [" + r.config + "]";
		CourseCrossValidator::print(r);

		reports.push_back(std::move(r));
	}

	CourseCrossValidator::printTable(reports);

	return 0;
}

int runLidar(const std::string& outDir)
{
	auto feats = FeatureExtractor::loadCSV(TSICommon::join(outDir, "features_dino.csv"));

	CourseCrossValidator::Options base;
	base.rf.maxDepth = 20;
	base.rf.minSampleCount = 5;
	base.rf.maxTrees = 200;
	base.rf.activeVarCount = 0;						// sqrt(nFeats): 5 for 25-dim, ~20 for 392-dim
	base.courseIdDivisor = 100000;				// TreeLoader offset step
	base.groupOf = &groupOf;
	base.lidarCount = TreeFeature::kLidarCount;
	base.runRandomKFold = true;

	std::vector<std::pair<std::string, CourseCrossValidator::Options>> variants;

	{
		auto otps = base;
		otps.target = CourseCrossValidator::Target::Group;
		otps.featureSet = CourseCrossValidator::FeatureSet::LidarOnly;
		otps.rf.activeVarCount = 3;		// sqrt(8)
		variants.push_back({ "I group lidar-only", otps });
	}

	{
		auto otps = base;
		otps.target = CourseCrossValidator::Target::Group;
		otps.featureSet = CourseCrossValidator::FeatureSet::PcaImagePlusLidar;
		otps.pcaDims = 32;
		variants.push_back({ "J group pca32+lidar", otps });
	}

	{
		auto otps = base;
		otps.target = CourseCrossValidator::Target::Group;
		otps.featureSet = CourseCrossValidator::FeatureSet::PcaImagePlusLidar;
		otps.pcaDims = 32;
		otps.classPriors = true;
		variants.push_back({ "K group pca32+lidar priors", otps });
	}

	{
		auto otps = base;
		otps.featureSet = CourseCrossValidator::FeatureSet::LidarOnly;
		otps.rf.activeVarCount = 3;
		variants.push_back({ "L species lidar-only", otps });
	}

	std::vector<CourseCrossValidator::Report> reports;

	for (auto& v : variants)
	{
		std::cout << "\n############ " << v.first << " ############\n";

		CourseCrossValidator::Report r = CourseCrossValidator::run(feats, v.second);
		r.config = v.first + "  [" + r.config + "]";
		CourseCrossValidator::print(r);
		reports.push_back(std::move(r));
	}

	CourseCrossValidator::printTable(reports);

	return 0;
}

// ----------------------------------------------------------------------------------------------------------------------------
//  export : trees inside the course -> <out_dir>/<course>/trees.json
//    id (TreeID in tree_info CSV), x, y (EPSG:5179), height, crownD,
//    embedding[384] (from <out_dir>/features_dino.csv), suggested = null, confidence = 0
//  Course interior:
//    courseBoundary set   : inside the boundary SHP polygons
//    courseBoundary unset : InsideForest == N in features_dino.csv (outside every forest polygon)
//  Trees without an embedding row (crop failed / empty) are skipped and counted.
// ----------------------------------------------------------------------------------------------------------------------------

static const int kExportEpsg = 5179;
static const int kExportDinoDim = 384;

static bool insideBoundary(const ForestLayer& boundary, double x, double y)
{
	OGRPoint pt(x, y);

	for (auto& ft : boundary.features)
	{
		if (!ft.geometry)
		{
			continue;
		}

		OGREnvelope env;
		ft.geometry->getEnvelope(&env);

		if ((x < env.MinX) || (x > env.MaxX) || (y < env.MinY) || (y > env.MaxY))
		{
			continue;
		}

		if (ft.geometry->Contains(&pt))
		{
			return true;
		}
	}

	return false;
}

// dino_0 .. dino_383 of the wanted TreeIDs (global, offset applied), kept as the CSV text
//  outsideForestOnly : keep only rows with InsideForest == N
static std::unordered_map<long long, std::vector<std::string>> loadDinoEmbeddings(const std::string& path, const std::unordered_set<long long>& wanted, const std::string& courseName, bool outsideForestOnly, int& wrongCourse)
{
	std::ifstream file(path);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open feature CSV: " + path);
	}

	std::string line;

	if (!std::getline(file, line))
	{
		throw std::runtime_error("Feature CSV is empty: " + path);
	}

	if (!line.empty() && line.back() == '\r')
	{
		line.pop_back();
	}

	std::vector<std::string> header = TSICommon::split(line);

	int first = -1;
	int dim = 0;

	for (int i = 0 ; i < (int)header.size() ; ++i)
	{
		if (header[i] == "dino_0")
		{
			first = i;

			while ((first + dim < (int)header.size()) && (header[first + dim] == "dino_" + std::to_string(dim)))
			{
				++dim;
			}

			break;
		}
	}

	if (dim != kExportDinoDim)
	{
		throw std::runtime_error("Expected " + std::to_string(kExportDinoDim) + " dino_* columns, found " + std::to_string(dim) + ": " + path);
	}

	const int kInsideForestCol = 1;
	const int kImagePathCol = 4;
	const std::string courseDir = "/" + courseName + "/";

	std::unordered_map<long long, std::vector<std::string>> result;
	wrongCourse = 0;

	while (std::getline(file, line))
	{
		size_t comma = line.find(',');

		if (comma == std::string::npos)
		{
			continue;
		}

		long long id = -1;

		try { id = std::stoll(line.substr(0, comma)); }
		catch (...) { continue; }

		if (!wanted.count(id))
		{
			continue;
		}

		if (!line.empty() && line.back() == '\r')
		{
			line.pop_back();
		}

		std::vector<std::string> cols = TSICommon::split(line);

		if ((int)cols.size() < first + dim)
		{
			continue;
		}

		// TreeID offsets follow the course order in the config; the crop path tells which course the row really is
		if (cols[kImagePathCol].find(courseDir) == std::string::npos)
		{
			++wrongCourse;

			continue;
		}

		if (outsideForestOnly && (cols[kInsideForestCol] != "N"))
		{
			continue;
		}

		std::vector<std::string> emb(cols.begin() + first, cols.begin() + first + dim);
		bool ok = true;

		for (auto& v : emb)
		{
			try
			{
				if (!std::isfinite(std::stof(v)))
				{
					ok = false;
				}
			}
			catch (...)
			{
				ok = false;
			}

			if (!ok)
			{
				break;
			}
		}

		if (ok)
		{
			result[id] = std::move(emb);
		}
	}

	return result;
}

static const CourseDataSet& findCourse(const CourseDataSetConfig& cfg, const std::string& courseName)
{
	for (auto& c : cfg.courses)
	{
		if (c.courseName == courseName)
		{
			return c;
		}
	}

	throw std::runtime_error("Course not found in config: " + courseName);
}

int runExport(const std::string& configPath, const std::string& outDir, const std::string& courseName)
{
	CourseDataSetConfig cfg = CourseDataSetConfigLoader::load(configPath);

	const CourseDataSet* course = &findCourse(cfg, courseName);

	bool useBoundary = !course->boundaryPath.empty();

	std::cout << "=== Export: " << courseName << " ===\n";

	if (!useBoundary)
	{
		std::cout << "  ! 'courseBoundary' is not set - trees with InsideForest == N are taken as the course interior\n";
	}

	// 1. Trees in EPSG:5179 (TreeID without course offset)
	TreeData trees = TreeLoader::load(course->treeInfoPath, course->treeSrcEPSG, kExportEpsg, 0.0, 0.0, course->treeExcludePath);

	// 2. Candidates: inside the boundary, or every tree (filtered by InsideForest in step 3)
	std::vector<const TreePoint*> inside;

	if (useBoundary)
	{
		ForestLayer boundary = ForestLoader::load(course->boundaryPath);

		int boundaryEpsg = boundary.epsg;

		if (boundaryEpsg <= 0)
		{
			std::cout << "  ! Boundary CRS unknown, assuming EPSG:" << kExportEpsg << "\n";

			boundaryEpsg = kExportEpsg;
		}

		OGRCoordinateTransformation* toBoundaryCT = nullptr;

		if (boundaryEpsg != kExportEpsg)
		{
			toBoundaryCT = CoordTransformer::createTransform(kExportEpsg, boundaryEpsg);

			if (!toBoundaryCT)
			{
				ForestLoader::release(boundary);

				throw std::runtime_error("Failed to create transform EPSG:" + std::to_string(kExportEpsg) + " -> EPSG:" + std::to_string(boundaryEpsg));
			}
		}

		for (auto& tp : trees.tps)
		{
			double bx = tp.x;
			double by = tp.y;

			if (toBoundaryCT)
			{
				toBoundaryCT->Transform(1, &bx, &by);
			}

			if (insideBoundary(boundary, bx, by))
			{
				inside.push_back(&tp);
			}
		}

		OGRCoordinateTransformation::DestroyCT(toBoundaryCT);
		ForestLoader::release(boundary);

		std::cout << "  Interior criterion: boundary SHP " << course->boundaryPath << " (EPSG:" << boundaryEpsg << ")\n";
		std::cout << "  Trees inside boundary: " << inside.size() << " / " << trees.tps.size() << "\n";
	}
	else
	{
		for (auto& tp : trees.tps)
		{
			inside.push_back(&tp);
		}

		std::cout << "  Interior criterion: InsideForest == N in features_dino.csv (no courseBoundary)\n";
	}

	// 3. Embeddings
	std::unordered_set<long long> wanted;

	for (auto* tp : inside)
	{
		wanted.insert(tp->treeId + course->treeIdOffset);
	}

	int wrongCourse = 0;
	auto embeddings = loadDinoEmbeddings(TSICommon::join(outDir, "features_dino.csv"), wanted, courseName, !useBoundary, wrongCourse);

	if (wrongCourse > 0)
	{
		std::cout << "  ! " << wrongCourse << " embedding rows belong to another course (TreeID offset / course order mismatch) - ignored\n";
	}

	// 4. trees.json
	std::string courseDir = TSICommon::join(outDir, courseName);
	TSICommon::mkdirs(courseDir);

	std::string jsonPath = TSICommon::join(courseDir, "trees.json");
	std::ofstream ofs(jsonPath, std::ios::binary);

	if (!ofs.is_open())
	{
		throw std::runtime_error("Failed to create: " + jsonPath);
	}

	ofs << std::fixed << "[\n";

	int written = 0;
	int noEmbedding = 0;

	for (auto* tp : inside)
	{
		auto it = embeddings.find(tp->treeId + course->treeIdOffset);

		if (it == embeddings.end())
		{
			++noEmbedding;

			continue;
		}

		if (written > 0)
		{
			ofs << ",\n";
		}

		ofs << "  {\"id\": " << tp->treeId
			<< ", \"x\": " << std::setprecision(3) << tp->x
			<< ", \"y\": " << tp->y
			<< ", \"height\": " << std::setprecision(2) << tp->height
			<< ", \"crownD\": " << tp->crownD
			<< ", \"embedding\": [";

		for (size_t i = 0 ; i < it->second.size() ; ++i)
		{
			ofs << (i ? "," : "") << it->second[i];
		}

		ofs << "], \"suggested\": null, \"confidence\": 0}";

		++written;
	}

	ofs << "\n]\n";
	ofs.close();

	std::cout << "  Written: " << written << " trees -> " << jsonPath << "\n";

	if (useBoundary)
	{
		std::cout << "  Skipped (no embedding): " << noEmbedding << "\n";
	}
	else
	{
		std::cout << "  Not exported (InsideForest == Y or no embedding): " << noEmbedding << "\n";
	}

	return 0;
}

// ----------------------------------------------------------------------------------------------------------------------------
//  render : export result drawn on the aerial photo (OverlayRenderer drawing functions)
//    green  (2px) : trees in <out_dir>/<course>/trees.json
//    yellow (1px) : InsideForest == N but not in trees.json (dropped by export filters)
//    gray   (1px) : InsideForest == Y (forest), omitted with --skip-forest
//  Outputs in <out_dir>/<course>/ :
//    render_overview.jpg          1/4 scale, whole image
//    render_tile_1..3.jpg         2048x2048 at full resolution, densest trees.json areas
//  Geo -> pixel uses the aerial GeoTransform inverse, as TreeLabeler does for crops.
// ----------------------------------------------------------------------------------------------------------------------------

static const int kRenderTileSize = 2048;
static const int kRenderTileCount = 3;
static const int kRenderDensityCell = 256;
static const double kRenderOverviewScale = 0.25;

// trees.json as written by runExport (one tree per line)
static std::vector<TreePoint> loadExportedTrees(const std::string& path)
{
	std::ifstream file(path);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open: " + path + " (run export first)");
	}

	std::vector<TreePoint> result;
	std::string line;

	while (std::getline(file, line))
	{
		TreePoint tp;

		if (sscanf_s(line.c_str(), " {\"id\": %lld, \"x\": %lf, \"y\": %lf, \"height\": %lf, \"crownD\": %lf", &tp.treeId, &tp.x, &tp.y, &tp.height, &tp.crownD) == 5)
		{
			result.push_back(tp);
		}
	}

	if (result.empty())
	{
		throw std::runtime_error("No trees parsed from: " + path);
	}

	return result;
}

// TreeID (global, offset applied) -> InsideForest, for the course's rows in features_dino.csv
static std::unordered_map<long long, bool> loadInsideForestFlags(const std::string& path, const std::string& courseName)
{
	std::ifstream file(path);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open feature CSV: " + path);
	}

	const std::string courseDir = "/" + courseName + "/";

	std::unordered_map<long long, bool> result;
	std::string line;

	std::getline(file, line);		// header

	while (std::getline(file, line))
	{
		// only the first 5 columns are needed: TreeID,InsideForest,Reliable,DistToEdge,ImagePath
		size_t c[5];
		size_t pos = 0;
		bool ok = true;

		for (int i = 0 ; i < 5 ; ++i)
		{
			c[i] = line.find(',', pos);

			if (c[i] == std::string::npos)
			{
				ok = false;

				break;
			}

			pos = c[i] + 1;
		}

		if (!ok)
		{
			continue;
		}

		std::string imagePath = line.substr(c[3] + 1, c[4] - c[3] - 1);

		if (imagePath.find(courseDir) == std::string::npos)
		{
			continue;
		}

		long long id = -1;

		try { id = std::stoll(line.substr(0, c[0])); }
		catch (...) { continue; }

		result[id] = (line.substr(c[0] + 1, c[1] - c[0] - 1) == "Y");
	}

	return result;
}

// Up to `count` non-overlapping tile x tile windows with the most points (cell-aligned)
static std::vector<cv::Rect> densestTiles(const std::vector<cv::Point2f>& pts, int imgW, int imgH, int tile, int cell, int count)
{
	int gw = (imgW + cell - 1) / cell;
	int gh = (imgH + cell - 1) / cell;
	int k = tile / cell;

	cv::Mat grid = cv::Mat::zeros(gh, gw, CV_64F);

	for (auto& p : pts)
	{
		int gx = (int)p.x / cell;
		int gy = (int)p.y / cell;

		if ((p.x >= 0) && (p.y >= 0) && (gx < gw) && (gy < gh))
		{
			grid.at<double>(gy, gx) += 1.0;
		}
	}

	cv::Mat sum;
	cv::integral(grid, sum, CV_64F);

	int maxCx = std::max(0, gw - k);
	int maxCy = std::max(0, gh - k);

	std::vector<cv::Rect> chosenCells;
	std::vector<cv::Rect> tiles;

	for (int n = 0 ; n < count ; ++n)
	{
		double best = 0.0;
		cv::Rect bestCells;

		for (int cy = 0 ; cy <= maxCy ; ++cy)
		{
			for (int cx = 0 ; cx <= maxCx ; ++cx)
			{
				cv::Rect r(cx, cy, std::min(k, gw - cx), std::min(k, gh - cy));

				bool overlaps = false;

				for (auto& c : chosenCells)
				{
					if ((r & c).area() > 0)
					{
						overlaps = true;

						break;
					}
				}

				if (overlaps)
				{
					continue;
				}

				double s = sum.at<double>(r.y + r.height, r.x + r.width) - sum.at<double>(r.y, r.x + r.width) - sum.at<double>(r.y + r.height, r.x) + sum.at<double>(r.y, r.x);

				if (s > best)
				{
					best = s;
					bestCells = r;
				}
			}
		}

		if (best <= 0.0)
		{
			break;
		}

		chosenCells.push_back(bestCells);

		cv::Rect px(bestCells.x * cell, bestCells.y * cell, tile, tile);
		tiles.push_back(px & cv::Rect(0, 0, imgW, imgH));
	}

	return tiles;
}

static cv::Mat toBGR8(const cv::Mat& img)
{
	cv::Mat out;

	if (img.channels() == 4)
	{
		cv::cvtColor(img, out, cv::COLOR_BGRA2BGR);
	}
	else if (img.channels() == 1)
	{
		cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
	}
	else
	{
		out = img.clone();
	}

	if (out.depth() != CV_8U)
	{
		out.convertTo(out, CV_8U);
	}

	return out;
}

int runRender(const std::string& configPath, const std::string& outDir, const std::string& courseName, bool skipForest)
{
	CourseDataSetConfig cfg = CourseDataSetConfigLoader::load(configPath);
	const CourseDataSet& course = findCourse(cfg, courseName);

	std::string courseDir = TSICommon::join(outDir, courseName);

	std::cout << "=== Render: " << courseName << (skipForest ? " (--skip-forest)" : "") << " ===\n";

	// 1. Tree sets (EPSG:5179)
	TreeData exported;
	exported.tps = loadExportedTrees(TSICommon::join(courseDir, "trees.json"));

	std::unordered_set<long long> exportedIds;

	for (auto& tp : exported.tps)
	{
		exportedIds.insert(tp.treeId);
	}

	TreeData all = TreeLoader::load(course.treeInfoPath, course.treeSrcEPSG, kExportEpsg, 0.0, 0.0, course.treeExcludePath);
	auto insideForest = loadInsideForestFlags(TSICommon::join(outDir, "features_dino.csv"), courseName);

	TreeData dropped;		// InsideForest == N, not exported
	TreeData forest;		// InsideForest == Y
	int noRow = 0;

	for (auto& tp : all.tps)
	{
		if (exportedIds.count(tp.treeId))
		{
			continue;
		}

		auto it = insideForest.find(tp.treeId + course.treeIdOffset);

		if (it == insideForest.end())
		{
			++noRow;

			continue;
		}

		(it->second ? forest : dropped).tps.push_back(tp);
	}

	std::cout << "  trees.json: " << exported.tps.size() << ", InsideForest=N not exported: " << dropped.tps.size() << ", InsideForest=Y: " << forest.tps.size() << ", no feature row: " << noRow << "\n";

	// 2. Aerial photo + inverse GeoTransform
	AerialPhoto aerial = AerialLoader::load(course.aerialImgPath);

	if ((aerial.epsg > 0) && (aerial.epsg != kExportEpsg))
	{
		OGRCoordinateTransformation* ct = CoordTransformer::createTransform(kExportEpsg, aerial.epsg);

		if (!ct)
		{
			throw std::runtime_error("Failed to create transform EPSG:" + std::to_string(kExportEpsg) + " -> EPSG:" + std::to_string(aerial.epsg));
		}

		for (TreeData* td : { &exported, &dropped, &forest })
		{
			for (auto& tp : td->tps)
			{
				ct->Transform(1, &tp.x, &tp.y);
			}
		}

		OGRCoordinateTransformation::DestroyCT(ct);
	}

	double gt[6] = { aerial.originX, aerial.pixelW, aerial.rotationX, aerial.originY, aerial.rotationY, aerial.pixelH };
	double igt[6];

	if (!CoordTransformer::computeInverseGT(gt, igt))
	{
		throw std::runtime_error("Failed to compute inverse GeoTransform");
	}

	double pixPerMeter = std::abs(igt[1]);

	auto geoToPx = [&](double gx, double gy) -> cv::Point2f
	{
		return cv::Point2f((float)(igt[0] + igt[1] * gx + igt[2] * gy), (float)(igt[3] + igt[4] * gx + igt[5] * gy));
	};

	// 3. Styles
	OverlayOptions base;
	base.drawPolygons = false;
	base.crownAlpha = 0.0;				// draw straight onto the canvas
	base.showTreeId = false;
	base.drawCenter = true;

	OverlayOptions greenOpts = base;
	greenOpts.crownColor = greenOpts.centerColor = cv::Scalar(0, 220, 0);
	greenOpts.crownOutlineWidth = 2;
	greenOpts.centerRadius = 2;

	OverlayOptions yellowOpts = base;
	yellowOpts.crownColor = yellowOpts.centerColor = cv::Scalar(0, 230, 255);
	yellowOpts.crownOutlineWidth = 1;
	yellowOpts.centerRadius = 1;

	OverlayOptions grayOpts = base;
	grayOpts.crownColor = grayOpts.centerColor = cv::Scalar(160, 160, 160);
	grayOpts.crownOutlineWidth = 1;
	grayOpts.centerRadius = 1;

	OverlayOptions legendOpts = base;
	legendOpts.legendX = 20;
	legendOpts.legendY = 20;
	legendOpts.resultTitle = courseName + " export: " + std::to_string(exported.tps.size()) + " trees";

	std::map<std::string, cv::Scalar> legend =
	{
		{ "1 trees.json: " + std::to_string(exported.tps.size()), greenOpts.crownColor },
		{ "2 InsideForest=N, not exported: " + std::to_string(dropped.tps.size()), yellowOpts.crownColor },
		{ "3 InsideForest=Y (forest): " + (skipForest ? std::string("not drawn") : std::to_string(forest.tps.size())), grayOpts.crownColor },
	};

	auto drawAll = [&](cv::Mat& canvas, std::function<cv::Point2f(double, double)> toPx, double ppm)
	{
		if (!skipForest)
		{
			OverlayRenderer::drawTreesOn(canvas, forest, toPx, ppm, grayOpts);
		}

		OverlayRenderer::drawTreesOn(canvas, dropped, toPx, ppm, yellowOpts);
		OverlayRenderer::drawTreesOn(canvas, exported, toPx, ppm, greenOpts);
		OverlayRenderer::drawLegendOn(canvas, legend, legendOpts);
	};

	TSICommon::mkdirs(courseDir);

	// 4. Overview (1/4)
	{
		cv::Mat small;
		cv::resize(aerial.image, small, cv::Size(), kRenderOverviewScale, kRenderOverviewScale, cv::INTER_AREA);

		cv::Mat canvas = toBGR8(small);
		small.release();

		drawAll(canvas, [&](double gx, double gy) { return geoToPx(gx, gy) * (float)kRenderOverviewScale; }, pixPerMeter * kRenderOverviewScale);

		std::string path = TSICommon::join(courseDir, "render_overview.jpg");
		cv::imwrite(path, canvas);

		std::cout << "  Saved: " << path << " (" << canvas.cols << "x" << canvas.rows << ")\n";
	}

	// 5. Tiles at full resolution, densest trees.json areas
	std::vector<cv::Point2f> exportedPx;

	for (auto& tp : exported.tps)
	{
		exportedPx.push_back(geoToPx(tp.x, tp.y));
	}

	std::vector<cv::Rect> tiles = densestTiles(exportedPx, aerial.image.cols, aerial.image.rows, kRenderTileSize, kRenderDensityCell, kRenderTileCount);

	for (size_t i = 0 ; i < tiles.size() ; ++i)
	{
		const cv::Rect& r = tiles[i];

		cv::Mat canvas = toBGR8(aerial.image(r));

		drawAll(canvas, [&](double gx, double gy) { return geoToPx(gx, gy) - cv::Point2f((float)r.x, (float)r.y); }, pixPerMeter);

		int inTile = 0;

		for (auto& p : exportedPx)
		{
			if (r.contains(cv::Point((int)p.x, (int)p.y)))
			{
				++inTile;
			}
		}

		std::string path = TSICommon::join(courseDir, "render_tile_" + std::to_string(i + 1) + ".jpg");
		cv::imwrite(path, canvas);

		std::cout << "  Saved: " << path << " (" << r.width << "x" << r.height << " at px " << r.x << "," << r.y << ", trees.json trees: " << inTile << ")\n";
	}

	return 0;
}

// ----------------------------------------------------------------------------------------------------------------------------
//  edit : mark false detections in <course>_tree_info.csv on a 1/4 aerial photo (TreeEditor / ImageViewer)
//    S saves <tree_info>_excluded.csv (or config tree_exclude); TreeLoader::load / FeatureExtractor::loadCSV drop action == remove.
//    The tree_info CSV is never modified.
// ----------------------------------------------------------------------------------------------------------------------------

static const double kEditImageScale = 0.25;

int runEdit(const std::string& configPath, const std::string& courseName)
{
	CourseDataSetConfig cfg = CourseDataSetConfigLoader::load(configPath);
	const CourseDataSet& course = findCourse(cfg, courseName);

	std::string exclusionPath = TreeExclusion::resolvePath(course.treeInfoPath, course.treeExcludePath);

	std::cout << "=== Edit: " << courseName << " ===\n";
	std::cout << "  tree_info : " << course.treeInfoPath << "\n";
	std::cout << "  exclusion : " << exclusionPath << (TSICommon::exists(exclusionPath) ? "" : " (new)") << "\n";

	// every tree, TreeID without course offset, no exclusion applied
	TreeData trees = TreeLoader::load(course.treeInfoPath, course.treeSrcEPSG, kExportEpsg, 0.0, 0.0, TreeExclusion::kNone);
	AerialPhoto aerial = AerialLoader::load(course.aerialImgPath, kEditImageScale);

	if ((aerial.epsg > 0) && (aerial.epsg != kExportEpsg))
	{
		OGRCoordinateTransformation* ct = CoordTransformer::createTransform(kExportEpsg, aerial.epsg);

		if (!ct)
		{
			throw std::runtime_error("Failed to create transform EPSG:" + std::to_string(kExportEpsg) + " -> EPSG:" + std::to_string(aerial.epsg));
		}

		for (auto& tp : trees.tps)
		{
			ct->Transform(1, &tp.x, &tp.y);
		}

		OGRCoordinateTransformation::DestroyCT(ct);
	}

	TreeEditor::run(aerial, trees.tps, exclusionPath, "TSI Tree Editor - " + courseName);

	return 0;
}

// TreeSpeciesIdentification.exe                                    ← 기존 파이프라인
// TreeSpeciesIdentification.exe experiment extract <입력> out\ models\dinov2_vits14.onnx
// TreeSpeciesIdentification.exe experiment eval out\
// TreeSpeciesIdentification.exe --experiment export ..\..\Resources\courses_dataset.json out\ Hwarang
// TreeSpeciesIdentification.exe --experiment render ..\..\Resources\courses_dataset.json out\ Hwarang [--skip-forest]

int runFeatureExperiment(int argc, char** argv)
{
	std::string cmd = argv[1];
	std::string outDir = (cmd == "extract") ? argv[3] : argv[2];

	OutputLogger logger(TSICommon::join(outDir, "logs"), "feature_experiment_" + cmd);

	try
	{
		if (cmd == "extract")
		{
			if (argc < 4)
			{
				std::cerr << "extract needs <labeled_input> <out_dir> <dino.onnx>\n";

				return 1;
			}

			return runExtract(argv[2], argv[3], argv[4]);
		}

		if (cmd == "eval")
		{
			return runEval(argv[2]);
		}

		std::cerr << "unknown command: " << cmd << "\n";

		return 1;
	}
	catch (const std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << "\n";

		return 2;
	}

	return -1;
}