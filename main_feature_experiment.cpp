
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

struct FeatureRowInfo
{
	bool insideForest = false;
	std::string imagePath;			// crown crop saved by TreeLabeler
};

// TreeID (global, offset applied) -> InsideForest + ImagePath, for the course's rows in a features_*.csv
static std::unordered_map<long long, FeatureRowInfo> loadFeatureRows(const std::string& path, const std::string& courseName)
{
	std::ifstream file(path);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open feature CSV: " + path);
	}

	const std::string courseDir = "/" + courseName + "/";

	std::unordered_map<long long, FeatureRowInfo> result;
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

		FeatureRowInfo info;
		info.insideForest = (line.substr(c[0] + 1, c[1] - c[0] - 1) == "Y");
		info.imagePath = imagePath;

		result[id] = std::move(info);
	}

	return result;
}

// TreeID (global, offset applied) -> InsideForest, for the course's rows in features_dino.csv
static std::unordered_map<long long, bool> loadInsideForestFlags(const std::string& path, const std::string& courseName)
{
	std::unordered_map<long long, bool> result;

	for (auto& kv : loadFeatureRows(path, courseName))
	{
		result[kv.first] = kv.second.insideForest;
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
//  review : check the trees removed by <course>_tree_info_excluded.csv (action == remove)
//    <out>/<course>/review_overview.jpg            1/4 aerial, removed = red circle + X, kept = thin green circle
//    <out>/<course>/review_tile_01..06.jpg         full-resolution 1536 px tiles on the grid cells with the most removed trees
//    <out>/<course>/review_excluded_sheet_NN.jpg   96x96 crown thumbnails, 20 columns, 400 per sheet, TreeID below
//    <out>/<course>/review_kept_sheet.jpg          400 random kept trees, same layout
//    <out>/<course>/review_summary.txt             height / crown diameter, InsideForest ratio, top 10 grid cells
//  Grid cell = tile size (1536 full-resolution px). auto_forest trees count as kept.
//  Thumbnails use the TreeLabeler crop (ImagePath in features_dino.csv) if it exists, otherwise a crop of the aerial photo.
// ----------------------------------------------------------------------------------------------------------------------------

static const int kReviewTileSize = 1536;
static const int kReviewTileCount = 6;
static const int kReviewTopCells = 10;
static const double kReviewClusterRatio = 0.9;
static const double kReviewOverviewScale = 0.25;
static const int kReviewThumb = 96;
static const int kReviewThumbLabelH = 14;
static const int kReviewSheetCols = 20;
static const int kReviewSheetPerPage = 400;
static const unsigned kReviewKeptSeed = 42;

struct ReviewTree
{
	TreePoint tp;					// x, y in the aerial CRS
	double gx = 0.0;				// EPSG:5179
	double gy = 0.0;
	cv::Point2f px;					// full-resolution aerial pixel
	bool removed = false;
	int forest = -1;				// InsideForest: 1 = Y, 0 = N, -1 = no feature row
	std::string imagePath;
};

static std::string reviewStatsLine(const char* name, std::vector<double> v)
{
	char buf[200];

	if (v.empty())
	{
		sprintf_s(buf, "  %-8s n=0\n", name);

		return buf;
	}

	std::sort(v.begin(), v.end());

	size_t n = v.size();
	double median = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
	double mean = std::accumulate(v.begin(), v.end(), 0.0) / n;

	sprintf_s(buf, "  %-8s n=%-6zu min %6.2f   median %6.2f   max %6.2f   mean %6.2f\n", name, n, v.front(), median, v.back(), mean);

	return buf;
}

static std::string reviewPercent(int a, int b)
{
	char buf[32];
	sprintf_s(buf, "%.1f%%", (b > 0) ? 100.0 * a / b : 0.0);

	return buf;
}

// TreeLabeler crop if present, otherwise the crown (x1.4) cut from the full-resolution aerial photo
static cv::Mat reviewThumb(const ReviewTree& t, const cv::Mat& aerial, double pixPerMeter, bool& fromCrop)
{
	cv::Mat crop;

	if (!t.imagePath.empty())
	{
		crop = cv::imread(t.imagePath, cv::IMREAD_COLOR);
	}

	fromCrop = !crop.empty();

	if (crop.empty())
	{
		int half = (int)std::lround(std::max(8.0, t.tp.crownD * 0.5 * 1.4 * pixPerMeter));
		half = std::min(half, 256);

		cv::Rect r = cv::Rect((int)t.px.x - half, (int)t.px.y - half, half * 2, half * 2) & cv::Rect(0, 0, aerial.cols, aerial.rows);

		if (r.area() > 0)
		{
			crop = toBGR8(aerial(r));
		}
	}

	cv::Mat thumb(kReviewThumb, kReviewThumb, CV_8UC3, cv::Scalar(40, 40, 40));

	if (!crop.empty())
	{
		cv::resize(toBGR8(crop), thumb, thumb.size(), 0, 0, (crop.cols > kReviewThumb) ? cv::INTER_AREA : cv::INTER_LINEAR);
	}

	return thumb;
}

// 20-column sheets of 96x96 thumbnails with TreeID below. numbered: <stem>_01.jpg, _02.jpg ... otherwise <stem>.jpg (first page only)
static std::vector<std::string> saveContactSheets(const std::vector<const ReviewTree*>& trees, const cv::Mat& aerial, double pixPerMeter, const std::string& courseDir, const std::string& stem, bool numbered, int& fromCropCount)
{
	std::vector<std::string> paths;
	const int cellH = kReviewThumb + kReviewThumbLabelH;
	size_t pages = numbered ? (trees.size() + kReviewSheetPerPage - 1) / kReviewSheetPerPage : std::min<size_t>(1, trees.size());

	for (size_t page = 0 ; page < pages ; ++page)
	{
		size_t begin = page * kReviewSheetPerPage;
		size_t end = std::min(trees.size(), begin + kReviewSheetPerPage);
		int count = (int)(end - begin);
		int rows = (count + kReviewSheetCols - 1) / kReviewSheetCols;

		cv::Mat sheet(rows * cellH, kReviewSheetCols * kReviewThumb, CV_8UC3, cv::Scalar(0, 0, 0));

		for (int i = 0 ; i < count ; ++i)
		{
			const ReviewTree& t = *trees[begin + i];
			int x = (i % kReviewSheetCols) * kReviewThumb;
			int y = (i / kReviewSheetCols) * cellH;
			bool fromCrop = false;

			reviewThumb(t, aerial, pixPerMeter, fromCrop).copyTo(sheet(cv::Rect(x, y, kReviewThumb, kReviewThumb)));
			fromCropCount += fromCrop ? 1 : 0;

			// TreeID (yellow = cut from the aerial photo, no saved crop)
			cv::putText(sheet, std::to_string(t.tp.treeId), { x + 3, y + kReviewThumb + 11 }, cv::FONT_HERSHEY_SIMPLEX, 0.36, fromCrop ? cv::Scalar(230, 230, 230) : cv::Scalar(0, 230, 255), 1, cv::LINE_AA);
		}

		char name[64];

		if (numbered)
		{
			sprintf_s(name, "%s_%02d.jpg", stem.c_str(), (int)page + 1);
		}
		else
		{
			sprintf_s(name, "%s.jpg", stem.c_str());
		}

		std::string path = TSICommon::join(courseDir, name);
		cv::imwrite(path, sheet);
		paths.push_back(path);

		std::cout << "  Saved: " << path << " (" << count << " trees)\n";
	}

	return paths;
}

int runReview(const std::string& configPath, const std::string& outDir, const std::string& courseName)
{
	CourseDataSetConfig cfg = CourseDataSetConfigLoader::load(configPath);
	const CourseDataSet& course = findCourse(cfg, courseName);

	std::string courseDir = TSICommon::join(outDir, courseName);
	std::string exclusionPath = TreeExclusion::resolvePath(course.treeInfoPath, course.treeExcludePath);

	std::cout << "=== Review: " << courseName << " ===\n";

	// 1. Exclusion file + every tree (no exclusion applied), EPSG:5179, TreeID without offset
	auto entries = TreeExclusion::load(exclusionPath);
	std::unordered_set<long long> removedIds = TreeExclusion::removedIds(entries);

	if (removedIds.empty())
	{
		throw std::runtime_error("No remove rows in: " + exclusionPath);
	}

	int autoForest = 0;

	for (auto& kv : entries)
	{
		autoForest += (kv.second.action == TreeExclusion::kAutoForest) ? 1 : 0;
	}

	TreeData all = TreeLoader::load(course.treeInfoPath, course.treeSrcEPSG, kExportEpsg, 0.0, 0.0, TreeExclusion::kNone);

	// InsideForest + crop path from the feature CSV (rows of removed trees are still in the file)
	std::string featPath = TSICommon::join(outDir, "features_dino.csv");

	if (!TSICommon::exists(featPath))
	{
		featPath = TSICommon::join(outDir, "features_handcrafted.csv");
	}

	std::unordered_map<long long, FeatureRowInfo> rows;

	if (TSICommon::exists(featPath))
	{
		rows = loadFeatureRows(featPath, courseName);
	}
	else
	{
		std::cout << "  ! No features_*.csv in " << outDir << " - InsideForest unknown, thumbnails cut from the aerial photo\n";
		featPath = "(none)";
	}

	// 2. Aerial photo (full resolution) + CRS
	AerialPhoto aerial = AerialLoader::load(course.aerialImgPath);

	OGRCoordinateTransformation* toAerial = nullptr;
	OGRCoordinateTransformation* toExport = nullptr;

	if ((aerial.epsg > 0) && (aerial.epsg != kExportEpsg))
	{
		toAerial = CoordTransformer::createTransform(kExportEpsg, aerial.epsg);
		toExport = CoordTransformer::createTransform(aerial.epsg, kExportEpsg);

		if (!toAerial || !toExport)
		{
			throw std::runtime_error("Failed to create transform EPSG:" + std::to_string(kExportEpsg) + " <-> EPSG:" + std::to_string(aerial.epsg));
		}
	}

	double gt[6] = { aerial.originX, aerial.pixelW, aerial.rotationX, aerial.originY, aerial.rotationY, aerial.pixelH };
	double igt[6];

	if (!CoordTransformer::computeInverseGT(gt, igt))
	{
		throw std::runtime_error("Failed to compute inverse GeoTransform");
	}

	double pixPerMeter = std::abs(igt[1]);
	int W = aerial.image.cols;
	int H = aerial.image.rows;

	auto geoToPx = [&](double gx, double gy) -> cv::Point2f
	{
		return cv::Point2f((float)(igt[0] + igt[1] * gx + igt[2] * gy), (float)(igt[3] + igt[4] * gx + igt[5] * gy));
	};

	// full-resolution pixel -> EPSG:5179
	auto pxToExport = [&](double px, double py, double& ex, double& ey)
	{
		ex = gt[0] + gt[1] * px + gt[2] * py;
		ey = gt[3] + gt[4] * px + gt[5] * py;

		if (toExport)
		{
			toExport->Transform(1, &ex, &ey);
		}
	};

	std::vector<ReviewTree> trees;
	trees.reserve(all.tps.size());

	TreeData removedTD;
	TreeData keptTD;
	int removedInTreeInfo = 0;

	for (auto& tp : all.tps)
	{
		ReviewTree rt;
		rt.tp = tp;
		rt.gx = tp.x;
		rt.gy = tp.y;
		rt.removed = removedIds.count(tp.treeId) > 0;

		if (toAerial)
		{
			toAerial->Transform(1, &rt.tp.x, &rt.tp.y);
		}

		rt.px = geoToPx(rt.tp.x, rt.tp.y);

		auto it = rows.find(tp.treeId + course.treeIdOffset);

		if (it != rows.end())
		{
			rt.forest = it->second.insideForest ? 1 : 0;
			rt.imagePath = it->second.imagePath;
		}

		removedInTreeInfo += rt.removed ? 1 : 0;
		(rt.removed ? removedTD : keptTD).tps.push_back(rt.tp);
		trees.push_back(std::move(rt));
	}

	if (toAerial)
	{
		OGRCoordinateTransformation::DestroyCT(toAerial);
	}

	std::cout << "  tree_info: " << trees.size() << ", removed: " << removedTD.tps.size() << ", kept: " << keptTD.tps.size() << "\n";

	// 3. Grid (cell = tile size): removed / total per cell
	const int cell = kReviewTileSize;
	int gw = (W + cell - 1) / cell;
	int gh = (H + cell - 1) / cell;

	struct Cell
	{
		int removed = 0;
		int total = 0;
		double sumX = 0.0;			// removed-tree pixel sums (centroid)
		double sumY = 0.0;
	};

	std::vector<Cell> grid((size_t)gw * gh);

	for (auto& t : trees)
	{
		if ((t.px.x < 0) || (t.px.y < 0) || (t.px.x >= W) || (t.px.y >= H))
		{
			continue;
		}

		Cell& c = grid[(size_t)((int)t.px.y / cell) * gw + (int)t.px.x / cell];
		c.total += 1;

		if (t.removed)
		{
			c.removed += 1;
			c.sumX += t.px.x;
			c.sumY += t.px.y;
		}
	}

	std::vector<int> cellOrder;

	for (int i = 0 ; i < (int)grid.size() ; ++i)
	{
		if (grid[i].removed > 0)
		{
			cellOrder.push_back(i);
		}
	}

	std::stable_sort(cellOrder.begin(), cellOrder.end(), [&](int a, int b) { return grid[a].removed > grid[b].removed; });

	// 4. Styles: kept = thin green circle, removed = red circle + X
	OverlayOptions base;
	base.drawPolygons = false;
	base.crownAlpha = 0.0;
	base.showTreeId = false;
	base.drawCenter = false;

	OverlayOptions keptOpts = base;
	keptOpts.crownColor = cv::Scalar(0, 220, 0);
	keptOpts.crownOutlineWidth = 1;

	OverlayOptions removedOpts = base;
	removedOpts.crownColor = cv::Scalar(0, 0, 255);
	removedOpts.crownOutlineWidth = 2;

	OverlayOptions legendOpts = base;
	legendOpts.legendX = 20;
	legendOpts.legendY = 20;
	legendOpts.resultTitle = courseName + " review: tree_info_excluded";

	std::map<std::string, cv::Scalar> legend =
	{
		{ "1 removed: " + std::to_string(removedTD.tps.size()), removedOpts.crownColor },
		{ "2 kept: " + std::to_string(keptTD.tps.size()), keptOpts.crownColor },
	};

	auto drawAll = [&](cv::Mat& canvas, std::function<cv::Point2f(double, double)> toPx, double ppm, bool legendOn)
	{
		OverlayRenderer::drawTreesOn(canvas, keptTD, toPx, ppm, keptOpts);
		OverlayRenderer::drawTreesOn(canvas, removedTD, toPx, ppm, removedOpts);

		for (auto& tp : removedTD.tps)
		{
			cv::Point2f p = toPx(tp.x, tp.y);

			if ((p.x < -20) || (p.y < -20) || (p.x > canvas.cols + 20) || (p.y > canvas.rows + 20))
			{
				continue;
			}

			int d = std::max(3, (int)(tp.crownD * 0.5 * ppm * 0.7));
			cv::Point c((int)p.x, (int)p.y);

			cv::line(canvas, c + cv::Point(-d, -d), c + cv::Point(d, d), removedOpts.crownColor, (d > 6) ? 2 : 1, cv::LINE_AA);
			cv::line(canvas, c + cv::Point(-d, d), c + cv::Point(d, -d), removedOpts.crownColor, (d > 6) ? 2 : 1, cv::LINE_AA);
		}

		if (legendOn)
		{
			OverlayRenderer::drawLegendOn(canvas, legend, legendOpts);
		}
	};

	TSICommon::mkdirs(courseDir);

	std::vector<std::string> savedFiles;

	// 5. Overview (1/4)
	{
		cv::Mat small;
		cv::resize(aerial.image, small, cv::Size(), kReviewOverviewScale, kReviewOverviewScale, cv::INTER_AREA);

		cv::Mat canvas = toBGR8(small);
		small.release();

		drawAll(canvas, [&](double gx, double gy) { return geoToPx(gx, gy) * (float)kReviewOverviewScale; }, pixPerMeter * kReviewOverviewScale, true);

		std::string path = TSICommon::join(courseDir, "review_overview.jpg");
		cv::imwrite(path, canvas);
		savedFiles.push_back(path);

		std::cout << "  Saved: " << path << " (" << canvas.cols << "x" << canvas.rows << ")\n";
	}

	// 6. Tiles: centred on the removed-tree centroid of the top cells, skipping cells already covered by a tile
	std::vector<cv::Rect> tiles;

	for (int ci : cellOrder)
	{
		if ((int)tiles.size() >= kReviewTileCount)
		{
			break;
		}

		const Cell& c = grid[ci];
		cv::Point centre((int)(c.sumX / c.removed), (int)(c.sumY / c.removed));
		bool covered = false;

		for (auto& r : tiles)
		{
			covered = covered || r.contains(centre);
		}

		if (covered)
		{
			continue;
		}

		int tw = std::min(kReviewTileSize, W);
		int th = std::min(kReviewTileSize, H);
		int x = std::min(std::max(0, centre.x - tw / 2), W - tw);
		int y = std::min(std::max(0, centre.y - th / 2), H - th);

		tiles.push_back(cv::Rect(x, y, tw, th));
	}

	for (size_t i = 0 ; i < tiles.size() ; ++i)
	{
		const cv::Rect& r = tiles[i];

		cv::Mat canvas = toBGR8(aerial.image(r));

		drawAll(canvas, [&](double gx, double gy) { return geoToPx(gx, gy) - cv::Point2f((float)r.x, (float)r.y); }, pixPerMeter, false);

		int inRemoved = 0;
		int inKept = 0;

		for (auto& t : trees)
		{
			if (r.contains(cv::Point((int)t.px.x, (int)t.px.y)))
			{
				(t.removed ? inRemoved : inKept) += 1;
			}
		}

		double ex = 0.0;
		double ey = 0.0;
		pxToExport(r.x + r.width * 0.5, r.y + r.height * 0.5, ex, ey);

		char line1[128];
		char line2[128];
		sprintf_s(line1, "Tile %02d  center X %.1f  Y %.1f  (EPSG:5179)", (int)i + 1, ex, ey);
		sprintf_s(line2, "removed %d  kept %d", inRemoved, inKept);

		cv::Mat roi = canvas(cv::Rect(0, 0, std::min(620, canvas.cols), std::min(64, canvas.rows)));
		cv::Mat dark(roi.size(), roi.type(), cv::Scalar(20, 20, 20));
		cv::addWeighted(dark, 0.7, roi, 0.3, 0, roi);

		cv::putText(canvas, line1, { 10, 26 }, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
		cv::putText(canvas, line2, { 10, 54 }, cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(120, 160, 255), 1, cv::LINE_AA);

		char name[32];
		sprintf_s(name, "review_tile_%02d.jpg", (int)i + 1);

		std::string path = TSICommon::join(courseDir, name);
		cv::imwrite(path, canvas);
		savedFiles.push_back(path);

		std::cout << "  Saved: " << path << " (" << r.width << "x" << r.height << " at px " << r.x << "," << r.y << ", removed " << inRemoved << ", kept " << inKept << ")\n";
	}

	// 7. Contact sheets: every removed tree (TreeID order), 400 random kept trees
	std::vector<const ReviewTree*> removedList;
	std::vector<const ReviewTree*> keptList;

	for (auto& t : trees)
	{
		(t.removed ? removedList : keptList).push_back(&t);
	}

	std::sort(removedList.begin(), removedList.end(), [](const ReviewTree* a, const ReviewTree* b) { return a->tp.treeId < b->tp.treeId; });

	std::mt19937 rng(kReviewKeptSeed);
	std::shuffle(keptList.begin(), keptList.end(), rng);
	keptList.resize(std::min<size_t>(keptList.size(), kReviewSheetPerPage));
	std::sort(keptList.begin(), keptList.end(), [](const ReviewTree* a, const ReviewTree* b) { return a->tp.treeId < b->tp.treeId; });

	int removedFromCrop = 0;
	int keptFromCrop = 0;

	for (auto& p : saveContactSheets(removedList, aerial.image, pixPerMeter, courseDir, "review_excluded_sheet", true, removedFromCrop))
	{
		savedFiles.push_back(p);
	}

	for (auto& p : saveContactSheets(keptList, aerial.image, pixPerMeter, courseDir, "review_kept_sheet", false, keptFromCrop))
	{
		savedFiles.push_back(p);
	}

	// 8. Summary
	std::vector<double> hRemoved, hKept, cRemoved, cKept;
	int forestCount[2][3] = {};		// [removed][Y, N, no row]

	for (auto& t : trees)
	{
		(t.removed ? hRemoved : hKept).push_back(t.tp.height);
		(t.removed ? cRemoved : cKept).push_back(t.tp.crownD);
		forestCount[t.removed ? 1 : 0][(t.forest == 1) ? 0 : (t.forest == 0) ? 1 : 2] += 1;
	}

	int nRemoved = (int)removedList.size();
	int nKept = (int)(trees.size() - removedList.size());

	int cellsWithRemoved = (int)cellOrder.size();
	int removedInCluster = 0;
	int cellsCluster = 0;

	for (int ci : cellOrder)
	{
		if (grid[ci].removed >= kReviewClusterRatio * grid[ci].total)
		{
			removedInCluster += grid[ci].removed;
			cellsCluster += 1;
		}
	}

	std::ostringstream ss;
	char buf[256];

	ss << "=== Review: " << courseName << " ===\n";
	ss << "  exclusion file : " << exclusionPath << "\n";
	ss << "  tree_info      : " << course.treeInfoPath << " (" << trees.size() << " trees)\n";
	ss << "  feature CSV    : " << featPath << "\n";
	ss << "  remove rows    : " << removedIds.size() << " (in tree_info " << removedInTreeInfo << ", not in tree_info " << (removedIds.size() - removedInTreeInfo) << ")\n";
	ss << "  auto_forest    : " << autoForest << " (counted as kept)\n";
	ss << "  removed / kept : " << nRemoved << " / " << nKept << "\n";

	ss << "\n[Height (m)]\n" << reviewStatsLine("removed", hRemoved) << reviewStatsLine("kept", hKept);
	ss << "\n[Crown diameter (m)]\n" << reviewStatsLine("removed", cRemoved) << reviewStatsLine("kept", cKept);

	ss << "\n[InsideForest (" << featPath << ")]\n";

	for (int r = 1 ; r >= 0 ; --r)
	{
		int n = r ? nRemoved : nKept;

		sprintf_s(buf, "  %-8s Y %6d (%s)   N %6d (%s)   no feature row %d\n", r ? "removed" : "kept",
			forestCount[r][0], reviewPercent(forestCount[r][0], n).c_str(), forestCount[r][1], reviewPercent(forestCount[r][1], n).c_str(), forestCount[r][2]);
		ss << buf;
	}

	sprintf_s(buf, "\n[Spatial: %d px grid cells (%.0f m), top %d by removed]\n", cell, cell / pixPerMeter, kReviewTopCells);
	ss << buf;
	ss << "  rank  cell(col,row)   center X      center Y     removed  total   ratio\n";

	for (int k = 0 ; k < std::min<int>(kReviewTopCells, (int)cellOrder.size()) ; ++k)
	{
		int ci = cellOrder[k];
		const Cell& c = grid[ci];
		int col = ci % gw;
		int row = ci / gw;
		double ex = 0.0;
		double ey = 0.0;

		pxToExport((col + 0.5) * cell, (row + 0.5) * cell, ex, ey);

		sprintf_s(buf, "  %4d  (%3d,%3d)    %11.1f  %11.1f  %7d  %5d  %6s%s\n", k + 1, col, row, ex, ey, c.removed, c.total,
			reviewPercent(c.removed, c.total).c_str(), (c.removed >= kReviewClusterRatio * c.total) ? "  block (>= 90%)" : "");
		ss << buf;
	}

	sprintf_s(buf, "  cells with removed trees: %d, of which >= 90%% removed: %d\n", cellsWithRemoved, cellsCluster);
	ss << buf;
	ss << "  removed trees in >= 90% cells: " << removedInCluster << " (" << reviewPercent(removedInCluster, nRemoved) << ")"
		<< ", in mixed cells: " << (nRemoved - removedInCluster) << " (" << reviewPercent(nRemoved - removedInCluster, nRemoved) << ")\n";

	ss << "\n[Thumbnails]\n";
	ss << "  removed: " << removedFromCrop << " from TreeLabeler crops, " << (nRemoved - removedFromCrop) << " cut from the aerial photo (yellow TreeID)\n";
	ss << "  kept   : " << keptFromCrop << " from TreeLabeler crops, " << ((int)keptList.size() - keptFromCrop) << " cut from the aerial photo (random " << keptList.size() << ", seed " << kReviewKeptSeed << ")\n";

	std::string summaryPath = TSICommon::join(courseDir, "review_summary.txt");
	savedFiles.push_back(summaryPath);

	ss << "\n[Files]\n";

	for (auto& p : savedFiles)
	{
		ss << "  " << p << "\n";
	}

	if (toExport)
	{
		OGRCoordinateTransformation::DestroyCT(toExport);
	}

	std::ofstream out(summaryPath, std::ios::binary);

	if (!out.is_open())
	{
		throw std::runtime_error("Failed to write: " + summaryPath);
	}

	std::string text = ss.str();
	std::string crlf;

	for (char ch : text)
	{
		crlf += (ch == '\n') ? std::string("\r\n") : std::string(1, ch);
	}

	out << crlf;

	std::cout << "\n" << text;

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

	// exclusions are manual only: rows from automatic filters (note auto_*) are reverted
	int remaining = 0;
	int reverted = TreeExclusion::dropAutoRows(exclusionPath, &remaining);

	std::cout << "  auto_* rows reverted: " << reverted << ", remaining rows: " << remaining << "\n";

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

	TreeEditor::run(aerial, trees.tps, exclusionPath, "TSI Tree Editor - " + courseName, course.aerialImgPath);

	return 0;
}

// TreeSpeciesIdentification.exe                                    ← 기존 파이프라인
// TreeSpeciesIdentification.exe experiment extract <입력> out\ models\dinov2_vits14.onnx
// TreeSpeciesIdentification.exe experiment eval out\
// TreeSpeciesIdentification.exe --experiment export ..\..\Resources\courses_dataset.json out\ Hwarang
// TreeSpeciesIdentification.exe --experiment render ..\..\Resources\courses_dataset.json out\ Hwarang [--skip-forest]
// TreeSpeciesIdentification.exe --experiment review ..\..\Resources\courses_dataset.json Hwarang

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