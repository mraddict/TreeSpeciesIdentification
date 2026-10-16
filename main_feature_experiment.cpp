
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

// ----------------------------------------------------------------------------------------------------------------------------
//  no_image trees in export: species from the nearest forest-map polygon with a species (groups C / D / E / M)
//    confidence: inside the polygon 0.9, within 50 m 0.6, farther 0.3, no such polygon 0
// ----------------------------------------------------------------------------------------------------------------------------

static const double kNoImageNearDist = 50.0;
static const double kNoImageConfInside = 0.9;
static const double kNoImageConfNear = 0.6;
static const double kNoImageConfFar = 0.3;

class ForestSpeciesLookup
{
public:

	// forest : loaded forest map (owned by the caller), points are given in EPSG pointEpsg
	ForestSpeciesLookup(const ForestLayer& forest, int pointEpsg)
	{
		int forestEpsg = (forest.epsg > 0) ? forest.epsg : pointEpsg;

		if (forestEpsg != pointEpsg)
		{
			ct_ = CoordTransformer::createTransform(pointEpsg, forestEpsg);

			if (!ct_)
			{
				throw std::runtime_error("Failed to create transform EPSG:" + std::to_string(pointEpsg) + " -> EPSG:" + std::to_string(forestEpsg));
			}
		}

		for (auto& ft : forest.features)
		{
			auto it = ft.attributes.find(SPECIES_CODE);

			if (!ft.geometry || (it == ft.attributes.end()))
			{
				continue;
			}

			std::string code = TSICommon::trim(it->second);

			if (!SpeciesCodes::isTrainable(code) && !SpeciesCodes::isMixed(code))
			{
				continue;
			}

			Candidate c;
			c.geometry = ft.geometry;
			c.code = code;
			ft.geometry->getEnvelope(&c.env);

			candidates_.push_back(c);
		}
	}

	~ForestSpeciesLookup()
	{
		if (ct_)
		{
			OGRCoordinateTransformation::DestroyCT(ct_);
		}
	}

	size_t polygonCount() const
	{
		return candidates_.size();
	}

	// false if there is no species polygon at all. dist = 0 inside the polygon.
	bool nearest(double x, double y, std::string& code, double& dist) const
	{
		if (ct_)
		{
			ct_->Transform(1, &x, &y);
		}

		OGRPoint pt(x, y);

		// inside a polygon (the common case): only polygons whose envelope holds the point
		for (auto& c : candidates_)
		{
			if ((x >= c.env.MinX) && (x <= c.env.MaxX) && (y >= c.env.MinY) && (y <= c.env.MaxY) && c.geometry->Contains(&pt))
			{
				code = c.code;
				dist = 0.0;

				return true;
			}
		}

		// growing search radius; the envelope distance is a lower bound of the polygon distance
		for (double radius : { kNoImageNearDist, 500.0, 5000.0, std::numeric_limits<double>::max() })
		{
			double best = std::numeric_limits<double>::max();
			const Candidate* bestC = nullptr;

			for (auto& c : candidates_)
			{
				double dx = std::max({ c.env.MinX - x, 0.0, x - c.env.MaxX });
				double dy = std::max({ c.env.MinY - y, 0.0, y - c.env.MaxY });

				if ((dx > radius) || (dy > radius) || (std::hypot(dx, dy) > std::min(radius, best)))
				{
					continue;
				}

				double d = c.geometry->Distance(&pt);

				if ((d >= 0.0) && (d < best))
				{
					best = d;
					bestC = &c;
				}
			}

			if (bestC && (best <= radius))
			{
				code = bestC->code;
				dist = best;

				return true;
			}
		}

		return false;
	}

private:

	struct Candidate
	{
		OGRGeometry* geometry = nullptr;
		OGREnvelope env;
		std::string code;
	};

	std::vector<Candidate> candidates_;
	OGRCoordinateTransformation* ct_ = nullptr;
};

// TreeIDs (without offset) with InsideForest == Y in a features_*.csv (defined after loadFeatureRows)
static std::unordered_set<long long> insideForestIdsOf(const std::string& path, const std::string& courseName, long long treeIdOffset);

// Per-course counts collected by prepare (-1 = not run)
struct CourseRunStats
{
	int total = -1;					// tree_info rows (no exclusion applied)
	int removed = -1;				// action == remove, present in tree_info
	int noImage = -1;				// action == no_image, present in tree_info
	int review = -1;				// action == review, written to trees.json
	int insideForest = -1;			// kept trees with InsideForest == Y
	int noFeatureRow = -1;			// kept trees without a features_dino.csv row
	int exported = -1;				// trees.json (review + auto_forest)
	bool hasExclusionFile = false;
};

static int exportCourse(const std::string& configPath, const std::string& outDir, const std::string& courseName, CourseRunStats* stats);

int runExport(const std::string& configPath, const std::string& outDir, const std::string& courseName)
{
	return exportCourse(configPath, outDir, courseName, nullptr);
}

static int exportCourse(const std::string& configPath, const std::string& outDir, const std::string& courseName, CourseRunStats* stats)
{
	CourseDataSetConfig cfg = CourseDataSetConfigLoader::load(configPath);

	const CourseDataSet* course = &findCourse(cfg, courseName);

	bool useBoundary = !course->boundaryPath.empty();

	std::cout << "=== Export: " << courseName << " ===\n";

	// 1. Trees in EPSG:5179 (TreeID without course offset); remove is dropped by TreeLoader
	TreeData trees = TreeLoader::load(course->treeInfoPath, course->treeSrcEPSG, kExportEpsg, 0.0, 0.0, course->treeExcludePath);

	std::map<long long, TreeExclusion::Entry> exclusion = TreeExclusion::load(TreeExclusion::resolvePath(course->treeInfoPath, course->treeExcludePath));
	std::unordered_set<long long> reviewIds = TreeExclusion::reviewIds(exclusion);
	std::unordered_set<long long> noImageIds = TreeExclusion::noImageIds(exclusion);

	std::cout << "  tree_info_excluded: " << TreeExclusion::countText(exclusion) << "\n";

	// InsideForest decides forest (track A, forest_trees.json) vs auto_forest for trees that are not in the file
	std::unordered_set<long long> insideForestIds;			// TreeID without offset
	std::string dinoPath = TSICommon::join(outDir, "features_dino.csv");

	if (TSICommon::exists(dinoPath))
	{
		insideForestIds = insideForestIdsOf(dinoPath, courseName, course->treeIdOffset);
	}
	else
	{
		std::cout << "  ! " << dinoPath << " not found - InsideForest unknown, every tree not in the file is auto_forest\n";
	}

	auto isForestState = [&](long long id)
	{
		auto e = exclusion.find(id);
		bool inFile = (e != exclusion.end()) && (e->second.action != TreeExclusion::kAutoForest);

		return !inFile && insideForestIds.count(id);
	};

	// 2. Every kept tree, or the ones inside courseBoundary if the config sets it
	std::vector<const TreePoint*> selected;

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
				selected.push_back(&tp);
			}
		}

		OGRCoordinateTransformation::DestroyCT(toBoundaryCT);
		ForestLoader::release(boundary);

		std::cout << "  Boundary SHP " << course->boundaryPath << " (EPSG:" << boundaryEpsg << "): " << selected.size() << " / " << trees.tps.size() << " trees\n";
	}
	else
	{
		for (auto& tp : trees.tps)
		{
			selected.push_back(&tp);
		}
	}

	// 3. Embeddings for review trees only (the designer's candidates)
	std::unordered_set<long long> wanted;

	for (auto* tp : selected)
	{
		if (reviewIds.count(tp->treeId))
		{
			wanted.insert(tp->treeId + course->treeIdOffset);
		}
	}

	std::unordered_map<long long, std::vector<std::string>> embeddings;

	if (!wanted.empty())
	{
		if (TSICommon::exists(dinoPath))
		{
			int wrongCourse = 0;
			embeddings = loadDinoEmbeddings(dinoPath, wanted, courseName, false, wrongCourse);

			if (wrongCourse > 0)
			{
				std::cout << "  ! " << wrongCourse << " embedding rows belong to another course (TreeID offset / course order mismatch) - ignored\n";
			}
		}
		else
		{
			std::cout << "  ! " << dinoPath << " not found - review trees are written without embeddings\n";
		}
	}

	// 4. Forest map species for auto_forest / no_image
	ForestLayer forest = ForestLoader::loadMultiple(course->shpFiles, "", false);

	std::string courseDir = TSICommon::join(outDir, courseName);
	TSICommon::mkdirs(courseDir);

	std::string jsonPath = TSICommon::join(courseDir, "trees.json");
	std::ofstream ofs(jsonPath, std::ios::binary);

	if (!ofs.is_open())
	{
		ForestLoader::release(forest);

		throw std::runtime_error("Failed to create: " + jsonPath);
	}

	std::string forestPath = TSICommon::join(courseDir, "forest_trees.json");
	std::ofstream ofsForest(forestPath, std::ios::binary);

	if (!ofsForest.is_open())
	{
		ForestLoader::release(forest);

		throw std::runtime_error("Failed to create: " + forestPath);
	}

	ofs << std::fixed << "[\n";
	ofsForest << std::fixed << "[\n";

	int written = 0;
	int forestWritten = 0;
	int reviewWritten = 0;
	int reviewNoEmbedding = 0;
	int noImageWritten = 0;
	int autoCount[4] = { 0, 0, 0, 0 };		// inside / within 50 m / farther / no species polygon

	{
		ForestSpeciesLookup lookup(forest, kExportEpsg);

		std::cout << "  Forest map species polygons (C/D/E/M): " << lookup.polygonCount() << "\n";

		for (auto* tp : selected)
		{
			// forest (track A): not in the file and inside the forest map -> forest_trees.json only
			if (isForestState(tp->treeId))
			{
				std::string code;
				double dist = -1.0;
				bool found = lookup.nearest(tp->x, tp->y, code, dist);

				if (forestWritten > 0)
				{
					ofsForest << ",\n";
				}

				ofsForest << "  {\"id\": " << tp->treeId
					<< ", \"x\": " << std::setprecision(3) << tp->x
					<< ", \"y\": " << tp->y
					<< ", \"height\": " << std::setprecision(2) << tp->height
					<< ", \"crownD\": " << tp->crownD;

				if (found)
				{
					ofsForest << ", \"species_code\": \"" << code << "\", \"asset\": \"" << SpeciesCodes::asset(code) << "\"}";
				}
				else
				{
					ofsForest << ", \"species_code\": null, \"asset\": null}";
				}

				++forestWritten;

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
				<< ", \"crownD\": " << tp->crownD;

			if (reviewIds.count(tp->treeId))
			{
				// review: the designer decides; embedding for the classifier, no suggestion yet
				auto it = embeddings.find(tp->treeId + course->treeIdOffset);

				if (it == embeddings.end())
				{
					ofs << ", \"embedding\": null";
					++reviewNoEmbedding;
				}
				else
				{
					ofs << ", \"embedding\": [";

					for (size_t i = 0 ; i < it->second.size() ; ++i)
					{
						ofs << (i ? "," : "") << it->second[i];
					}

					ofs << "]";
				}

				ofs << ", \"mode\": \"review\", \"suggested\": null, \"source\": null, \"confidence\": 0}";
				++reviewWritten;
			}
			else
			{
				// auto_forest (default) and no_image: asset from the nearest forest-map species polygon
				std::string code;
				double dist = -1.0;
				double conf = 0.0;
				int bucket = 3;

				if (lookup.nearest(tp->x, tp->y, code, dist))
				{
					bucket = (dist <= 1e-9) ? 0 : (dist <= kNoImageNearDist) ? 1 : 2;
					conf = (bucket == 0) ? kNoImageConfInside : (bucket == 1) ? kNoImageConfNear : kNoImageConfFar;
				}

				++autoCount[bucket];
				noImageWritten += noImageIds.count(tp->treeId) ? 1 : 0;

				ofs << ", \"embedding\": null, \"mode\": \"auto_forest\"";

				if (code.empty())
				{
					ofs << ", \"suggested\": null, \"source\": \"forest:none\"";
				}
				else
				{
					ofs << ", \"suggested\": \"" << SpeciesCodes::asset(code) << "\", \"source\": \"forest:" << code << "\"";
				}

				ofs << ", \"confidence\": " << std::setprecision(1) << conf << "}";
			}

			++written;
		}
	}

	ForestLoader::release(forest);

	ofs << "\n]\n";
	ofs.close();
	ofsForest << "\n]\n";
	ofsForest.close();

	int autoWritten = written - reviewWritten;
	int removedCount = (int)TreeExclusion::removedIds(exclusion).size();

	std::cout << "  forest " << forestWritten << " / auto_forest " << (autoWritten - noImageWritten) << " / review " << reviewWritten
		<< " / no_image " << noImageWritten << " / remove " << removedCount << " (remove: rows in the file)\n";
	std::cout << "  Written: " << written << " trees -> " << jsonPath << " (review " << reviewWritten << ", auto_forest " << autoWritten << " incl. no_image " << noImageWritten << ")\n";
	std::cout << "  Written: " << forestWritten << " trees -> " << forestPath << " (forest, track A)\n";
	std::cout << "  auto_forest confidence: inside polygon " << autoCount[0] << " (" << kNoImageConfInside << "), <= " << kNoImageNearDist << " m " << autoCount[1]
		<< " (" << kNoImageConfNear << "), farther " << autoCount[2] << " (" << kNoImageConfFar << "), no species polygon " << autoCount[3] << " (0)\n";

	if (reviewNoEmbedding > 0)
	{
		std::cout << "  ! review trees without an embedding: " << reviewNoEmbedding << "\n";
	}

	if (stats)
	{
		stats->exported = written;
		stats->review = reviewWritten;
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

static std::unordered_set<long long> insideForestIdsOf(const std::string& path, const std::string& courseName, long long treeIdOffset)
{
	std::unordered_set<long long> ids;

	for (auto& kv : loadFeatureRows(path, courseName))
	{
		if (kv.second.insideForest)
		{
			ids.insert(kv.first - treeIdOffset);
		}
	}

	return ids;
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

// Crown crop for a tree without a saved TreeLabeler crop (empty Mat = none)
using ThumbFallback = std::function<cv::Mat(const ReviewTree&)>;

// Crown (x1.4) cut from an in-memory full-resolution aerial photo
static ThumbFallback cropFromAerialImage(const cv::Mat& aerial, double pixPerMeter)
{
	return [&aerial, pixPerMeter](const ReviewTree& t) -> cv::Mat
	{
		int half = (int)std::lround(std::max(8.0, t.tp.crownD * 0.5 * 1.4 * pixPerMeter));
		half = std::min(half, 256);

		cv::Rect r = cv::Rect((int)t.px.x - half, (int)t.px.y - half, half * 2, half * 2) & cv::Rect(0, 0, aerial.cols, aerial.rows);

		return (r.area() > 0) ? toBGR8(aerial(r)) : cv::Mat();
	};
}

// TreeLabeler crop if present, otherwise fallback(t)
static cv::Mat reviewThumb(const ReviewTree& t, const ThumbFallback& fallback, bool& fromCrop)
{
	cv::Mat crop;

	if (!t.imagePath.empty())
	{
		crop = cv::imread(t.imagePath, cv::IMREAD_COLOR);
	}

	fromCrop = !crop.empty();

	if (crop.empty() && fallback)
	{
		crop = fallback(t);
	}

	cv::Mat thumb(kReviewThumb, kReviewThumb, CV_8UC3, cv::Scalar(40, 40, 40));

	if (!crop.empty())
	{
		cv::resize(toBGR8(crop), thumb, thumb.size(), 0, 0, (crop.cols > kReviewThumb) ? cv::INTER_AREA : cv::INTER_LINEAR);
	}

	return thumb;
}

// 20-column sheets of 96x96 thumbnails with TreeID below. numbered: <stem>_01.jpg, _02.jpg ... otherwise <stem>.jpg (first page only)
static std::vector<std::string> saveContactSheets(const std::vector<const ReviewTree*>& trees, const ThumbFallback& fallback, const std::string& courseDir, const std::string& stem, bool numbered, int& fromCropCount)
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

			reviewThumb(t, fallback, fromCrop).copyTo(sheet(cv::Rect(x, y, kReviewThumb, kReviewThumb)));
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

static int reviewCourse(const std::string& configPath, const std::string& outDir, const std::string& courseName, CourseRunStats* stats);

int runReview(const std::string& configPath, const std::string& outDir, const std::string& courseName)
{
	return reviewCourse(configPath, outDir, courseName, nullptr);
}

// No remove rows (or no file): everything is drawn as kept, no tiles / removed sheets
static int reviewCourse(const std::string& configPath, const std::string& outDir, const std::string& courseName, CourseRunStats* stats)
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
		std::cout << "  ! No remove rows in " << exclusionPath << " - all trees are drawn as kept\n";
	}

	int autoForest = 0;
	std::unordered_set<long long> noImageIds = TreeExclusion::noImageIds(entries);

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
	ThumbFallback thumbFallback = cropFromAerialImage(aerial.image, pixPerMeter);

	for (auto& p : saveContactSheets(removedList, thumbFallback, courseDir, "review_excluded_sheet", true, removedFromCrop))
	{
		savedFiles.push_back(p);
	}

	for (auto& p : saveContactSheets(keptList, thumbFallback, courseDir, "review_kept_sheet", false, keptFromCrop))
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

	int noImageKept = 0;
	int noImageForest = 0;

	for (auto& t : trees)
	{
		if (!t.removed && noImageIds.count(t.tp.treeId))
		{
			++noImageKept;
			noImageForest += (t.forest == 1) ? 1 : 0;
		}
	}

	ss << "  no_image       : " << noImageKept << " (counted as kept; InsideForest Y " << noImageForest << ", image unusable -> not trained, exported as auto_forest)\n";
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

	if (stats)
	{
		stats->total = (int)trees.size();
		stats->removed = nRemoved;
		stats->noImage = noImageKept;
		stats->insideForest = forestCount[0][0] - noImageForest;		// trainable images only
		stats->noFeatureRow = forestCount[0][2];
		stats->hasExclusionFile = TSICommon::exists(exclusionPath);
	}

	return 0;
}

// ----------------------------------------------------------------------------------------------------------------------------
//  prepare : export -> review for one course, or for every enabled course ("all")
//    <out>/<course>/trees.json, review_*      as export / review
//    <out>/prepare_summary.txt                 all: per-course table (total / removed / InsideForest Y / exported)
//  No tree_info_excluded.csv : warning only.
//  No features_dino.csv rows for the course : export skipped, extract needed (not run here).
// ----------------------------------------------------------------------------------------------------------------------------

struct PrepareRow
{
	std::string course;
	CourseRunStats stats;
	std::string note;
};

static void appendNote(std::string& note, const std::string& s)
{
	note += (note.empty() ? "" : "; ") + s;
}

static PrepareRow prepareCourse(const std::string& configPath, const std::string& outDir, const CourseDataSet& course)
{
	PrepareRow row;
	row.course = course.courseName;

	std::cout << "\n##### Prepare: " << course.courseName << " #####\n";

	std::string exclusionPath = TreeExclusion::resolvePath(course.treeInfoPath, course.treeExcludePath);

	if (!TSICommon::exists(exclusionPath))
	{
		std::cout << "  ! 제외 파일 없음 - edit 로 먼저 검토 권장 (" << exclusionPath << ")\n";
		appendNote(row.note, "no exclusion file");
	}

	std::string featPath = TSICommon::join(outDir, "features_dino.csv");
	size_t featRows = TSICommon::exists(featPath) ? loadFeatureRows(featPath, course.courseName).size() : 0;

	try
	{
		if (featRows == 0)
		{
			std::cout << "  ! " << featPath << " has no rows for " << course.courseName << " - extract needed (not run), export skipped\n";
			appendNote(row.note, "extract needed");
		}
		else
		{
			exportCourse(configPath, outDir, course.courseName, &row.stats);
		}

		reviewCourse(configPath, outDir, course.courseName, &row.stats);
	}
	catch (const std::exception& e)
	{
		std::cout << "  ERROR: " << e.what() << "\n";
		appendNote(row.note, std::string("error: ") + e.what());
	}

	if (featRows == 0)
	{
		row.stats.insideForest = -1;		// unknown without feature rows
	}

	return row;
}

static std::string prepareCell(int v)
{
	return (v < 0) ? "-" : std::to_string(v);
}

int runPrepare(const std::string& configPath, const std::string& outDir, const std::string& target)
{
	CourseDataSetConfig cfg = CourseDataSetConfigLoader::load(configPath);

	if (target != "all")
	{
		PrepareRow row = prepareCourse(configPath, outDir, findCourse(cfg, target));

		return (row.note.find("error") == std::string::npos) ? 0 : 2;
	}

	std::vector<PrepareRow> rows;

	for (auto& c : cfg.courses)
	{
		if (c.enabled)
		{
			rows.push_back(prepareCourse(configPath, outDir, c));
		}
		else
		{
			std::cout << "\n  Skipped (disabled in config): " << c.courseName << "\n";
		}
	}

	// Summary table
	std::ostringstream ss;
	char buf[512];
	int sum[6] = { 0, 0, 0, 0, 0, 0 };

	ss << "=== Prepare summary: " << configPath << " ===\n";
	sprintf_s(buf, "  %-10s %8s %8s %9s %7s %10s %8s  %s\n", "course", "total", "removed", "no_image", "review", "forest(Y)", "export", "note");
	ss << buf;

	for (auto& r : rows)
	{
		sprintf_s(buf, "  %-10s %8s %8s %9s %7s %10s %8s  %s\n", r.course.c_str(), prepareCell(r.stats.total).c_str(), prepareCell(r.stats.removed).c_str(),
			prepareCell(r.stats.noImage).c_str(), prepareCell(r.stats.review).c_str(), prepareCell(r.stats.insideForest).c_str(), prepareCell(r.stats.exported).c_str(), r.note.c_str());
		ss << buf;

		sum[0] += std::max(0, r.stats.total);
		sum[1] += std::max(0, r.stats.removed);
		sum[2] += std::max(0, r.stats.noImage);
		sum[3] += std::max(0, r.stats.review);
		sum[4] += std::max(0, r.stats.insideForest);
		sum[5] += std::max(0, r.stats.exported);
	}

	sprintf_s(buf, "  %-10s %8d %8d %9d %7d %10d %8d\n", "sum", sum[0], sum[1], sum[2], sum[3], sum[4], sum[5]);
	ss << buf;
	ss << "  total     : tree_info rows (no exclusion)\n";
	ss << "  removed   : tree_info_excluded.csv action == remove (dropped)\n";
	ss << "  no_image  : action == no_image (kept, image unusable, not trained; exported as auto_forest)\n";
	ss << "  review    : action == review (exported as mode review)\n";
	ss << "  forest(Y) : kept trees with InsideForest == Y and a usable image (features_dino.csv)\n";
	ss << "  export    : trees.json = review + auto_forest + no_image (forest-state trees go to forest_trees.json)\n";

	std::string text = ss.str();
	std::string path = TSICommon::join(outDir, "prepare_summary.txt");
	std::ofstream out(path, std::ios::binary);

	if (!out.is_open())
	{
		throw std::runtime_error("Failed to write: " + path);
	}

	for (char ch : text)
	{
		if (ch == '\n')
		{
			out << '\r';
		}

		out << ch;
	}

	std::cout << "\n" << text << "  Saved: " << path << "\n";

	return 0;
}

// ----------------------------------------------------------------------------------------------------------------------------
//  migrate-excluded : old remove rows -> remove (structure) or no_image (low-quality-aerial)
//    structure rule : V_mean >= tv and S_mean <= ts (features_handcrafted.csv, crown crop HSV 0-255)
//    tv / ts        : grid search on Hwarang's rows (any action, the .bak file if present):
//                     TreeID 35518..35890 = greenhouse (structure), 36166..37367 = forest; best balanced accuracy
//    remove rows without a feature row -> no_image ("low-quality-aerial; no feature row")
//    no_image / review / auto_forest rows are not touched
//    backup         : <tree_info stem>_excluded.bak.csv (an existing backup is kept, never overwritten)
//    check sheets   : <out>/<course>/review_remove_sheet.jpg (first 400 remove), review_noimage_sheet_01.jpg (first 400 no_image)
// ----------------------------------------------------------------------------------------------------------------------------

static const long long kMigrateStructureFrom = 35518;
static const long long kMigrateStructureTo = 35890;
static const long long kMigrateForestFrom = 36166;
static const long long kMigrateForestTo = 37367;
static const char* kMigrateCalibCourse = "Hwarang";
static const char* kNoteStructure = "structure";
static const char* kNoteLowQuality = "low-quality-aerial";

struct CrownHsv
{
	float s = 0.0f;
	float v = 0.0f;
	std::string imagePath;
};

// TreeID (global, offset applied) -> S_mean / V_mean / ImagePath from features_handcrafted.csv
static std::unordered_map<long long, CrownHsv> loadCrownHsv(const std::string& path)
{
	std::ifstream file(path);

	if (!file.is_open())
	{
		throw std::runtime_error("Failed to open: " + path + " (run experiment extract first)");
	}

	std::string line;
	std::getline(file, line);

	if (!line.empty() && (line.back() == '\r'))
	{
		line.pop_back();
	}

	std::vector<std::string> header = TSICommon::split(line);
	int colS = -1;
	int colV = -1;
	int colPath = -1;

	for (int i = 0 ; i < (int)header.size() ; ++i)
	{
		colS = (header[i] == "S_mean") ? i : colS;
		colV = (header[i] == "V_mean") ? i : colV;
		colPath = (header[i] == "ImagePath") ? i : colPath;
	}

	if ((colS < 0) || (colV < 0))
	{
		throw std::runtime_error("S_mean / V_mean columns not found in " + path);
	}

	int lastCol = std::max({ colS, colV, colPath });
	std::unordered_map<long long, CrownHsv> result;

	while (std::getline(file, line))
	{
		// only the leading columns up to lastCol are needed
		std::vector<std::string> cols;
		size_t pos = 0;

		while ((int)cols.size() <= lastCol)
		{
			size_t c = line.find(',', pos);
			cols.push_back(line.substr(pos, (c == std::string::npos) ? std::string::npos : c - pos));

			if (c == std::string::npos)
			{
				break;
			}

			pos = c + 1;
		}

		if ((int)cols.size() <= lastCol)
		{
			continue;
		}

		try
		{
			CrownHsv h;
			h.s = std::stof(cols[colS]);
			h.v = std::stof(cols[colV]);
			h.imagePath = (colPath >= 0) ? cols[colPath] : "";

			result[std::stoll(cols[0])] = h;
		}
		catch (...)
		{
		}
	}

	return result;
}

static bool isStructure(const CrownHsv& h, int tv, int ts)
{
	return (h.v >= tv) && (h.s <= ts);
}

static std::string bakPathOf(const std::string& exclusionPath)
{
	size_t dot = exclusionPath.find_last_of('.');

	return ((dot == std::string::npos) ? exclusionPath : exclusionPath.substr(0, dot)) + ".bak.csv";
}

// Picks tv / ts on the calibration course; prints the evidence
static void calibrateStructureRule(const CourseDataSet& calib, const std::unordered_map<long long, CrownHsv>& hsv, int& tv, int& ts)
{
	std::string path = TreeExclusion::resolvePath(calib.treeInfoPath, calib.treeExcludePath);
	std::string bak = bakPathOf(path);
	std::string source = TSICommon::exists(bak) ? bak : path;
	auto entries = TreeExclusion::load(source);

	std::vector<CrownHsv> structure;
	std::vector<CrownHsv> forest;
	std::vector<long long> structureIds;
	std::vector<long long> forestIds;

	for (auto& kv : entries)
	{
		bool inS = (kv.first >= kMigrateStructureFrom) && (kv.first <= kMigrateStructureTo);
		bool inF = (kv.first >= kMigrateForestFrom) && (kv.first <= kMigrateForestTo);

		if (!inS && !inF)
		{
			continue;
		}

		auto it = hsv.find(kv.first + calib.treeIdOffset);

		if (it == hsv.end())
		{
			continue;
		}

		(inS ? structure : forest).push_back(it->second);
		(inS ? structureIds : forestIds).push_back(kv.first);
	}

	if (structure.empty() || forest.empty())
	{
		throw std::runtime_error("Calibration needs rows in both ID ranges of " + calib.courseName + " (" + source + ")");
	}

	// grid search: best balanced accuracy, ties -> nearest to the centre of the optimal region
	double best = -1.0;
	std::vector<std::pair<int, int>> optimal;

	for (int v = 0 ; v <= 255 ; ++v)
	{
		for (int s = 0 ; s <= 255 ; ++s)
		{
			int tp = 0;
			int tn = 0;

			for (auto& h : structure)
			{
				tp += isStructure(h, v, s) ? 1 : 0;
			}

			for (auto& h : forest)
			{
				tn += isStructure(h, v, s) ? 0 : 1;
			}

			double bal = 0.5 * ((double)tp / structure.size() + (double)tn / forest.size());

			if (bal > best + 1e-12)
			{
				best = bal;
				optimal.clear();
			}

			if (std::abs(bal - best) <= 1e-12)
			{
				optimal.push_back({ v, s });
			}
		}
	}

	double cv = 0.0;
	double cs = 0.0;

	for (auto& o : optimal)
	{
		cv += o.first;
		cs += o.second;
	}

	cv /= optimal.size();
	cs /= optimal.size();

	double bestD = std::numeric_limits<double>::max();

	for (auto& o : optimal)
	{
		double d = std::hypot(o.first - cv, o.second - cs);

		if (d < bestD)
		{
			bestD = d;
			tv = o.first;
			ts = o.second;
		}
	}

	auto pct = [](std::vector<float> x, double q) -> double
	{
		std::sort(x.begin(), x.end());

		return x[(size_t)std::min<double>((double)(x.size() - 1), std::floor(q * (x.size() - 1) + 0.5))];
	};

	auto line = [&](const char* name, const std::vector<CrownHsv>& g)
	{
		std::vector<float> vv, ss;

		for (auto& h : g)
		{
			vv.push_back(h.v);
			ss.push_back(h.s);
		}

		char buf[256];
		sprintf_s(buf, "    %-10s n=%-4zu  V_mean p10 %6.1f  p50 %6.1f  p90 %6.1f   S_mean p10 %6.1f  p50 %6.1f  p90 %6.1f\n",
			name, g.size(), pct(vv, 0.1), pct(vv, 0.5), pct(vv, 0.9), pct(ss, 0.1), pct(ss, 0.5), pct(ss, 0.9));
		std::cout << buf;
	};

	int sOk = 0;
	int fOk = 0;
	std::vector<long long> missS;
	std::vector<long long> missF;

	for (size_t i = 0 ; i < structure.size() ; ++i)
	{
		if (isStructure(structure[i], tv, ts)) ++sOk; else missS.push_back(structureIds[i]);
	}

	for (size_t i = 0 ; i < forest.size() ; ++i)
	{
		if (!isStructure(forest[i], tv, ts)) ++fOk; else missF.push_back(forestIds[i]);
	}

	auto ids = [](const std::vector<long long>& v)
	{
		std::string s;

		for (size_t i = 0 ; i < std::min<size_t>(v.size(), 15) ; ++i)
		{
			s += (i ? " " : "") + std::to_string(v[i]);
		}

		return s + ((v.size() > 15) ? " ..." : "");
	};

	std::cout << "=== Structure rule calibration (" << calib.courseName << ", " << source << ") ===\n";
	std::cout << "  greenhouse = rows with TreeID " << kMigrateStructureFrom << ".." << kMigrateStructureTo << ", forest = " << kMigrateForestFrom << ".." << kMigrateForestTo << "\n";
	line("greenhouse", structure);
	line("forest", forest);
	std::cout << "  rule: structure = V_mean >= " << tv << " and S_mean <= " << ts << "  (grid search 0..255, best balanced accuracy, " << optimal.size() << " optimal pairs, centre picked)\n";
	std::cout << "  greenhouse -> structure " << sOk << "/" << structure.size() << ", forest -> no_image " << fOk << "/" << forest.size()
		<< "  (balanced accuracy " << std::fixed << std::setprecision(1) << best * 100.0 << "%)\n" << std::setprecision(6);
	std::cout << "  greenhouse rows not matched: " << (missS.empty() ? "-" : ids(missS)) << "\n";
	std::cout << "  forest rows matched as structure: " << (missF.empty() ? "-" : ids(missF)) << "\n";
}

// Geo info of an aerial file without loading the raster
struct AerialFileGeo
{
	std::string path;
	double gt[6] = { 0, 1, 0, 0, 0, -1 };
	double igt[6] = { 0, 1, 0, 0, 0, -1 };
	int epsg = 0;
	bool ok = false;
};

static AerialFileGeo readAerialGeo(const std::string& path)
{
	AerialFileGeo g;
	g.path = path;

	GDALDataset* ds = (GDALDataset*)GDALOpen(path.c_str(), GA_ReadOnly);

	if (!ds)
	{
		return g;
	}

	if ((ds->GetGeoTransform(g.gt) == CE_None) && CoordTransformer::computeInverseGT(g.gt, g.igt))
	{
		g.ok = true;
	}

	const OGRSpatialReference* srs = ds->GetSpatialRef();

	if (srs)
	{
		OGRSpatialReference copy(*srs);
		copy.AutoIdentifyEPSG();

		const char* code = copy.GetAuthorityCode(nullptr);
		g.epsg = code ? std::atoi(code) : 0;
	}

	GDALClose(ds);

	return g;
}

static int migrateCourse(const CourseDataSet& course, const std::string& outDir, const std::unordered_map<long long, CrownHsv>& hsv, int tv, int ts, std::ostringstream& table)
{
	std::string path = TreeExclusion::resolvePath(course.treeInfoPath, course.treeExcludePath);
	char buf[256];

	std::cout << "\n=== Migrate: " << course.courseName << " ===\n";

	if (!TSICommon::exists(path))
	{
		std::cout << "  no exclusion file (" << path << ") - skipped\n";
		sprintf_s(buf, "  %-10s %8s %10s %10s %12s %10s  %s\n", course.courseName.c_str(), "-", "-", "-", "-", "-", "no file");
		table << buf;

		return 0;
	}

	// 1. Backup (the first one is kept)
	std::string bak = bakPathOf(path);

	if (TSICommon::exists(bak))
	{
		std::cout << "  backup exists, kept: " << bak << "\n";
	}
	else
	{
		std::filesystem::copy_file(path, bak);
		std::cout << "  backup: " << bak << "\n";
	}

	// 2. remove rows -> structure (remove) / no_image
	auto entries = TreeExclusion::load(path);
	int before = 0;
	int keptRemove = 0;
	int toNoImage = 0;
	int noRow = 0;

	for (auto& kv : entries)
	{
		if (kv.second.action != TreeExclusion::kRemove)
		{
			continue;
		}

		++before;

		auto it = hsv.find(kv.first + course.treeIdOffset);

		if ((it != hsv.end()) && isStructure(it->second, tv, ts))
		{
			kv.second.note = kNoteStructure;
			++keptRemove;
		}
		else
		{
			kv.second.action = TreeExclusion::kNoImage;
			kv.second.note = (it == hsv.end()) ? std::string(kNoteLowQuality) + "; no feature row" : kNoteLowQuality;
			++toNoImage;
			noRow += (it == hsv.end()) ? 1 : 0;
		}
	}

	TreeExclusion::save(path, entries);

	int noImageTotal = (int)TreeExclusion::noImageIds(entries).size();

	std::cout << "  remove " << before << " -> remove (structure) " << keptRemove << ", no_image " << toNoImage << " (no feature row " << noRow << ")\n";
	std::cout << "  saved: " << path << " (" << TreeExclusion::countText(entries) << ")\n";

	sprintf_s(buf, "  %-10s %8d %10d %10d %12d %10d\n", course.courseName.c_str(), before, keptRemove, toNoImage, noRow, noImageTotal);
	table << buf;

	// 3. Check sheets: first 400 remove, first 400 no_image (TreeID order)
	TreeData all = TreeLoader::load(course.treeInfoPath, course.treeSrcEPSG, kExportEpsg, 0.0, 0.0, TreeExclusion::kNone);
	AerialFileGeo geo = readAerialGeo(course.aerialImgPath);
	OGRCoordinateTransformation* toAerial = nullptr;

	if (geo.ok && (geo.epsg > 0) && (geo.epsg != kExportEpsg))
	{
		toAerial = CoordTransformer::createTransform(kExportEpsg, geo.epsg);
	}

	std::vector<ReviewTree> removeTrees;
	std::vector<ReviewTree> noImageTrees;

	for (auto& tp : all.tps)
	{
		auto e = entries.find(tp.treeId);

		if (e == entries.end())
		{
			continue;
		}

		bool isRemove = (e->second.action == TreeExclusion::kRemove);
		bool isNoImage = (e->second.action == TreeExclusion::kNoImage);

		if ((!isRemove || (removeTrees.size() >= (size_t)kReviewSheetPerPage)) && (!isNoImage || (noImageTrees.size() >= (size_t)kReviewSheetPerPage)))
		{
			continue;
		}

		ReviewTree rt;
		rt.tp = tp;
		rt.gx = tp.x;
		rt.gy = tp.y;

		double ax = tp.x;
		double ay = tp.y;

		if (toAerial)
		{
			toAerial->Transform(1, &ax, &ay);
		}

		rt.px = cv::Point2f((float)(geo.igt[0] + geo.igt[1] * ax + geo.igt[2] * ay), (float)(geo.igt[3] + geo.igt[4] * ax + geo.igt[5] * ay));

		auto h = hsv.find(tp.treeId + course.treeIdOffset);
		rt.imagePath = (h != hsv.end()) ? h->second.imagePath : "";

		(isRemove ? removeTrees : noImageTrees).push_back(rt);
	}

	if (toAerial)
	{
		OGRCoordinateTransformation::DestroyCT(toAerial);
	}

	double ppm = geo.ok ? std::abs(geo.igt[1]) : 1.0;

	// no saved crop: read the crown window from the original file
	ThumbFallback fromFile = [&geo, ppm](const ReviewTree& t) -> cv::Mat
	{
		if (!geo.ok)
		{
			return cv::Mat();
		}

		int half = std::min(256, (int)std::lround(std::max(8.0, t.tp.crownD * 0.5 * 1.4 * ppm)));

		return AerialLoader::readWindow(geo.path, cv::Rect((int)t.px.x - half, (int)t.px.y - half, half * 2, half * 2));
	};

	std::vector<const ReviewTree*> removeList;
	std::vector<const ReviewTree*> noImageList;

	for (auto& t : removeTrees) removeList.push_back(&t);
	for (auto& t : noImageTrees) noImageList.push_back(&t);

	std::string courseDir = TSICommon::join(outDir, course.courseName);
	TSICommon::mkdirs(courseDir);

	int fromCrop = 0;

	if (!removeList.empty())
	{
		saveContactSheets(removeList, fromFile, courseDir, "review_remove_sheet", false, fromCrop);
	}

	if (!noImageList.empty())
	{
		saveContactSheets(noImageList, fromFile, courseDir, "review_noimage_sheet", true, fromCrop);
	}

	return 0;
}

int runMigrateExcluded(const std::string& configPath, const std::string& outDir, const std::string& target)
{
	CourseDataSetConfig cfg = CourseDataSetConfigLoader::load(configPath);

	std::string hsvPath = TSICommon::join(outDir, "features_handcrafted.csv");
	std::cout << "  Loading " << hsvPath << " ...\n";
	auto hsv = loadCrownHsv(hsvPath);
	std::cout << "  rows: " << hsv.size() << "\n";

	int tv = 0;
	int ts = 0;
	calibrateStructureRule(findCourse(cfg, kMigrateCalibCourse), hsv, tv, ts);

	std::ostringstream table;
	char buf[256];

	sprintf_s(buf, "  %-10s %8s %10s %10s %12s %10s\n", "course", "remove", "->remove", "->no_image", "(no featrow)", "no_image");
	table << "=== migrate-excluded (structure = V_mean >= " << tv << " and S_mean <= " << ts << ") ===\n" << buf;

	for (auto& c : cfg.courses)
	{
		if ((target == "all") ? c.enabled : (c.courseName == target))
		{
			migrateCourse(c, outDir, hsv, tv, ts, table);
		}
	}

	table << "  remove     : remove rows before migration\n"
		<< "  ->remove   : kept as remove, note \"structure\"\n"
		<< "  ->no_image : changed to no_image, note \"low-quality-aerial\" (no featrow: no features_handcrafted.csv row)\n"
		<< "  no_image   : no_image rows after migration (including earlier ones)\n";

	std::cout << "\n" << table.str();

	return 0;
}

// ----------------------------------------------------------------------------------------------------------------------------
//  edit : mark false detections in <course>_tree_info.csv on a 1/4 aerial photo (TreeEditor / ImageViewer)
//    S saves <tree_info>_excluded.csv (or config tree_exclude); TreeLoader::load / FeatureExtractor::loadCSV drop action == remove.
//    The tree_info CSV is never modified.
// ----------------------------------------------------------------------------------------------------------------------------

static const double kEditImageScale = 0.25;
static const char* kEditOutDir = "out";				// features_dino.csv location (edit takes no out_dir)

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

	// InsideForest per tree (TreeID without offset) for B (select removed trees inside the forest map)
	std::unordered_map<long long, bool> insideForest;
	std::string featPath = TSICommon::join(kEditOutDir, "features_dino.csv");

	if (TSICommon::exists(featPath))
	{
		for (auto& kv : loadFeatureRows(featPath, courseName))
		{
			insideForest[kv.first - course.treeIdOffset] = kv.second.insideForest;
		}
	}

	std::cout << "  InsideForest flags: " << insideForest.size() << " (" << featPath << ")\n";

	TreeEditor::run(aerial, trees.tps, exclusionPath, "TSI Tree Editor - " + courseName, course.aerialImgPath, &insideForest);

	return 0;
}

// TreeSpeciesIdentification.exe                                    ← 기존 파이프라인
// TreeSpeciesIdentification.exe experiment extract <입력> out\ models\dinov2_vits14.onnx
// TreeSpeciesIdentification.exe experiment eval out\
// TreeSpeciesIdentification.exe --experiment export ..\..\Resources\courses_dataset.json out\ Hwarang
// TreeSpeciesIdentification.exe --experiment render ..\..\Resources\courses_dataset.json out\ Hwarang [--skip-forest]
// TreeSpeciesIdentification.exe --experiment review ..\..\Resources\courses_dataset.json Hwarang
// TreeSpeciesIdentification.exe --experiment prepare ..\..\Resources\courses_dataset.json Hwarang|all
// TreeSpeciesIdentification.exe --experiment migrate-excluded ..\..\Resources\courses_dataset.json Hwarang|all

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