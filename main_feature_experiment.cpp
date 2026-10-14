
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

// TreeSpeciesIdentification.exe                                    ← 기존 파이프라인
// TreeSpeciesIdentification.exe experiment extract <입력> out\ models\dinov2_vits14.onnx
// TreeSpeciesIdentification.exe experiment eval out\

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