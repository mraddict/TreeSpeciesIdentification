
#include "TSICommon.h"
/*
#include "AerialLoader.h"
#include "ForestLoader.h"
#include "TreeLoader.h"
*/
#include "CourseDataSetLoader.h"

#include "OverlayRenderer.h"
#include "TreeLabeler.h"
#include "FeatureExtractor.h"
#include "SpeciesClassifier.h"
#include "RefinedClassifier.h"
#include "Hierarchicalclassifier.h"
#include "OutputLogger.h"

#include "AerialReproject.h"
#include "AerialHistogramMatcher.h"

#include "CourseCrossValidator.h"
#include "DinoEmbedder.h"

// Merge features from multiple courses 
// normalizePhoto: true이면 항공사진 기반 특징(0~16)을 코스별 z-score 정규화
ExtractionResult mergeFeatures_old(const std::vector<ExtractionResult>& perCourse, const std::vector<CourseDataSet>& courseDSs)
{
	ExtractionResult merged;

	for (auto& feat : perCourse)
	{
		for (auto& tf : feat.all)
		{
			merged.all.push_back(tf);
		}
	}

	// Rebuild train/predict pointers
	for (auto& tf : merged.all)
	{
		if (tf.insideForest && tf.reliable)
		{
			merged.train.push_back(&tf);
		}
		else if (!tf.insideForest)
		{
			merged.predict.push_back(&tf);
		}
	}

	return merged;
}

// normalizePhoto: true이면 항공사진 기반 특징(0~16)을 코스별 z-score 정규화
ExtractionResult mergeFeatures(const std::vector<ExtractionResult>& perCourse, bool normalizePhoto = false)
{
	// 항공사진 기반 특징 수 (BGR 6 + HSV 3 + 텍스처 8 = 17)
	const int PHOTO_FEATURES = 17;
	ExtractionResult merged;

	// 공통 수종 찾기 (모든 코스의 학습 풀에 존재하는 수종) 
	std::set<std::string> commonSpecies;

	if (normalizePhoto && perCourse.size() > 1)
	{
		// 첫 코스의 학습 풀 수종 수집
		for (auto& tf : perCourse[0].all)
		{
			if (tf.insideForest && tf.reliable && !tf.speciesCode.empty())
			{
				commonSpecies.insert(tf.speciesCode);
			}
		}

		// 나머지 코스와 교집합
		for (size_t c = 1 ; c < perCourse.size() ; ++c)
		{
			std::set<std::string> courseSpecies;

			for (auto& tf : perCourse[c].all)
			{
				if (tf.insideForest && tf.reliable && !tf.speciesCode.empty())
				{
					courseSpecies.insert(tf.speciesCode);
				}
			}

			// 교집합: commonSpecies에 있지만 courseSpecies에 없는 것 제거
			std::set<std::string> intersection;

			for (auto& sp : commonSpecies)
			{
				if (courseSpecies.count(sp))
				{
					intersection.insert(sp);
				}
			}

			commonSpecies = intersection;
		}

		std::cout << "  Common species for normalization: " << commonSpecies.size() << " -";
		
		for (auto& sp : commonSpecies)
		{
			std::cout << " " << sp;
		}

		std::cout << "\n";
	}

	for (size_t c = 0; c < perCourse.size(); ++c)
	{
		auto& feat = perCourse[c];

		if (!normalizePhoto || feat.all.empty())
		{
			// 정규화 없이 그대로 복사

			for (auto& tf : feat.all)
			{
				merged.all.push_back(tf);
			}

			continue;
		}

		// 코스별 평균/표준편차 계산 (공통 수종의 학습 풀만 대상) 
		int nFeat = std::min(PHOTO_FEATURES, (int)feat.all[0].feature.size());
		std::vector<double> sum(nFeat, 0.0);
		std::vector<double> sumSq(nFeat, 0.0);
		int count = 0;

		for (auto& tf : feat.all)
		{
			if (!tf.insideForest || !tf.reliable)
			{
				continue;
			}

			// 공통 수종만 사용
			if (!commonSpecies.count(tf.speciesCode))
			{
				continue;
			}

			for (int i = 0; i < nFeat; ++i)
			{
				sum[i] += tf.feature[i];
				sumSq[i] += tf.feature[i] * tf.feature[i];
			}

			++count;
		}

		std::vector<double> mean(nFeat, 0.0);
		std::vector<double> stddev(nFeat, 0.0);

		if (count > 1)
		{
			for (int i = 0 ; i < nFeat ; ++i)
			{
				mean[i] = sum[i] / count;
				double var = sumSq[i] / count - mean[i] * mean[i];
				stddev[i] = (var > 0.0) ? std::sqrt(var) : 1.0;
			}
		}

		std::cout << "  Course " << c << ": normalization base = " << count << " trees (common species only)\n";

		// 정규화 적용하여 복사 
		for (auto& tf : feat.all)
		{
			TreeFeature copy = tf;

			for (int i = 0 ; (i < nFeat) && (i < (int)copy.feature.size()) ; ++i)
			{
				copy.feature[i] = (float)((copy.feature[i] - mean[i]) / stddev[i]);
			}

			// 특징 17~24 (LiDAR)는 그대로 유지
			merged.all.push_back(copy);
		}
	}

	// Rebuild train/predict pointers
	for (auto& tf : merged.all)
	{
		if (tf.insideForest && tf.reliable)
		{
			merged.train.push_back(&tf);
		}
		else if (!tf.insideForest)
		{
			merged.predict.push_back(&tf);
		}
	}

	return merged;
}

// Save merged features CSV with CourseID column 
void saveMergedCSV(const std::vector<ExtractionResult>& perCourse, const std::vector<CourseDataSet>& courseDSs, const std::string& path)
{
	std::ofstream out(path);

	out << "CourseID,TreeID,InsideForest,Reliable,DistToEdge,ImagePath,"
		<< "SpeciesCode,SpeciesName,"
		<< "B_mean,G_mean,R_mean,B_std,G_std,R_std,"
		<< "H_mean,S_mean,V_mean,Contrast,Entropy,"
		<< "LBP_mean,LBP_std,GLCM_energy,GLCM_corr,GLCM_homo,GLCM_dissim,"
		<< "Height,CrownD,H_CrownD\n";

	for (size_t c = 0 ; c < perCourse.size() ; ++c)
	{
		long long offset = courseDSs[c].treeIdOffset;

		for (auto& tf : perCourse[c].all)
		{
			out << courseDSs[c].courseName << ","
				<< (tf.treeId + offset) << ","
				<< (tf.insideForest ? "Y" : "N") << ","
				<< (tf.reliable ? "Y" : "N") << ","
				<< std::fixed << std::setprecision(2) << tf.distToEdge << ","
				<< tf.imagePath << ","
				<< tf.speciesCode << ","
				<< tf.speciesName << ",";

			for (size_t f = 0 ; f < tf.feature.size() ; ++f)
			{
				if (f > 0)
				{
					out << ",";
				}

				out << std::fixed << std::setprecision(4) << tf.feature[f];
			}

			out << "\n";
		}
	}

	std::cout << "  Merged features saved: " << path << "\n";
}

void reprojectAerialPhoto(const std::string& srcPath, const std::string& dstPath)
{
	ReprojectOptions opts;
	opts.srcEpsg = 32652;		// 원본: WGS84 UTM 52N
	opts.dstEpsg = 5179;		// 대상: UTM-K GRS80
	opts.dstResX = 0.05;		// 5cm로 리샘플링 (3cm → 5cm)
	opts.dstResY = 0.05;
	opts.numThreads = 4;
	opts.memoryLimit = 1024;  // 13GB 이미지이니 메모리 넉넉히}

	auto report = AerialReproject::reproject(srcPath, dstPath, opts);

	AerialReproject::printReport(report);
}

int runFeatureExperiment(int argc, char** argv);
int runExtract(std::vector<LabeledTree>& labeled, const std::string& outDir, const std::string& onnxPath);
int runEval(const std::string& outDir);
int runSweep(const std::string& outDir);
int runLidar(const std::string& outDir);
int runExport(const std::string& configPath, const std::string& outDir, const std::string& courseName);
int runRender(const std::string& configPath, const std::string& outDir, const std::string& courseName, bool skipForest);
int runEdit(const std::string& configPath, const std::string& courseName);
int runReview(const std::string& configPath, const std::string& outDir, const std::string& courseName);
int runPrepare(const std::string& configPath, const std::string& outDir, const std::string& target);
int runMigrateExcluded(const std::string& configPath, const std::string& outDir, const std::string& target);

void printUsage(const char* prog)
{
	std::cerr
		<< "Usage:\n"
		<< "  " << prog << " --config <courses.json> [options]\n"
		<< "  " << prog << " --resume features <merged_features.csv> [options]\n\n"
		<< "Options:\n"
		<< "  --output <dir>          Output directory (default: results_multi)\n"
		<< "  --edge-dist <m>         Min edge distance (default: 10)\n"
		<< "  --trees <N>             RF tree count (default: 200)\n"
		<< "  --max-depth <N>         RF max depth (default: 20)\n"
		<< "  --no-self-train         Disable self-training\n"
		<< "  --save-overlay          Save result overlay images per course\n"
		<< "  --normalize-photo       Z-score normalize photo features (0-16) per course\n"
		<< "  --result-csv <path>     Output CSV path\n";
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		printUsage(argv[0]);

		return 1;
	}

	OutputLogger logger("results", "log");

	std::string outputDir = ""; //"/labeled_trees_05";
	std::string featureCSV = "/tree_features.csv";
	std::string mergedFeatureCSV = "/merged_tree_features.csv";
	std::string modelDir = "/species_model";
	std::string modelCSV = "/species_result.csv";

	_mkdir(outputDir.c_str());

	bool resumeFeature = false;
	bool experiment = false;
	bool isSweepMethod = false;
	bool isLidarMethod = false;

	std::string courseDataSetConfigFilename = "../../Resources/courses_dataset.json";
	int argIdx = 1;

	if (std::string(argv[1]) == "--config" && argc >= 3)
	{
		courseDataSetConfigFilename = argv[2];
		argIdx = 3;
	}
	else if (std::string(argv[1]) == "--resume" && argc >= 4 && std::string(argv[2]) == "features")
	{
		featureCSV = argv[3];
		argIdx = 4;

		resumeFeature = true;
	}
	else 	if (std::string(argv[1]) == "--experiment" && argc >= 2)
	{
		experiment = true;
		resumeFeature = false;

		if (std::string(argv[2]) == "eval")
		{
			resumeFeature = true;
		}

		if (std::string(argv[2]) == "sweep")
		{
			resumeFeature = true;
			isSweepMethod = true;
		}

		if (std::string(argv[2]) == "lidar")
		{
			resumeFeature = true;
			isLidarMethod = true;
		}
	}
	else
	{
		printUsage(argv[0]);

		return 1;
	}

 	TSICommon::initialize();
	ForestCommon::initialize();

	if (experiment && (std::string(argv[2]) == "edit"))
	{
		if (argc < 5)
		{
			std::cerr << "edit needs <courses_dataset.json> <course>\n";

			return 1;
		}

		try
		{
			return runEdit(argv[3], argv[4]);
		}
		catch (const std::exception& e)
		{
			std::cerr << "ERROR: " << e.what() << "\n";

			return 2;
		}
	}

	if (experiment && (std::string(argv[2]) == "migrate-excluded"))
	{
		if (argc < 5)
		{
			std::cerr << "migrate-excluded needs <courses_dataset.json> <course|all>\n";

			return 1;
		}

		try
		{
			return runMigrateExcluded(argv[3], "out", argv[4]);
		}
		catch (const std::exception& e)
		{
			std::cerr << "ERROR: " << e.what() << "\n";

			return 2;
		}
	}

	if (experiment && (std::string(argv[2]) == "prepare"))
	{
		if (argc < 5)
		{
			std::cerr << "prepare needs <courses_dataset.json> <course|all>\n";

			return 1;
		}

		try
		{
			return runPrepare(argv[3], "out", argv[4]);
		}
		catch (const std::exception& e)
		{
			std::cerr << "ERROR: " << e.what() << "\n";

			return 2;
		}
	}

	if (experiment && (std::string(argv[2]) == "review"))
	{
		if (argc < 5)
		{
			std::cerr << "review needs <courses_dataset.json> <course>\n";

			return 1;
		}

		try
		{
			return runReview(argv[3], "out", argv[4]);
		}
		catch (const std::exception& e)
		{
			std::cerr << "ERROR: " << e.what() << "\n";

			return 2;
		}
	}

	if (experiment && (std::string(argv[2]) == "render"))
	{
		if (argc < 6)
		{
			std::cerr << "render needs <courses_dataset.json> <out_dir> <course> [--skip-forest]\n";

			return 1;
		}

		bool skipForest = false;

		for (int i = 6 ; i < argc ; ++i)
		{
			if (std::string(argv[i]) == "--skip-forest")
			{
				skipForest = true;
			}
		}

		try
		{
			return runRender(argv[3], argv[4], argv[5], skipForest);
		}
		catch (const std::exception& e)
		{
			std::cerr << "ERROR: " << e.what() << "\n";

			return 2;
		}
	}

	if (experiment && (std::string(argv[2]) == "export"))
	{
		if (argc < 6)
		{
			std::cerr << "export needs <courses_dataset.json> <out_dir> <course>\n";

			return 1;
		}

		try
		{
			return runExport(argv[3], argv[4], argv[5]);
		}
		catch (const std::exception& e)
		{
			std::cerr << "ERROR: " << e.what() << "\n";

			return 2;
		}
	}
	
	//reprojectAerialPhoto("../../Resources/Diamond/DiamondCC_Orthomosic_BackGround.tif", "../../Resources/Diamond/DiamondCC_orthomosaic_5179_05cm.tif");

	// Options
	HierarchicalOptions hOpts;
	
	bool saveOverlay = true;
	bool normalizePhoto = true;

	for (int i = argIdx; i < argc; ++i)
	{
		std::string arg = argv[i];

		if (arg == "--output" && i + 1 < argc)
		{
			outputDir = argv[++i];
		}
		else if (arg == "--edge-dist" && i + 1 < argc)
		{
			hOpts.minEdgeDist = std::stod(argv[++i]);
		}
		else if (arg == "--trees" && i + 1 < argc)
		{
			hOpts.level1Opts.maxTrees = std::stoi(argv[i + 1]);
			hOpts.level2Opts.maxTrees = std::stoi(argv[++i]);
		}
		else if (arg == "--max-depth" && i + 1 < argc)
		{
			hOpts.level1Opts.maxDepth = std::stoi(argv[i + 1]);
			hOpts.level2Opts.maxDepth = std::stoi(argv[++i]);
		}
		else if (arg == "--no-self-train")
		{
			hOpts.selfTraining = false;
		}
		else if (arg == "--save-overlay")
		{
			saveOverlay = true;
		}
		else if (arg == "--normalize-photo")
		{
			normalizePhoto = true;
		}
		else if (arg == "--result-csv" && i + 1 < argc)
		{
			hOpts.resultCsv = argv[++i];
		}
	}

	//hOpts.maxRemoveRatio = 0.3;	

/*
	AerialHistogramMatcher matcher;
	matcher.runWithTrees(courseDataSetConfigFilename, "AnseongW", "../../Resources/matched_images");

	return 0;
*/

	CourseDataSetConfig courseDSConfig = CourseDataSetConfigLoader::load(courseDataSetConfigFilename);
	TreeExclusion::registerConfig(courseDSConfig);

	try
	{
		ExtractionResult features;

		std::vector<ExtractionResult> perCourseFeatures;
		std::vector<std::unique_ptr<AerialPhoto>> perCourseAerial;
		std::vector<std::unique_ptr<ForestLayer>> perCourseForeset;
		std::vector<std::unique_ptr<TreeData>> perCourseTrees;

		if (resumeFeature)
		{
			std::cout << "  [Resuming from trained model]\n\n";

			if (experiment)
			{
				if (isSweepMethod)
				{
					return runSweep(argv[3]);
				}
				else if (isLidarMethod)
				{
					return runLidar(argv[3]);
				}

				return runEval(argv[3]);
			}

			features = FeatureExtractor::loadCSV(featureCSV);
		}
		else
		{
			// Full pipeline: process each course 
			
			outputDir = courseDSConfig.outputDir;

			if (hOpts.resultCsv.empty())
			{
				hOpts.resultCsv = courseDSConfig.outputDir + "/hierarchical_result.csv";
			}

			std::vector<LabeledTree> allLabeledTree;

			for (auto& course : courseDSConfig.courses)
			{
				std::unique_ptr<AerialPhoto> pAerialImg;
				std::unique_ptr<ForestLayer> pForest;
				std::unique_ptr<TreeData> pTrees;				

				if (!course.enabled)
				{
					std::cout << "\n=== Skipping " << course.courseName << " (disabled) ===\n";

					continue;
				}

				std::cout << "\n========================================\n";
				std::cout << "  Processing: " << course.courseName << "\n";
				std::cout << "========================================\n";

				// 1. Load data

				// 항공사진
				std::cout << "\n=== Loading aerial ===\n";
				//auto pAerialImg = std::make_unique<AerialPhoto>(AerialLoader::load(course.aerialImgPath));
				pAerialImg.reset(new AerialPhoto(AerialLoader::load(course.aerialImgPath)));
				// 메타데이터 출력
				AerialLoader::printInfo(*pAerialImg);

				// 임상도
				std::cout << "\n=== Loading forest ===\n";
				pForest.reset(new ForestLayer(ForestLoader::loadMultiple(course.shpFiles, "", false)));
				// 메타데이터 출력
				ForestLoader::printInfo(*pForest);

				// 나무 CSV 로드 + 좌표 변환
				pTrees.reset(new TreeData(TreeLoader::load(course.treeInfoPath, course.treeSrcEPSG, course.targetEPSG, 0.0, 0.0, course.treeExcludePath)));

				// Apply ID offset
				pTrees->treeIdOffset = course.treeIdOffset;

				for (auto& tp : pTrees->tps)
				{
					tp.treeId += course.treeIdOffset;
				}
				
				// 메타데이터 출력
				TreeLoader::printInfo(*pTrees);

				// 2. Label trees
				TSICommon::mkdirs(course.outputDir);

				LabelOptions labelOpts;

				labelOpts.outputDir = course.outputDir;
				labelOpts.aerialCrsEpsg = pAerialImg->epsg;
				labelOpts.forestCrsEpsg = pForest->epsg;

				std::cout << "\n=== Labeling ===\n";
				auto labeled = TreeLabeler::label(*pAerialImg, *pForest, *pTrees, labelOpts);

				// 3. Extract features
				if (experiment)
				{
					allLabeledTree.insert(allLabeledTree.end(), labeled.begin(), labeled.end());
				}

				std::cout << "\n=== Extracting features ===\n";
				ExtractionResult courseFeatures = FeatureExtractor::extractFromFiles(labeled);
				//features = FeatureExtractor::extractFromAerialImage(labeled, *pAerialImg);

				// 4. Save feature CSV
				FeatureExtractor::saveCSV(courseFeatures, course.outputDir + featureCSV);

				// 5. Feature statistics
				FeatureExtractor::printFeatureStats(courseFeatures);

				OverlayOptions overlayOpt;
				overlayOpt.saveScale = 0.5;
				OverlayRenderer::saveOverlay(*pAerialImg, *pForest, *pTrees, course.outputDir + "/aerial_forest_tree_combined.png");

				perCourseFeatures.push_back(std::move(courseFeatures));
				perCourseAerial.push_back(std::move(pAerialImg));
				perCourseForeset.push_back(std::move(pForest));
				perCourseTrees.push_back(std::move(pTrees));
			}

			if (experiment)
			{
				// TreeSpeciesIdentification.exe experiment extract <입력> out\ models\dinov2_vits14.onnx
				return runExtract(allLabeledTree, "out", "../../Resources/models/dinov2_vits14.onnx");
			}

			// Merge all features
			std::cout << "\n========================================\n";
			std::cout << "  Merging " << perCourseFeatures.size() << " datasets\n";
			std::cout << "========================================\n";

			// Get enabled configs for merge
			std::vector<CourseDataSet> enabledDSs;

			for (auto& c : courseDSConfig.courses)
			{
				if (c.enabled)
				{
					enabledDSs.push_back(c);
				}
			}

			features = mergeFeatures(perCourseFeatures, normalizePhoto);

			std::cout << "  Total trees: " << features.all.size() << "\n";
			std::cout << "  Train pool:  " << features.train.size() << "\n";
			std::cout << "  Predict:     " << features.predict.size() << "\n";

			if (normalizePhoto)
			{
				std::cout << "  Photo feature normalization: ON (features 0-16)\n";
			}
			
			// Save merged CSV
			TSICommon::mkdirs(courseDSConfig.outputDir);
			saveMergedCSV(perCourseFeatures, enabledDSs, courseDSConfig.outputDir + mergedFeatureCSV);

			// statistics
			FeatureExtractor::printFeatureStats(features);
		}

/*
		// Run refined classification
		RefinementOptions refOpts;
		refOpts.rfOpts.modelPath = modelDir;
		refOpts.rfOpts.resultCsv = modelCSV;
		refOpts.saveDebugImages = true;
		refOpts.debugOutputDir = "./labeled_trees_05";
		refOpts.cropSourceDir = "./labeled_trees_05";

		auto classifier = RefinedClassifier::run(features, refOpts);
*/
		
		// Train hierarchical classifier
		HierarchicalClassifier hClf;
		hClf.train(features, hOpts);

		// Predict
		auto results = hClf.predict(features);

		// Evaluate
		HierarchicalClassifier::evaluate(results);

		// Save
		HierarchicalClassifier::saveResults(results, hOpts.resultCsv);
		
		// Per-course overlay 
		if (saveOverlay && !perCourseAerial.empty())
		{
			std::cout << "\n=== Rendering overlays per course ===\n";

			// Build result lookup by treeId
			std::map<long long, const HierarchicalResult*> resultMap;

			for (auto& r : results)
			{
				resultMap[r.treeId] = &r;
			}

			size_t courseIdx = 0;

			for (size_t ci = 0 ; ci < courseDSConfig.courses.size() ; ++ci)
			{
				if (!courseDSConfig.courses[ci].enabled)
				{
					continue;
				}

				if (courseIdx >= perCourseAerial.size())
				{
					break;
				}

				auto& course = courseDSConfig.courses[ci];
				std::cout << "\n  Overlay: " << course.courseName << "\n";

				// Collect results for this course
				// All IDs already have offset from loading

				std::vector<HierarchicalResult> courseResults;

				for (auto& tf : perCourseFeatures[courseIdx].all)
				{
					auto it = resultMap.find(tf.treeId);

					if (it != resultMap.end())
					{
						courseResults.push_back(*it->second);
					}
				}

				OverlayOptions roOpts;

				roOpts.drawPolygons = true;
				roOpts.polyAlpha = 0.15;

				roOpts.drawLegend = true;
				roOpts.polyLineWidth = 1;
				roOpts.crownScale = 0.5;
				roOpts.crownAlpha = 0.4;
				roOpts.crownOutlineWidth = 1;

				roOpts.saveScale = 0.5;

				// Species overlay
				std::string ovSpeciesPath = course.outputDir + "/result_species_overlay.png";
				roOpts.resultTitle = course.courseName + " - Predicted Species";
				roOpts.resultColorMode = OverlayOptions::RESULT_BY_SPECIES;

				OverlayRenderer::saveResultOverlay(*perCourseAerial[courseIdx], *perCourseForeset[courseIdx], *perCourseTrees[courseIdx], courseResults, ovSpeciesPath, roOpts);

				// Group overlay
				std::string ovGroupPath = course.outputDir + "/result_group_overlay.png";
				roOpts.resultTitle = course.courseName + "Predicted Group (C/D/E/N)";
				roOpts.resultColorMode = OverlayOptions::RESULT_BY_GROUP;

				OverlayRenderer::saveResultOverlay(*perCourseAerial[courseIdx], *perCourseForeset[courseIdx], *perCourseTrees[courseIdx], courseResults, ovGroupPath, roOpts);

				// Correctness overlay
				std::string ovCorrPath = course.outputDir + "/result_correctness_overlay.png";
				roOpts.resultTitle = course.courseName + "Correctness";
				roOpts.resultColorMode = OverlayOptions::RESULT_CORRECTNESS;

				OverlayRenderer::saveResultOverlay(*perCourseAerial[courseIdx], *perCourseForeset[courseIdx], *perCourseTrees[courseIdx], courseResults, ovCorrPath, roOpts);

				++courseIdx;
			}
		}

		std::cout << "\n=== Done ===\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << "\nError: " << e.what() << "\n";

		return 1;
	}

/*
	// Predict
	auto results = classifier.predict(features);

	// Evaluate
	std::cout << "\n=== Step 6: Evaluation ===\n";
	SpeciesClassifier::evaluate(results);
	SpeciesClassifier::saveResults(results, clsOpts.resultCsv);
*/
	
	return 0;
}
