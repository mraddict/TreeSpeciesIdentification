
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

// ─── Merge features from multiple courses ───
ExtractionResult mergeFeatures(const std::vector<ExtractionResult>& perCourse, const std::vector<CourseDataSet>& courseDSs)
{
	ExtractionResult merged;

	for (size_t c = 0 ; c < perCourse.size() ; ++c)
	{
		const auto& f = perCourse[c];
		long long offset = courseDSs[c].treeIdOffset;

		// Deep copy all TreeFeature with offset applied
		for (auto& tf : f.all)
		{
			TreeFeature copy = tf;
			copy.treeId += offset;
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

// ─── Save merged features CSV with CourseID column ───
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

int main()
{
 	TSICommon::initialize();
	ForestCommon::initialize();

	//reprojectAerialPhoto("../../Resources/Diamond/DiamondCC_Orthomosic_BackGround.tif", "../../Resources/Diamond/DiamondCC_orthomosaic_5179_05cm.tif");

	OutputLogger logger("../results", "log");

	std::string outputDir = ""; //"/labeled_trees_05";
	std::string featureCSV = "/tree_features.csv";
	std::string mergedFeatureCSV = "/merged_tree_features.csv";
	std::string modelDir = "/species_model";
	std::string modelCSV = "/species_result.csv";
	/*
	std::string featureCSV = outputDir + "/tree_features.csv";
	std::string mergedFeatureCSV = outputDir + "/merged_tree_features.csv";
	std::string modelDir = outputDir + "/species_model";
	std::string modelCSV = modelDir + "/species_result.csv";
*/
	 
	_mkdir(outputDir.c_str());

	// 항공사진
	std::string aerialImgPath = "../../Resources/AnseongW/AnSung_W_orthomosaic_5179_05cm.tif";
	//std::string aerialImgPath = "../../Resources/AnseongW/AnSung_W_orthomosaic_5179_20cm.tif";

	// 임상도
	std::vector<std::string> SHPFiles =
	{
		"../../Resources/AnseongW/treemap/37713068/37713068.shp",
		"../../Resources/AnseongW/treemap/37713069/37713069.shp",
		"../../Resources/AnseongW/treemap/37713078/37713078.shp",
		"../../Resources/AnseongW/treemap/37713079/37713079.shp",
		//"./forest_merged.shp"
	};

	// 나무위치
	std::string treeInfoPath = "../../Resources/AnseongW/tree_info.csv";

	std::string courseDataSetConfigFilename = "../../Resources/courses_dataset.json";
	CourseDataSetConfig courseDSConfig = CourseDataSetConfigLoader::load(courseDataSetConfigFilename);

	bool resumeFeature = false;

	std::vector<ExtractionResult> perCourseFeatures;
	std::vector<std::unique_ptr<AerialPhoto>> perCourseAerial;
	std::vector<std::unique_ptr<ForestLayer>> perCourseForeset;
	std::vector<std::unique_ptr<TreeData>> perCourseTrees;

	try
	{
		ExtractionResult features;

		if (resumeFeature)
		{
			std::cout << "  [Resuming from trained model]\n\n";
			features = FeatureExtractor::loadCSV(featureCSV);
		}
		else
		{
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
				pTrees.reset(new TreeData(TreeLoader::load(course.treeInfoPath, course.treeSrcEPSG, course.targetEPSG)));

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
				std::cout << "\n=== Extracting features ===\n";
				ExtractionResult courseFeatures = FeatureExtractor::extractFromFiles(labeled);
				//features = FeatureExtractor::extractFromAerialImage(labeled, *pAerialImg);

				// 4. Save feature CSV
				FeatureExtractor::saveCSV(courseFeatures, course.outputDir + featureCSV);

				// 5. Feature statistics
				FeatureExtractor::printFeatureStats(features);

				perCourseFeatures.push_back(std::move(courseFeatures));
				perCourseAerial.push_back(std::move(pAerialImg));
				perCourseForeset.push_back(std::move(pForest));
				perCourseTrees.push_back(std::move(pTrees));

				OverlayRenderer::saveOverlay(*pAerialImg, *pForest, *pTrees, outputDir + "/aerial_forest_tree_combined.png");
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

			features = mergeFeatures(perCourseFeatures, enabledDSs);

			std::cout << "  Total trees: " << features.all.size() << "\n";
			std::cout << "  Train pool:  " << features.train.size() << "\n";
			std::cout << "  Predict:     " << features.predict.size() << "\n";

			// Save merged CSV
			TSICommon::mkdirs(courseDSConfig.outputDir);
			saveMergedCSV(perCourseFeatures, enabledDSs, courseDSConfig.outputDir + mergedFeatureCSV);
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
		HierarchicalOptions hOpts;
		hOpts.selfTraining = false;
		//hOpts.maxRemoveRatio = 0.3;
		hOpts.resultCsv = courseDSConfig.outputDir + "/hierarchical_result.csv";

		// Train hierarchical classifier
		HierarchicalClassifier hClf;
		hClf.train(features, hOpts);

		// Predict
		auto results = hClf.predict(features);

		// Evaluate
		HierarchicalClassifier::evaluate(results);

		// Save
		HierarchicalClassifier::saveResults(results, hOpts.resultCsv);

/*

		
		// Render overlay if requested
		bool saveOverlay = true;
		if (saveOverlay)
		{
			if (aerialImgPath.empty() || SHPFiles.empty() || treeInfoPath.empty())
			{
				std::cout << "\n  Overlay skipped: need --overlay-aerial, --overlay-forest, --overlay-tree\n";
			}
			else
			{
				std::cout << "\n=== Rendering overlay ===\n";

				// 항공사진
				if (!pAerialImg)
				{
					pAerialImg.reset(new AerialPhoto(AerialLoader::load(aerialImgPath)));
				}
				
				// 임상도
				if (!pForest)
				{
					pForest.reset(new ForestLayer(ForestLoader::loadMultiple(SHPFiles, "", false)));
				}
				
				// 나무 CSV 로드 + 좌표 변환 (5186 → 5179)
				if (!pTrees)
				{
					pTrees.reset(new TreeData(TreeLoader::load(treeInfoPath, 5186, 5179)));
				}

				if (!pAerialImg || !pForest || !pTrees)
				{
					std::cout << "  Overlay skipped: need aerial, forest, tree data\n"
						<< "  Use --overlay-aerial, --overlay-forest, --overlay-tree\n";
				}
				else
				{
					OverlayOptions roOpts;

					roOpts.drawPolygons = true;
					roOpts.polyAlpha = 0.15;

					roOpts.drawLegend = true;
					roOpts.polyLineWidth = 1;
					roOpts.crownScale = 0.5;
					roOpts.crownAlpha = 0.4;
					roOpts.crownOutlineWidth = 1;

					// Species overlay
					std::string ovSpeciesPath = outputDir + "/result_species_overlay.png";
					roOpts.resultTitle = "Predicted Species";
					roOpts.resultColorMode = OverlayOptions::RESULT_BY_SPECIES;

					OverlayRenderer::saveResultOverlay(*pAerialImg, *pForest, *pTrees, results, ovSpeciesPath, roOpts);

					// Group overlay
					std::string ovGroupPath = outputDir + "/result_group_overlay.png";
					roOpts.resultTitle = "Predicted Group (C/D/E/N)";
					roOpts.resultColorMode = OverlayOptions::RESULT_BY_GROUP;

					OverlayRenderer::saveResultOverlay(*pAerialImg, *pForest, *pTrees, results, ovGroupPath, roOpts);

					// Correctness overlay
					std::string ovCorrPath = outputDir + "/result_correctness_overlay.png";
					roOpts.resultTitle = "Correctness";
					roOpts.resultColorMode = OverlayOptions::RESULT_CORRECTNESS;

					OverlayRenderer::saveResultOverlay(*pAerialImg, *pForest, *pTrees, results, ovCorrPath, roOpts);
				}
				
			}
		}

		std::cout << "\n=== Done ===\n";
*/
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
