
#include "TSICommon.h"
/*
#include "AerialLoader.h"
#include "ForestLoader.h"
#include "TreeLoader.h"
*/
#include "OverlayRenderer.h"
#include "TreeLabeler.h"
#include "FeatureExtractor.h"
#include "SpeciesClassifier.h"
#include "RefinedClassifier.h"
#include "Hierarchicalclassifier.h"
#include "OutputLogger.h"

int main()
{
	TSICommon::initialize();
	ForestCommon::initialize();

	OutputLogger logger("results", "log");

	std::string outputDir = "./labeled_trees_05";
	std::string featureCSV = outputDir + "/tree_features.csv";
	std::string modelDir = outputDir + "/species_model";
	std::string modelCSV = modelDir + "/species_result.csv";

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

	bool resumeFeature = false;

	std::unique_ptr<AerialPhoto> pAerialImg;
	std::unique_ptr<ForestLayer> pForest;
	std::unique_ptr<TreeData> pTrees;

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
			// 1. Load data
	
			// 항공사진
			pAerialImg.reset(new AerialPhoto(AerialLoader::load(aerialImgPath)));
			// 메타데이터 출력
			AerialLoader::printInfo(*pAerialImg);

			// 임상도
			pForest.reset(new ForestLayer(ForestLoader::loadMultiple(SHPFiles, "", false)));
			// 메타데이터 출력
			ForestLoader::printInfo(*pForest);

			// 나무 CSV 로드 + 좌표 변환 (5186 → 5179)
			pTrees.reset(new TreeData(TreeLoader::load(treeInfoPath, 5186, 5179)));
			// 메타데이터 출력
			TreeLoader::printInfo(*pTrees);

			OverlayRenderer::saveOverlay(*pAerialImg, *pForest, *pTrees, outputDir + "/aerial_forest_tree_combined.png");

			// 2. Label trees

			LabelOptions labelOpts;

			labelOpts.outputDir = outputDir;
			labelOpts.aerialCrsEpsg = pAerialImg->epsg;
			labelOpts.forestCrsEpsg = pForest->epsg;

			auto result = TreeLabeler::label(*pAerialImg, *pForest, *pTrees, labelOpts);

			// 3. Extract features
			std::cout << "\n";

			features = FeatureExtractor::extractFromFiles(result);

			// 4. Save feature CSV
			FeatureExtractor::saveCSV(features, featureCSV);

			// 5. Feature statistics
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

		HierarchicalOptions hOpts;
		hOpts.selfTraining = false;
		//hOpts.maxRemoveRatio = 0.3;
		hOpts.resultCsv = outputDir + "/hierarchical_result.csv";
		
		// Train hierarchical classifier
		HierarchicalClassifier hClf;
		hClf.train(features, hOpts);

		// Predict
		auto results = hClf.predict(features);

		// Evaluate
		HierarchicalClassifier::evaluate(results);

		// Save
		HierarchicalClassifier::saveResults(results, hOpts.resultCsv);

		
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
