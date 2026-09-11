#pragma once

#include "SpeciesClassifier.h"

// ─────────────────────────────────────────────
//  Species group definitions
// ─────────────────────────────────────────────

struct SpeciesGroup
{
	std::string groupName;
	std::string groupCode;									// "C", "D", "E", "N"
	std::vector<std::string> speciesCode;
};

// ─────────────────────────────────────────────
//  Hierarchical prediction result
// ─────────────────────────────────────────────

struct HierarchicalResult
{
	long long treeId = -1;

	// Level 1
	std::string predictedGroup;
	std::string predictedGroupName;					// Conifer(C) / Deciduous(D) / Evergreen(E) / Non-forest(N)
	float groupConfidence = 0.0f;

	// Level 2
	std::string predictedCode;							// detailed species code
	std::string predictedName;
	float speciesConfidence = 0.0f;

	// Combined confidence
	float combinedConfidence = 0.0f;

	// Truth (if available)
	std::string trueCode;
	std::string trueGroup;
	bool groupCorrect = false;
	bool speciesCorrect = false;
	double distToEdge = 0.0;
};

// ─────────────────────────────────────────────
//  Options
// ─────────────────────────────────────────────

struct HierarchicalOptions
{
	// Group definitions
	// Conifers (침엽수)
	std::vector<std::string> coniferCodes =
	{
		"11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "10"
	};
	// Deciduous broadleaf (낙엽활엽수)
	std::vector<std::string> deciduousCodes =
	{
		"31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "30"
	};
	// Evergreen broadleaf (상록활엽수)
	std::vector<std::string> evergreenCodes =
	{
		"61", "62", "63", "64", "65", "66", "67", "68", "60"
	};
	// Non-forest / mixed (혼효·비산림)
	std::vector<std::string> nonForestCodes =
	{
		"77", "78", "81", "82", "83", "91", "92", "93", "94", "95", "99"
	};

	// RF parameters per level
	ClassifierOptions level1Opts;
	ClassifierOptions level2Opts;

	// Edge distance filter
	double minEdgeDist = 10.0;

	// Self-training (label noise reduction)
	bool selfTraining = true;					// enable self-training per group
	int selfTrainingIters = 2;					// max iterations
	float maxRemoveRatio = 1.0f;			// max removal per species (1.0 = no limit)

	// Output
	std::string resultCsv = "hierarchical_result.csv";
};

// ─────────────────────────────────────────────
//  HierarchicalClassifier
// ─────────────────────────────────────────────

class HierarchicalClassifier
{
public:

	SpeciesClassifier level1Clf;										// group classifier
	SpeciesClassifier coniferClf;									// conifer detail classifier
	SpeciesClassifier deciduousClf;								// deciduous detail classifier
	SpeciesClassifier evergreenClf;								// evergreen broadleaf detail classifier
	SpeciesClassifier nonForestClf;								// non-forest detail classifier

	std::map<std::string, std::string> codeToGroup;		// species code -> group code

	// ─── Train ───
	void train(ExtractionResult& features, const HierarchicalOptions& opts = HierarchicalOptions())
	{
		std::cout << "========================================\n";
		std::cout << "  Hierarchical Classification\n";
		std::cout << "========================================\n\n";

		// Build code -> group mapping
		buildGroupMapping(opts);

		// Filter by edge distance
		std::vector<TreeFeature*> trainPool;

		for (auto* tf : features.train)
		{
			if (tf->distToEdge >= opts.minEdgeDist)
			{
				trainPool.push_back(tf);
			}
		}

		std::cout << "  Training pool (edge >= " << opts.minEdgeDist << "m): " << trainPool.size() << "\n\n";

		// ════════════════════════════════════
		 //  Level 1: Group classification
		 // ════════════════════════════════════
		std::cout << "=== Level 1: Group Classification ===\n";
		std::cout << "  Conifer / Deciduous / Evergreen / Non-forest\n\n";
	
		// Create group-labeled features
		ExtractionResult groupFeatures = features;
		groupFeatures.train.clear();

		// Temporarily replace speciesCode with group code
		std::map<long long, std::string> originalCodes;			// backup

		for (auto* tf : trainPool)
		{
			auto it = codeToGroup.find(tf->speciesCode);

			if (it == codeToGroup.end())
			{
				continue;
			}

			originalCodes[tf->treeId] = tf->speciesCode;
			tf->speciesCode = it->second;
			groupFeatures.train.push_back(tf);
		}
	
		// Print group distribution
		std::map<std::string, int> groupCounts;

		for (auto* tf : groupFeatures.train)
		{
			groupCounts[tf->speciesCode]++;
		}

		std::cout << "  Group distribution:\n";
		
		for (auto& kv : groupCounts)
		{
			std::cout << "    " << std::setw(3) << kv.first << " (" << groupCodeToName(kv.first) << "): " << kv.second << "\n";
		}

		std::cout << "\n";

		ClassifierOptions l1Opts = opts.level1Opts;
		l1Opts.crossValidate = true;
		level1Clf.train(groupFeatures, l1Opts);

		// Restore original codes
		for (auto* tf : groupFeatures.train)
		{
			auto it = originalCodes.find(tf->treeId);

			if (it != originalCodes.end())
			{
				tf->speciesCode = it->second;
			}
		}
	
		// ════════════════════════════════════
		//  Level 2: Conifer detail
		// ════════════════════════════════════
		std::cout << "\n=== Level 2a: Conifer Species ===\n";

		ExtractionResult coniferFeatures = features;
		coniferFeatures.train.clear();

		for (auto* tf : trainPool)
		{
			if (codeToGroup[tf->speciesCode] == "C")
			{
				coniferFeatures.train.push_back(tf);
			}
		}

		if (coniferFeatures.train.size() >= 20)
		{
			std::cout << "  Conifer training: " << coniferFeatures.train.size() << "\n";
			printDistribution(coniferFeatures.train);

			// Self-training to remove mislabeled trees
			if (opts.selfTraining)
			{
				coniferFeatures.train = selfTrainPool(coniferFeatures.train, features, opts.level2Opts, opts.selfTrainingIters, opts.maxRemoveRatio, "Conifer");

				std::cout << "  After self-training: " << coniferFeatures.train.size() << "\n";
				printDistribution(coniferFeatures.train);
			}

			ClassifierOptions l2Opts = opts.level2Opts;
			l2Opts.crossValidate = true;
			l2Opts.balanceClasses = true;
			coniferClf.train(coniferFeatures, l2Opts);
			hasConiferClf_ = true;
		}
		else
		{
			std::cout << "  Skipped: too few conifer samples (" << coniferFeatures.train.size() << ")\n";
		}
	
		// ════════════════════════════════════
		//  Level 2: Deciduous detail
		// ════════════════════════════════════
		std::cout << "\n=== Level 2b: Deciduous Species ===\n";

		ExtractionResult deciduousFeatures = features;
		deciduousFeatures.train.clear();

		for (auto* tf : trainPool)
		{
			if (codeToGroup[tf->speciesCode] == "D")
			{
				deciduousFeatures.train.push_back(tf);
			}
		}

		if (deciduousFeatures.train.size() >= 20)
		{
			std::cout << "  Deciduous training: " << deciduousFeatures.train.size() << "\n";
			printDistribution(deciduousFeatures.train);

			// Self-training to remove mislabeled trees
			if (opts.selfTraining)
			{
				deciduousFeatures.train = selfTrainPool(deciduousFeatures.train, features, opts.level2Opts, opts.selfTrainingIters, opts.maxRemoveRatio, "Deciduous");
			
				std::cout << "  After self-training: " << deciduousFeatures.train.size() << "\n";
				printDistribution(deciduousFeatures.train);
			}

			ClassifierOptions l2Opts = opts.level2Opts;
			l2Opts.crossValidate = true;
			l2Opts.balanceClasses = true;
			deciduousClf.train(deciduousFeatures, l2Opts);
			hasDeciduousClf_ = true;
		}
		else
		{
			std::cout << "  Skipped: too few deciduous samples (" << deciduousFeatures.train.size() << ")\n";
		}

		// ════════════════════════════════════
		//  Level 2: Evergreen broadleaf detail
		// ════════════════════════════════════
		std::cout << "\n=== Level 2c: Evergreen Broadleaf Species ===\n";

		ExtractionResult evergreenFeatures = features;
		evergreenFeatures.train.clear();

		for (auto* tf : trainPool)
		{
			if (codeToGroup[tf->speciesCode] == "E")
			{
				evergreenFeatures.train.push_back(tf);
			}
		}

		if (evergreenFeatures.train.size() >= 20)
		{
			std::cout << "  Evergreen training: " << evergreenFeatures.train.size() << "\n";
			printDistribution(evergreenFeatures.train);

			// Self-training to remove mislabeled trees
			if (opts.selfTraining)
			{
				evergreenFeatures.train = selfTrainPool(evergreenFeatures.train, features, opts.level2Opts, opts.selfTrainingIters, opts.maxRemoveRatio, "Evergreen");

				std::cout << "  After self-training: " << evergreenFeatures.train.size() << "\n";
				printDistribution(evergreenFeatures.train);
			}

			ClassifierOptions l2Opts = opts.level2Opts;
			l2Opts.crossValidate = true;
			l2Opts.balanceClasses = true;
			evergreenClf.train(evergreenFeatures, l2Opts);
			hasEvergreenClf_ = true;
		}
		else
		{
			std::cout << "  Skipped: too few evergreen samples (" << evergreenFeatures.train.size() << ")\n";
		}
	
		// ════════════════════════════════════
		//  Level 2: Non-forest detail
		// ════════════════════════════════════
		std::cout << "\n=== Level 2d: Non-forest Detail ===\n";

		ExtractionResult nonForestFeatures = features;
		nonForestFeatures.train.clear();

		for (auto* tf : trainPool)
		{
			if (codeToGroup[tf->speciesCode] == "N")
			{
				nonForestFeatures.train.push_back(tf);
			}
		}

		if (nonForestFeatures.train.size() >= 20)
		{
			std::cout << "  Non-forest training: " << nonForestFeatures.train.size() << "\n";
			printDistribution(nonForestFeatures.train);

			// Self-training to remove mislabeled trees
			if (opts.selfTraining)
			{
				nonForestFeatures.train = selfTrainPool(nonForestFeatures.train, features, opts.level2Opts, opts.selfTrainingIters, opts.maxRemoveRatio, "Non-forest");

				std::cout << "  After self-training: " << nonForestFeatures.train.size() << "\n";
				printDistribution(nonForestFeatures.train);
			}

			ClassifierOptions l2Opts = opts.level2Opts;
			l2Opts.crossValidate = true;
			l2Opts.balanceClasses = true;
			nonForestClf.train(nonForestFeatures, l2Opts);
			hasNonForestClf_ = true;
		}
		else
		{
			std::cout << "  Skipped: too few non-forest samples (" << nonForestFeatures.train.size() << ")\n";
		}
}

	// ─── Predict ───
	std::vector<HierarchicalResult> predict(ExtractionResult& features)
	{
		std::cout << "\n=== Hierarchical Prediction ===\n";

		std::vector<HierarchicalResult> results;

		// Save original species codes and distToEdge before any modification
		std::map<long long, std::string> originalCodes;
		std::map<long long, double> originalDistToEdge;
		
		for (auto& tf : features.all)
		{
			originalCodes[tf.treeId] = tf.speciesCode;
			originalDistToEdge[tf.treeId] = tf.distToEdge;
		}

		// Step 1: Predict group for all trees
		// Replace species codes with group codes for Level 1
		for (auto& tf : features.all)
		{
			auto it = codeToGroup.find(tf.speciesCode);
			tf.speciesCode = (it != codeToGroup.end()) ? it->second : "";
		}

		auto groupResults = level1Clf.predict(features);

		// Restore original codes immediately
		for (auto& tf : features.all)
		{
			auto it = originalCodes.find(tf.treeId);

			if (it != originalCodes.end())
			{
				tf.speciesCode = it->second;
			}
		}

		// Step 2: Predict detail species
		std::map<long long, PredictResult> coniferPreds;
		std::map<long long, PredictResult> deciduousPreds;
		std::map<long long, PredictResult> evergreenPreds;
		std::map<long long, PredictResult> nonForestPreds;

		if (hasConiferClf_)
		{
			auto cp = coniferClf.predict(features);
		
			for (auto& pr : cp)
			{
				coniferPreds[pr.treeId] = pr;
			}
		}
		
		if (hasDeciduousClf_)
		{
			auto dp = deciduousClf.predict(features);

			for (auto& pr : dp)
			{
				deciduousPreds[pr.treeId] = pr;
			}
		}

		if (hasEvergreenClf_)
		{
			auto ep = evergreenClf.predict(features);

			for (auto& pr : ep)
			{
				evergreenPreds[pr.treeId] = pr;
			}
		}

		if (hasNonForestClf_)
		{
			auto np = nonForestClf.predict(features);

			for (auto& pr : np)
			{
				nonForestPreds[pr.treeId] = pr;
			}
		}

		// Combine results using original codes for truth
		for (auto& gr : groupResults)
		{
			HierarchicalResult hr;
			hr.treeId = gr.treeId;

			// Level 1 prediction
			hr.predictedGroup = gr.predictedCode;
			hr.predictedGroupName = groupCodeToName(gr.predictedCode);
			hr.groupConfidence = gr.confidence;

			// Truth: use original species code, not the group-replaced one
			auto origIt = originalCodes.find(gr.treeId);

			std::string origCode = (origIt != originalCodes.end()) ? origIt->second : "";
			hr.trueCode = origCode;

			auto groupIt = codeToGroup.find(origCode);
			hr.trueGroup = (groupIt != codeToGroup.end()) ? groupIt->second : "";

			hr.groupCorrect = (!hr.trueGroup.empty() && (hr.predictedGroup == hr.trueGroup));

			// Level 2: based on predicted group
			if (gr.predictedCode == "C" && hasConiferClf_)
			{
				auto cit = coniferPreds.find(gr.treeId);

				if (cit != coniferPreds.end())
				{
					hr.predictedCode = cit->second.predictedCode;
					hr.predictedName = cit->second.predictedName;
					hr.speciesConfidence = cit->second.confidence;
				}
			}
			else if (gr.predictedCode == "D" && hasDeciduousClf_)
			{
				auto dit = deciduousPreds.find(gr.treeId);
			
				if (dit != deciduousPreds.end())
				{
					hr.predictedCode = dit->second.predictedCode;
					hr.predictedName = dit->second.predictedName;
					hr.speciesConfidence = dit->second.confidence;
				}
			}
			else if (gr.predictedCode == "E" && hasEvergreenClf_)
			{
				auto eit = evergreenPreds.find(gr.treeId);
			
				if (eit != evergreenPreds.end())
				{
					hr.predictedCode = eit->second.predictedCode;
					hr.predictedName = eit->second.predictedName;
					hr.speciesConfidence = eit->second.confidence;
				}
			}
			else if (gr.predictedCode == "N" && hasNonForestClf_)
			{
				auto nit = nonForestPreds.find(gr.treeId);

				if (nit != nonForestPreds.end())
				{
					hr.predictedCode = nit->second.predictedCode;
					hr.predictedName = nit->second.predictedName;
					hr.speciesConfidence = nit->second.confidence;
				}
			}
			else
			{
				// Non-forest or no detail classifier
				hr.predictedCode = gr.predictedCode;
				hr.predictedGroupName = groupCodeToName(gr.predictedCode);
				hr.speciesConfidence = gr.confidence;
			}

			hr.combinedConfidence = hr.groupConfidence * hr.speciesConfidence;
			hr.speciesCorrect = (!hr.trueCode.empty() && (hr.predictedCode == hr.trueCode));

			auto distIt = originalDistToEdge.find(gr.treeId);
			hr.distToEdge = (distIt != originalDistToEdge.end()) ? distIt->second : 0.0;

			results.push_back(hr);
		}

		std::cout << "  Predicted: " << results.size() << " trees\n";

		return results;
	}

	// ─── Evaluate ───
	static void evaluate(const std::vector<HierarchicalResult>& results)
	{
		std::cout << "\n=== Hierarchical Evaluation ===\n";
	
		// Level 1: Group accuracy
		int gTotal = 0, gCorrect = 0;
		std::map<std::string, int> gPerClass, gPerClassCorrect;

		// Level 2: Species accuracy
		int sTotal = 0, sCorrect = 0;
		std::map<std::string, int> sPerClass, sPerClassCorrect;

		for (auto& r : results)
		{
			if (r.trueGroup.empty())
			{
				continue;
			}

			++gTotal;
			++gPerClass[r.trueGroup];

			if (r.groupCorrect)
			{
				++gCorrect;
				++gPerClassCorrect[r.trueGroup];
			}

			if (r.trueCode.empty())
			{
				continue;
			}

			++sTotal;
			++sPerClass[r.trueCode];
			
			if (r.speciesCorrect)
			{
				++sCorrect;
				++sPerClassCorrect[r.trueCode];
			}
		}

		// Group accuracy
		std::cout << "\n  --- Level 1: Group Accuracy ---\n";

		double gAcc = (gTotal > 0) ? 100.0 * gCorrect / gTotal : 0;
		std::cout << "  Overall: " << std::fixed << std::setprecision(1) << gAcc << "% (" << gCorrect << "/" << gTotal << ")\n";

		std::cout << "  " << std::left << std::setw(15) << "Group" << std::setw(10) << "Correct" << std::setw(10) << "Total" << "Accuracy\n";
		std::cout << "  " << std::string(45, '-') << "\n";

		for (auto& kv : gPerClass)
		{
			int c = gPerClassCorrect.count(kv.first) ? gPerClassCorrect[kv.first] : 0;
			double acc = (kv.second > 0) ? 100.0 * c / kv.second : 0;

			std::cout << "  " << std::setw(15) << groupCodeToNameStatic(kv.first)
				<< std::setw(10) << c
				<< std::setw(10) << kv.second
				<< std::setprecision(1) << acc << "%\n";
		}

		// Species accuracy
		std::cout << "\n  --- Level 2: Species Accuracy ---\n";
		double sAcc = (sTotal > 0) ? 100.0 * sCorrect / sTotal : 0;
		std::cout << "  Overall: " << std::setprecision(1) << sAcc << "% (" << sCorrect << "/" << sTotal << ")\n";

		std::cout << "  " << std::left << std::setw(15) << "Species" << std::setw(10) << "Correct" << std::setw(10) << "Total" << "Accuracy\n";
		std::cout << "  " << std::string(45, '-') << "\n";

		std::vector<std::pair<std::string, int>> sorted(sPerClass.begin(), sPerClass.end());
		std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });

		for (auto& kv : sorted)
		{
			int c = sPerClassCorrect.count(kv.first) ? sPerClassCorrect[kv.first] : 0;
			double acc = (kv.second > 0) ? 100.0 * c / kv.second : 0;

			std::cout << "  " << std::setw(15) << kv.first
				<< std::setw(10) << c
				<< std::setw(10) << kv.second
				<< std::setprecision(1) << acc << "%\n";
		}

		// Error analysis: group confusion vs species confusion
		int groupWrong_speciesWrong = 0;
		int groupRight_speciesWrong = 0;

		for (auto& r : results)
		{
			if (r.trueCode.empty())
			{
				continue;
			}

			if (!r.speciesCorrect)
			{
				if (r.groupCorrect)
				{
					++groupRight_speciesWrong;  // right group, wrong species
				}
				else
				{
					++groupWrong_speciesWrong;  // wrong group entirely
				}
			}
		}

		std::cout << "\n  --- Error Analysis ---\n";
		std::cout << "  Wrong group (big error)   : " << groupWrong_speciesWrong << "\n";
		std::cout << "  Right group, wrong species : " << groupRight_speciesWrong << "\n";
		std::cout << "  Correct species            : " << sCorrect << "\n";

		// ─── Accuracy by distance to polygon edge ───
		std::cout << "\n  --- Accuracy by Distance to Edge ---\n";

		struct DistBand
		{
			const char* label;
			double minDist;
			double maxDist;
			int total;
			int correct;
		};
	
		DistBand bands[] =
		{
			{"  < 0m (boundary)", -999.0,   0.0, 0, 0},
			{"  0-5m (near edge)", 0.0,    5.0, 0, 0},
			{"  5-10m", 5.0,   10.0, 0, 0},
			{"  10-20m", 10.0,   20.0, 0, 0},
			{"  >= 20m (deep)", 20.0, 9999.0, 0, 0},
		};

		for (auto& r : results)
		{
			if (r.trueCode.empty())
			{
				continue;
			}

			for (auto& b : bands)
			{
				if ((r.distToEdge >= b.minDist) && (r.distToEdge < b.maxDist))
				{
					b.total++;

					if (r.speciesCorrect)
					{
						b.correct++;

						break;
					}
				}
			}
		}

		std::cout << "  " << std::left << std::setw(25) << "Distance" << std::setw(10) << "Total" << std::setw(10) << "Correct" << "Accuracy\n";
		std::cout << "  " << std::string(55, '-') << "\n";

		for (auto& b : bands)
		{
			if (b.total == 0)
			{
				continue;
			}

			double acc = 100.0 * b.correct / b.total;
			std::cout << "  " << std::setw(25) << b.label << std::setw(10) << b.total << std::setw(10) << b.correct << std::fixed << std::setprecision(1) << acc << "%\n";
		}

		// Overall reliable (>= 10m)
		int relTotal = 0;
		int relCorrect = 0;

		for (auto& r : results)
		{
			if (r.trueCode.empty())
			{
				continue;
			}

			if (r.distToEdge >= 10.0)
			{
				relTotal++;

				if (r.speciesCorrect)
				{
					relCorrect++;
				}
			}
		}

		if (relTotal > 0)
		{
			std::cout << "  " << std::string(55, '-') << "\n";
			std::cout << "  " << std::setw(25) << "Reliable (>= 10m)"
				<< std::setw(10) << relTotal
				<< std::setw(10) << relCorrect
				<< std::fixed << std::setprecision(1)
				<< (100.0 * relCorrect / relTotal) << "%\n";
			std::cout << "  " << std::setw(25) << "All"
				<< std::setw(10) << sTotal
				<< std::setw(10) << sCorrect
				<< std::fixed << std::setprecision(1)
				<< (100.0 * sCorrect / sTotal) << "%\n";
		}
	
		// ─── Confidence-based label correction for edge trees ───
		// For trees < 10m from edge: if model confidence is high,
		// trust the model prediction over the forest map label
		std::cout << "\n  --- Label Correction Analysis (< 10m trees) ---\n";

		double confThresholds[] = { 0.5, 0.6, 0.7, 0.8, 0.9 };
		int edgeTotal = 0;
		int edgeOriginalCorrect = 0;

		// Count edge trees
		for (auto& r : results)
		{
			if (r.trueCode.empty())
			{
				continue;
			}

			if (r.distToEdge < 10.0)
			{
				++edgeTotal;

				if (r.speciesCorrect)
				{
					++edgeOriginalCorrect;
				}
			}
		}

		if (edgeTotal > 0)
		{
			std::cout << "  Edge trees (< 10m): " << edgeTotal << ", original accuracy: " << std::fixed << std::setprecision(1) << (100.0 * edgeOriginalCorrect / edgeTotal) << "%\n\n";
			std::cout << "  " << std::left << std::setw(12) << "Threshold" << std::setw(12) << "Corrected" << std::setw(12) << "Kept" << std::setw(12) << "Uncertain" << "Note\n";
			std::cout << "  " << std::string(60, '-') << "\n";

			for (double thresh : confThresholds)
			{
				// For each edge tree:
				//   if model confidence >= threshold AND prediction != label
				//     → assume model is correct (label was wrong)
				//   if model confidence < threshold AND prediction != label
				//     → mark as "uncertain"

				int corrected = 0;			// mismatch + high conf → trust model
				int uncertain = 0;			// mismatch + low conf → unknown
				int kept = 0;					// match → already correct

				for (auto& r : results)
				{
					if (r.trueCode.empty())
					{
						continue;
					}

					if (r.distToEdge >= 10.0)
					{
						continue;
					}

					if (r.speciesCorrect)
					{
						++kept;
					}
					else
					{
						// Model and label disagree
						// Use combined confidence (group × species)
						float conf = r.combinedConfidence;

						if (conf >= thresh)
						{
							++corrected;
						}
						else
						{
							++uncertain;
						}
					}
				}

				int resolved = kept + corrected;
				int total_resolved = kept + corrected + uncertain;
				double resolvedPct = (total_resolved > 0) ? 100.0 * resolved / total_resolved : 0;

				std::cout << "  " << std::setw(12) << (">= " + std::to_string((int)(thresh * 100)) + "%")
					<< std::setw(12) << corrected
					<< std::setw(12) << kept
					<< std::setw(12) << uncertain
					<< std::fixed << std::setprecision(1)
					<< resolved << "/" << total_resolved
					<< " (" << resolvedPct << "% resolved)\n";
			}

			// Best estimate: combine reliable + corrected edge
			std::cout << "\n  --- Combined Accuracy Estimate ---\n";

			for (double thresh : confThresholds)
			{
				int totalResolved = relCorrect;			// reliable trees correct
				int totalEval = relTotal;

				for (auto& r : results)
				{
					if (r.trueCode.empty())
					{
						continue;
					}

					if (r.distToEdge >= 10.0)
					{
						continue;
					}

					if (r.speciesCorrect)
					{
						++totalResolved;
						++totalEval;
					}
					else if (r.combinedConfidence >= thresh)
					{
						// Trust model → count as correct

						++totalResolved;
						++totalEval;
					}
					else
					{
						// else: uncertain, exclude from evaluation
					}
				}

				double combinedAcc = (totalEval > 0) ? 100.0 * totalResolved / totalEval : 0;
				std::cout << "  Conf >= " << (int)(thresh * 100) << "%: " << std::fixed << std::setprecision(1) << combinedAcc << "% (" << totalResolved << "/" << totalEval << ")\n";
			}
		
			// ─── Predicted species distribution for unlabeled trees ───
			std::cout << "\n  --- Predicted Species (Outside Forest Map) ---\n"; 

			std::map<std::string, int> unlabeledByCode;
			std::map<std::string, std::string> unlabeledNames;
			std::map<std::string, int> unlabeledByGroup;
			int unlabeledTotal = 0;

			for (auto& r : results)
			{
				if (!r.trueCode.empty())
				{
					// skip labeled trees
					continue;
				}

				unlabeledByCode[r.predictedCode]++;
				unlabeledNames[r.predictedCode] = r.predictedName;
				unlabeledByGroup[r.predictedGroup]++;

				++unlabeledTotal;
			}

			if (unlabeledTotal > 0)
			{
				// Group summary
				std::cout << "  Total unlabeled trees: " << unlabeledTotal << "\n\n";
				std::cout << "  [By Group]\n";
				
				std::vector<std::pair<std::string, int>> gSorted(unlabeledByGroup.begin(), unlabeledByGroup.end());
				std::sort(gSorted.begin(), gSorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
			
				for (auto& kv : gSorted)
				{
					double pct = 100.0 * kv.second / unlabeledTotal;
					std::cout << "    " << std::setw(15) << std::left
						<< groupCodeToNameStatic(kv.first)
						<< std::setw(8) << kv.second
						<< std::fixed << std::setprecision(1) << pct << "%\n";
				}

				// Species detail
				std::cout << "\n  [By Species]\n";
				std::cout << "  " << std::left << std::setw(8) << "Code" << std::setw(20) << "Species" << std::setw(8) << "Count" << "Percent\n";
				std::cout << "  " << std::string(50, '-') << "\n";
			
				std::vector<std::pair<std::string, int>> sSorted(unlabeledByCode.begin(), unlabeledByCode.end());
				std::sort(sSorted.begin(), sSorted.end(), [](auto& a, auto& b) { return a.second > b.second; });

				for (auto& kv : sSorted)
				{
					double pct = 100.0 * kv.second / unlabeledTotal;
					std::string name = unlabeledNames.count(kv.first) ? unlabeledNames[kv.first] : "";
					std::cout << "  " << std::setw(8) << kv.first
						<< std::setw(20) << name
						<< std::setw(8) << kv.second
						<< std::fixed << std::setprecision(1) << pct << "%\n";
				}
			}
			else
			{
				std::cout << "  No unlabeled trees.\n";
			}
		}
	}



	// ─── Save results CSV ───
	static void saveResults(const std::vector<HierarchicalResult>& results, const std::string& path)
	{
		std::ofstream out(path);

		out << "TreeID,PredGroup,PredGroupName,GroupConf,"
			<< "PredCode,PredName,SpeciesConf,CombinedConf,"
			<< "TrueCode,TrueGroup,GroupCorrect,SpeciesCorrect,DistToEdge\n";

		for (auto& r : results)
		{
			out << r.treeId << ","
				<< r.predictedGroup << ","
				<< r.predictedGroupName << ","
				<< std::fixed << std::setprecision(3) << r.groupConfidence << ","
				<< r.predictedCode << ","
				<< r.predictedName << ","
				<< r.speciesConfidence << ","
				<< r.combinedConfidence << ","
				<< r.trueCode << ","
				<< r.trueGroup << ","
				<< (r.trueGroup.empty() ? "-" : (r.groupCorrect ? "Y" : "N")) << ","
				<< (r.trueCode.empty() ? "-" : (r.speciesCorrect ? "Y" : "N"))
				<< std::setprecision(2) << r.distToEdge
				<< "\n";
		}

		out.close();
		std::cout << "  Results saved: " << path << "\n";
	}

private:

	bool hasConiferClf_ = false;
	bool hasDeciduousClf_ = false;
	bool hasEvergreenClf_ = false;
	bool hasNonForestClf_ = false;

	// ─── Self-training: remove likely mislabeled trees ───
	// Train a model, predict on training data, remove trees where
	// model disagrees with label. Repeat for maxIter rounds.
	static std::vector<TreeFeature*> selfTrainPool(
		std::vector<TreeFeature*>& pool,
		ExtractionResult& features,
		const ClassifierOptions& clsOpts,
		int maxIter,
		float maxRemoveRatio,
		const std::string& groupName)
	{
		std::vector<TreeFeature*> current = pool;

		for (int iter = 0 ; iter < maxIter ; ++iter)
		{
			// Train on current pool
			SpeciesClassifier tmpClf;
			ExtractionResult tmpFeatures = features;
			tmpFeatures.train = current;

			ClassifierOptions tmpOpts = clsOpts;
			tmpOpts.crossValidate = false;
			tmpOpts.balanceClasses = true;
			
			tmpClf.train(tmpFeatures, tmpOpts);

			// Predict on current pool
			auto preds = tmpClf.predict(tmpFeatures);
			std::map<long long, PredictResult> predMap;

			for (auto& pr : preds)
			{
				predMap[pr.treeId] = pr;
			}

			// Count per-species totals
			std::map<std::string, int> speciesTotal;
			std::map<std::string, int> speciesRemoveCount;

			for (auto* tf : current)
			{
				speciesTotal[tf->speciesCode]++;
			}

			// Identify removals
			std::vector<bool> shouldRemove(current.size(), false);

			for (size_t i = 0 ; i < current.size() ; ++i)
			{
				auto it = predMap.find(current[i]->treeId);

				if ((it != predMap.end()) && (it->second.predictedCode != current[i]->speciesCode))
				{
					shouldRemove[i] = true;
					speciesRemoveCount[(current[i]->speciesCode)]++;
				}
			}

			// Apply removal with per-species limit
			std::vector<TreeFeature*> cleaned;
			int removed = 0;
			int prot = 0;

			for (size_t i = 0 ; i < current.size() ; ++i)
			{
				if (!shouldRemove[i])
				{
					cleaned.push_back(current[i]);

					continue;
				}

				int total = speciesTotal[(current[i]->speciesCode)];
				int remCnt = speciesRemoveCount[(current[i]->speciesCode)];
				float ratio = (total > 0) ? (float)remCnt / total : 0;

				if (ratio > maxRemoveRatio)
				{
					cleaned.push_back(current[i]);
					++prot;
				}
				else
				{
					++removed;
				}
			}

			std::cout << "    Self-training " << groupName << " iter " << (iter + 1) << ": " << removed << " removed";
			
			if (prot > 0)
			{
				std::cout << ", " << prot << " protected";
			}

			std::cout << " (" << cleaned.size() << " remain)\n";

			if (removed == 0)
			{
				break;
			}

			current = cleaned;
		}

		return current;
	}

	void buildGroupMapping(const HierarchicalOptions& opts)
	{
		for (auto& code : opts.coniferCodes)
		{
			codeToGroup[code] = "C";
		}

		for (auto& code : opts.deciduousCodes)
		{
			codeToGroup[code] = "D";
		}

		for (auto& code : opts.evergreenCodes)
		{
			codeToGroup[code] = "E";
		}

		for (auto& code : opts.nonForestCodes)
		{
			codeToGroup[code] = "N";
		}
	}

	std::string groupCodeToName(const std::string& code) const
	{
		return groupCodeToNameStatic(code);
	}

	static std::string groupCodeToNameStatic(const std::string& code)
	{
		if (code == "C")
		{
			return "Conifer";
		}

		if (code == "D")
		{
			return "Deciduous";
		}

		if (code == "E")
		{
			return "Evergreen";
		}

		if (code == "N")
		{
			return "Non-forest";
		}

		return code;
	}

	static void printDistribution(const std::vector<TreeFeature*>& pool)
	{
		std::map<std::string, int> counts;

		for (auto* tf : pool)
		{
			counts[tf->speciesCode]++;
		}

		std::vector<std::pair<std::string, int>> sorted(counts.begin(), counts.end());
		std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });

		for (auto& kv : sorted)
		{
			std::cout << "    " << std::setw(10) << std::left << kv.first << " : " << kv.second << "\n";
		}
	}
};