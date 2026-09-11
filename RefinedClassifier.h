#pragma once

#include "SpeciesClassifier.h"

// ─────────────────────────────────────────────
//  Refinement options
// ─────────────────────────────────────────────

struct RefinementOptions
{
	// Round 1: stricter edge distance
	// distToEdge가 10m 미만인 나무를 학습 데이터에서 제외.
	// 폴리곤 깊숙이 있는 나무만 남기면 해당 수종일 확률이 높아짐
	double round1_minEdgeDist = 10.0;			// increased from default 5m

	// Round 2: self-training
	// "소나무 영역에 있는데 모델이 참나무라고 예측한" 나무를 제거
	bool round2_enabled = true;
	int round2_iterations = 2;								// how many self-training passes
	// Remove trees where model predicts differently from label
	float round2_maxRemoveRatio = 1.0f;		// max removal ratio per species (0~1)
																		// 1.0 = no limit (default)
																		// 0.3 = keep at least 70% per species

	// Round 3: confidence filtering
	// Round 2 모델의 예측 중 confidence가 낮은 나무를 추가로 제거
	bool round3_enabled = true;
	float round3_minConfidence = 0.6f;			// keep only predictions above this

	// Debug image export
	bool saveDebugImages = false;					// save crop images per round
	std::string debugOutputDir = "debug_rounds";
	std::string cropSourceDir;							// where crop images are stored (from TreeLabeler)

	// RF parameters (same for all rounds)
	ClassifierOptions rfOpts;
};

// ─────────────────────────────────────────────
//  RefinedClassifier
// ─────────────────────────────────────────────

class RefinedClassifier
{
public:

	static SpeciesClassifier run(ExtractionResult& features, const RefinementOptions& opts = RefinementOptions())
	{
		std::cout << "========================================\n";
		std::cout << "  Refined Classification Pipeline\n";
		std::cout << "========================================\n\n";

		// Keep a working copy of training pointers
		 // (we'll filter this list across rounds)
		std::vector<TreeFeature*> trainPool = features.train;

		std::cout << "  Initial training pool: " << trainPool.size() << " trees\n\n";

		// ─── Round 1: Stricter edge distance ───
		std::cout << "=== Round 1: Edge distance filter (" << opts.round1_minEdgeDist << "m) ===\n";

		std::vector<TreeFeature*> round1Pool;
		filterByEdgeDist(features, opts.round1_minEdgeDist, round1Pool);

		std::cout << "  Before: " << trainPool.size() << " trees\n";
		std::cout << "  After:  " << round1Pool.size() << " trees (" << (trainPool.size() - round1Pool.size()) << " removed)\n";

		printSpeciesDistribution(round1Pool, "Round 1");

		SpeciesClassifier clf1;

		// Train Round 1 model
		ExtractionResult round1Features = features;
		round1Features.train = round1Pool;

		ClassifierOptions r1Opts = opts.rfOpts;
		r1Opts.crossValidate = true;
		clf1.train(round1Features, r1Opts);

		// Predict on all training data to find mismatches
		auto round1Results = clf1.predict(round1Features);
		reportAccuracy(round1Results, "Round 1");

		// Debug images: Round 1
		if (opts.saveDebugImages)
		{
			std::vector<TreeFeature*> r1Removed;
			std::set<long long> r1KeptIds;

			for (auto* tf : round1Pool)
			{
				r1KeptIds.insert(tf->treeId);
			}

			for (auto* tf : trainPool)
			{
				if (r1KeptIds.find(tf->treeId) == r1KeptIds.end())
				{
					r1Removed.push_back(tf);
				}
			}

			exportRoundImages("R1", round1Pool, r1Removed, "removed_edge", opts.debugOutputDir, opts.cropSourceDir);
		}

		trainPool = round1Pool;

		// ─── Round 2: Self-training ───

		if (opts.round2_enabled)
		{
			for (int iter = 0 ; iter < opts.round2_iterations ; ++iter)
			{
				std::cout << "\n=== Round 2: Self-training (iteration " << (iter + 1) << "/" << opts.round2_iterations << ") ===\n";

				// Use current model to predict training data
				// Remove trees where prediction != label
				// But respect per-species removal limit
				std::vector<TreeFeature*> cleanPool;
				int removed = 0;
				int protected_ = 0;

				// Build a quick lookup of predictions
				std::map<long long, PredictResult> predMap;

				for (auto& pr : round1Results)
				{
					predMap[pr.treeId] = pr;
				}

				// Count original per-species totals
				std::map<std::string, int> speciesTotal;
				std::map<std::string, int> speciesRemoved;

				for (auto* tf : trainPool)
				{
					speciesTotal[tf->speciesCode]++;
				}

				// First pass: identify which trees to remove
				std::vector<bool> shouldRemove(trainPool.size(), false);

				for (size_t idx = 0 ; idx < trainPool.size() ; ++idx)
				{
					auto* tf = trainPool[idx];
					auto it = predMap.find(tf->treeId);

					if (it != predMap.end() && (it->second.predictedCode != tf->speciesCode))
					{
						shouldRemove[idx] = true;
						speciesRemoved[tf->speciesCode]++;
					}
				}

				// Second pass: apply removal with per-species limit
				for (size_t idx = 0 ; idx < trainPool.size() ; ++idx)
				{
					auto* tf = trainPool[idx];

					if (!shouldRemove[idx])
					{
						cleanPool.push_back(tf);

						continue;
					}

					// Check if this species hit the removal limit
					int total = speciesTotal[tf->speciesCode];
					int removedSoFar = speciesRemoved[tf->speciesCode];
					float removeRatio = (total > 0) ? (float)removedSoFar / total : 0.0f;

					if (removeRatio > opts.round2_maxRemoveRatio)
					{
						// Protect: too many removed from this species
						cleanPool.push_back(tf);
						++protected_;
					}
					else
					{
						++removed;
					}
				}

				std::cout << "  Before: " << trainPool.size() << " trees\n";
				std::cout << "  Removed (model disagrees): " << removed << "\n";

				if (protected_ > 0)
				{
					std::cout << "  Protected (removal limit " << (int)(opts.round2_maxRemoveRatio * 100) << "%): " << protected_ << "\n";
				}

				std::cout << "  After:  " << cleanPool.size() << " trees\n";

				// Per-species removal detail
				if (!speciesRemoved.empty())
				{
					std::cout << "  [Per-species removal]\n";

					for (auto& kv : speciesRemoved)
					{
						int total = speciesTotal[kv.first];
						float pct = (total > 0) ? 100.0f * kv.second / total : 0.0f;
						bool limited = (pct > opts.round2_maxRemoveRatio * 100);

						std::cout << "    " << std::setw(12) << std::left << kv.first
							<< " : " << kv.second << "/" << total
							<< " (" << std::fixed << std::setprecision(1)
							<< pct << "%)"
							<< (limited ? " *limited*" : "") << "\n";
					}
				}

				if (cleanPool.size() < 50)
				{
					std::cout << "  WARNING: Too few samples remaining. Stopping.\n";

					break;
				}

				printSpeciesDistribution(cleanPool, "Round 2 iter " + std::to_string(iter + 1));

				// Retrain on cleaned data
				SpeciesClassifier clf2;
				ExtractionResult round2Features = features;
				round2Features.train = cleanPool;

				ClassifierOptions r2Opts = opts.rfOpts;
				r2Opts.crossValidate = true;
				clf2.train(round2Features, r2Opts);

				// Predict on all training data to find mismatches
				auto round2Results = clf2.predict(round2Features);
				reportAccuracy(round2Results, "Round 2 iter" + std::to_string(iter + 1));

				if (opts.saveDebugImages && (removed > 0))
				{
					std::vector<TreeFeature*> r2Removed;
					std::set<long long> cleanIds;

					for (auto* tf : cleanPool)
					{
						cleanIds.insert(tf->treeId);
					}

					for (auto* tf : trainPool)
					{
						if (cleanIds.find(tf->treeId) == cleanIds.end())
						{
							r2Removed.push_back(tf);
						}
					}

					std::string round2Lable = "R2_iter" + std::to_string(iter + 1);
					exportRoundImages(round2Lable, cleanPool, r2Removed, "removed_disagree", opts.debugOutputDir, opts.cropSourceDir);
				}

				trainPool = cleanPool;
				clf1 = clf2;
			}
		}

		// ─── Round 3: Confidence filtering ───
		if (opts.round3_enabled)
		{
			std::cout << "\n=== Round 3: Confidence filtering (>=" << opts.round3_minConfidence << ") ===\n";

			// Predict current training pool with current model
			// Keep only trees where model is confident AND agrees with label
			std::map<long long, PredictResult> predMap;

			for (auto& pr : round1Results)
			{
				predMap[pr.treeId] = pr;
			}

			std::vector<TreeFeature*> confPool;
			int lowConf = 0;
			int disagree = 0;

			for (auto* tf : trainPool)
			{
				auto it = predMap.find(tf->treeId);

				if (it == predMap.end())
				{
					confPool.push_back(tf);

					continue;
				}

				if (it->second.confidence < opts.round3_minConfidence)
				{
					++lowConf;

					continue;
				}

				if (it->second.predictedCode != tf->speciesCode)
				{
					++disagree;

					continue;
				}

				confPool.push_back(tf);
			}

			std::cout << "  Before: " << trainPool.size() << " trees\n";
			std::cout << "  Removed (low confidence): " << lowConf << "\n";
			std::cout << "  Removed (disagree): " << disagree << "\n";
			std::cout << "  After:  " << confPool.size() << " trees\n";

			if (confPool.size()>= 50)
			{
				printSpeciesDistribution(confPool, "Round 3");

				// Final training

				SpeciesClassifier clf3;
				ExtractionResult round3Features = features;
				round3Features.train = confPool;

				ClassifierOptions r3Opts = opts.rfOpts;
				r3Opts.crossValidate = true;
				clf3.train(round3Features, r3Opts);

				auto round3Results = clf3.predict(round3Features);
				reportAccuracy(round3Results, "Round 3 (final)");

				// Debug images: Round 3
				if (opts.saveDebugImages)
				{
					std::vector<TreeFeature*> r3Removed;
					std::set<long long> confIds;

					for (auto* tf : confPool)
					{
						confIds.insert(tf->treeId);
					}

					for (auto* tf : trainPool)
					{
						if (confIds.find(tf->treeId) == confIds.end())
						{
							r3Removed.push_back(tf);
						}
					}

					exportRoundImages("R3", confPool, r3Removed, "removed_lowconf", opts.debugOutputDir, opts.cropSourceDir);
				}

				trainPool = confPool;
				clf1 = clf3;
			}
			else
			{
				std::cout << "  WARNING: Too few samples. Keeping Round 2 model.\n";
			}
		}

		// ─── Summary ───
		std::cout << "\n========================================\n";
		std::cout << "  Refinement Complete\n";
		std::cout << "========================================\n";
		std::cout << "  Initial pool : " << features.train.size() << " trees\n";
		std::cout << "  Final pool   : " << trainPool.size() << " trees\n";
		std::cout << "  Removed total: "
			<< (features.train.size() - trainPool.size()) << " ("
			<< std::fixed << std::setprecision(1)
			<< (100.0 * (features.train.size() - trainPool.size()) / features.train.size())
			<< "%)\n";

		// Update features.train to final pool
		features.train = trainPool;

		return clf1;
	}

private:

	// Filter training pool by stricter edge distance
	static void filterByEdgeDist(const ExtractionResult& features, double minDist, std::vector<TreeFeature*>& out)
	{
		out.clear();
		int removed = 0;

		for (auto* tf : const_cast<ExtractionResult&>(features).train)
		{
			if (tf->distToEdge >= minDist)
			{
				out.push_back(tf);
			}
			else
			{
				++removed;
			}
		}

		std::cout << "  Filtered by distToEdge >= " << minDist << "m: " << removed << " removed\n";
	}

	// Print species distribution
	static void printSpeciesDistribution(const std::vector<TreeFeature*>& pool, const std::string& label)
	{
		std::map<std::string, int> counts;

		for (auto* tf : pool)
		{
			counts[tf->speciesCode]++;
		}

		std::vector<std::pair<std::string, int>> sorted(counts.begin(), counts.end());
		std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });

		std::cout << "  [" << label << " species distribution]\n";

		for (auto& kv : sorted)
		{
			std::cout << "    " << std::setw(12) << std::left << kv.first << " : " << kv.second << "\n";
		}
	}

	// Quick accuracy report on labeled trees
	static void reportAccuracy(const std::vector<PredictResult>& results, const std::string& label)
	{
		int total = 0;
		int correct = 0;

		for (auto& r : results)
		{
			if (r.trueCode.empty())
			{
				continue;
			}

			++total;

			if (r.correct)
			{
				++correct;
			}
		}

		double acc = (total > 0) ? 100.0 * correct / total : 0.0;

		std::cout << "  " << label << " accuracy: "
			<< std::fixed << std::setprecision(1) << acc << "% ("
			<< correct << "/" << total << ")\n";
	}

	// ─── Export debug images per round ───
	// Copies crop images into round-specific folders:
	//   debug_rounds/
	//     R1_kept/<species>/tree_<id>.png
	//     R1_removed/tree_<id>.png
	//     R2_iter1_kept/<species>/tree_<id>.png
	//     R2_iter1_removed/tree_<id>.png
	//     R3_kept/<species>/tree_<id>.png
	//     R3_removed_lowconf/tree_<id>.png

	static void exportRoundImages(const std::string& roundName, 
		const std::vector<TreeFeature*> kept, 
		const std::vector<TreeFeature*> removed,
		const std::string& removedSubfolder,
		const std::string& debugDir,
		const std::string& cropSourceDir)
	{
		std::string keptDir = TSICommon::join(debugDir, roundName + "_kept");
		std::string removedDir = TSICommon::join(debugDir, roundName + "_" + removedSubfolder);
		TSICommon::mkdirs(removedDir);
	
		// Build lookup of existing image files by tree ID
		// Images are in: cropSourceDir/train/<species>/tree_<id>.png
		//            or: cropSourceDir/predict/unknown/tree_<id>.png

		int copiedKept = 0;
		int copiedRemoved = 0;
		std::set<std::string> createdDirs;

		// Copy kept images (organized by species)

		for (auto* tf : kept)
		{
			if (tf->imagePath.empty())
			{
				continue;
			}

			std::string filename = extractFilename(tf->imagePath);
			std::string specDir = tf->speciesCode.empty() ? "unknown" : tf->speciesCode;
			std::string dstDir = TSICommon::join(keptDir, specDir);

			if (createdDirs.find(dstDir) == createdDirs.end())
			{
				TSICommon::mkdirs(dstDir);
				createdDirs.insert(dstDir);
			}

			copyFile(tf->imagePath, TSICommon::join(dstDir, filename));
			++copiedKept;
		}

		// Copy removed images (flat, no species subfolder)

		for (auto* tf : removed)
		{
			if (tf->imagePath.empty())
			{
				continue;
			}

			std::string filename = extractFilename(tf->imagePath);
			
			copyFile(tf->imagePath, TSICommon::join(removedDir, filename));
			++copiedRemoved;
		}
	
		std::cout << "  Debug images [" << roundName << "]: "
			<< copiedKept << " kept, " << copiedRemoved << " removed"
			<< " -> " << debugDir << "/\n";
	}

	// Extract filename from full path
	static std::string extractFilename(const std::string& path)
	{
		size_t pos = path.find_last_of("/\\");

		return ((pos != std::string::npos) ? path.substr(pos + 1) : path);
	}

	// Simple file copy
	static void copyFile(const std::string& src, const std::string& dst)
	{
		std::ifstream in(src, std::ios::binary);

		if (!in.is_open())
		{
			return;
		}

		std::ofstream out(dst, std::ios::binary);

		if (!out.is_open())
		{
			return;
		}

		out << in.rdbuf();
	}
};