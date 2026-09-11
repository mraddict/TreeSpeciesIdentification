#pragma once

#include "FeatureExtractor.h"

// 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
//  Classifier options
// 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式

struct ClassifierOptions
{
	// Random Forest parameters
	int maxDepth = 20;							// max tree depth
	int minSampleCount = 5;					// min samples per leaf
	int maxTrees = 200;							// number of trees
	int activeVarCount = 0;					// features per split (0 = sqrt(nFeatures))

	// Class balancing
	bool balanceClasses = false;			// weight samples inversely proportional to class frequency

	// Evaluation
	bool crossValidate = true;				// run k-fold cross validation
	int cvFolds = 5;								// number of CV folds
	float trainRatio = 0.8f;						// train/test split ratio (if CV disabled)

	// Output
	std::string modelPath = "species_model";
	std::string resultCsv = "species_result.csv";
};

// 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
//  Prediction result for a single tree
// 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式

struct PredictResult
{
	long long treeId = -1;
	std::string predictedCode;			// predicted species
	std::string predictedName;
	std::string trueCode;					// actual species (if known)
	float confidence = 0.0f;				// vote ratio (0~1)
	bool correct = false;						// prediction matches truth
};

// 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
//  SpeciesClassifier
// 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式

class SpeciesClassifier
{
public:

	// Label mapping
	std::map<std::string, int> codeToLabel;
	std::map<int, std::string> labelToCode;
	std::map<std::string, std::string> codeToName;	// code -> species name

	// 式式式 Train from ExtractionResult 式式式
	void train(const ExtractionResult& features, const ClassifierOptions& opts = ClassifierOptions())
	{
		std::cout << "=== Random Forest Training ===\n";

		if (features.train.empty())
		{
			throw std::runtime_error("No training data available.");
		}

		// Build label mapping
		buildLabelMap(features.train);

		int nSamples = (int)features.train.size();
		int nFeats = (int)features.train[0]->feature.size();

		// Build OpenCV matrices
		cv::Mat X(nSamples, nFeats, CV_32F);
		cv::Mat y(nSamples, 1, CV_32S);

		for (int i = 0 ; i < nSamples ; ++i)
		{
			auto* tf = features.train[i];

			for (int j = 0 ; j < nFeats ; ++j)
			{
				X.at<float>(i , j) = tf->feature[j];
			}

			y.at<int>(i, 0) = codeToLabel[tf->speciesCode];
		}

		// Print training data distribution
		std::map<std::string, int> cnt;

		for (auto* tf : features.train)
		{
			cnt[tf->speciesCode]++;
		}

		std::cout << "  Training samples: " << nSamples << "\n";
		std::cout << "  Species count   : " << codeToLabel.size() << "\n";
		std::cout << "  Features        : " << nFeats << "\n";
		std::cout << "  Distribution:\n";

		std::vector<std::pair<std::string, int>> sorted(cnt.begin(), cnt.end());
		std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });

		for (auto& kv : sorted)
		{
			std::string name = codeToName.count(kv.first) ? codeToName[kv.first] : "";

			std::cout << "    " << std::setw(10) << std::left << kv.first << " " << std::setw(16) << name << " : " << kv.second << "\n";
		}

		// Create and configure Random Forest
		rf_ = cv::ml::RTrees::create();

		rf_->setMaxDepth(opts.maxDepth);
		rf_->setMinSampleCount(opts.minSampleCount);
		rf_->setCalculateVarImportance(true);

		// Active variable count (features per split)
		int activeVars = opts.activeVarCount;

		if (activeVars <= 0)
		{
			activeVars = (int)std::round(std::sqrt((double)nFeats));
		}

		rf_->setActiveVarCount(activeVars);

		// Termination criteria (number of trees)
		rf_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, opts.maxTrees, 1e-6));
	
		std::cout << "\n  RF parameters:\n"
			<< "    Max depth      : " << opts.maxDepth << "\n"
			<< "    Min samples    : " << opts.minSampleCount << "\n"
			<< "    Trees          : " << opts.maxTrees << "\n"
			<< "    Active vars    : " << activeVars << "\n"
			<< "    Balance classes: " << (opts.balanceClasses ? "YES" : "no") << "\n";

		// Train
		std::cout << "  Training...\n";

		// Compute sample weights if balancing is enabled
		cv::Mat sampleWeights;

		if (opts.balanceClasses)
		{
			sampleWeights = cv::Mat::ones(nSamples, 1, CV_32F);

			// Count per class
			std::map<int, int> classCounts;

			for (int i = 0 ; i < nSamples ; ++i)
			{
				classCounts[y.at<int>(i, 0)]++;
			}

			// Weight = totalSamples / (numClasses * classCount)
			std::cout << "  Class weights:\n";

			int numClasses = (int)classCounts.size();

			for (auto& kv : classCounts)
			{
				float weight = (float)nSamples / (float)(numClasses * kv.second);
				std::string code = labelToCode.count(kv.first) ? labelToCode[kv.first] : "?";

				std::cout << "    " << std::setw(10) << std::left << code << " : " << kv.second << " samples, weight=" << std::fixed << std::setprecision(2) << weight << "\n";

				for (int i = 0 ; i < nSamples ; ++i)
				{
					if (y.at<int>(i, 0) == kv.first)
					{
						sampleWeights.at<float>(i, 0) = weight;
					}
				}
			}
		}

		auto trainSet = cv::ml::TrainData::create(X, cv::ml::ROW_SAMPLE, y, cv::noArray(), cv::noArray(), (opts.balanceClasses ? sampleWeights : cv::noArray()));
		rf_->train(trainSet);

		// Training accuracy
		cv::Mat pred;

		rf_->predict(X, pred);

		int correct = 0;

		for (int i = 0 ; i < nSamples ; ++i)
		{
			if ((int)pred.at<float>(i) == y.at<int>(i))
			{
				++correct;
			}
		}

		std::cout << "  Training accuracy: " << std::fixed << std::setprecision(1) << (100.0 * correct / nSamples) << "% (" << correct << "/" << nSamples << ")\n";

		// Feature importance
		printFeatureImportance();

		// Cross-validation
		if (opts.crossValidate && nSamples >= opts.cvFolds * 5)
		{
			crossValidate(X, y, opts);
		}

		trained_ = true;
	}

	// 式式式 Predict species for all trees 式式式
	std::vector<PredictResult> predict(const ExtractionResult& features)
	{
		if (!trained_)
		{
			throw std::runtime_error("Model not trained. Call train() first.");
		}

		std::cout << "\n=== Predicting species ===\n";

		std::vector<PredictResult> results;
		results.reserve(features.all.size());

		for (auto& tf : features.all)
		{
			if (tf.feature.empty())
			{
				continue;
			}

			int nFeats = (int)tf.feature.size();
			cv::Mat x(1, nFeats, CV_32F);

			for (int j = 0 ; j < nFeats ; ++j)
			{
				x.at<float>(0, j) = tf.feature[j];
			}

			// Get votes from all trees
			cv::Mat votes;
			rf_->getVotes(x, votes, 0);

			// Find best class + confidence
			int bestLabel = 0;
			float bestVotes = 0.0f;
			float totalVotes = 0.0f;

			for (int c = 0 ; c <votes.cols ; ++c)
			{
				totalVotes += votes.at<int>(1, c);
			}

			for (int c = 0 ; c < votes.cols ; ++c)
			{
				float v = (float)votes.at<int>(1, c);

				if (v > bestVotes)
				{
					bestVotes = v;
					bestLabel = c;
				}
			}

			PredictResult pr;
			
			pr.treeId = tf.treeId;
			pr.predictedCode = labelToCode.count(bestLabel) ? labelToCode[bestLabel] : "?";
			pr.predictedName = codeToName.count(pr.predictedCode) ? codeToName[pr.predictedCode] : "";
			pr.trueCode = tf.speciesCode;
			pr.confidence = (totalVotes > 0) ? bestVotes / totalVotes : 0.0f;
			pr.correct = (!pr.trueCode.empty() && (pr.predictedCode == pr.trueCode));

			results.push_back(pr);
		}

		std::cout << "  Predicted: " << results.size() << " trees\n";

		return results;
	}

	// 式式式 Evaluate prediction accuracy 式式式
	static void evaluate(const std::vector<PredictResult>& results)
	{
		std::cout << "\n=== Prediction Evaluation ===\n";

		int total = 0;
		int correct = 0;
		std::map<std::string, int> perClassCorrect;
		std::map<std::string, int> perClassTotal;

		// Confusion tracking
		std::map<std::string, std::map<std::string, int>> confusion;

		for (auto& r : results)
		{
			if (r.trueCode.empty())
			{
				continue;
			}

			++total;
			++perClassTotal[r.trueCode];
			confusion[r.trueCode][r.predictedCode]++;

			if (r.correct)
			{
				++correct;
				++perClassCorrect[r.trueCode];
			}
		}

		if (total == 0)
		{
			std::cout << "  No labeled data for evaluation.\n";

			return;
		}

		double accuracy = 100.0 * correct / total;
		std::cout << "  Overall accuracy: " << std::fixed << std::setprecision(1) << accuracy << "% (" << correct << "/" << total << ")\n\n";

		// Per-class accuracy
		std::cout << "  Per-species accuracy:\n";
		std::cout << "  " << std::left << std::setw(12) << "Species"
			<< std::setw(10) << "Correct"
			<< std::setw(10) << "Total"
			<< std::setw(10) << "Accuracy"
			<< "\n";
		std::cout << "  " << std::string(42, '-') << "\n";

		std::vector<std::pair<std::string, int>> sortedSpec(perClassTotal.begin(), perClassTotal.end());
		std::sort(sortedSpec.begin(), sortedSpec.end(), [](auto& a, auto& b) { return a.second > b.second; });

		for (auto& kv : sortedSpec)
		{
			int c = perClassCorrect.count(kv.first) ? perClassCorrect[kv.first] : 0;
			int t = kv.second;
			double acc = (t > 0) ? 100.0 * c / t : 0.0;

			std::cout << "  " << std::setw(12) << kv.first
				<< std::setw(10) << c
				<< std::setw(10) << t
				<< std::setw(9) << std::setprecision(1) << acc << "%\n";
		}

		// Confidence statistics
		float sumConf = 0.0f;
		float minConf = 1.0f;
		float maxConf = 0.0f;
		int confCount = 0;

		for (auto& r : results)
		{
			sumConf += r.confidence;
			minConf = std::min(minConf, r.confidence);
			maxConf = std::max(maxConf, r.confidence);

			++confCount;
		}

		if (confCount > 0)
		{
			std::cout << "\n  Confidence: mean="
				<< std::setprecision(2) << (sumConf / confCount)
				<< " min=" << minConf
				<< " max=" << maxConf << "\n";
		}

		std::cout << std::setprecision(6);
	}

	// 式式式 Save model + label mapping 式式式
	void save(const std::string& path)
	{
		if (!trained_)
		{
			std::cerr << "  WARNING: Model not trained.\n";

			return;
		}

		rf_->save(path + "_rf.xml");

		// Save label mapping
		std::ofstream f(path + "_labels.csv");

		f << "Code,Label,Name\n";

		for (auto& kv : codeToLabel)
		{
			std::string name = codeToName.count(kv.first) ? codeToName[kv.first] : "";
			f << kv.first << "," << kv.second << "," << name << "\n";
		}

		f.close();
	
		std::cout << "  Model saved: " << path << "_rf.xml\n";
		std::cout << "  Labels saved: " << path << "_labels.csv\n";
	}

private:

	cv::Ptr<cv::ml::RTrees> rf_;
	bool trained_ = false;

	void buildLabelMap(const std::vector<TreeFeature*>& data)
	{
		std::set<std::string> codes;

		for (auto* tf : data)
		{
			codes.insert(tf->speciesCode);

			if (!tf->speciesName.empty())
			{
				codeToName[tf->speciesCode] = tf->speciesName;
			}
		}

		int idx = 0;

		for (auto& code : codes)
		{
			codeToLabel[code] = idx;
			labelToCode[idx] = code;

			++idx;
		}
	}

	// Feature importance ranking
	void printFeatureImportance()
	{
		cv::Mat imp = rf_->getVarImportance();

		if (imp.empty())
		{
			return;
		}

		auto& names = TreeFeature::featureNames();

		std::cout << "\n  Feature importance:\n";

		std::vector<std::pair<float, int>> vi;

		for (int i = 0 ; i < imp.rows ; ++i)
		{
			vi.push_back({imp.at<float>(i), i});
		}

		std::sort(vi.begin(), vi.end());

		float maxImp = vi.empty() ? 1.0f : vi[0].first;

		for (auto& p : vi)
		{
			std::string name = (p.second < (int)names.size()) ? names[p.second] : ("feat_" + std::to_string(p.second));

			// Visual bar
			int barLen = (maxImp > 0) ? (int)(20.0f * p.first / maxImp) : 0;
			std::string bar(barLen, '#');

			std::cout << "    " << std::setw(12) << std::left << name
				<< " " << std::setw(20) << std::left << bar
				<< " " << std::fixed << std::setprecision(4) << p.first << "\n";
		}
	}

	// K-fold cross-validation
	void crossValidate(const cv::Mat& X, const cv::Mat& y, const ClassifierOptions& opts)
	{
		int nSamples = X.rows;
		int nFolds = opts.cvFolds;
		int foldSize = nSamples / nFolds;

		std::cout << "\n  " << nFolds << "-fold cross-validation:\n";

		// Shuffle indices
		std::vector<int> indices(nSamples);
		std::iota(indices.begin(), indices.end(), 0);
		std::mt19937 rng(42);
		std::shuffle(indices.begin(), indices.end(), rng);

		double totalAcc = 0;
		std::vector<double> foldAccs;

		for (int fold = 0 ; fold < nFolds ; ++fold)
		{
			// Split into train/test
			int testStart = fold * foldSize;
			int testEnd = (fold == nFolds - 1) ? nSamples : testStart + foldSize;
			int testCount = testEnd - testStart;
			int trainCount = nSamples - testCount;

			cv::Mat trainX(trainCount, X.cols, CV_32F);
			cv::Mat trainY(trainCount, 1, CV_32S);
			cv::Mat testX(testCount, X.cols, CV_32F);
			cv::Mat testY(testCount, 1, CV_32S);

			int ti = 0;
			int vi = 0;

			for (int i = 0 ; i < nSamples ; ++i)
			{
				int idx = indices[i];

				if ((i >= testStart) && (i < testEnd))
				{
					X.row(idx).copyTo(testX.row(vi));
					testY.at<int>(vi, 0) = y.at<int>(idx, 0);

					++vi;
				}
				else
				{
					X.row(idx).copyTo(trainX.row(ti));
					trainY.at<int>(ti, 0) = y.at<int>(idx, 0);

					++ti;
				}
			}

			// Train fold model
			auto foldRF = cv::ml::RTrees::create();

			foldRF->setMaxDepth(opts.maxDepth);
			foldRF->setMinSampleCount(opts.minSampleCount);
			foldRF->setActiveVarCount(rf_->getActiveVarCount());
			foldRF->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, opts.maxTrees, 1e-6));

			auto foldData = cv::ml::TrainData::create(trainX, cv::ml::ROW_SAMPLE, trainY);
			foldRF->train(foldData);

			// Evaluate on test set
			cv::Mat pred;
			foldRF->predict(testX, pred);

			int correct = 0;

			for (int i = 0 ; i < testCount ; ++i)
			{
				if ((int)pred.at<float>(i) == testY.at<int>(i, 0))
				{
					++correct;
				}
			}

			double acc = 100.0 * correct / testCount;
			totalAcc += acc;
			foldAccs.push_back(acc);
			
			std::cout << "    Fold " << (fold + 1) << ": "
				<< std::fixed << std::setprecision(1) << acc << "% ("
				<< correct << "/" << testCount << ")\n";
		}

		double meanAcc = totalAcc / nFolds;
		double variance = 0;

		for (double a : foldAccs)
		{
			variance += (a - meanAcc) * (a - meanAcc);
		}

		double stdAcc = std::sqrt(variance / nFolds);

		std::cout << "    式式式式式式式式式式式式式式式\n";
		std::cout << "    Mean: " << std::setprecision(1) << meanAcc << "% (+/- " << std::setprecision(1) << stdAcc << "%)\n";
	}
};