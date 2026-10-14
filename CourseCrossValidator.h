#pragma once

// ----------------------------------------------------------------------------------------------------------------------------
//  CourseCrossValidator.h
//  Leave-one-course-out (LOCO) evaluation for species / asset classifiers
//
//  Why:
//    Random k-fold puts neighboring trees from the same polygon into both
//    train and test, which inflates accuracy. Holding out a whole course
//    measures what actually matters: generalization to a course the model
//    has never seen (different flight date, camera, lighting).
//
//  Usage:
//    auto feats = FeatureExtractor::loadCSV("features_dino.csv");
//    CourseCV::Options o;
//    o.rf = ClassifierOptions();            // same RF settings as SpeciesClassifier
//    o.groupOf = &ForestCommon::groupOf;    // optional: species code -> L1 group
//    CourseCV::Report r = CourseCV::run(feats, o);
//    CourseCV::print(r);
//
//  Course ID:
//    Derived from TreeID offset (treeId / courseIdDivisor). Set
//    opts.courseIdDivisor to the offset step used in TreeLoader (100000),
//    or supply opts.courseOf to map a TreeFeature to a course id yourself.
//
//  Dependencies: FeatureExtractor.h (TreeFeature, ExtractionResult),
//                SpeciesClassifier.h (ClassifierOptions), OpenCV ml
// ----------------------------------------------------------------------------------------------------------------------------

#include "TSICommon.h"

class CourseCrossValidator
{
public:

	enum class Target
	{
		Species,
		Group
	};

	enum class Classifier
	{
		RandomForest,
		Logistic
	};

	enum class FeatureSet
	{
		All,										// every column as stored (default)
		LidarOnly,							// last lidarCount columns only
		PcaImagePlusLidar				// PCA(pcaDims) on image columns, standardized, + standardized LiDAR
	};

	struct Options
	{
		// data

		long long courseIdDivisor = 100000;								// treeId / divisor = course id (TreeLoader offset step)
		std::function<int(const TreeFeature&)> courseOf;			// optional override
		std::function<std::string(const std::string&)> groupOf;	// optional: species code -> group code (for L1 accuracy)
		std::set<std::string> excludeCodes;								// e.g. {"77"}
		Target target = Target::Species;

		// feature

		FeatureSet featureSet = FeatureSet::All;
		int lidarCount = 8;															// LiDAR columns are always the LAST lidarCount
		int pcaDims = 0;																// 0 = off

		// Classifier

		Classifier classifier = Classifier::RandomForest;
		ClassifierOptions rf;															// RF hyper-parameters (maxDepth, minSampleCount, maxTrees, activeVarCount)
		bool classPriors = false;													// inverse-frequency priors (RF only)
		int logisticIters = 1500;														// Logistic only
		double logisticLR = 0.05;
		int logisticBatch = 256;

		// evaluation

		bool runRandomKFold = true;											// also run shuffled k-fold for comparison
		int randomFolds = 5;
		unsigned seed = 42;

		std::string discribe() const
		{
			std::ostringstream s;
			s << ((target == Target::Group) ? "group" : "species");

			if (!excludeCodes.empty())
			{
				s << " -{";

				for (auto& c : excludeCodes)
				{
					s << c << " ";
				}

				s << "}";
			}

			if (featureSet == FeatureSet::LidarOnly)
			{
				s << " lidar-only";
			}
			else if (featureSet == FeatureSet::PcaImagePlusLidar)
			{
				s << " pca" << pcaDims << "+lidar(std)";
			}
			else if (pcaDims > 0)
			{
				s << " pca" << pcaDims;
			}

			s << (classifier == Classifier::Logistic ? " logistic" : " rf");

			if (classPriors && classifier == Classifier::RandomForest)
			{
				s << "+priors";
			}

			return s.str();
		}
	};

	struct FoldResult
	{
		int heldOutCourse = -1;
		int testCount = 0;
		int correct = 0;
		int groupCorrect = 0;
		int unseenLabelCount = 0;												// test trees whose label never appears in training folds
		std::map<std::string, std::pair<int, int>> perLabel;			// label -> (correct, total)

		double accuracy() const
		{
			return testCount ? 100.0 * correct / testCount : 0.0;
		}

		double groupAccuracy() const
		{
			return testCount ? 100.0 * groupCorrect / testCount : 0.0;
		}
	};

	struct Report
	{
		std::string config;
		int nFeatures = 0;
		int nFeaturesUsed = 0;														// after PCA
		int nSamples = 0;
		int nLabels = 0;
		std::vector<FoldResult> loco;												// one per held-out course
		std::vector<double> randomFoldAcc;									// shuffled k-fold accuracies
		double locoMean = 0;
		double locoStd = 0;
		double locoWeighted = 0.0;													// weighted by test count (= overall held-out accuracy)
		double locoGroupWeighted = 0.0;										// only meaningful for Target::Species with groupOf
		double randomMean = 0.0;
		std::map<std::string, std::pair<int, int>> perLabel;				// aggregated over all held-out courses
		std::map<std::string, std::string> labelName;
	};

	// Run LOCO (+ optional random k-fold baseline) on the reliable training set 
	static Report run(const ExtractionResult& features, const Options& opts = Options())
	{
		Report rep;
		rep.config = opts.discribe();

		if ((opts.target == Target::Group) && !opts.groupOf)
		{
			throw std::runtime_error("CourseCV: Target::Group requires opts.groupOf");
		}

		// Select rows
		std::vector<const TreeFeature*> data;

		for (auto* tf : features.train)
		{
			if (opts.excludeCodes.count(tf->speciesCode))
			{
				continue;
			}

			data.push_back(tf);
		}

		if (data.empty())
		{
			throw std::runtime_error("CourseCrossValidation: no training data");
		}

		// Label 

		auto labelOf = [&](const TreeFeature& tf) -> std::string
		{
			return (opts.target == Target::Group) ? opts.groupOf(tf.speciesCode) : tf.speciesCode;
		};

		std::map<std::string, int> labelToIdx;
		std::map<int, std::string> idxToLabel;

		{
			std::set<std::string> labels;

			for (auto* tf : data)
			{
				std::string l = labelOf(*tf);
				labels.insert(l);

				if ((opts.target == Target::Species) && !tf->speciesName.empty())
				{
					rep.labelName[l] = tf->speciesName;
				}
				else if (opts.target == Target::Group)
				{
					rep.labelName[l] = SpeciesCodes::groupName(l);
				}
			}

			int idx = 0;

			for (auto& l : labels)
			{
				labelToIdx[l] = idx;
				idxToLabel[idx] = l;

				++idx;
			}
		}

		const int n = (int)data.size();
		const int nFeats = (int)data[0]->feature.size();
		const int nLabels = (int)labelToIdx.size();

		rep.nSamples = n;
		rep.nFeatures = nFeats;

		if (opts.featureSet == FeatureSet::LidarOnly)
		{
			rep.nFeaturesUsed = opts.lidarCount;
		}
		else if (opts.featureSet == FeatureSet::PcaImagePlusLidar)
		{
			rep.nFeaturesUsed = std::min(opts.pcaDims, nFeats - opts.lidarCount) + opts.lidarCount;
		}
		else
		{
			rep.nFeaturesUsed = (opts.pcaDims > 0) ? std::min(opts.pcaDims, nFeats) : nFeats;
		}

		rep.nLabels = nLabels;

		// Matrices
		cv::Mat X(n, nFeats, CV_32F);
		cv::Mat y(n, 1, CV_32S);

		std::vector<int> course(n);

		for (int i = 0 ; i < n ; ++i)
		{
			auto* tf = data[i];

			for (int j = 0 ; j < nFeats ; j++)
			{
				X.at<float>(i, j) = tf->feature[j];
			}

			y.at<int>(i, 0) = labelToIdx[labelOf(*tf)];
			course[i] = opts.courseOf ? opts.courseOf(*tf) : (int)(tf->treeId / opts.courseIdDivisor);
		}

		std::set<int> courses(course.begin(), course.end());

		std::cout << "=== LOCO [" << rep.config << "] ===\n"
			<< "  Samples : " << n << "   Features: " << nFeats << (opts.pcaDims > 0 ? " -> " + std::to_string(rep.nFeaturesUsed) : "") << "   Labels: " << nLabels << "   Courses: " << courses.size() << "\n";

		// LOCO 

		if (courses.size() < 2)
		{
			std::cout << "CourseCV: need at least 2 courses (check courseIdDivisor / TreeID offsets)\n";
		}
		else
		{
			// LOCO folds 
			double sumAcc = 0.0;
			long long wCorrect  = 0;
			long long wGroupCorrect = 0;
			long long wTotal = 0;

			for (int held : courses)
			{
				std::vector<int> trainIdx, testIdx;

				for (int i = 0 ; i < n ; ++i)
				{
					(course[i] == held ? testIdx : trainIdx).push_back(i);
				}

				FoldResult fr;
				fr.heldOutCourse = held;
				fr.testCount = (int)testIdx.size();

				// Species seen in training folds
				std::set<int> seen;

				for (int i : trainIdx)
				{
					seen.insert(y.at<int>(i, 0));
				}

				std::vector<int> pred = fitPredict(X, y, trainIdx, testIdx, nLabels, opts);

				for (int k = 0; k < fr.testCount; ++k)
				{
					int trueL = y.at<int>(testIdx[k], 0);
					int predL = pred[k];

					auto& pl = fr.perLabel[idxToLabel[trueL]];
					++pl.second;

					if (!seen.count(trueL))
					{
						++fr.unseenLabelCount;
					}

					if (predL == trueL)
					{
						++fr.correct;
						++pl.first;
					}

					if ((opts.target == Target::Species) && opts.groupOf)
					{
						if (opts.groupOf(idxToLabel[predL]) == opts.groupOf(idxToLabel[trueL]))
						{
							++fr.groupCorrect;
						}
					}
				}

				std::cout << "  Held-out course " << std::setw(3) << held << ": "
					<< std::fixed << std::setprecision(1) << std::setw(5) << fr.accuracy() << "%  ("
					<< fr.correct << "/" << fr.testCount << ")";

				if ((opts.target == Target::Species) && opts.groupOf)
				{
					std::cout << "   group " << std::setw(5) << fr.groupAccuracy() << "%";
				}

				if (fr.unseenLabelCount > 0)
				{
					std::cout << "   [" << fr.unseenLabelCount << " trees of label unseen in training]";
				}

				std::cout << "\n";

				sumAcc += fr.accuracy();
				wCorrect += fr.correct;
				wGroupCorrect += fr.groupCorrect;
				wTotal += fr.testCount;

				for (auto& kv : fr.perLabel)
				{
					rep.perLabel[kv.first].first += kv.second.first;
					rep.perLabel[kv.first].second += kv.second.second;
				}

				rep.loco.push_back(std::move(fr));
			}

			rep.locoMean = sumAcc / rep.loco.size();

			double var = 0.0;

			for (auto& fr : rep.loco)
			{
				var += (fr.accuracy() - rep.locoMean) * (fr.accuracy() - rep.locoMean);
			}

			rep.locoStd = std::sqrt(var / rep.loco.size());
			rep.locoWeighted = wTotal ? 100.0 * wCorrect / wTotal : 0.0;
			rep.locoGroupWeighted = wTotal ? 100.0 * wGroupCorrect / wTotal : 0.0;
		}

		// Random k-fold baseline (same RF settings) 
		if (opts.runRandomKFold && n >= opts.randomFolds * 5)
		{
			std::cout << "\n  Random " << opts.randomFolds << "-fold (for comparison):\n";

			std::vector<int> perm(n);
			std::iota(perm.begin(), perm.end(), 0);
			std::mt19937 rng(opts.seed);
			std::shuffle(perm.begin(), perm.end(), rng);

			int foldSize = n / opts.randomFolds;
			double sum = 0.0;

			for (int f = 0 ; f < opts.randomFolds ; ++f)
			{
				int s = f * foldSize;
				int e = (f == opts.randomFolds - 1) ? n : s + foldSize;

				std::vector<int> trainIdx, testIdx;

				for (int i = 0 ; i < n ; ++i)
				{
					((i >= s) && (i < e) ? testIdx : trainIdx).push_back(perm[i]);
				}

				std::vector<int> pred = fitPredict(X, y, trainIdx, testIdx, nLabels, opts);

				int correct = 0;

				for (int k = 0 ; k < (int)testIdx.size() ; ++k)
				{
					if (pred[k] == y.at<int>(testIdx[k], 0))
					{
						++correct;
					}
				}

				double acc = 100.0 * correct / testIdx.size();
				rep.randomFoldAcc.push_back(acc);
				sum += acc;

				std::cout << "    Fold " << (f + 1) << ": " << std::fixed << std::setprecision(1) << acc << "%\n";
			}

			rep.randomMean = sum / opts.randomFolds;

			std::cout << "  Random " << opts.randomFolds << "-fold mean: " << std::fixed << std::setprecision(1) << rep.randomMean << "%\n";
		}

		return rep;
	}

	// Summary 
	static void print(const Report& r, const std::string& title = "")
	{
		std::cout << "\n  =============================================================\n";

		if (!title.empty())
		{
			std::cout << "  " << title << "\n";
		}

		std::cout << "  Config: " << r.config << "\n";
		std::cout << "  Features: " << r.nFeatures;

		if (r.nFeaturesUsed != r.nFeatures)
		{
			std::cout << " -> " << r.nFeaturesUsed;
		}

		std::cout << "   Samples: " << r.nSamples << "   Labels: " << r.nLabels << "\n";
		std::cout << "  =============================================================\n";
		std::cout << std::fixed << std::setprecision(1);

		if (!r.loco.empty())
		{
			std::cout << "  LOCO  mean of courses : " << r.locoMean << "%  (+/- " << r.locoStd << "%)\n";
			std::cout << "  LOCO  overall (pooled): " << r.locoWeighted << "%\n";

			if (r.locoGroupWeighted > 0.0)
			{
				std::cout << "  LOCO  group  (pooled) : " << r.locoGroupWeighted << "%\n";
			}
		}

		if (!r.randomFoldAcc.empty())
		{
			std::cout << "  Random k-fold mean    : " << r.randomMean << "%\n";

			if (!r.loco.empty())
			{
				std::cout << "  Inflation (random-LOCO): " << (r.randomMean - r.locoMean) << " pt\n";
			}
		}

		std::cout << "  =============================================================\n";
		std::cout << "  Per-label (pooled over held-out courses):\n";

		std::vector<std::pair<std::string, std::pair<int, int>>> rows(r.perLabel.begin(), r.perLabel.end());
		std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second.second > b.second.second; });

		for (auto& kv : rows)
		{
			double acc = kv.second.second ? 100.0 * kv.second.first / kv.second.second : 0.0;
			std::string name = r.labelName.count(kv.first) ? r.labelName.at(kv.first) : "";

			std::cout << "    " << std::setw(8) << std::left << kv.first
				<< std::setw(16) << name
				<< std::right << std::setw(6) << acc << "%  ("
				<< kv.second.first << "/" << kv.second.second << ")\n";
		}

		std::cout << "  =============================================================\n";
	}

	// Compact table across many reports (variant sweep) 
	static void printTable(const std::vector<Report>& reports)
	{
		std::cout << "\n  ================================ Variant summary ================================\n";
		std::cout << "  " << std::left << std::setw(44) << "config"
			<< std::right << std::setw(6) << "dim"
			<< std::setw(9) << "LOCO" << std::setw(9) << "+-" << std::setw(9) << "pooled"
			<< std::setw(9) << "group" << std::setw(9) << "random" << std::setw(9) << "infl" << "\n";
		std::cout << "  " << std::string(104, '-') << "\n";

		for (auto& r : reports)
		{
			std::cout << "  " << std::left << std::setw(44) << r.config.substr(0, 43)
				<< std::right << std::setw(6) << r.nFeaturesUsed
				<< std::fixed << std::setprecision(1)
				<< std::setw(8) << r.locoMean << "%"
				<< std::setw(8) << r.locoStd << "%"
				<< std::setw(8) << r.locoWeighted << "%";

			if (r.locoGroupWeighted > 0.0)
			{
				std::cout << std::setw(8) << r.locoGroupWeighted << "%";
			}
			else
			{
				std::cout << std::setw(9) << "-";
			}

			if (!r.randomFoldAcc.empty())
			{
				std::cout << std::setw(8) << r.randomMean << "%" << std::setw(8) << (r.randomMean - r.locoMean) << " ";
			}
			else
			{
				std::cout << std::setw(9) << "-" << std::setw(9) << "-";
			}

			std::cout << "\n";
		}

		std::cout << "  " << std::string(104, '-') << "\n";
	}

	// Side-by-side comparison of two reports (e.g. Handcrafted vs Dino)
	static void compare(const Report& a, const std::string& nameA, const Report& b, const std::string& nameB)
	{
		std::cout << "  ======================= Comparison ======================================\n";
		std::cout << std::fixed << std::setprecision(1);
		std::cout << "  " << std::setw(26) << std::left << "" << std::setw(14) << std::right << nameA << std::setw(14) << nameB << std::setw(10) << "diff" << "\n";

		// dims: plain integers (no %)
		std::cout << "  " << std::setw(26) << std::left << "Features (dim)"
			<< std::setw(14) << std::right << a.nFeaturesUsed
			<< std::setw(14) << b.nFeaturesUsed
			<< std::setw(9) << std::showpos << (b.nFeaturesUsed - a.nFeaturesUsed) << std::noshowpos << "\n";
		
		auto row = [&](const char* label, double va, double vb)
		{
			std::cout << "  " << std::setw(26) << std::left << label
				<< std::setw(13) << std::right << va << "%"
				<< std::setw(13) << vb << "%"
				<< std::setw(9) << std::showpos << (vb - va) << std::noshowpos << "\n";
		};

		row("LOCO mean", a.locoMean, b.locoMean);
		row("LOCO pooled", a.locoWeighted, b.locoWeighted);

		if ((a.locoGroupWeighted > 0.0) || (b.locoGroupWeighted > 0.0))
		{
			row("LOCO group pooled", a.locoGroupWeighted, b.locoGroupWeighted);
		}

		if (!a.randomFoldAcc.empty() && !b.randomFoldAcc.empty())
		{
			row("Random k-fold mean", a.randomMean, b.randomMean);
		}

		// Per held-out course
		std::cout << "\n  Per held-out course:\n";

		for (size_t i = 0; (i < a.loco.size()) && (i < b.loco.size()); ++i)
		{
			std::cout << "    course " << std::setw(3) << a.loco[i].heldOutCourse
				<< std::setw(13) << a.loco[i].accuracy() << "%"
				<< std::setw(13) << b.loco[i].accuracy() << "%"
				<< std::setw(9) << std::showpos << (b.loco[i].accuracy() - a.loco[i].accuracy()) << std::noshowpos << "\n";
		}

		std::cout << "  =============================================================\n";
	}

private:

	// Fit on trainIdx, predict testIdx. PCA / standardization / priors fitted on train only 
	static std::vector<int> fitPredict(const cv::Mat& X, const cv::Mat& y, const std::vector<int>& trainIdx, const std::vector<int>& testIdx, int nLabels, const Options& opts)
	{
		cv::Mat trX, trY, teX;
		gatherXY(X, y, trainIdx, trX, trY);
		gatherX(X, testIdx, teX);

		const int lidarStart = X.cols - opts.lidarCount;

		// PCA (fit on train)

		if (opts.featureSet == FeatureSet::LidarOnly)
		{
			trX = trX.colRange(lidarStart, X.cols).clone();
			teX = teX.colRange(lidarStart, X.cols).clone();
		}
		else if (opts.featureSet == FeatureSet::PcaImagePlusLidar)
		{
			// Split image / LiDAR columns
			cv::Mat trImg = trX.colRange(0, lidarStart);
			cv::Mat teImg = teX.colRange(0, lidarStart);
			cv::Mat trLid = trX.colRange(lidarStart, X.cols).clone();
			cv::Mat teLid = teX.colRange(lidarStart, X.cols).clone();

			int k = std::min(opts.pcaDims > 0 ? opts.pcaDims : 32, lidarStart);
			cv::PCA pca(trImg, cv::Mat(), cv::PCA::DATA_AS_ROW, k);
			cv::Mat trP = pca.project(trImg), teP = pca.project(teImg);

			// Standardize each part on train statistics so neither dominates by scale
			cv::Mat m1, s1, m2, s2;
			meanStd(trP, m1, s1);
			meanStd(trLid, m2, s2);
			standardize(trP, m1, s1);
			standardize(teP, m1, s1);
			standardize(trLid, m2, s2);
			standardize(teLid, m2, s2);

			cv::hconcat(trP, trLid, trX);
			cv::hconcat(teP, teLid, teX);

		}
		else if ((opts.pcaDims > 0) && (opts.pcaDims < trX.cols))
		{
			// PCA on everything (fit on train)
			cv::PCA pca(trX, cv::Mat(), cv::PCA::DATA_AS_ROW, opts.pcaDims);

			trX = pca.project(trX);
			teX = pca.project(teX);
		}

		std::cout << "    [fitPredict] trX " << trX.rows << "x" << trX.cols << " type=" << trX.type()
			<< "  teX " << teX.rows << "x" << teX.cols << " type=" << teX.type()
			<< "  (CV_32F=" << CV_32F << ")\n";

		if (trX.empty() || teX.empty())
		{
			throw std::runtime_error("fitPredict: empty feature matrix (check lidarCount / featureSet)");
		}

		if (trX.type() != CV_32F)
		{
			trX.convertTo(trX, CV_32F);
		}

		if (teX.type() != CV_32F)
		{
			teX.convertTo(teX, CV_32F);
		}

		if (!trX.isContinuous())
		{
			trX = trX.clone();
		}

		if (!teX.isContinuous())
		{
			teX = teX.clone();
		}
		
		std::vector<int> pred(testIdx.size());

		if (opts.classifier == Classifier::Logistic)
		{
			// Standardize (fit on train)
			cv::Mat mean, stdv;
			cv::Mat sq;

			meanStd(trX, mean, stdv);
			standardize(trX, mean, stdv);
			standardize(teX, mean, stdv);

			cv::Mat trYf;
			trY.convertTo(trYf, CV_32F);

			auto lr = cv::ml::LogisticRegression::create();
			lr->setLearningRate(opts.logisticLR);
			lr->setIterations(opts.logisticIters);
			lr->setRegularization(cv::ml::LogisticRegression::REG_L2);
			lr->setTrainMethod(cv::ml::LogisticRegression::MINI_BATCH);
			lr->setMiniBatchSize(opts.logisticBatch);
			lr->train(trX, cv::ml::ROW_SAMPLE, trYf);

			cv::Mat out;
			lr->predict(teX, out);

			for (int k = 0 ; k < out.rows ; ++k)
			{
				// OpenCV LogisticRegression::predict returns CV_32S labels (CV_32F on some builds)
				pred[k] = (out.type() == CV_32S) ? out.at<int>(k, 0) : (int)std::lround(out.at<float>(k, 0));
			}

			return pred;
		}

		// Random Forest

		int activeVars = opts.rf.activeVarCount > 0 ? opts.rf.activeVarCount : (int)std::round(std::sqrt((double)trX.cols));

		auto rf = makeRF(opts.rf, activeVars);

		if (opts.classPriors)
		{
			// Inverse-frequency priors over the labels present in this training fold.
			// Absent labels get a tiny prior so the vector length still matches nLabels.

			// OpenCV requires: CV_64F, length == number of DISTINCT labels in this training fold,
			// ordered by ascending label value (the order DTrees uses internally).

			std::map<int, int> count;			// label -> count (std::map keeps ascending order)
			//std::vector<int> count(nLabels, 0);

			for (int i = 0 ; i < trY.rows ; ++i)
			{
				++count[trY.at<int>(i, 0)];
			}

			cv::Mat priors(1, (int)count.size(), CV_64F);
			double sum = 0.0;
			int c = 0;

			for (auto& kv : count)
			{
				//priors.at<double>(0, c) = (count[c] > 0) ? 1.0f / count[c] : 1e-6f;
				priors.at<double>(0, c) = 1.0 / kv.second;
				sum += priors.at<double>(0, c);

				++c;
			}

			priors /= sum;
			rf->setPriors(priors);
		}

		rf->train(cv::ml::TrainData::create(trX, cv::ml::ROW_SAMPLE, trY));

		cv::Mat out;
		rf->predict(teX, out);

		for (int k = 0 ; k < out.rows ; ++k)
		{
			pred[k] = (int)out.at<float>(k, 0);
		}

		return pred;
	}

	static void meanStd(const cv::Mat& M, cv::Mat& mean, cv::Mat& stdv)
	{
		cv::reduce(M, mean, 0, cv::REDUCE_AVG, CV_32F);

		cv::Mat sq;
		cv::multiply(M, M, sq);
		cv::reduce(sq, stdv, 0, cv::REDUCE_AVG, CV_32F);
		stdv = stdv - mean.mul(mean);
		cv::max(stdv, cv::Mat(stdv.size(), CV_32F, cv::Scalar(1e-12f)), stdv);
		cv::sqrt(stdv, stdv);
	}

	static void standardize(cv::Mat& M, const cv::Mat& mean, const cv::Mat& stdv)
	{
		for (int i = 0 ; i < M.rows ; ++i)
		{
			float* row = M.ptr<float>(i);

			for (int j = 0 ; j < M.cols ; ++j)
			{
				row[j] = (row[j] - mean.at<float>(0, j)) / stdv.at<float>(0, j);
			}
		}
	}

	static cv::Ptr<cv::ml::RTrees> makeRF(const ClassifierOptions& opts, int activeVars)
	{
		auto rf = cv::ml::RTrees::create();

		rf->setMaxDepth(opts.maxDepth);
		rf->setMinSampleCount(opts.minSampleCount);
		rf->setActiveVarCount(activeVars);
		rf->setCalculateVarImportance(false);
		rf->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, opts.maxTrees, 1e-6));

		return rf;
	}

	// Copy selected rows of X into a new matrix
	static void gatherX(const cv::Mat& X, const std::vector<int>& idx, cv::Mat& outX)
	{
		outX.create((int)idx.size(), X.cols, CV_32F);

		for (int k = 0 ; k < (int)idx.size() ; ++k)
		{
			X.row(idx[k]).copyTo(outX.row(k));
		}
	}

	// Copy selected rows of X and y into new matrices
	static void gatherXY(const cv::Mat& X, const cv::Mat& y, const std::vector<int>& idx, cv::Mat& outX, cv::Mat& outY)
	{
		gatherX(X, idx, outX);
		outY.create((int)idx.size(), 1, CV_32S);

		for (int k = 0 ; k < (int)idx.size() ; ++k)
		{
			outY.at<int>(k, 0) = y.at<int>(idx[k], 0);
		}
	}
};

