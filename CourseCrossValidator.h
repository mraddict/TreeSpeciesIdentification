#pragma once

// 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
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
// 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式


#include "TSICommon.h"

class CourseCrossValidator
{
public:

	struct Options
	{
		ClassifierOptions rf;															// RF hyper-parameters (maxDepth, minSampleCount, maxTrees, activeVarCount)
		long long courseIdDivisor = 100000;								// treeId / divisor = course id (TreeLoader offset step)
		std::function<int(const TreeFeature&)> courseOf;			// optional override
		std::function<std::string(const std::string&)> groupOf;	// optional: species code -> group code (for L1 accuracy)
		bool runRandomKFold = true;											// also run shuffled k-fold for comparison
		int randomFolds = 5;
		unsigned seed = 42;
	};

	struct FoldResult
	{
		int heldOutCourse = -1;
		int testCount = 0;
		int correct = 0;
		int groupCorrect = 0;
		int unseenSpeciesCount = 0;											// test trees whose species never appears in training folds
		std::map<std::string, std::pair<int, int>> perSpecies;		// code -> (correct, total)

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
		int nFeatures = 0;
		int nSamples = 0;
		int nSpecies = 0;
		std::vector<FoldResult> loco;												// one per held-out course
		std::vector<double> randomFoldAcc;									// shuffled k-fold accuracies
		double locoMean = 0;
		double locoStd = 0;
		double locoWeighted = 0.0;													// weighted by test count (= overall held-out accuracy)
		double locoGroupWeighted = 0.0;
		double randomMean = 0.0;
		std::map<std::string, std::pair<int, int>> perSpecies;			// aggregated over all held-out courses
		std::map<std::string, std::string> codeToName;
	};

	// 式式式 Run LOCO (+ optional random k-fold baseline) on the reliable training set 式式式
	static Report run(const ExtractionResult& features, const Options& opts = Options())
	{
		Report rep;

		const auto& data = features.train;

		if (data.empty())
		{
			throw std::runtime_error("CourseCrossValidation: no training data");
		}

		// Label map
		std::map<std::string, int> codeToLabel;
		std::map<int, std::string> labelToCode;

		{
			std::set<std::string> codes;

			for (auto* tf : data)
			{
				codes.insert(tf->speciesCode);

				if (!tf->speciesName.empty())
				{
					rep.codeToName[tf->speciesCode] = tf->speciesName;
				}
			}

			int idx = 0;

			for (auto&c : codes)
			{
				codeToLabel[c] = idx;
				labelToCode[idx] = c;

				++idx;
			}
		}

		const int n = (int)data.size();
		const int nFeats = (int)data[0]->feature.size();

		rep.nSamples = n;
		rep.nFeatures = nFeats;
		rep.nSpecies = (int)codeToLabel.size();

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

			y.at<int>(i, 0) = codeToLabel[tf->speciesCode];
			course[i] = opts.courseOf ? opts.courseOf(*tf) : (int)(tf->treeId / opts.courseIdDivisor);
		}

		std::set<int> courses(course.begin(), course.end());

		std::cout << "=== Leave-one-course-out evaluation ===\n"
			<< "  Samples : " << n << "\n" << "  Features: " << nFeats << "\n" << "  Species : " << rep.nSpecies << "\n" 
			<< "  Courses : " << courses.size() << " (";

		for (int c : courses)
		{
			std::cout << " " << c;
		}

		std::cout << " )\n";

		if (courses.size() < 2)
		{
			throw std::runtime_error("CourseCV: need at least 2 courses (check courseIdDivisor / TreeID offsets)");
		}

		int activeVars = opts.rf.activeVarCount > 0 ? opts.rf.activeVarCount : (int)std::round(std::sqrt((double)nFeats));

		// 式式式 LOCO folds 式式式
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
			std::set<int> seenLabels;

			for (int i : trainIdx)
			{
				seenLabels.insert(y.at<int>(i, 0));
			}

			cv::Mat trX, trY, teX;
			gatherXY(X, y, trainIdx, trX, trY);
			gatherX(X, testIdx, teX);

			auto rf = makeRF(opts.rf, activeVars);
			rf->train(cv::ml::TrainData::create(trX, cv::ml::ROW_SAMPLE, trY));

			cv::Mat pred;
			rf->predict(teX, pred);

			for (int k = 0 ; k < fr.testCount ; ++k)
			{
				int trueL = y.at<int>(testIdx[k], 0);
				int predL = (int)pred.at<float>(k);

				const std::string& trueCode = labelToCode[trueL];
				const std::string& predCode = labelToCode[predL];

				auto& ps = fr.perSpecies[trueCode];
				++ps.second;

				if (!seenLabels.count(trueL))
				{
					++fr.unseenSpeciesCount;
				}

				if (predL == trueL)
				{
					++fr.correct;
					++ps.first;
				}

				if (opts.groupOf)
				{
					if (opts.groupOf(predCode) == opts.groupOf(trueCode))
					{
						++fr.groupCorrect;
					}
				}
			}
			
			std::cout << "  Held-out course " << std::setw(3) << held << ": "
				<< std::fixed << std::setprecision(1) << std::setw(5) << fr.accuracy() << "%  ("
				<< fr.correct << "/" << fr.testCount << ")";

			if (opts.groupOf)
			{
				std::cout << "   group " << std::setw(5) << fr.groupAccuracy() << "%";
			}

			if (fr.unseenSpeciesCount > 0)
			{
				std::cout << "   [" << fr.unseenSpeciesCount << " trees of species unseen in training]";
			}

			std::cout << "\n";

			sumAcc += fr.accuracy();
			wCorrect += fr.correct;
			wGroupCorrect += fr.groupCorrect;
			wTotal += fr.testCount;

			for (auto& kv : fr.perSpecies)
			{
				rep.perSpecies[kv.first].first += kv.second.first;
				rep.perSpecies[kv.first].second += kv.second.second;
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

		// 式式式 Random k-fold baseline (same RF settings) 式式式
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

				cv::Mat trX, trY, teX;
				gatherXY(X, y, trainIdx, trX, trY);
				gatherX(X, testIdx, teX);

				auto rf = makeRF(opts.rf, activeVars);
				rf->train(cv::ml::TrainData::create(trX, cv::ml::ROW_SAMPLE, trY));

				cv::Mat pred;
				rf->predict(teX, pred);

				int correct = 0;

				for (int k = 0 ; k < (int)testIdx.size() ; ++k)
				{
					if ((int)pred.at<float>(k) == y.at<int>(testIdx[k], 0))
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
		}

		return rep;
	}

	// 式式式 Summary 式式式
	static void print(const Report& r, const std::string& title = "")
	{
		std::cout << "\n  =============================================================\n";

		if (!title.empty())
		{
			std::cout << "  " << title << "\n";
		}

		std::cout << "  Features: " << r.nFeatures << "   Samples: " << r.nSamples << "   Species: " << r.nSpecies << "\n";
		std::cout << "  ----------------------------------------------------------------------------------------------------------\n";
		std::cout << std::fixed << std::setprecision(1);
		std::cout << "  LOCO  mean of courses : " << r.locoMean << "%  (+/- " << r.locoStd << "%)\n";
		std::cout << "  LOCO  overall (pooled): " << r.locoWeighted << "%\n";

		if (r.locoGroupWeighted > 0.0)
		{
			std::cout << "  LOCO  group  (pooled) : " << r.locoGroupWeighted << "%\n";
		}

		if (!r.randomFoldAcc.empty())
		{
			std::cout << "  Random k-fold mean    : " << r.randomMean << "%\n";
			std::cout << "  Inflation (random-LOCO): " << (r.randomMean - r.locoMean) << " pt\n";
		}

		std::cout << "  ----------------------------------------------------------------------------------------------------------\n";
		std::cout << "  Per-species (pooled over held-out courses):\n";

		std::vector<std::pair<std::string, std::pair<int, int>>> rows(r.perSpecies.begin(), r.perSpecies.end());
		std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second.second > b.second.second; });

		for (auto& kv : rows)
		{
			double acc = kv.second.second ? 100.0 * kv.second.first / kv.second.second : 0.0;
			std::string name = r.codeToName.count(kv.first) ? r.codeToName.at(kv.first) : "";

			std::cout << "    " << std::setw(8) << std::left << kv.first
				<< std::setw(14) << name
				<< std::right << std::setw(6) << acc << "%  ("
				<< kv.second.first << "/" << kv.second.second << ")\n";
		}

		std::cout << "  =============================================================\n";
	}

	// 式式式 Side-by-side comparison of two reports (e.g. Handcrafted vs Dino) 式式式
	static void compare(const Report& a, const std::string& nameA, const Report& b, const std::string& nameB)
	{
		std::cout << "  ======================= Comparison ======================================\n";
		std::cout << std::fixed << std::setprecision(1);
		std::cout << "  " << std::setw(26) << std::left << "" << std::setw(14) << std::right << nameA << std::setw(14) << nameB << std::setw(10) << "diff" << "\n";

		auto row = [&](const char* label, double va, double vb)
			{
				std::cout << "  " << std::setw(26) << std::left << label
					<< std::setw(13) << std::right << va << "%"
					<< std::setw(13) << vb << "%"
					<< std::setw(9) << std::showpos << (vb - va) << std::noshowpos << "\n";
			};

		row("Features (dim)", (double)a.nFeatures, (double)b.nFeatures);
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

