#pragma once

#include "CourseDataSetLoader.h"

// CDF: Cumulative Distribution Function (누적 분포 함수)

class AerialHistogramMatcher
{
public:

	// ─── 메인 실행 함수 ───
	// configPath  : courses.json 경로
	// refCourseName : 기준 코스 이름 (예: "AnseongW")
	// outputDir   : 변환된 이미지 저장 폴더
	// suffix      : 저장 파일명 접미사 (기본: "_matched")
	void run(const std::string& configPath, const std::string& refCourseName, const std::string& outputDir, const std::string& suffix = "_matched")
	{
		// 1. 설정 로드
		//std::string courseDataSetConfigFilename = "../../Resources/courses_dataset.json";
		CourseDataSetConfig courseDSConfig = CourseDataSetConfigLoader::load(configPath);
		std::vector<CourseDataSet> enabled;

		for (auto& c : courseDSConfig.courses)
		{
			if (c.enabled)
			{
				enabled.push_back(c);
			}
		}

		if (enabled.empty())
		{
			std::cerr << "Error: No enabled courses in config.\n";
			return;
		}

		// 2. 기준 코스 찾기
		const CourseDataSet* refDS = nullptr;

		for (auto& c : enabled)
		{
			if (c.courseName == refCourseName)
			{
				refDS = &c;

				break;
			}
		}

		if (!refDS)
		{
			std::cerr << "Error: Reference course '" << refCourseName << "' not found.\n";
			return;
		}
	
		std::cout << "========================================\n"
			<< "  Histogram Matching\n"
			<< "  Reference: " << refCourseName << "\n"
			<< "  Courses:   " << enabled.size() << "\n"
			<< "  Output:    " << outputDir << "\n"
			<< "========================================\n\n";

		// 3. 기준 이미지의 히스토그램(CDF) 계산
		//    전체 이미지를 로드하지 않고 히스토그램만 계산
	
		std::cout << "=== Computing reference histogram ===\n" << "  Loading: " << refDS->aerialImgPath << "\n";

		std::vector<std::vector<float>> refCDF;
		{
			cv::Mat refImg = loadImageBGR(refDS->aerialImgPath);

			if (refImg.empty())
			{
				std::cerr << "Error: Cannot load reference image.\n";

				return;
			}

			std::cout << "  Size: " << refImg.cols << "x" << refImg.rows << " (" << (refImg.total() * refImg.elemSize() / 1048576) << " MB)\n";

			refCDF = computeCDF(refImg);

			std::cout << "  Reference CDF computed.\n\n";
			// refImg 메모리 해제됨 (스코프 종료)
		}

		// 4. 각 코스 처리
		for (auto& courseDS : enabled)
		{
			std::string outDir = outputDir + "/" + courseDS.courseName;
			TSICommon::mkdirs(outDir);

			// 파일명 추출
			std::string basename = TSICommon::getBaseName(courseDS.aerialImgPath);
			std::string ext = TSICommon::getExtension(courseDS.aerialImgPath);
			std::string outPath = outDir + "/" + basename + suffix + ext;

			if (courseDS.courseName == refCourseName)
			{
				// 기준 코스는 원본 복사
				std::cout << "=== " << courseDS.courseName << " (reference) ===\n" << "  Copying original to: " << outPath << "\n\n";
				TSICommon::copyFile(courseDS.aerialImgPath, outPath);

				continue;
			}

			std::cout << "=== Matching: " << courseDS.courseName << " ===\n" << "  Loading: " << courseDS.aerialImgPath << "\n";

			cv::Mat srcImg = loadImageBGR(courseDS.aerialImgPath);

			if (srcImg.empty())
			{
				std::cerr << "  Error: Cannot load image. Skipping.\n\n";

				continue;
			}

			std::cout << "  Size: " << srcImg.cols << "x" << srcImg.rows << " (" << (srcImg.total() * srcImg.elemSize() / 1048576) << " MB)\n";

			// 소스 CDF 계산
			auto srcCDF = computeCDF(srcImg);
			// LUT 생성
			auto luts = buildLUTs(srcCDF, refCDF);

			// LUT 적용 (in-place)
			applyLUTs(srcImg, luts);

			// 저장
			std::cout << "  Saving: " << outPath << "\n";
			cv::imwrite(outPath, srcImg);
		
			std::cout << "  Done.\n\n";
			// srcImg 메모리 해제됨 (루프 반복)
		}

		std::cout << "========================================\n"
			<< "  Histogram matching complete.\n"
			<< "  Update courses.json aerial paths to\n"
			<< "  use matched images, then run pipeline.\n"
			<< "========================================\n";
	}

	void runGDAL(const std::string& configPath, const std::string& refCourseName, const std::string& outputDir, const std::string& suffix = "_matched")
	{
		GDALAllRegister();

		// 1. 설정 로드
		//std::string courseDataSetConfigFilename = "../../Resources/courses_dataset.json";
		CourseDataSetConfig courseDSConfig = CourseDataSetConfigLoader::load(configPath);
		std::vector<CourseDataSet> enabled;

		for (auto& c : courseDSConfig.courses)
		{
			if (c.enabled)
			{
				enabled.push_back(c);
			}
		}

		if (enabled.empty())
		{
			std::cerr << "Error: No enabled courses in config.\n";
			return;
		}

		// 2. 기준 코스 찾기
		const CourseDataSet* refDS = nullptr;

		for (auto& c : enabled)
		{
			if (c.courseName == refCourseName)
			{
				refDS = &c;

				break;
			}
		}

		if (!refDS)
		{
			std::cerr << "Error: Reference course '" << refCourseName << "' not found.\n";
			return;
		}

		std::cout << "========================================\n"
			<< "  Histogram Matching\n"
			<< "  Reference: " << refCourseName << "\n"
			<< "  Courses:   " << enabled.size() << "\n"
			<< "  Output:    " << outputDir << "\n"
			<< "========================================\n\n";

		// 1. 기준 이미지 히스토그램 계산 (줄 단위, 메모리 ~width bytes)

		std::cout << "=== Reference: " << refCourseName << " ===\n" << "  Path: " << refDS->aerialImgPath << "\n";

		Histogram refHist;

		if (!computeHistogramGDAL(refDS->aerialImgPath, refHist))
		{
			return;
		}

		CDF refCDF = calcHistogramToCDF(refHist);
		std::cout << "  CDF computed.\n\n";

		// 2. 각 코스 처리

		for (auto& courseDS : enabled)
		{
			std::string outDir = outputDir + "/" + courseDS.courseName;
			TSICommon::mkdirs(outDir);
	
			// 파일명 추출
			std::string basename = TSICommon::getBaseName(courseDS.aerialImgPath);
			std::string ext = TSICommon::getExtension(courseDS.aerialImgPath);
			std::string outPath = outDir + "/" + basename + suffix + ext;

			if (courseDS.courseName == refCourseName)
			{
				// 기준 코스는 원본 복사
				std::cout << "=== " << courseDS.courseName << " (reference) ===\n" << "  Copying original to: " << outPath << "\n\n";
				TSICommon::copyFile(courseDS.aerialImgPath, outPath);

				continue;
			}
	
			std::cout << "=== Matching: " << courseDS.courseName << " ===\n" << "  Path: " << courseDS.aerialImgPath << "\n";

			// 소스 히스토그램 계산 (줄 단위)

			Histogram srcHist;

			if (!computeHistogramGDAL(courseDS.aerialImgPath, srcHist))
			{
				std::cerr << "  Skipping.\n\n";
				continue;
			}

			CDF srcCDF = calcHistogramToCDF(srcHist);
		
			// LUT 생성
			LUT lut = buildLUT(srcCDF, refCDF);
			std::cout << "  LUT computed.\n";

			// LUT 적용 + 저장 (줄 단위)
			if (applyLUTandSave(courseDS.aerialImgPath, outPath, lut))
			{
				std::cout << "  Saved: " << outPath << "\n\n";
			}
			else
			{
				std::cerr << "  Error saving. Skipping.\n\n";
			}
		}

		std::cout << "========================================\n"
			<< "  Histogram matching complete.\n"
			<< "  Update courses.json aerial paths to\n"
			<< "  use matched images, then run pipeline.\n"
			<< "========================================\n";
	}

	private:

	// BGR 이미지 로드 (4밴드면 앞 3밴드만 사용)
	static cv::Mat loadImageBGR(const std::string& path)
	{
		cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);

		if (img.empty())
		{
			return img;
		}

		if (img.channels() == 4)
		{
			// BGRA → BGR
			cv::Mat bgr;
			cv::cvtColor(img, bgr, cv::COLOR_BGRA2BGR);

			return bgr;
		}

		if (img.channels() == 1)
		{
			cv::Mat bgr;
			cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);

			return bgr;
		}

		return img;
	}

	// 채널별 CDF 계산 (B, G, R 3개)
	static std::vector<std::vector<float>> computeCDF(const cv::Mat& img)
	{
		std::vector<cv::Mat> channels;
		cv::split(img, channels);

		std::vector<std::vector<float>> cdfs;
		int histSize = 256;
		float range[] = {0, 256};
		const float* histRange = {range};

		for (int c = 0 ; c < 3 && c < (int)channels.size() ; ++c)
		{
			cv::Mat hist;
			cv::calcHist(&channels[c], 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange);

			// 누적
			std::vector<float> cdf(256);
			cdf[0] = hist.at<float>(0);

			for (int i = 1 ; i < 256 ; ++i)
			{
				cdf[i] = cdf[i - 1] + hist.at<float>(i);
			}

			// 정규화 (0~1)
			float total = cdf[255];

			if (total > 0)
			{
				for (int i = 0 ; i < 256 ; ++i)
				{
					cdf[i] /= total;
				}
			}

			cdfs.push_back(cdf);
		}

		return cdfs;
	}

	// src CDF → ref CDF 매핑 LUT(Look up table) 생성
	static std::vector<cv::Mat> buildLUTs(const std::vector<std::vector<float>>& srcCDF, const std::vector<std::vector<float>>& refCDF)
	{
		std::vector<cv::Mat> luts;
		int nCh = std::min(srcCDF.size(), refCDF.size());

		for (int c = 0 ; c < nCh ; ++c)
		{
			cv::Mat lut(1, 256, CV_8U);
			uchar* p = lut.ptr<uchar>();

			for (int i = 0 ; i < 256 ; ++i)
			{
				float val = srcCDF[c][i];

				// refCDF에서 가장 가까운 값 찾기 (이진 탐색)
				int lo = 0;
				int hi = 255;
				int best = 0;

				while (lo <= hi)
				{
					int mid = (lo + hi) / 2;

					if (refCDF[c][mid] <= val)
					{
						best = mid;
						lo = mid + 1;
					}
					else
					{
						hi = mid - 1;
					}
				}

				// best와 best+1 중 더 가까운 쪽
				if (best < 255)
				{
					float d1 = std::abs(val - refCDF[c][best]);
					float d2 = std::abs(val - refCDF[c][best + 1]);

					if (d2 < d1)
					{
						best = best + 1;
					}
				}

				p[i] = (uchar)best;
			}

			luts.push_back(lut);
		}

		return luts;
	}

	// LUT 적용 (in-place, 메모리 절약)
	static void applyLUTs(cv::Mat& img, const std::vector<cv::Mat>& luts)
	{
		std::vector<cv::Mat> channels;
		cv::split(img, channels);

		for (int c = 0 ; c < (int)luts.size() && c < (int)channels.size() ; ++c)
		{
			cv::LUT(channels[c], luts[c], channels[c]);
		}

		cv::merge(channels, img);
	}

	// ─── 채널별 히스토그램 (3채널, 각 256) ───
	struct Histogram
	{
		long long bins[3][256] = {};
	};

	// ─── 히스토그램 → CDF (정규화 0~1) ───
	struct CDF
	{
		float values[3][256] = {};
	};

	static CDF calcHistogramToCDF(const Histogram& hist)
	{
		CDF cdf;

		for (int b = 0 ; b < 3 ; ++b)
		{
			cdf.values[b][0] = (float)hist.bins[b][0];

			for (int i = 1 ; i < 256 ; ++i)
			{
				cdf.values[b][i] = cdf.values[b][i - 1] + (float)hist.bins[b][i];
			}

			float total = cdf.values[b][255];

			if (total > 0)
			{
				for (int i = 0 ; i < 256 ; ++i)
				{
					cdf.values[b][i] /= total;
				}
			}
		}

		return cdf;
	}

	// ─── CDF → LUT 생성 ───
	struct LUT
	{
		unsigned char table[3][256] = {};
	};

	// ─── GDAL로 줄 단위 히스토그램 계산 ───
	static bool computeHistogramGDAL(const std::string& path, Histogram& hist)
	{
		GDALDataset* ds = (GDALDataset*)GDALOpen(path.c_str(), GA_ReadOnly);

		if (!ds)
		{
			std::cerr << "  Error: Cannot open " << path << "\n";
			return false;
		}

		int width = ds->GetRasterXSize();
		int height = ds->GetRasterYSize();
		int nBands = ds->GetRasterCount();
		int useBands = std::min(nBands, 3);

		std::cout << "  Size: " << width << "x" << height << " (" << nBands << " bands)\n";

		// 줄 버퍼 (1줄 × width 픽셀)
		std::vector<unsigned char> lineBuf(width);

		// 밴드별 히스토그램 계산
		for (int b = 0; b < useBands; ++b)
		{
			GDALRasterBand* pBand = ds->GetRasterBand(b + 1);

			for (int y = 0; y < height; ++y)
			{
				CPLErr err = pBand->RasterIO(GF_Read, 0, y, width, 1,
					lineBuf.data(), width, 1,
					GDT_Byte, 0, 0);

				if (err != CE_None)
				{
					throw std::runtime_error("RasterIO failed at row " + std::to_string(y));
				}

				for (int x = 0; x < width; ++x)
				{
					hist.bins[b][lineBuf[x]]++;
				}
			}

			// 진행률 (밴드당)
			if (b == 0)
			{
				std::cout << "  Histogram: band 1/" << useBands;
			}
			else
			{
				std::cout << ", " << (b + 1) << "/" << useBands;
			}
		}

		std::cout << " done.\n";

		GDALClose(ds);

		return true;
	}

	static LUT buildLUT(const CDF& srcCDF, const CDF& refCDF)
	{
		LUT lut;

		for (int b = 0; b < 3; ++b)
		{
			for (int i = 0; i < 256; ++i)
			{
				float val = srcCDF.values[b][i];

				// 이진 탐색
				int lo = 0;
				int hi = 256;
				int best = 0;

				while (lo <= hi)
				{
					int mid = (lo + hi) / 2;

					if (refCDF.values[b][mid] <= val)
					{
						best = mid;
						lo = mid + 1;
					}
					else
					{
						hi = mid - 1;
					}
				}

				if (best < 255)
				{
					float d1 = std::abs(val - refCDF.values[b][best]);
					float d2 = std::abs(val - refCDF.values[b][best + 1]);

					if (d2 < d1)
					{
						best = best + 1;
					}
				}

				lut.table[b][i] = (unsigned char)best;
			}
		}

		return lut;
	}

	// ─── GDAL로 줄 단위 LUT 적용 + 저장 ───
	static bool applyLUTandSave(const std::string& srcPath, const std::string& outPath, const LUT& lut)
	{
		GDALDataset* srcDs = (GDALDataset*)GDALOpen(srcPath.c_str(), GA_ReadOnly);

		if (!srcDs)
		{
			std::cerr << "  Error: Cannot open " << srcPath << "\n";
			return false;
		}

		int width = srcDs->GetRasterXSize();
		int height = srcDs->GetRasterYSize();
		int nBands = srcDs->GetRasterCount();
		int useBands = std::min(nBands, 3);

		// 출력 파일 생성 (원본과 동일한 크기, 밴드 수, GeoTransform)
		GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");

		if (!driver)
		{
			GDALClose(srcDs);

			std::cerr << "  Error: GTiff driver not available.\n";
			return false;
		}

		// 압축 옵션
		char** createOpts = nullptr;
		createOpts = CSLSetNameValue(createOpts, "COMPRESS", "LZW");
		createOpts = CSLSetNameValue(createOpts, "TILED", "YES");
		createOpts = CSLSetNameValue(createOpts, "BIGTIFF", "YES");

		GDALDataset* outDs = driver->Create(outPath.c_str(), width, height, nBands, GDT_Byte, createOpts);
		CSLDestroy(createOpts);

		if (!outDs)
		{
			GDALClose(srcDs);

			std::cerr << "  Error: Cannot create " << outPath << "\n";
			return false;
		}

		// GeoTransform, Projection 복사
		double gt[6];

		if (srcDs->GetGeoTransform(gt) == CE_None)
		{
			outDs->SetGeoTransform(gt);
		}

		const char* proj = srcDs->GetProjectionRef();

		if (proj && proj[0])
		{
			outDs->SetProjection(proj);
		}

		// 줄 단위 처리
		std::vector<unsigned char> lineBuf(width);
		int progressPct = 0;

		for (int b = 0 ; b < nBands ; ++b)
		{
			GDALRasterBand* srcBand = srcDs->GetRasterBand(b + 1);
			GDALRasterBand* outBand = outDs->GetRasterBand(b + 1);

			for (int y = 0 ; y < height ; ++y)
			{
				CPLErr err = srcBand->RasterIO(GF_Read, 0, y, width, 1,
					lineBuf.data(), width, 1,
					GDT_Byte, 0, 0);

				if (err != CE_None)
				{
					throw std::runtime_error("RasterIO failed at row " + std::to_string(y));
				}

				// RGB 밴드(1~3)에만 LUT 적용, 나머지(Alpha 등)는 그대로
				if (b < useBands)
				{
					for (int x = 0 ; x < width ; ++x)
					{
						lineBuf[x] = lut.table[b][lineBuf[x]];
					}
				}
			
				outBand->RasterIO(GF_Write, 0, y, width, 1,
					lineBuf.data(), width, 1,
					GDT_Byte, 0, 0);
			}

			// 진행률
			int pct = (int)((b + 1) * 100.0 / nBands);

			if (pct != progressPct)
			{
				progressPct = pct;
				std::cout << "  Applying LUT: " << pct << "%\r" << std::flush;
			}
		}

		std::cout << "  Applying LUT: 100%   \n";

		GDALClose(outDs);
		GDALClose(srcDs);

		return true;
	}
};
