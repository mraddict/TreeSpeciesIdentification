#pragma once

#include "ForestCommon.h"

struct AerialPhoto
{
	cv::Mat image;

	// 지리 참조 정보 (GeoTransform 6개 파라미터)
	double originX = 0;					// 좌상단 X 좌표
	double originY = 0;					// 좌상단 Y 좌표
	double pixelW = 0;					// X 방향 픽셀 크기 (m 또는 도)
	double pixelH = 0;					// Y 방향 픽셀 크기 (보통 음수)
	double rotationX = 0;				// X 방향 회전 (보통 0)
	double rotationY = 0;				// Y 방향 회전 (보통 0)
	// 계산된 범위 (Bounding Box)
	double geoMinX = 0;
	double geoMaxX = 0;
	double geoMinY = 0;
	double geoMaxY = 0;

	// 메타 정보
	double resolution = 0;				// m/픽셀 (gt[1])
	std::string prj;
	int width = 0;							// 픽셀 너비 (m/px)
	int height = 0;							// 픽셀 높이
	std::string projection;				// WKT 투영 정보
	std::string projName;				// 투영 이름 (간략)
	int epsg = -1;							// EPSG 코드
	std::string datum;						// 데이텀 이름
	std::string units;						// 좌표 단위 (metre, degree 등)
	std::string driverName;			// GDAL 드라이버 이름
	int bands = 0;							// 밴드 수
	std::vector<GDALDataType> bandTypes;			// 밴드별 데이터 타입
	std::vector<double> bandNoData;						// 밴드별 NoData 값
	std::vector<bool> bandHasNoData;
};

class AerialLoader
{
public:

	// ─── GeoTIFF loading (image + metadata) ───
	static AerialPhoto load(const std::string& path)
	{
		// Enable GDAL error reporting
		CPLSetErrorHandler(CPLQuietErrorHandler);

		GDALDataset* pDS = (GDALDataset*)GDALOpen(path.c_str(), GA_ReadOnly);

		if (!pDS)
		{
			std::string errMsg = CPLGetLastErrorMsg();
			throw std::runtime_error("Failed to open GeoTIFF: " + path + (errMsg.empty() ? "" : " (" + errMsg + ")"));
		}

		struct DSGuard
		{
			GDALDataset* p;

			~DSGuard()
			{
				GDALClose(p);
			}
		} g{ pDS };
		
		AerialPhoto photo;

		// 기본 정보
		photo.width = pDS->GetRasterXSize();
		photo.height = pDS->GetRasterYSize();
		photo.bands = pDS->GetRasterCount();
		photo.driverName = pDS->GetDriver()->GetDescription();

		// GeoTransform → 지리 범위
		double gt[6];

		if (pDS->GetGeoTransform(gt) != CE_None)
		{
			throw std::runtime_error("Can't find GeoTransform: 좌표 정보가 없는 이미지");
		}

		photo.originX = gt[0];
		photo.originY = gt[3];
		photo.pixelW = gt[1];
		photo.pixelH = gt[5];
		photo.rotationX = gt[2];
		photo.rotationY = gt[4];

		photo.geoMinX = gt[0];
		photo.geoMaxY = gt[3];
		photo.geoMaxX = gt[0] + photo.width * gt[1] + photo.height * gt[2];
		photo.geoMinY = gt[3] + photo.width * gt[4] + photo.height * gt[5];

		// pixelH가 음수일 때 MinY/MaxY 정렬
		if (photo.geoMinY > photo.geoMaxY)
		{
			std::swap(photo.geoMinY, photo.geoMaxY);
		}

		if (photo.geoMinX > photo.geoMaxX)
		{
			std::swap(photo.geoMinX, photo.geoMaxX);
		}

		// 좌표계 정보 파싱
		photo.projection = pDS->GetProjectionRef() ? pDS->GetProjectionRef() : "";
		parseProjection(photo);

/*
		std::cout << "gt[0]=" << gt[0] << " gt[1]=" << gt[1] << "\n";
		std::cout << "gt[2]=" << gt[2] << " gt[4]=" << gt[4] << "\n";
		std::cout << "gt[3]=" << gt[3] << " gt[5]=" << gt[5] << "\n";
		std::cout << "W=" << photo.width << " H=" << photo.height << "\n";
		std::cout << "gt[0]+gt[1]*W=" << gt[0] + gt[1] * photo.width << "\n";
		std::cout << "gt[0]+gt[1]*W+gy[2]*H=" << gt[0] + gt[1] * photo.width + gt[2] * photo.height << "\n";
		std::cout << "gt[3]+gt[5]*H=" << gt[3] + gt[5] * photo.height << "\n";
		std::cout << "gt[3]+gt[4]*W+gt[5]*H=" << gt[3] + gt[4] * photo.width + gt[5] * photo.height << "\n";
*/

		// 래스터 밴드 정보
		for (int b = 0; b < photo.bands; ++b)
		{
			GDALRasterBand* band = pDS->GetRasterBand(b + 1);
			photo.bandTypes.push_back(band->GetRasterDataType());

			int hasNoData = 0;
			double nd = band->GetNoDataValue(&hasNoData);

			photo.bandHasNoData.push_back(hasNoData != 0);
			photo.bandNoData.push_back(hasNoData ? nd : 0.0);
		}

		// 이미지 데이터를 OpenCV Mat으로 변환
		photo.image = readImageData(pDS, photo);

		return photo;
	}

	// ─── 좌표 변환 유틸리티: 픽셀 ↔ 지리 좌표 ───
	static void pixelToGeo(const AerialPhoto& p, int px, int py, double& gx, double& gy)
	{
		gx = p.originX + px * p.pixelW + py * p.rotationX;
		gy = p.originY + px * p.rotationY + py * p.pixelH;
	}

	static void geoToPixel(const AerialPhoto& p, double gx, double gy, int& px, int& py)
	{
		// 회전이 없는 경우 (일반적)
		if ((std::abs(p.rotationX) < 1e-10) && (std::abs(p.rotationY) < 1e-10))
		{
			px = (int)std::round((gx - p.originX) / p.pixelW);
			py = (int)std::round((gy - p.originY) / p.pixelH);
		}
		else
		{
			// 회전이 있는 경우: 역행렬 계산
			double det = p.pixelW * p.pixelH - p.rotationX * p.rotationY;

			if (std::abs(det) < 1e-20)
			{
				throw std::runtime_error("GeoTransform 역변환 불가 (det≈0)");
			}

			double dx = gx - p.originX;
			double dy = gy - p.originY;

			px = (int)std::round((p.pixelH * dx - p.rotationX * dy) / det);
			py = (int)std::round((p.pixelW * dy - p.rotationY * dx) / det);
		}
	}

	// ─── 메타데이터 출력 ───
	static void printInfo(const AerialPhoto& p)
	{
		std::cout << std::fixed << std::setprecision(6);
		std::cout << "╔══════════════════════════════════════════════╗\n";
		std::cout << "║     항공사진 GeoTIFF 메타데이터 정보         ║\n";
		std::cout << "╠══════════════════════════════════════════════╣\n";

		// 1. 기본 정보
		std::cout << "║ [파일 기본 정보]                             ║\n";
		std::cout << "╟──────────────────────────────────────────────╢\n";
		std::cout << "  드라이버     : " << p.driverName << "\n";
		std::cout << "  이미지 크기  : " << p.width << " x " << p.height << " pixels\n";
		std::cout << "  밴드 수      : " << p.bands << "\n";

		if (!p.image.empty())
		{
			double sizeMB = (double)(p.image.total() * p.image.elemSize()) / (1024.0 * 1024.0);
			std::cout << "  메모리 크기  : " << std::setprecision(1) << sizeMB << " MB\n";
		}

		std::cout << std::setprecision(6);

		// 2. 밴드 정보
		std::cout << "╟──────────────────────────────────────────────╢\n";
		std::cout << "║ [밴드 정보]                                  ║\n";
		std::cout << "╟──────────────────────────────────────────────╢\n";
		for (int b = 0; b < p.bands; ++b)
		{
			std::cout << "  밴드 " << (b + 1) << "  타입: " << GDALGetDataTypeName(p.bandTypes[b]);

			if (p.bandHasNoData[b])
			{
				std::cout << "  NoData: " << p.bandNoData[b];
			}
			else
			{
				std::cout << "  NoData: 없음";
			}

			std::cout << "\n";
		}

		// 3. 좌표계 정보
		std::cout << "╟──────────────────────────────────────────────╢\n";
		std::cout << "║ [좌표계 / 투영 정보]                         ║\n";
		std::cout << "╟──────────────────────────────────────────────╢\n";
		std::cout << "  EPSG 코드    : " << (p.epsg > 0 ? std::to_string(p.epsg) : "확인 불가") << "\n";
		std::cout << "  투영 이름    : " << (p.projName.empty() ? "N/A" : p.projName) << "\n";
		std::cout << "  데이텀       : " << (p.datum.empty() ? "N/A" : p.datum) << "\n";
		std::cout << "  좌표 단위    : " << (p.units.empty() ? "N/A" : p.units) << "\n";

		// 4. GeoTransform (지리 참조)
		std::cout << "╟──────────────────────────────────────────────╢\n";
		std::cout << "║ [GeoTransform 파라미터]                      ║\n";
		std::cout << "╟──────────────────────────────────────────────╢\n";
		std::cout << "  원점 X (좌상단) : " << p.originX << "\n";
		std::cout << "  원점 Y (좌상단) : " << p.originY << "\n";
		std::cout << "  픽셀 크기 X     : " << p.pixelW << "\n";
		std::cout << "  픽셀 크기 Y     : " << p.pixelH << "\n";
		std::cout << "  회전 X          : " << p.rotationX << "\n";
		std::cout << "  회전 Y          : " << p.rotationY << "\n";

		// 5. 범위 (Bounding Box)
		std::cout << "╟──────────────────────────────────────────────╢\n";
		std::cout << "║ [영역 범위 (Bounding Box)]                   ║\n";
		std::cout << "╟──────────────────────────────────────────────╢\n";
		std::cout << "  Min X (서쪽)    : " << p.geoMinX << "\n";
		std::cout << "  Max X (동쪽)    : " << p.geoMaxX << "\n";
		std::cout << "  Min Y (남쪽)    : " << p.geoMinY << "\n";
		std::cout << "  Max Y (북쪽)    : " << p.geoMaxY << "\n";

		double extentX = p.geoMaxX - p.geoMinX;
		double extentY = p.geoMaxY - p.geoMinY;
		std::cout << "  가로 범위       : " << extentX << "\n";
		std::cout << "  세로 범위       : " << extentY << "\n";

		// 미터 단위인 경우 면적 표시
		if ((p.units.find("metre") != std::string::npos) || (p.units.find("meter") != std::string::npos) || (p.units.find("Meter") != std::string::npos))
		{
			double areaSqKm = (extentX * extentY) / 1e6;
			std::cout << "  영역 면적       : " << std::setprecision(3) << areaSqKm << " km²\n";
		}

		// 6. 해상도 (GSD 추정)
		std::cout << "╟──────────────────────────────────────────────╢\n";
		std::cout << "║ [공간 해상도]                                ║\n";
		std::cout << "╟──────────────────────────────────────────────╢\n";

		double absPixW = std::abs(p.pixelW);
		double absPixH = std::abs(p.pixelH);
		std::cout << "  GSD (X방향)     : " << absPixW << "\n";
		std::cout << "  GSD (Y방향)     : " << absPixH << "\n";

		if ((p.units.find("metre") != std::string::npos) || (p.units.find("meter") != std::string::npos))
		{
			std::cout << "  → 약 " << std::setprecision(1) << (absPixW * 100.0) << " cm/pixel\n";
		}
		else if (p.units.find("degree") != std::string::npos)
		{
			// 도 단위인 경우 대략적 m 변환 (중위도 기준)
			double midLat = (p.geoMinY + p.geoMaxY) / 2.0;
			double mPerDegX = 111320.0 * std::cos(midLat * M_PI / 180.0);
			double mPerDegY = 110540.0;
			std::cout << "  → 약 " << std::setprecision(2) << (absPixW * mPerDegX) << " m/pixel (X, 위도 " << std::setprecision(1) << midLat << "° 기준)\n";
			std::cout << "  → 약 " << std::setprecision(2) << (absPixH * mPerDegY) << " m/pixel (Y)\n";
		}

		// 7. 적합성 판단
		std::cout << "╟──────────────────────────────────────────────╢\n";
		std::cout << "║ [수종 판별 적합성 검토]                      ║\n";
		std::cout << "╟──────────────────────────────────────────────╢\n";
		
		checkSuitability(p);

		std::cout << "╚══════════════════════════════════════════════╝\n";
	}

private:

	static void parseProjection(AerialPhoto& photo)
	{
		if (photo.projection.empty())
		{
			return;
		}

		// 좌표계
		OGRSpatialReference srs;

		if (srs.importFromWkt(photo.projection.c_str()) != OGRERR_NONE)
		{
			return;
		}

		// EPSG 코드
		const char* auth = srs.GetAuthorityCode(nullptr);

		if (auth)
		{
			photo.epsg = std::atoi(auth);
		}

		// 투영 이름
		const char* projName = srs.GetAuthorityCode("PROJCS");

		if (projName)
		{
			photo.projName = projName;
		}
		else
		{
			const char* geogName = srs.GetAuthorityCode("GEOGCS");

			if (geogName)
			{
				photo.projName = geogName;
			}
		}

		// 데이텀
		const char* datumStr = srs.GetAttrValue("DATUM");

		if (datumStr)
		{
			photo.datum = datumStr;
		}

		// 좌표 단위
		char* unitName = nullptr;

		if (srs.IsProjected())
		{
			srs.GetLinearUnits(&unitName);
		}
		else
		{
			srs.GetAngularUnits(&unitName);
		}

		if (unitName)
		{
			photo.units = unitName;
		}
	}

	// 이미지 데이터 읽기 → OpenCV BGR Mat
	static cv::Mat readImageData(GDALDataset* pDS, const AerialPhoto& photo)
	{
		int W = photo.width;
		int H = photo.height;
		int bandCount = photo.bands;

		// Memory estimate
		double memMB = (double)W * H * 3 / (1024.0 * 1024.0);
		std::cout << "  Loading image: " << W << "x" << H << " (" << std::fixed << std::setprecision(0) << memMB << " MB)...\n";
		std::cout << std::setprecision(6);

		// Sanity check: refuse images that would need > 4GB
		if (memMB > 4096.0)
		{
			std::cerr << "  WARNING: Image too large (" << memMB << " MB). Consider downsampling.\n";
		}

		// Allocate BGR image
		cv::Mat bgr;

		try
		{
			bgr = cv::Mat(H, W, CV_8UC3);
		}
		catch (const cv::Exception& e)
		{
			throw std::runtime_error("Failed to allocate image (" + std::to_string((int)memMB) + " MB): " + e.what());
		}
		
		// Determine which bands to read (max 3: R, G, B)
		int bandsToRead = std::min(bandCount, 3);
		// Band indices for GDAL (1-based): R=1, G=2, B=3
		int bandMap[3] = { 3, 2, 1 };		// BGR order for OpenCV (band3->B, band2->G, band1->R)
														// Actually: GDAL band1=R -> OpenCV channel2(R)
														//           GDAL band2=G -> OpenCV channel1(G)
														//           GDAL band3=B -> OpenCV channel0(B)
														// 밴드가 3개 이상이면 BGR로, 아니면 그레이스케일로
		// So we read bands [1,2,3] into pixel-interleaved BGR using bandMap

		if (bandsToRead >= 3)
		{
			// Read all 3 bands at once using RasterIO with pixel interleave
			// This reads R,G,B bands directly into BGR layout
			// bandMap: read band3(B) first, band2(G) second, band1(R) third
			int gdalBands[3] = { 3, 2, 1 };  // B, G, R order

			// Row-by-row to reduce peak memory
			int rowBytes = W * 3;

			for (int y = 0 ; y < H ; ++y)
			{
				uint8_t* rowPtr = bgr.ptr<uint8_t>(y);
				CPLErr err = pDS->RasterIO(
					GF_Read, 
					0, y, W, 1,				// source: full width, 1 row at y
					rowPtr, W, 1,		// dest: same size
					GDT_Byte,
					3, gdalBands,		// 3 bands in B,G,R order
					3, rowBytes, 1,		// pixel/line/band spacing
					nullptr);

				if (err != CE_None)
				{
					throw std::runtime_error("RasterIO failed at row " + std::to_string(y));
				}

				// Progress (every 10%)
				if ((y > 0) && ((y % (H / 10)) == 0))
				{
					std::cout << "  " << (100 * y / H) << "%\r" << std::flush;
				};
			}

			std::cout << "  100%   \n";
		}
		else if (bandCount == 1)
		{
			cv::Mat gray(H, W, CV_8UC1);

			GDALRasterBand* pBand = pDS->GetRasterBand(1);

			for (int y = 0 ; y < H ; ++y)
			{
				CPLErr err = pBand->RasterIO(
					GF_Read, 
					0, y, W, 1,
					gray.ptr<uint8_t>(y), W, 1,
					GDT_Byte, 
					0, 0);

				if (err != CE_None)
				{
					throw std::runtime_error("RasterIO failed at row " + std::to_string(y));
				}
			}

			cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
		}
		else
		{
			// 4밴드 (RGBN 등) → RGB만 사용

			cv::Mat bgr(H, W, CV_8UC3);
			std::vector<uint8_t> buf(W * H);

			for (int bi = 0; bi < 3; ++bi)
			{
				GDALRasterBand* band = pDS->GetRasterBand(bi + 1);
				band->RasterIO(GF_Read, 0, 0, W, H, buf.data(), W, H, GDT_Byte, 0, 0);

				int cvCh = (bi == 0) ? 2 : (bi == 1) ? 1 : 0;

				for (int y = 0; y < H; ++y)
				{
					for (int x = 0; x < W; ++x)
					{
						bgr.at<cv::Vec3b>(y, x)[cvCh] = buf[y * W + x];
					}
				}
			}

			std::cout << " >" << bandCount << "밴드 중 RGB(1~3)만 로딩\n";
		}

		if (bandCount > 3)
		{
			std::cout << "  Note: " << bandCount << " bands, using RGB (bands 1-3)\n";
		}

		return bgr;
	}

	// 수종 판별 적합성 검토
	static void checkSuitability(const AerialPhoto& p)
	{
		bool allGood = true;

		// 좌표계 확인
		if (p.epsg == 0)
		{
			std::cout << "  > 좌표계(EPSG)를 식별할 수 없습니다.\n"
				<< "    > LiDAR/임상도와 좌표계 일치 여부를 수동 확인하세요.\n";

			allGood = false;
		}
		else if ((p.epsg == 5179) || (p.epsg == 5186) || (p.epsg == 5174))
		{
			std::cout << "  > 좌표계 EPSG:" << p.epsg << " > 한국 공간정보 표준 좌표계입니다.\n";
		}
		else if (p.epsg == 4326)
		{
			std::cout << "  > WGS84 경위도(EPSG:4326)입니다.\n"
				<< "    > 한국 투영좌표계(5179/5186)로 변환이 필요할 수 있습니다.\n";
			
			allGood = false;
		}
		else
		{
			std::cout << "  > EPSG:" << p.epsg << " > LiDAR/임상도와 좌표계 일치 여부를 확인하세요.\n";
		}

		// 밴드 수 확인
		if (p.bands >= 3)
		{
			std::cout << "  > RGB 밴드 확보 > 색상 특징 추출 가능\n";
		}
		else
		{
			std::cout << "  > 밴드 수 " << p.bands << "개 > RGB 색상 특징 추출이 제한됩니다.\n";

			allGood = false;
		}

		if (p.bands >= 4)
		{
			std::cout << "  > 4밴드 이상 > NIR(근적외선) 활용 가능성 있음\n";
		}

		// 해상도 확인
		double gsd = std::abs(p.pixelW);
		
		if (p.units.find("degree") != std::string::npos)
		{
			double midLat = (p.geoMinY + p.geoMaxY) / 2.0;
			gsd = gsd * 111320.0 * std::cos(midLat * M_PI / 180.0);
		}
		
		if (gsd <= 0.25)
		{
			std::cout << "  > 고해상도 (<=25cm) > 개별 수관 식별에 우수\n";
		}
		else if (gsd <= 0.5)
		{
			std::cout << "  > 해상도 적합 (<=50cm) > 수관 식별 가능\n";
		}
		else if (gsd <= 1.0)
		{
			std::cout << "  > 해상도 보통 (<=1m)  큰 수관만 식별 가능\n";
		}
		else
		{
			std::cout << "  > 해상도 부족 (>" << gsd << "m) > 개별 나무 수관 식별 어려움\n";

			allGood = false;
		}

		// 회전 확인
		if ((std::abs(p.rotationX) > 1e-6) || (std::abs(p.rotationY) > 1e-6))
		{
			std::cout << "  > 이미지에 회전이 적용되어 있습니다.\n"
				<< "    > 좌표 변환 시 회전 파라미터를 반영해야 합니다.\n";
		}

		if (allGood)
		{
			std::cout << "\n  > 종합: 수종 판별 프로젝트에 적합한 항공사진입니다.\n";
		}
		else
		{
			std::cout << "\n  > 종합: 위 주의사항을 확인/보완한 후 사용하세요.\n";
		}
	}
};
