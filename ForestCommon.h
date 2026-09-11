#pragma once

#include "TSICommon.h"

//수종 코드
#define SPECIES_CODE			"KOFTR_GROU"
//수종 이름
#define SPECIES_NAME			"KOFTR_NM"
// 수종 기본 색상
#define SPECIES_DEFAULT_COLOR	cv::Scalar(200, 200, 200)

struct Point2D
{
	double x, y;
};

struct ForestFeature
{
	OGRGeometry* geometry = nullptr;					// 원본 포인터(레이어가 소유)

	long long fid = -1;
	int sourceIndex = 0;												// 어느 SHP에서 왔는지 (0~N-1)
	std::map<std::string, std::string> attributes;			// 모든 속성 필드

	double area = 0.0;												// 면적 (m² 또는 좌표 단위²)
	double perimeter = 0.0;										// 둘레
};

struct ForestLayer
{
	std::vector<ForestFeature> features;

	// 메타 정보
	std::string layerName;
	std::string projection;											// WKT 좌표계
	std::string projName;											// 투영 이름 (간략)
	int epsg = -1;														// EPSG 코드
	std::string datum;													// 데이텀
	std::string units;													// 좌표 단위
	OGRwkbGeometryType geomType = wkbUnknown;
	std::string geomTypeName;
	
	// 범위 (Bounding Box)
	OGREnvelope extent = {};									// 통합 Bounding Box
	std::vector<std::string> sourcePaths;											// 로드한 SHP 경로 목록

	// 필드 정보
	struct FieldInfo
	{
		std::string name;
		std::string sourcePath;										// 로드한 SHP 경로 목록
		OGRFieldType type;
		std::string typeName;
		int width = 0;
		int precision = 0;
	};
	std::vector<FieldInfo> fields;

	// 수종 관련 통계
	std::map<std::string, int> speciesCodeCount;			// KOFTR_GROU 값별 개수
	std::map<std::string, int> speciesNameCount;		// KOFTR_NM 값별 개수
	std::map<std::string, double> speciesAreaSum;		// KOFTR_GROU 값별 면적 합
	int totalFeatures = 0;
	int validGeomCount = 0;
	int emptyGeomCount = 0;
	int nullSpeciesCount = 0;
};

class ForestCommon
{
public:

	static void initialize()
	{
		// [중요] GDAL 초기화 전에 PROJ_LIB 경로를 먼저 지정합니다.
		// 슬래시(/)를 사용하거나 역슬래시 두 번(\\)을 사용하세요.
		CPLSetConfigOption("PROJ_LIB", "C:/DevelopmentTools/vcpkg/installed/x64-windows/share/proj");

		GDALAllRegister();
	}
};
