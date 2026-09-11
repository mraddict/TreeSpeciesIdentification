#pragma once

#include "ForestCommon.h"

// 수종그룹코드(KOFTR_GROU) → BGR 색상 매핑
// 산림청 임상도(1:5,000) 코드 정의서 기준
// 색상 테이블에 등록된 코드인지 확인

inline bool hasSpeciesColor(const std::string& code)
{
	static const std::unordered_map<std::string, cv::Scalar> table =
	{
		{"11",{}},{"12",{}},{"13",{}},{"14",{}},{"15",{}},{"16",{}},
		{"17",{}},{"18",{}},{"19",{}},{"20",{}},{"21",{}},{"10",{}},
		{"31",{}},{"32",{}},{"33",{}},{"34",{}},{"35",{}},{"36",{}},
		{"37",{}},{"38",{}},{"39",{}},{"40",{}},{"41",{}},{"42",{}},
		{"43",{}},{"44",{}},{"45",{}},{"46",{}},{"47",{}},{"48",{}},
		{"49",{}},{"30",{}},{"61",{}},{"62",{}},{"63",{}},{"64",{}},
		{"65",{}},{"66",{}},{"67",{}},{"68",{}},{"60",{}},{"77",{}},
		{"78",{}},{"81",{}},{"82",{}},{"83",{}},{"91",{}},{"92",{}},
		{"93",{}},{"94",{}},{"95",{}},{"99",{}},

	};

	return (table.count(code) > 0);
}

inline cv::Scalar getSpeciesColor(const std::string& code)
{
	// 수종그룹코드(KOFTR_GROU) → BGR 색상 매핑
	static const std::unordered_map<std::string, cv::Scalar> table =
	{
		// ── 침엽수 (녹색 계열) ──────────────────────────
		{"C", { 34, 139,  34}},   // 침엽수
		{"11", { 34, 139,  34}},   // 소나무
		{"12", {  0,  80,   0}},   // 잣나무
		{"13", { 19,  69, 139}},   // 낙엽송
		{"14", {  0, 165, 255}},   // 리기다소나무
		{"15", { 60, 179, 113}},   // 곰솔
		{"16", {128, 200,  80}},   // 전나무
		{"17", { 80, 220, 160}},   // 편백나무
		{"18", {100, 240, 100}},   // 삼나무
		{"19", { 50, 150,  50}},   // 가문비나무
		{"20", {  0, 120,  80}},   // 비자나무
		{"21", {180, 230, 180}},   // 은행나무
		{"10", {150, 200, 150}},   // 기타침엽수
		// ── 낙엽활엽수 (황/갈색 계열) ──────────────────
		{"D", { 30, 144, 255}},   // 낙엽수
		{"31", { 30, 144, 255}},   // 상수리나무
		{"32", { 65, 105, 225}},   // 신갈나무
		{"33", { 99, 149, 234}},   // 굴참나무
		{"34", {130, 170, 250}},   // 기타참나무류
		{"35", { 42, 200, 180}},   // 오리나무
		{"36", {  0, 210, 210}},   // 고로쇠나무
		{"37", {200, 230, 255}},   // 자작나무
		{"38", {120, 160, 200}},   // 박달나무
		{"39", { 20, 100, 200}},   // 밤나무
		{"40", { 80, 180, 230}},   // 물푸레나무
		{"41", { 50, 130, 180}},   // 서어나무
		{"42", {160, 200, 230}},   // 때죽나무
		{"43", { 90, 160, 220}},   // 호두나무
		{"44", {110, 190, 240}},   // 백합나무
		{"45", {170, 220, 255}},   // 포플러
		{"46", {120,  80, 200}},   // 벚나무
		{"47", { 80,  50, 160}},   // 느티나무
		{"48", {100,  70, 180}},   // 층층나무
		{"49", {150, 120, 210}},   // 아까시나무
		{"30", {180, 160, 230}},   // 기타활엽수
		// ── 상록활엽수 (진녹/청록 계열) ────────────────
		{"E", {  0,  80,  60}},   // 상록수
		{"61", {  0,  80,  60}},   // 가시나무
		{"62", {  0, 100,  70}},   // 구실잣밤나무
		{"63", { 20, 120,  80}},   // 녹나무
		{"64", { 40, 130,  90}},   // 굴거리나무
		{"65", { 60, 140, 100}},   // 황칠나무
		{"66", { 80, 150, 110}},   // 사스레피나무
		{"67", {100, 160, 120}},   // 후박나무
		{"68", {120, 170, 130}},   // 새덕이
		{"60", {140, 180, 140}},   // 기타상록활엽수
		// ── 혼효·죽림·비산림 (회색/특수 계열) ─────────
		{"N", {128, 128,   0}},   // 비산림
		{"77", {128, 128,   0}},   // 침활혼효림
		{"78", {  0, 200, 100}},   // 죽림
		{"81", {211, 211, 211}},   // 미립목지
		{"82", {169, 169, 169}},   // 제지
		{"83", {144, 238, 144}},   // 관목덤불
		{"91", {192, 192, 192}},   // 주거지
		{"92", {200, 230, 200}},   // 초지
		{"93", {220, 220, 180}},   // 경작지
		{"94", {255, 200,   0}},   // 수체
		{"95", {180, 255, 180}},   // 과수원
		{"99", {230, 230, 230}},   // 비산림
	};

	auto it = table.find(code);
	return (it != table.end()) ? it->second : cv::Scalar(200, 200, 200);

/*
	std::hash<std::string> hasher;
	size_t h = hasher(code);
	cv::Scalar hc = cv::Scalar((h * 97) % 200 + 55, (h * 53) % 200 + 55, (h * 31) % 200 + 55);
*/
}

// Forest polygon fill: lighter/desaturated version of getSpeciesColor
// // Shifts toward white by blending, so tree points (vivid) stand out
inline cv::Scalar getPolygonFillColor(const std::string& code)
{
	cv::Scalar baseColor = getSpeciesColor(code);

	double blend = 0.6;

	return cv::Scalar(baseColor[0] + (255 - baseColor[0]) * blend, baseColor[1] + (255 - baseColor[1]) * blend, baseColor[2] + (255 - baseColor[2]) * blend);
}

// ─── Species code -> English name mapping ───

inline const std::string& getSpeciesName(const std::string& code, bool getKo = false)
{
	struct SpeciesEntry
	{
		const char* code;
		const char* nameKo;
		const char* nameEn;
	};

	static const std::vector<SpeciesEntry> entries =
	{
		// 침엽수
		{"C","침엽수",         "Conifer"},
		{"11","소나무",         "Pine"},
		{"12","잣나무",         "Korean Pine"},
		{"13","낙엽송",         "Larch"},
		{"14","리기다소나무",   "Pitch Pine"},
		{"15","곰솔",           "Black Pine"},
		{"16","전나무",         "Fir"},
		{"17","편백나무",       "Hinoki Cypress"},
		{"18","삼나무",         "Cedar"},
		{"19","가문비나무",     "Spruce"},
		{"20","비자나무",       "Japanese Torreya"},
		{"21","은행나무",       "Ginkgo"},
		{"10","기타침엽수",     "Other Conifers"},
		// 낙엽활엽수
		{"D","낙엽수",     "Deciduous"},
		{"31","상수리나무",     "Sawtooth Oak"},
		{"32","신갈나무",       "Mongolian Oak"},
		{"33","굴참나무",       "Oriental Oak"},
		{"34","기타참나무류",   "Other Oaks"},
		{"35","오리나무",       "Alder"},
		{"36","고로쇠나무",     "Painted Maple"},
		{"37","자작나무",       "Birch"},
		{"38","박달나무",       "Asian White Birch"},
		{"39","밤나무",         "Chestnut"},
		{"40","물푸레나무",     "Ash"},
		{"41","서어나무",       "Hornbeam"},
		{"42","때죽나무",       "Snowbell"},
		{"43","호두나무",       "Walnut"},
		{"44","백합나무",       "Tulip Tree"},
		{"45","포플러",         "Poplar"},
		{"46","벚나무",         "Cherry"},
		{"47","느티나무",       "Zelkova"},
		{"48","층층나무",       "Dogwood"},
		{"49","아까시나무",     "Black Locust"},
		{"30","기타활엽수",     "Other Deciduous"},
		// 상록활엽수
		{"E","상록수",       "Evergreen"},
		{"61","가시나무",       "Evergreen Oak"},
		{"62","구실잣밤나무",   "Castanopsis"},
		{"63","녹나무",         "Camphor Tree"},
		{"64","굴거리나무",     "Daphniphyllum"},
		{"65","황칠나무",       "Dendropanax"},
		{"66","사스레피나무",   "Eurya"},
		{"67","후박나무",       "Machilus"},
		{"68","새덕이",         "Neolitsea"},
		{"60","기타상록활엽수", "Other Evergreen"},
		// 혼효·비산림
		{"N","비산림",     "Non-Forest"},
		{"77","침활혼효림",     "Mixed Forest"},
		{"78","죽림",           "Bamboo"},
		{"81","미립목지",       "Unstocked"},
		{"82","제지",           "Bare Land"},
		{"83","관목덤불",       "Shrub"},
		{"91","주거지",         "Residential"},
		{"92","초지",           "Grassland"},
		{"93","경작지",         "Farmland"},
		{"94","수체",           "Water"},
		{"95","과수원",         "Orchard"},
		{"99","비산림",         "Non-Forest"},
	};

	static std::map<std::string, std::string> tableEn;
	static std::map<std::string, std::string> tableKo;
	static bool initialized = false;

	if (!initialized)
	{
		for (auto& e : entries)
		{
			tableEn[e.code] = e.nameEn;
			tableKo[e.code] = e.nameKo;
		}

		initialized = true;
	}

	static const std::string empty;

	if (getKo)
	{
		auto it = tableKo.find(code);
		
		return (it != tableKo.end()) ? it->second : empty;
	}

	auto it = tableEn.find(code);

	return (it != tableEn.end()) ? it->second : empty;
}

class ForestLoader
{
	/**
	  * path : .shp 파일 경로 (또는 GeoJSON, GPKG 등 어떤 OGR 소스도 OK)
	  * filterSQL : OGR SQL WHERE 절 (선택). 예) "FRTP_CD = '01'"
	  */

public:
	
	static ForestLayer load(const std::string& path, const std::string& filterSQL = "", bool debug = false)
	{
		return loadMultiple({path}, filterSQL, debug);
	}

	// ─── 하나 이상의 SHP 로딩 (핵심 로딩 함수) ───
	static ForestLayer loadMultiple(const std::vector<std::string>& paths, const std::string& filterSQL, bool debug)
	{
		if (paths.empty())
		{
			throw std::runtime_error("로딩할 SHP 파일 목록이 비어있습니다.");
		}

		ForestLayer result;
		result.sourcePaths = paths;

		bool bFirstLayer = true;
		long long fidOffset = 0;			// FID 중복 방지용 오프셋

		for (int si = 0 ; si < (int)paths.size() ; ++si)
		{
			const auto& path = paths[si];

			// ── 데이터소스 열기 ──
			GDALDataset* pDS = (GDALDataset*)GDALOpenEx(path.c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY, nullptr, nullptr, nullptr);

			if (!pDS)
			{
				throw std::runtime_error("SHP 열기 실패: " + path);
			}

			struct DSGuard
			{
				GDALDataset* p;

				~DSGuard()
				{
					GDALClose(p);
				}
			} g{ pDS };

			OGRLayer* pLayer = pDS->GetLayer(0);

			if (!pLayer)
			{
				throw std::runtime_error("레이어를 찾을 수 없습니다: " + path);
			}

			if (!filterSQL.empty())
			{
				pLayer->SetAttributeFilter(filterSQL.c_str());
			}

			// 필드 정보
			OGRFeatureDefn* pDefn = pLayer->GetLayerDefn();
			int iFieldCount = pDefn->GetFieldCount();

			// ── 첫 번째 파일에서 메타 정보 수집 ──
			if (bFirstLayer)
			{
				result.layerName = pLayer->GetName();
				result.totalFeatures = 0; //(int)pLayer->GetFeatureCount(); 밑에서 설정

				if (const OGRSpatialReference* pSRS = pLayer->GetSpatialRef())
				{
					char* wkt = nullptr;
					pSRS->exportToWkt(&wkt);
					result.projection = wkt ? wkt : "";

					CPLFree(wkt);

					parseProjection(pSRS, result);
				}

				// 지오메트리 타입
				result.geomType = pLayer->GetGeomType();
				result.geomTypeName = OGRGeometryTypeToName(result.geomType);

				// 범위
				OGREnvelope env;
				if (pLayer->GetExtent(&env) == OGRERR_NONE)
				{
					mergeExtent(result.extent, env, true);
				}

				for (int i = 0; i < iFieldCount; i++)
				{
					OGRFieldDefn* fd = pDefn->GetFieldDefn(i);

					ForestLayer::FieldInfo fi;

					fi.name = fd->GetNameRef();
					fi.type = fd->GetType();
					fi.typeName = fd->GetFieldTypeName(fi.type);
					fi.width = fd->GetWidth();
					fi.precision = fd->GetPrecision();

					result.fields.push_back(fi);
				}

				bFirstLayer = false;
			}
			else
			{
				// ── 추가 파일: 좌표계 일치 확인 + 범위 갱신 ──
				const OGRSpatialReference* pSRS = pLayer->GetSpatialRef();

				if (pSRS && (result.epsg > 0))
				{
					const char* auth = pSRS->GetAuthorityCode(nullptr);
					int otherEpsg = auth ? std::atoi(auth) : 0;

					if ((otherEpsg > 0) && (otherEpsg != result.epsg))
					{
						std::cerr << "  ⚠ 좌표계 불일치! EPSG:" << result.epsg << " vs EPSG:" << otherEpsg << " (" << path << ")\n";
					}
				}

				// ── Extent 누적 ──
				OGREnvelope env;
				if (pLayer->GetExtent(&env) == OGRERR_NONE)
				{
					mergeExtent(result.extent, env, false);
				}
			}

			// ── 피처 읽기 (모든 파일 공통) ──

			pLayer->ResetReading();

			int dbgLoaded = 0;
			int fileFeatureCount = 0;
			OGRFeature* pFeature;

			while ((pFeature = pLayer->GetNextFeature()) != nullptr)
			{
				++result.totalFeatures;
				++fileFeatureCount;

				ForestFeature ff;

				ff.fid = pFeature->GetFID() + fidOffset;
				ff.sourceIndex = si;

				// 지오메트리 복사
				// 도형 복사 (DS 닫히면 원본 포인터 무효 → clone 필수)
				OGRGeometry* pGeometry = pFeature->GetGeometryRef();

				if (pGeometry && !pGeometry->IsEmpty())
				{
					ff.geometry = pGeometry ? pGeometry->clone() : nullptr;
					++result.validGeomCount;

					// 면적/둘레 (폴리곤인 경우)
					OGRwkbGeometryType flatType = wkbFlatten(pGeometry->getGeometryType());

					if ((flatType == wkbPolygon) || (flatType== wkbMultiPolygon))
					{
						// 면적: OGR C API 사용 (GEOS 내부 lock 우회)
						try
						{
							ff.area = OGR_G_Area(OGRGeometry::ToHandle(ff.geometry));
						}
						catch (...)
						{
							ff.area = 0.0;
						}

						// 둘레: Boundary → Length (별도 블록으로 분리)
						try
						{
							OGRGeometry* boundary = pGeometry->Boundary();

							if (boundary)
							{
								ff.perimeter = OGR_G_Length(OGRGeometry::ToHandle(boundary));
								OGRGeometryFactory::destroyGeometry(boundary);
							}
						}
						catch (...)
						{
							ff.perimeter = 0.0;
						}
					}
					else
					{
						++result.emptyGeomCount;
					}

					for (int i = 0; i < iFieldCount; i++)
					{
						const char* key = pDefn->GetFieldDefn(i)->GetNameRef();
						const char* val = pFeature->GetFieldAsString(i);
						ff.attributes[key] = val ? val : "";
					}

					// 수종 통계 집계
					auto itCode = ff.attributes.find(SPECIES_CODE);

					if (itCode != ff.attributes.end() && !itCode->second.empty())
					{
						result.speciesCodeCount[itCode->second]++;
						result.speciesAreaSum[itCode->second] += ff.area;

						// Replace KOFTR_NM with English name based on code
						const std::string& enName = getSpeciesName(itCode->second);

						//if (!enName.empty())
						{
							ff.attributes[SPECIES_NAME] = enName;
						}
					}
					else
					{
						++result.nullSpeciesCount;
					}

					auto itName = ff.attributes.find(SPECIES_NAME);

					if (itName != ff.attributes.end() && !itName->second.empty())
					{
						result.speciesNameCount[itName->second]++;
					}

					result.features.push_back(std::move(ff));
					OGRFeature::DestroyFeature(pFeature);
				}
				
				// 진행 상황 출력 (파일이 2개 이상일 때만)
				if ((paths.size() > 1) && debug)
				{
					std::cout << "  [" << (si + 1) << "/" << paths.size() << "] " << path << " → " << fileFeatureCount << " 피처\n";
				}
			}
			
			// 다음 파일의 FID가 겹치지 않도록 오프셋 갱신
			fidOffset += pLayer->GetFeatureCount(FALSE);
		}

		if (paths.size() > 1)
		{
			std::cout << "  → 병합 완료: 총 " << result.features.size() << " 피처\n";
		}
		
		return result;
	}

	// ─── 특정 좌표가 어떤 폴리곤에 속하는지 검색 ───
	static bool queryPoint(const ForestLayer& forest, double gx, double gy)
	{
		std::cout << "\n── 좌표 질의: (" << gx << ", " << gy << ") ──\n";

		OGRPoint pt(gx, gy);
		bool found = false;

		for (size_t i = 0; i < forest.features.size(); ++i)
		{
			auto& feat = forest.features[i];

			if (!feat.geometry)
			{
				continue;
			}

			if (!feat.geometry->Contains(&pt))
			{
				continue;
			}

			found = true;
			std::cout << "  → 피처 #" << i << " 에 포함됨\n";

			// 주요 속성 출력
			for (auto& kv : feat.attributes)
			{
				if (!kv.second.empty())
				{
					std::cout << "    " << std::setw(15) << std::left << kv.first << ": " << kv.second << "\n";
				}
			}

			// 경계까지 거리
			OGRGeometry* boundary = feat.geometry->Boundary();

			if (boundary)
			{
				double dist = pt.Distance(boundary);
				std::cout << "    경계까지 거리 : " << std::setprecision(2) << dist << " (단위: 좌표계 단위)\n";
				
				OGRGeometryFactory::destroyGeometry(boundary);
			}

			std::cout << "    면적          : " << std::setprecision(1) << feat.area << "\n" << std::setprecision(6);
		}

		if (!found)
		{
			std::cout << "  → 어떤 폴리곤에도 포함되지 않습니다.\n";
			std::cout << "  임상도 범위: X[" << forest.extent.MinX << " ~ " << forest.extent.MaxX << "] Y[" << forest.extent.MinY << " ~ " << forest.extent.MaxY << "]\n";
		}

		return found;
	}
	
	// ─── 메타데이터 출력 ───
	static void printInfo(const ForestLayer& f)
	{
		std::cout << std::fixed << std::setprecision(6);
		std::cout << "╔══════════════════════════════════════════════════╗\n";
		std::cout << "║       임상도 SHP 메타데이터 정보                ║\n";
		std::cout << "╠══════════════════════════════════════════════════╣\n";

		// 1. 기본 정보
		std::cout << "║ [파일 기본 정보]                                 ║\n";
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "  레이어 이름   : " << f.layerName << "\n";
		std::cout << "  지오메트리 타입: " << f.geomTypeName << "\n";
		std::cout << "  전체 피처 수  : " << f.totalFeatures << "\n";
	
		if (f.validGeomCount > 0)
		{
			std::cout << "  유효 지오메트리: " << f.validGeomCount << "\n";
			std::cout << "  빈 지오메트리  : " << f.emptyGeomCount << "\n";
		}

		// 2. 좌표계 정보
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "║ [좌표계 / 투영 정보]                             ║\n";
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "  EPSG 코드    : " << (f.epsg > 0 ? std::to_string(f.epsg) : "확인 불가") << "\n";
		std::cout << "  투영 이름    : " << (f.projName.empty() ? "N/A" : f.projName) << "\n";
		std::cout << "  데이텀       : " << (f.datum.empty() ? "N/A" : f.datum) << "\n";
		std::cout << "  좌표 단위    : " << (f.units.empty() ? "N/A" : f.units) << "\n";

		// 3. 영역 범위
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "║ [영역 범위 (Bounding Box)]                       ║\n";
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "  Min X (서쪽)  : " << f.extent.MinX << "\n";
		std::cout << "  Max X (동쪽)  : " << f.extent.MaxX << "\n";
		std::cout << "  Min Y (남쪽)  : " << f.extent.MinY << "\n";
		std::cout << "  Max Y (북쪽)  : " << f.extent.MaxY << "\n";

		double extentX = f.extent.MaxX - f.extent.MinX;
		double extentY = f.extent.MaxY - f.extent.MinY;
		std::cout << "  가로 범위     : " << extentX << "\n";
		std::cout << "  세로 범위     : " << extentY << "\n";

		if ((f.units.find("metre") != std::string::npos) || (f.units.find("meter") != std::string::npos) || (f.units.find("Meter") != std::string::npos))
		{
			double areaSqKm = (extentX * extentY) / 1e6;
			std::cout << "  범위 면적     : " << std::setprecision(3) << areaSqKm << " km² (BBox)\n";
			std::cout << std::setprecision(6);
		}

		// 4. 필드 (속성) 정보
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "║ [속성 필드 목록] (" << f.fields.size() << "개)    ║\n";
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "  " << std::left << std::setw(20) << "필드명" << std::setw(12) << "타입" << std::setw(8) << "너비" << "정밀도\n";
		std::cout << "  " << std::string(48, '-') << "\n";
		
		for (auto& fi : f.fields)
		{
			// 수종 관련 필드 강조
			bool isSpecies = (fi.name == SPECIES_CODE || fi.name == SPECIES_NAME || fi.name == "FROR_NM" || fi.name == "DMCLS_CD");
			std::cout << (isSpecies ? "★ " : "  ") << std::setw(20) << fi.name << std::setw(12) << fi.typeName << std::setw(8) << fi.width << fi.precision << "\n";
		}

		// 5. 수종 관련 필드 검증
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "║ [수종 판별 관련 필드 검증]                       ║\n";
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		
		ForestLoader::checkRequiredFields(f);

		// 6. 수종 분포 통계
		if (!f.speciesCodeCount.empty())
		{
			std::cout << "╟──────────────────────────────────────────────────╢\n";
			std::cout << "║ [수종 코드 분포 (KOFTR_GROU)]                   ║\n";
			std::cout << "╟──────────────────────────────────────────────────╢\n";
		
			//printSpeciesStats(f);
		}

		// 7. 적합성 종합 판단
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "║ [수종 판별 적합성 종합 검토]                     ║\n";
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		
		checkSuitability(f);

		std::cout << "╚══════════════════════════════════════════════════╝\n";
	}

	static void release(ForestLayer& layer)
	{
		for (auto& f : layer.features)
		{
			OGRGeometryFactory::destroyGeometry(f.geometry);
			f.geometry = nullptr;
		}

		layer.features.clear();
	}

	// 로드된 ForestLayer 를 단일 SHP 파일로 저장
	static void exportToSHP(const ForestLayer& layer, const std::string& outPath)
	{
		// ── ESRI Shapefile 드라이버 ──
		GDALDriver* pDriver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");

		if (!pDriver)
		{
			std::cerr << "ESRI Shapefile 드라이버를 찾을 수 없습니다.\n";

			return;
		}

		// ── 출력 데이터소스 생성 ──
		GDALDataset* pOutDS = pDriver->Create(outPath.c_str(), 0, 0, 0, GDT_Unknown, nullptr);

		if (!pOutDS)
		{
			throw std::runtime_error("출력 파일 생성 실패: " + outPath);
		}

		struct DSGuard
		{
			GDALDataset* p;

			~DSGuard()
			{
				GDALClose(p);
			}
		} g{pOutDS};

		// ── 좌표계 설정 (첫 번째 소스 PRJ 사용) ──
		OGRSpatialReference* pSRS = nullptr;

		if (!layer.projection.empty())
		{
			pSRS = new OGRSpatialReference();
			pSRS->importFromWkt(layer.projection.c_str());
		}

		// ── geometry 타입 결정 (첫 번째 피처 기준) ──
		OGRwkbGeometryType geomType = wkbPolygon;

		for (auto& f : layer.features)
		{
			if (f.geometry)
			{
				geomType = f.geometry->getGeometryType();

				break;
			}
		}

		// ── 레이어 생성 ──
		OGRLayer* pOutLayer = pOutDS->CreateLayer("forest_merged", pSRS, geomType, nullptr);

		if (!pOutLayer)
		{
			if (pSRS)
			{
				pSRS->Release();

				throw std::runtime_error("레이어 생성 실패");
			}
		}

		if (pSRS)
		{
			pSRS->Release();
		}
	
		// ── 필드 정의 추가 ──
		// 원본 SHP 를 열어 필드 타입 정보를 가져옴
		GDALDataset* pSrcDS = nullptr;

		if (!layer.sourcePaths.empty())
		{
			pSrcDS = (GDALDataset*)GDALOpenEx(layer.sourcePaths[0].c_str(), GDAL_OF_VECTOR | GDAL_OF_READONLY, nullptr, nullptr, nullptr);
		}

		OGRFeatureDefn* pSrcDefn = nullptr;

		if (pSrcDS)
		{
			OGRLayer* pSrcLayer = pSrcDS->GetLayer(0);

			if (pSrcLayer)
			{
				pSrcDefn = pSrcLayer->GetLayerDefn();
			}
		}

		if (pSrcDefn)
		{
			for (int i = 0 ; i < pSrcDefn->GetFieldCount() ; ++i)
			{
				std::cout << "[SRC] Field: " << pSrcDefn->GetFieldDefn(i)->GetNameRef() << std::endl;

				OGRFieldDefn fieldDef(*pSrcDefn->GetFieldDefn(i));

				// 필드명에 ASCII 범위 밖 문자가 있으면 영문명으로 대체
				// (DBF는 ISO-8859-1만 허용 → 한글 필드명 저장 불가)
				const std::string szOriginName = fieldDef.GetNameRef();
				bool bHasNonAscii = false;

				for (unsigned char c : szOriginName)
				{
					if (c > 127)
					{
						bHasNonAscii = true;

						break;
					}
				}

				if (bHasNonAscii)
				{
					// 로딩 디버그 출력에서 확인한 필드 순서 기준으로 매핑
					// 실제 데이터의 한글 필드가 추가되면 여기에 추가
					static const std::map<int, std::string> fallbackNames =
					{
						{10, "UPDATE_YR"},
					};

					auto it = fallbackNames.find(i);
					std::string newName = (it != fallbackNames.end()) ? it->second : "FIELD_" + std::to_string(i);

					OGRFieldDefn newFieldDef(newName.c_str(), fieldDef.GetType());
					pOutLayer->CreateField(&newFieldDef);
				}
				else
				{
					pOutLayer->CreateField(&fieldDef);
				}
			}
		}
		else
		{
			// 원본 못 열면 모든 필드를 문자열로 생성
			for (auto& fd : layer.fields)
			{
				OGRFieldDefn fieldDef(fd.name.c_str(), OFTString);
				fieldDef.SetWidth(254);
				pOutLayer->CreateField(&fieldDef);
			}
		}

		if (pSrcDS)
		{
			GDALClose(pSrcDS);
		}

		// ── 피처 쓰기 ──
		OGRFeatureDefn* pOutDefn = pOutLayer->GetLayerDefn();

		for (int i = 0 ; i < pOutDefn->GetFieldCount() ; ++i)
		{
			std::cout << "[OUT] Field: " << pOutDefn->GetFieldDefn(i)->GetNameRef() << std::endl;
		}

		int written = 0;
		int skipped = 0;

		for (auto& f : layer.features)
		{
			if (!f.geometry)
			{
				++skipped;

				continue;
			}

			OGRFeature* pOutFeature = OGRFeature::CreateFeature(pOutDefn);

			// 속성 설정
			for (int i = 0 ; i < pOutDefn->GetFieldCount() ; ++i)
			{
				const char* pszFName = pOutDefn->GetFieldDefn(i)->GetNameRef();
				auto it = f.attributes.find(pszFName);

				if (it != f.attributes.end())
				{
					pOutFeature->SetField(pszFName, it->second.c_str());
				}
			}

			// geometry 설정
			pOutFeature->SetGeometry(f.geometry);

			if (pOutLayer->CreateFeature(pOutFeature) != OGRERR_NONE)
			{
				++skipped;
			}
			else
			{
				++written;
			}

			OGRFeature::DestroyFeature(pOutFeature);
		}

		std::cout << "SHP 저장 완료: " << outPath << "\n" << "  저장된 피처 : " << written << "\n" << "  건너뜀      : " << skipped << "\n";

/*
		for (auto& f : layer.features)
		{
			OGRGeometryFactory::destroyGeometry(f.geometry);
		}
*/

		//layer.features.clear();
	}


private:

	static void parseProjection(const OGRSpatialReference* pSRS, ForestLayer& fl)
	{
		// EPSG 코드
		const char* auth = pSRS->GetAuthorityCode(nullptr);

		if (auth)
		{
			fl.epsg = std::atoi(auth);
		}

		// 투영 이름
		const char* projName = pSRS->GetAttrValue("PROJCS");

		if (projName)
		{
			fl.projName = projName;
		}
		else
		{
			const char* geogName = pSRS->GetAttrValue("GEOGCS");

			if (geogName)
			{
				fl.projName = geogName;
			}
		}

		// 데이텀
		const char* datumStr = pSRS->GetAttrValue("DATUM");

		if (datumStr)
		{
			fl.datum = datumStr;
		}

		// 좌표 단위
		char* unitName = nullptr;

		if (pSRS->IsProjected())
		{
			pSRS->GetLinearUnits(&unitName);
		}
		else
		{
			pSRS->GetAngularUnits(&unitName);
		}

		if (unitName)
		{
			fl.units = unitName;
		}
	}

	static void mergeExtent(OGREnvelope& combinedEnvelope, const OGREnvelope& part, bool init)
	{
		if (init)
		{
			combinedEnvelope = part;

			return;
		}

		combinedEnvelope.MinX = std::min(combinedEnvelope.MinX, part.MinX);
		combinedEnvelope.MinY = std::min(combinedEnvelope.MinY, part.MinY);
		combinedEnvelope.MaxX = std::max(combinedEnvelope.MaxX, part.MaxX);
		combinedEnvelope.MaxY = std::max(combinedEnvelope.MaxY, part.MaxY);
	}

	// 필수 필드 검증
	static void checkRequiredFields(const ForestLayer& f)
	{
		// 수종 판별에 필요한 핵심 필드 목록
		struct RequiredField
		{
			std::string name;
			std::string description;
			bool critical;						// true=필수, false=권장
		};

		std::vector<RequiredField> required =
		{
			 {"KOFTR_GROU", "수종 그룹 코드 (학습 레이블)",  true},
			 {"KOFTR_NM",   "수종 이름",                     true},
			 {"FROR_NM",    "임상 이름",                     false},
			 {"DMCLS_CD",   "영급 코드",                     false},
			 {"DNST_CD",    "밀도 코드",                     false},
			 {"KOFTR_CD",   "수종 코드 (상세)",              false},
			 {"AGE_CLS",    "영급",                          false},
			 {"FRTP_CD",    "임상 코드",                     false},
		};

		std::set<std::string> fieldNames;

		for (auto& fi : f.fields)
		{
			fieldNames.insert(fi.name);
		}

		for (auto& req : required)
		{
			bool exists = fieldNames.count(req.name) > 0;
			std::string icon = exists ? "✓" : (req.critical ? "✗ [필수]" : "- [선택]");

			std::cout << "  " << icon << " " << std::setw(15) << std::left << req.name << " : " << req.description << (exists ? "" : " → 없음") << "\n";
		}

		// 필수 필드 누락 경고
		bool hasCritical = fieldNames.count(SPECIES_CODE) > 0;

		if (!hasCritical)
		{
			std::cout << "\n  ⚠ 핵심 필드 KOFTR_GROU 가 없습니다!\n"
				<< "    → 임상도에서 수종 레이블을 부여할 수 없습니다.\n"
				<< "    → 필드 이름이 다를 수 있으니 위 목록을 확인하세요.\n";
		}
	}

	// 수종 분포 통계
	static void printSpeciesStats(const ForestLayer& f)
	{
		// 피처 수 기준 정렬 (내림차순)
		std::vector<std::pair<std::string, int>> sorted(f.speciesCodeCount.begin(), f.speciesCodeCount.end());
		std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });

		int totalLabeled = 0;
		for (auto& kv : sorted)
		{
			totalLabeled += kv.second;
		}

		std::cout << "  " << std::left << std::setw(12) << "코드" << std::setw(10) << "피처 수" << std::setw(10) << "비율(%)" << "면적 합\n";
		std::cout << "  " << std::string(48, '-') << "\n";

		int showCount = std::min((int)sorted.size(), 20);

		for (int i = 0; i < showCount; ++i)
		{
			auto& kv = sorted[i];
			double pct = (totalLabeled > 0) ? (100.0 * kv.second / totalLabeled) : 0.0;

			auto areaIt = f.speciesAreaSum.find(kv.first);
			double area = (areaIt != f.speciesAreaSum.end()) ? areaIt->second : 0.0;

			std::cout << "  " << std::setw(12) << kv.first << std::setw(10) << kv.second << std::setw(10) << std::setprecision(1) << pct << std::setprecision(1) << area << "\n";
		}

		std::cout << std::setprecision(6);

		if ((int)sorted.size() > showCount)
		{
			std::cout << "  ... 외 " << (sorted.size() - showCount) << "개 수종\n";
		}

		std::cout << "  ────────\n";
		std::cout << "  전체 레이블 피처: " << totalLabeled << "\n";

		if (f.nullSpeciesCount > 0)
		{
			std::cout << "  수종 미지정 피처 : " << f.nullSpeciesCount << "\n";
		}

		// KOFTR_NM 매핑 (코드 ↔ 이름)
		if (!f.speciesNameCount.empty())
		{
			std::cout << "\n  [수종 이름 (KOFTR_NM) 분포]\n";
			std::vector<std::pair<std::string, int>> nameSorted(f.speciesNameCount.begin(), f.speciesNameCount.end());
			std::sort(nameSorted.begin(), nameSorted.end(), [](auto& a, auto& b) { return a.second > b.second; });

			int nameShow = std::min((int)nameSorted.size(), 15);

			for (int i = 0; i < nameShow; ++i)
			{
				std::cout << "  " << std::setw(20) << nameSorted[i].first << " : " << nameSorted[i].second << "\n";
			}

			if ((int)nameSorted.size() > nameShow)
			{
				std::cout << "  ... 외 " << (nameSorted.size() - nameShow) << "개\n";
			}
		}
	}

	// 적합성 종합 판단
	static void checkSuitability(const ForestLayer& f)
	{
		bool allGood = true;

		// 좌표계 확인
		if (f.epsg == 0)
		{
			std::cout << "  ⚠ 좌표계(EPSG)를 식별할 수 없습니다.\n"
				<< "    → .prj 파일이 없거나 비표준 좌표계일 수 있습니다.\n";

			allGood = false;
		}
		else if (f.epsg == 5179 || f.epsg == 5186 || f.epsg == 5174)
		{
			std::cout << "  ✓ 좌표계 EPSG:" << f.epsg << " → 한국 표준 좌표계입니다.\n";
		}
		else if (f.epsg == 4326)
		{
			std::cout << "  ⚠ WGS84 경위도(EPSG:4326)입니다.\n"
				<< "    → 면적/거리 계산에 투영좌표계 변환이 필요합니다.\n";

			allGood = false;
		}
		else
		{
			std::cout << "  △ EPSG:" << f.epsg << " → 항공사진/LiDAR와 좌표계 일치를 확인하세요.\n";
		}

		// 지오메트리 타입 확인
		if (f.geomType == wkbPolygon || f.geomType == wkbMultiPolygon || f.geomType == wkbPolygon25D || f.geomType == wkbMultiPolygon25D)
		{
			std::cout << "  ✓ 폴리곤 지오메트리 → Contains() 연산 가능\n";
		}
		else
		{
			std::cout << "  ⚠ 지오메트리 타입이 " << f.geomTypeName << " → 폴리곤이 아닙니다.\n";

			allGood = false;
		}

		// 피처 수 확인
		if (f.totalFeatures > 0)
		{
			std::cout << "  ✓ 피처 " << f.totalFeatures << "개 확인\n";
		}
		else
		{
			std::cout << "  ⚠ 피처가 없습니다.\n";

			allGood = false;
		}

		// 수종 필드 확인
		bool hasKoftrGrou = false;

		for (auto& fi : f.fields)
		{
			if (fi.name == "KOFTR_GROU")
			{
				hasKoftrGrou = true;

				break;
			}
		}

		if (hasKoftrGrou)
		{
			std::cout << "  ✓ KOFTR_GROU 필드 있음 → 수종 레이블 부여 가능\n";

			if (!f.speciesCodeCount.empty())
			{
				std::cout << "  ✓ " << f.speciesCodeCount.size() << "개 수종 코드 확인됨\n";
			}
		}
		else
		{
			std::cout << "  ⚠ KOFTR_GROU 필드 없음 → 수종 레이블 부여 불가\n";
			allGood = false;
		}

		// 빈 지오메트리 비율
		if (f.emptyGeomCount > 0 && f.totalFeatures > 0)
		{
			double emptyPct = 100.0 * f.emptyGeomCount / f.totalFeatures;

			if (emptyPct > 10.0)
			{
				std::cout << "  ⚠ 빈 지오메트리 " << std::setprecision(1) << emptyPct << "% → 데이터 품질 확인 필요\n";
				std::cout << std::setprecision(6);

				allGood = false;
			}
		}

		if (allGood)
		{
			std::cout << "\n  ★ 종합: 수종 판별 프로젝트에 적합한 임상도입니다.\n";
		}
		else
		{
			std::cout << "\n  ★ 종합: 위 주의사항을 확인/보완한 후 사용하세요.\n";
		}
	}
};

inline void printSummary(const ForestLayer & layer)
{
	std::cout << "=== 임상도 레이어 요약 ===\n" << "레이어 이름  : " << layer.layerName << "\n" << "소스 파일 수 : " << layer.sourcePaths.size() << "\n";

	for ( int i = 0 ; i < (int)layer.sourcePaths.size() ; ++i)
	{
		std::cout << "  [" << i << "] " << layer.sourcePaths[i] << "\n";
	}

	std::cout << "총 피처 수   : " << layer.features.size() << "\n"
		<< "통합 범위    : ("
		<< layer.extent.MinX << ", " << layer.extent.MinY << ") ~ ("
		<< layer.extent.MaxX << ", " << layer.extent.MaxY << ")\n"
		<< "좌표계(WKT)  : " << layer.projection.substr(0, 80) << "...\n"
		<< "속성 필드    : ";

	for (auto& fd : layer.fields)
	{
		std::cout << fd.name << "  ";
	}

	std::cout << "\n";
}

inline std::vector<const ForestFeature*> filterByField(const ForestLayer& layer, const std::string& field, const std::string& value)
{
	std::vector<const ForestFeature*> out;

	for (auto& f : layer.features)
	{
		auto it = f.attributes.find(field);
	
		if (it != f.attributes.end() && it->second == value)
		{
			out.push_back(&f);
		}
	}

	return out;
}

inline std::vector<const ForestFeature*> filterBySource(const ForestLayer& layer, int si)
{
	std::vector<const ForestFeature*> out;

	for (auto& f : layer.features)
	{
		if (f.sourceIndex == si)
		{
			out.push_back(&f);
		}
	}

	return out;
}
