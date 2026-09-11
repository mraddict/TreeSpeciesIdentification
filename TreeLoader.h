#pragma once

#include "ForestLoader.h"

// ─────────────────────────────────────────────
//  구조체: 개별 나무 데이터
// ─────────────────────────────────────────────
struct TreePoint
{
	long long treeId = -1;			// TreeID
	double x , y;							// 지리 좌표 (임상도와 같은 좌표계로 변환된 값. EPSG:5179)
	double height = 0.0;				// TreeHeight(metre)
	double crownD = 0.0;			// CrownDiameter(metre)
	double crownDSN = 0.0;		// CrownDiameter S-N (m)
	double crownDEW = 0.0;		// CrownDiameter E-W (m)
	double crownArea = 0.0;		// CrownArea (m²)
	double crownVol = 0.0;		// CrownVolume (m³)
	long long oldId = -1;				// OldID
};

// ─────────────────────────────────────────────
//  로딩 결과 + 통계
// ─────────────────────────────────────────────
struct TreeData
{
	std::vector<TreePoint> tps;

	// 범위
	OGREnvelope extent = {};

	// 통계
	int totalRows = 0;
	int loadedCount = 0;
	int skippedCount = 0;

	// 좌표 변환 정보
	int srcEpsg = 0;					// 원본 좌표계 (0이면 불명)
	int dstEpsg = 0;					// 변환 후 좌표계 (0이면 변환 안 함)
	bool transformed = false;

	// CSV 헤더 정보
	std::vector<std::string> headers;
};

// CSV 로드 + EPSG 변환
class TreeLoader
{
public:

	typedef std::map<std::string, int> TreeDataColumnMap;

	// csvPath  : CSV 파일 경로
	// srcEPSG  : CSV 좌표계 (예: 5186)
	// dstEPSG  : 임상도 좌표계 (예: 5179)
	//static std::vector<TreePoint> load(const std::string& csvPath, int srcEPSG, int dstEPSG, double offsetX = 0.0, double offsetY = 0.0)
	static TreeData load(const std::string& csvPath, int srcEPSG = 0, int dstEPSG = 0, double offsetX = 0.0, double offsetY = 0.0)
	{
		std::ifstream ifs(csvPath);

		if (!ifs)
		{
			throw std::runtime_error("Failed to load CSV: " + csvPath);
		}

		TreeData data;
		data.srcEpsg = srcEPSG;
		data.dstEpsg = dstEPSG;

		// 헤더 읽기 (BOM 제거 포함)
		std::string header;
		if (!std::getline(ifs, header))
		{
			throw std::runtime_error("CSV가 비어있습니다: " + csvPath);
		}

		header = stripBOM(header);
		header = trimCR(header);

		data.headers = splitCSV(header);

		// 컬럼 인덱스 매핑
		auto colMap = buildColumnMap(data.headers);

		int colTreeId = getCol(colMap, { "TreeID", "treeID", "treeid", "TREEID" });
		int colX = getCol(colMap, { "TreeLocationX", "X", "x", "LocationX" });
		int colY = getCol(colMap, { "TreeLocationY", "Y", "y", "LocationY" });
		int colHeight = getCol(colMap, { "TreeHeight(metre)", "TreeHeight", "Height", "height" });
		int colCrownD = getCol(colMap, { "CrownDiameter(metre)", "CrownDiameter", "CrownD" });
		int colCrownSN = getCol(colMap, { "CrownDiameter(S-N)(metre)", "CrownDiameter_SN" });
		int colCrownEW = getCol(colMap, { "CrownDiameter(E-W)(metre)", "CrownDiameter_EW" });
		int colArea = getCol(colMap, { "CrownArea(square metre)", "CrownArea" });
		int colVol = getCol(colMap, { "CrownVolume(cubic metre)", "CrownVolume" });
		int colOldId = getCol(colMap, { "OldID", "oldID", "OldId" });

		if ((colX < 0) || (colY < 0))
		{
			throw std::runtime_error("필수 컬럼(X, Y)을 찾을 수 없습니다.");
		}

		// 데이터 읽기

		std::string line;

		while (std::getline(ifs, line))
		{
			++data.totalRows;
			line = trimCR(line);

			if (line.empty())
			{
				continue;
			}

			auto cols = splitCSV(line);

			try
			{
				TreePoint tp;

				tp.treeId = readLong(cols, colTreeId, data.totalRows);
				tp.x = readDouble(cols, colX);
				tp.y = readDouble(cols, colY);
				tp.height = readDouble(cols, colHeight);
				tp.crownD = readDouble(cols, colCrownD);
				tp.crownDSN = readDouble(cols, colCrownSN);
				tp.crownDEW = readDouble(cols, colCrownEW);
				tp.crownArea = readDouble(cols, colArea);
				tp.crownVol = readDouble(cols, colVol);
				tp.oldId = readLongOpt(cols, colOldId);

				// ID가 없으면 행 번호 사용
				if (tp.treeId < 0)
				{
					tp.treeId = data.totalRows;
				}

				data.tps.push_back(tp);
				++data.loadedCount;
			}
			catch (...)
			{
				++data.skippedCount;
			}
		}

		ifs.close();

		for (auto& tp : data.tps)
		{
			tp.x += offsetX;
			tp.y += offsetY;
		}

		if (data.tps.empty())
		{
			throw std::runtime_error("로딩된 나무가 없습니다: " + csvPath);
		}

		// 좌표 변환

		if ((srcEPSG > 0) && (dstEPSG > 0) && (srcEPSG != dstEPSG))
		{
			transformCoordinates(data, srcEPSG, dstEPSG);
		}

		// 통계 계산
		computeStats(data);

		return data;
	}

	// ─── 메타데이터 + 통계 출력 ───
	static void printInfo(const TreeData& d)
	{
		std::cout << std::fixed;
		std::cout << "╔══════════════════════════════════════════════════╗\n";
		std::cout << "║       개별목 CSV 메타데이터 정보                ║\n";
		std::cout << "╠══════════════════════════════════════════════════╣\n";

		// 1. 기본 정보
		std::cout << "║ [파일 기본 정보]                                 ║\n";
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "  전체 행       : " << d.totalRows << "\n";
		std::cout << "  로딩 성공     : " << d.loadedCount << "\n";
	
		if (d.skippedCount > 0)
		{
			std::cout << "  스킵 (파싱실패): " << d.skippedCount << "\n";
		}

		std::cout << "  컬럼 수       : " << d.headers.size() << "\n";
		std::cout << "  컬럼 목록     : ";

		for (size_t i = 0; i < d.headers.size(); ++i)
		{
			if (i > 0)
			{
				std::cout << ", ";
			}

			std::cout << d.headers[i];
		}

		std::cout << "\n";

		// 2. 좌표 정보
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "║ [좌표 범위]                                      ║\n";
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << std::setprecision(3);
		std::cout << "  Min X (서쪽)  : " << d.extent.MinX << "\n";
		std::cout << "  Max X (동쪽)  : " << d.extent.MaxX << "\n";
		std::cout << "  Min Y (남쪽)  : " << d.extent.MinY << "\n";
		std::cout << "  Max Y (북쪽)  : " << d.extent.MaxY << "\n";
		std::cout << "  가로 범위     : " << (d.extent.MaxX - d.extent.MinX) << " m\n";
		std::cout << "  세로 범위     : " << (d.extent.MaxY - d.extent.MinY) << " m\n";

		if (d.transformed)
		{
			std::cout << "  좌표 변환     : EPSG:" << d.srcEpsg << " → EPSG:" << d.dstEpsg << " (변환 완료)\n";
		}
		else if (d.srcEpsg > 0)
		{
			std::cout << "  원본 좌표계   : EPSG:" << d.srcEpsg << "\n";
		}
		else
		{
			std::cout << "  좌표계        : 미지정 (좌표 범위로 추정 필요)\n";
		}

		// 6. 적합성 검토
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		std::cout << "║ [수종 판별 적합성 검토]                          ║\n";
		std::cout << "╟──────────────────────────────────────────────────╢\n";
		
		checkSuitability(d);

		std::cout << "╚══════════════════════════════════════════════════╝\n";
	}

private:

	static std::string stripBOM(const std::string& s)
	{
		// UTF-8 BOM: 0xEF 0xBB 0xBF
		if ((s.size() >= 3) && ((unsigned char)s[0] == 0xFF) && ((unsigned char)s[1] == 0xBB) && ((unsigned char)s[2] == 0xBF))
		{
			return s.substr(3);
		}

		return s;
	}

	static std::string trimCR(const std::string& s)
	{
		if (!s.empty() && (s.back() == '\r'))
		{
			return s.substr(0, s.size() - 1);
		}

		return s;
	}

	static std::string trim(const std::string& s)
	{
		// 앞뒤 공백·따옴표 제거

		size_t start = s.find_first_not_of(" \t\r\n");

		if (start == std::string::npos)
		{
			return "";
		}

		size_t end = s.find_last_not_of(" \t\r\n");

		return s.substr(start, end - start + 1);
	}

	static std::vector<std::string> splitCSV(const std::string& line)
	{
		std::vector<std::string> cols;
		std::string cell;

		std::stringstream ss(line);
		
		while (std::getline(ss, cell, ','))
		{
			cols.push_back(trim(cell));
		}

		return cols;
	}

	static int findCol(const std::vector<std::string>& cols, const std::string& name)
	{
		for (int i = 0 ; i < (int)cols.size() ; ++i)
		{
			if (cols[i] == name)
			{
				return i;
			}
		}

		return -1;
	}

	static TreeDataColumnMap buildColumnMap(const std::vector<std::string>& headers)
	{
		TreeDataColumnMap m;

		for (int i = 0; i < (int)headers.size(); ++i)
		{
			m[headers[i]] = i;
		}

		return m;
	}

	// 여러 이름 후보 중 매칭되는 컬럼 인덱스 반환
	static int getCol(const TreeDataColumnMap& colMap, const std::vector<std::string>& names)
	{
		for (auto& name : names)
		{
			auto it = colMap.find(name);

			if (it != colMap.end())
			{
				return it->second;
			}
		}

		return -1;
	}

	static double readDouble(const std::vector<std::string>& cols, int idx)
	{
		if (idx < 0 || idx >= (int)cols.size() || cols[idx].empty())
		{
			return 0.0;
		}

		try
		{
			return std::stod(cols[idx]);
		}
		catch (...)
		{
			return 0.0;
		}
	}

	static long long readLong(const std::vector<std::string>& cols, int idx, int fallback)
	{
		if (idx < 0 || idx >= (int)cols.size() || cols[idx].empty())
		{
			return fallback;
		}

		try
		{
			return std::stoll(cols[idx]);
		}
		catch (...)
		{
			return fallback;
		}
	}

	static long long readLongOpt(const std::vector<std::string>& cols, int idx)
	{
		if (idx < 0 || idx >= (int)cols.size() || cols[idx].empty())
		{
			return -1;
		}

		try
		{
			return std::stoll(cols[idx]);
		}
		catch (...)
		{
			return -1;
		}
	}

	// ─── 좌표 변환 ───
	static void transformCoordinates(TreeData& data, int srcEPSG, int dstEPSG)
	{
		OGRSpatialReference srcSRS;
		srcSRS.importFromEPSG(srcEPSG);
		srcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

		OGRSpatialReference dstSRS;
		dstSRS.importFromEPSG(dstEPSG);
		dstSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

		OGRCoordinateTransformation* pCT = OGRCreateCoordinateTransformation(&srcSRS, &dstSRS);

		if (!pCT)
		{
			throw std::runtime_error("좌표 변환기 생성 실패: EPSG:" + std::to_string(srcEPSG) + " → EPSG:" + std::to_string(dstEPSG));
		}

		int total = (int)data.tps.size();
		int success = 0;
		int fail = 0;

		// 배치 변환 (1000개씩)
		const int batchSize = 1000;

		for (int i = 0; i < total; i += batchSize)
		{
			int remaining = total - i;
			int n = (remaining < batchSize) ? remaining : batchSize;

			std::vector<double> xs(n);
			std::vector<double> ys(n);
			std::vector<double> zs(n, 0.0);
			std::vector<int> successes(n, 0);

			for (int j = 0; j < n; ++j)
			{
				xs[j] = data.tps[i + j].x;
				ys[j] = data.tps[i + j].y;
			}

			pCT->Transform(n, xs.data(), ys.data(), zs.data(), successes.data());

			for (int j = 0; j < n; ++j)
			{
				if (successes[j])
				{
					data.tps[i + j].x = xs[j];
					data.tps[i + j].y = ys[j];
					
					++success;
				}
				else
				{
					++fail;
				}
			}
		}

		OGRCoordinateTransformation::DestroyCT(pCT);

		data.transformed = true;
		std::cout << "=== 좌표 변환 완료 ===\n"
			<< "  EPSG:" << srcEPSG << " → EPSG:" << dstEPSG << "\n"
			<< "  성공: " << success << " / 실패: " << fail << "\n";
	}

	// ─── 통계 계산 ───
	static void computeStats(TreeData& d)
	{
		auto& trees = d.tps;
		int n = (int)trees.size();

		// 범위
		d.extent.MinX = d.extent.MaxX = trees[0].x;
		d.extent.MinY = d.extent.MaxY = trees[0].y;

		for (auto& t : trees)
		{
			d.extent.MinX = std::min(d.extent.MinX, t.x);
			d.extent.MaxX = std::max(d.extent.MaxX, t.x);
			d.extent.MinY = std::min(d.extent.MinY, t.y);
			d.extent.MaxY = std::max(d.extent.MaxY, t.y);
		}
	}

	// ─── 적합성 검토 ───
	static void checkSuitability(const TreeData& d)
	{
		bool allGood = true;

		// 나무 수
		if (d.loadedCount > 100)
		{
			std::cout << "  ✓ 나무 " << d.loadedCount << "그루 → 충분한 학습 데이터\n";
		}
		else
		{
			std::cout << "  ⚠ 나무 " << d.loadedCount << "그루 → 학습 데이터가 부족할 수 있습니다.\n";
			allGood = false;
		}

		// 좌표계 확인
		if (d.transformed)
		{
			std::cout << "  ✓ 좌표 변환 완료 (EPSG:" << d.srcEpsg << " → EPSG:" << d.dstEpsg << ")\n";
		}
		else if (d.srcEpsg == 0)
		{
			std::cout << "  ⚠ 좌표계 미지정\n"
				<< "    → 좌표 범위: X[" << std::setprecision(0) << d.extent.MinX << "~" << d.extent.MaxX << "] Y[" << d.extent.MinY << "~" << d.extent.MaxY << "]\n"
				<< "    → 임상도/항공사진과 좌표계가 일치하는지 확인하세요.\n";
			
			allGood = false;
		}

		// 스킵 비율
		if ((d.totalRows > 0) && (d.skippedCount > 0))
		{
			double skipPct = 100.0 * d.skippedCount / d.totalRows;

			if (skipPct > 5.0)
			{
				std::cout << "  ⚠ 파싱 실패 " << std::setprecision(1) << skipPct << "% → CSV 데이터 품질 확인 필요\n";
				allGood = false;
			}
		}

		if (allGood)
		{
			std::cout << "\n  ★ 종합: 수종 판별 프로젝트에 적합한 개별목 데이터입니다.\n";
		}
		else
		{
			std::cout << "\n  ★ 종합: 위 주의사항을 확인/보완한 후 사용하세요.\n";
		}
	}
};