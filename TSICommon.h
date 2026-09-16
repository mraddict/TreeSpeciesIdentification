#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <numeric>
#include <random>

#include <direct.h>   // _mkdir (Windows)
#include <sys/stat.h> // stat

#include <ogrsf_frmts.h>
#include <opencv2/opencv.hpp>

// TreeSpeciesIdentification Common
class TSICommon
{
public:

	static void initialize()
	{
		
	}

	// LiDAR 범위(EPSG:5179)를 WGS84(위경도)로 변환 후 KML 저장
	static void exportLidarRangeToKML(double minX, double minY, double maxX, double maxY, int srcEPSG, const std::string& outPath)
	{
		// ── EPSG:srcEPSG → WGS84(4326) 변환기 ──
		OGRSpatialReference src;
		OGRSpatialReference dst;

		src.importFromEPSG(srcEPSG);
		src.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

		dst.importFromEPSG(4326);
		dst.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

		OGRCoordinateTransformation* ct = OGRCreateCoordinateTransformation(&src , &dst);

		if (!ct)
		{
			throw std::runtime_error("좌표 변환기 생성 실패");
		}

		// 4 코너 변환
		double xs[4] = {minX, maxX, minX, maxX};
		double ys[4] = {minY, minY, maxY, maxY};

		ct->Transform(4, xs, ys);

		OGRCoordinateTransformation::DestroyCT(ct);

		// 변환 후 범위
		double lonMin = *std::min_element(xs, xs + 4);
		double lonMax = *std::max_element(xs, xs + 4);
		double latMin = *std::min_element(ys, ys + 4);
		double latMax = *std::max_element(ys, ys + 4);

		std::cout << "WGS84 변환 결과\n"
			<< "  위도 범위: " << latMin << " ~ " << latMax << "\n"
			<< "  경도 범위: " << lonMin << " ~ " << lonMax << "\n";

		// ── KML 생성 ──
		std::ofstream f(outPath);
		if (!f) throw std::runtime_error("KML 파일 생성 실패: " + outPath);

		f << R"(<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>LiDAR 범위</name>

  <!-- 범위 박스 -->
  <Placemark>
    <name>LiDAR BBox</name>
    <Style>
      <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>
      <PolyStyle><color>330000ff</color></PolyStyle>
    </Style>
    <Polygon>
      <outerBoundaryIs><LinearRing><coordinates>
)";
		// 반시계 방향으로 꼭짓점 (KML 표준)
		f << lonMin << "," << latMin << ",0\n"
			<< lonMax << "," << latMin << ",0\n"
			<< lonMax << "," << latMax << ",0\n"
			<< lonMin << "," << latMax << ",0\n"
			<< lonMin << "," << latMin << ",0\n";

		f << R"(      </coordinates></LinearRing></outerBoundaryIs>
    </Polygon>
  </Placemark>

  <!-- 중심점 -->
  <Placemark>
    <name>LiDAR 중심</name>
    <Point>
      <coordinates>)";
		f << (lonMin + lonMax) / 2.0 << ","
			<< (latMin + latMax) / 2.0 << ",0";
		f << R"(</coordinates>
    </Point>
  </Placemark>

</Document>
</kml>
)";

		std::cout << "KML 저장 완료: " << outPath << "\n"
			<< "Google Earth Pro 에서 열어 해당 영역 확인 후\n"
			<< "파일 → 저장 → 이미지 저장 으로 항공사진 추출\n"; 
	}

	static void findMedian(cv::Mat& img, float& median, float& p25, float& p75)
	{
		median = 0.0f;
		p25 = 0.0f;
		p75 = 0.0f;

		std::vector<float> vals;
		vals.reserve(img.rows * img.cols);

		for (int r = 0; r < img.rows; ++r)
		{
			for (int c = 0; c < img.cols; ++c)
			{
				vals.push_back(img.at<uchar>(r, c));
			}
		}

		std::sort(vals.begin(), vals.end());

		median = vals[(vals.size() / 2)];
		p25 = vals[(vals.size() / 4)];
		p75 = vals[(vals.size() * 3 / 4)];
	}

	// ─── Directory utility (C++14 compatible, Windows) ───

	// Check if path exists
	static bool exists(const std::string& path)
	{
		struct _stat info;

		return (_stat(path.c_str(), &info) == 0);
	}

	// Create directory recursively (like mkdir -p)
	static void mkdirs(const std::string& path)
	{
		if (path.empty() || exists(path))
		{
			return;
		}

		// Find parent
		size_t pos = path.find_last_of("/\\");

		if ((pos != std::string::npos) && (pos > 0))
		{
			mkdirs(path.substr(0, pos));
		}

		_mkdir(path.c_str());
	}

	// Join path components
	static std::string join(const std::string& a, const std::string& b)
	{
		if (a.empty())
		{
			return b;
		}

		char last = a.back();

		if (last == '/' || last == '\\')
		{
			return a + b;
		}

		return a + "/" + b;
	}

	static std::string join(const std::string& a, const std::string& b, const std::string& c)
	{
		return join(join(a, b), c);
	}

	static std::string trim(const std::string& s)
	{
		size_t start = s.find_first_not_of(" \t\r\n");

		if (start == std::string::npos)
		{
			return "";
		}

		size_t end = s.find_last_not_of(" \t\r\n");

		return s.substr(start, end - start + 1);
	}

	static size_t findMatchingBracket(const std::string& s, size_t start, char open, char close)
	{
		if ((start == std::string::npos) || (s[start] != open))
		{
			return std::string::npos;
		}

		int depth = 1;
		bool inString = false;

		for (size_t i = start + 1 ; i < s.size() ; ++i)
		{
			if ((s[i] == '"') && (i == 0 || s[i - 1] != '\\'))
			{
				inString = !inString;
			}

			if (inString)
			{
				continue;
			}

			if (s[i] == open)
			{
				++depth;
			}

			if (s[i] == close)
			{
				--depth;
			}

			if (depth == 0)
			{
				return i;
			}
		}

		return std::string::npos;
	}

	// json

	// Extract a string value: "key": "value"
	static std::string extractString(const std::string& json, const std::string& key)
	{
		std::string search = "\"" + key + "\"";
		size_t pos = json.find(search);

		if (pos == std::string::npos)
		{
			return "";
		}

		// Find the colon after the key
		size_t colon = json.find(':', pos + search.size());

		if (colon == std::string::npos)
		{
			return "";
		}

		// Find the opening quote of the value
		size_t qStart = json.find('"', colon + 1);

		if (qStart == std::string::npos)
		{
			return "";
		}

		size_t qEnd = json.find('"', qStart + 1);

		if (qEnd == std::string::npos)
		{
			return "";
		}

		return json.substr(qStart + 1, qEnd - qStart - 1);
	}

	// Extract a number value: "key": 123
	static double extractNumber(const std::string& json, const std::string& key, double defaultVal = 0)
	{
		std::string search = "\"" + key + "\"";
		size_t pos = json.find(search);

		if (pos == std::string::npos)
		{
			return defaultVal;
		}

		size_t colon = json.find(':', pos + search.size());

		if (colon == std::string::npos)
		{
			return defaultVal;
		}

		// Skip whitespace after colon
		size_t numStart = json.find_first_not_of(" \t\r\n", colon + 1);

		if (numStart == std::string::npos)
		{
			return defaultVal;
		}

		try
		{
			return std::stod(json.substr(numStart));
		}
		catch (...)
		{
			return defaultVal;
		}
	}

	// Extract a bool value: "key": true
	static bool extractBool(const std::string& json, const std::string& key, bool defaultVal = true)
	{
		std::string search = "\"" + key + "\"";
		size_t pos = json.find(search);
	
		if (pos == std::string::npos)
		{
			return defaultVal;
		}

		size_t colon = json.find(':', pos + search.size());

		if (colon == std::string::npos)
		{
			return defaultVal;
		}

		size_t valStart = json.find_first_not_of(" \t\r\n", colon + 1);

		if (valStart == std::string::npos)
		{
			return defaultVal;
		}

		if (json.substr(valStart, 4) == "true")
		{
			return true;
		}

		if (json.substr(valStart, 5) == "false")
		{
			return false;
		}

		return defaultVal;
	}

	// Extract a string array: "key": ["a", "b", "c"]
	static std::vector<std::string> extractStringArray(const std::string& json, const std::string& key)
	{
		std::vector<std::string> result;

		std::string search = "\"" + key + "\"";
		size_t pos = json.find(search);

		if (pos == std::string::npos)
		{
			return result;
		}

		size_t arrStart = json.find('[', pos);
		size_t arrEnd = findMatchingBracket(json, arrStart, '[', ']');

		if (arrStart == std::string::npos || arrEnd == std::string::npos)
		{
			return result;
		}

		std::string arr = json.substr(arrStart + 1, arrEnd - arrStart - 1);

		// Extract each quoted string
		size_t p = 0;

		while (p < arr.size())
		{
			size_t qStart = arr.find('"', p);

			if (qStart == std::string::npos)
			{
				break;
			}

			size_t qEnd = arr.find('"', qStart + 1);

			if (qEnd == std::string::npos)
			{
				break;
			}

			result.push_back(arr.substr(qStart + 1, qEnd - qStart - 1));
			p = qEnd + 1;
		}

		return result;
	}
};