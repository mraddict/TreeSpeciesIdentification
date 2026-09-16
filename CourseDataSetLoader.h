#pragma once

#include "TSICommon.h"

struct CourseDataSet
{
	std::string courseName;

	// 파일 경로
	std::vector<std::string> shpFiles;		// 임상도 SHP 파일 목록
	std::string aerialImgPath;					// 항공사진 정사영상 경로 (.tif/.tiff)
	std::string treeInfoPath;					// 나무 위치 CSV 경로

	// 좌표계 정보
	int treeSrcEPSG = 5179;
	int targetEPSG = 5179;

	long long treeIdOffset = 0;				// TreeID 오프셋 (여러 코스 합칠 때 ID 충돌 방지)
	double crownPadding = 1.4;			// 크롭 패딩

	std::string outputDir = "";					// 출력 디렉토리 (코스별). 비어있으면 자동 생성
	bool enabled = true;							// 활성화 여부
};

struct CourseDataSetConfig
{
	std::vector<CourseDataSet> courses;

	std::string outputDir = "results";			// 전체 출력 디렉토리
	std::string mergedFeatureCsv = "";		// 합친 features CSV 경로

	// ID 오프셋 자동 계산
	void assignTreeIdOffsets(long long step = 100000)
	{
		for (size_t i = 0 ; i < courses.size() ; ++i)
		{
			courses[i].treeIdOffset = (long long)i * step;
			
			if (courses[i].outputDir.empty())
			{
				courses[i].outputDir = outputDir + "/" + courses[i].courseName;
			}
		}
	}

	// 활성화된 코스 수
	int enabledCount() const
	{
		int n = 0;

		for (auto& c : courses)
		{
			if (c.enabled)
			{
				++n;
			}
		}

		return n;
	}

	void printSummary() const
	{
		std::cout << "=== Multi-Course Configuration ===\n";
		std::cout << "  Total courses: " << courses.size() << " (" << enabledCount() << " enabled)\n";

		for (auto& c : courses)
		{
			std::cout << "  " << (c.enabled ? "[ON] " : "[OFF]") << c.courseName << " (SHP:" << c.shpFiles.size() << ", offset:" << c.treeIdOffset << ")\n";
		}

		std::cout << "  Output: " << outputDir << "\n\n";
	}
};

class CourseDataSetConfigLoader
{
public:

	static CourseDataSetConfig load(const std::string& configPath)
	{
		CourseDataSetConfig cfg;

		std::ifstream file(configPath);

		if (!file.is_open())
		{
			std::cerr << "Cannot open config: " << configPath << "\n";

			return cfg;
		}

		// Read entire file
		std::string json((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		std::string cleaned;

		std::istringstream stream(json.c_str());
		std::string line;

		while (std::getline(stream, line))
		{
			std::string trimmed = TSICommon::trim(line);

			if (trimmed.substr(0, 2) != "//")
			{
				cleaned += line + "\n";
			}
		}

		// Parse top-level output
		std::string outputVal = TSICommon::extractString(cleaned, "output");

		if (!outputVal.empty())
		{
			cfg.outputDir = outputVal;
		}

		// Parse courses array
		size_t coursesStart = cleaned.find("\"courses\"");

		if (coursesStart == std::string::npos)
		{
			std::cerr << "No 'courses' array found in config\n";

			return cfg;
		}

		// Find the array brackets
		size_t arrStart = cleaned.find('[', coursesStart);
		size_t arrEnd = TSICommon::findMatchingBracket(cleaned, arrStart, '[', ']');

		if ((arrStart == std::string::npos) || (arrEnd == std::string::npos))
		{
			std::cerr << "Invalid courses array\n";

			return cfg;
		}

		std::string coursesArr = cleaned.substr(arrStart + 1, arrEnd - arrStart - 1);

		// Parse each course object { ... }
		size_t pos = 0;

		while (pos < coursesArr.size())
		{
			size_t objStart = coursesArr.find('{', pos);

			if (objStart == std::string::npos)
			{
				break;
			}

			size_t objEnd = TSICommon::findMatchingBracket(coursesArr, objStart, '{', '}');

			if (objEnd == std::string::npos)
			{
				break;
			}

			std::string obj = coursesArr.substr(objStart, objEnd - objStart + 1);

			CourseDataSet course = parseCourseDataSet(obj);

			if (!course.courseName.empty())
			{
				cfg.courses.push_back(course);
			}

			pos = objEnd + 1;
		}

		cfg.assignTreeIdOffsets();

		std::cout << "Loaded config: " << configPath << "\n";
		cfg.printSummary();

		return cfg;

	}

	// Parse one course object
	static CourseDataSet parseCourseDataSet(const std::string& obj)
	{
		CourseDataSet c;
	
		c.courseName = TSICommon::extractString(obj, "name");
		c.shpFiles = TSICommon::extractStringArray(obj, "shp_files");
		c.aerialImgPath = TSICommon::extractString(obj, "aerial_img");
		c.treeInfoPath = TSICommon::extractString(obj, "tree_info");
		c.treeSrcEPSG = (int)TSICommon::extractNumber(obj, "tree_source_epsg", 5179);
		c.targetEPSG = (int)TSICommon::extractNumber(obj, "target_epsg", 5179);
		c.crownPadding = TSICommon::extractNumber(obj, "padding", 1.4);
		c.enabled = TSICommon::extractBool(obj, "enabled", true);

		return c;
	}
};