#pragma once

#include <string>
#include <vector>

struct CourseDataSetConfig
{
	std::string courseName;

	// 파일 경로
	std::vector<std::string> shpFiles;		// 임상도 SHP 파일 목록
	std::string aerialImgPath;					// 항공사진 정사영상 경로 (.tif/.tiff)
	std::string treeInfoPath;					// 나무 위치 CSV 경로

	// 좌표계 정보
	int srcEPSG = 5186;
	int dstEPSG = 5179;

	// 3. 정합 오프셋
	double globalOffsetX = 0.0;
	double globalOffsetY = 0.0;
	double fineOffsetX = 0.0;
	double fineOffsetY = 0.0;

	double getTotalOffsetX() const { return globalOffsetX + fineOffsetX; }
	double getTotalOffsetY() const { return globalOffsetY + fineOffsetY; }
};

