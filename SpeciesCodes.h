#pragma once

// ----------------------------------------------------------------------------------------------------------------------------
//  SpeciesCodes.h
//  Single source of truth for 산림청 임상도 수종그룹코드 (KOFTR_GROU)
//
//  One row per code: group, Korean name, English name, BGR colour.
//
//  Groups:
//    C  침엽수    (10-21)
//    D  낙엽활엽수 (30-49)
//    E  상록활엽수 (60-68)
//    M  혼효림    (77)          - a species-less label; excluded from training by default
//    N  비산림    (78, 81-83, 91-99) - no species information at all
//
//  API (all static, header-only, no init needed):
//    SpeciesCodes::group(code)        -> "C" / "D" / "E" / "M" / "N" / "" (unknown)
//    SpeciesCodes::isTrainable(code)  -> true for C/D/E
//    SpeciesCodes::isNonForest(code)  -> true for N
//    SpeciesCodes::isMixed(code)      -> true for M (77)
//    SpeciesCodes::name(code, ko)     -> English (default) or Korean name
//    SpeciesCodes::groupName(g, ko)   -> name of a group code
//    SpeciesCodes::color(code)        -> BGR colour (group codes "C","D","E","N" also accepted)
//    SpeciesCodes::fillColor(code)    -> lighter colour for polygon fills
//    SpeciesCodes::has(code)          -> code is in the table
//    SpeciesCodes::codesInGroup(g)    -> all codes of a group
//
//  Legacy free functions (hasSpeciesColor, getSpeciesColor, getPolygonFillColor,
//  getSpeciesName) are kept in ForestLoader.h as thin wrappers over this class.
// ----------------------------------------------------------------------------------------------------------------------------

#include "TSICommon.h"

class SpeciesCodes
{
public:

	struct Entry
	{
		const char* code;
		const char* group;			// C / D / E / M / N
		const char* nameKo;
		const char* nameEn;
		cv::Scalar color;				// BGR
	};

	// Group definitions
	struct GroupEntry
	{
		const char* code;
		const char* nameKo;
		const char* nameEn;
		cv::Scalar color;
	};

	static const std::vector<GroupEntry>& groups()
	{
		static const std::vector<GroupEntry> g =
		{
			{ "C", "침엽수", "Conifer", {  34, 139,  34 } },
			{ "D", "낙엽활엽수", "Deciduous", {  30, 144, 255 } },
			{ "E", "상록활엽수", "Evergreen", {   0,  80,  60 } },
			{ "M", "침활혼효림", "Mixed", { 128, 128,   0 } },
			{ "N", "비산림", "Non-Forest", { 160, 160, 160 } },
		};

		return g;
	}

	// Species table (산림청 임상도 1:5,000 코드 정의서 기준)
	static const std::vector<Entry>& entries()
	{
		static const std::vector<Entry> e =
		{
			// code / grp / nameKo / nameEn / BGR
			{ "11", "C", "소나무", "Pine", {  34, 139,  34 } },
			{ "12", "C", "잣나무", "Korean Pine", {   0,  80,   0 } },
			{ "13", "C", "낙엽송", "Larch", {  19,  69, 139 } },
			{ "14", "C", "리기다소나무", "Pitch Pine", {   0, 165, 255 } },
			{ "15", "C", "곰솔", "Black Pine", {  60, 179, 113 } },
			{ "16", "C", "전나무", "Fir", { 128, 200,  80 } },
			{ "17", "C", "편백나무", "Hinoki Cypress", {  80, 220, 160 } },
			{ "18", "C", "삼나무", "Cedar", { 100, 240, 100 } },
			{ "19", "C", "가문비나무", "Spruce", {  50, 150,  50 } },
			{ "20", "C", "비자나무", "Japanese Torreya", {   0, 120,  80 } },
			{ "21", "C", "은행나무", "Ginkgo", { 180, 230, 180 } },
			{ "10", "C", "기타침엽수", "Other Conifers", { 150, 200, 150 } },
			{ "31", "D", "상수리나무", "Sawtooth Oak", {  30, 144, 255 } },
			{ "32", "D", "신갈나무", "Mongolian Oak", {  65, 105, 225 } },
			{ "33", "D", "굴참나무", "Oriental Oak", {  99, 149, 234 } },
			{ "34", "D", "기타참나무류", "Other Oaks", { 130, 170, 250 } },
			{ "35", "D", "오리나무", "Alder", {  42, 200, 180 } },
			{ "36", "D", "고로쇠나무", "Painted Maple", {   0, 210, 210 } },
			{ "37", "D", "자작나무", "Birch", { 200, 230, 255 } },
			{ "38", "D", "박달나무", "Asian White Birch", { 120, 160, 200 } },
			{ "39", "D", "밤나무", "Chestnut", {  20, 100, 200 } },
			{ "40", "D", "물푸레나무", "Ash", {  80, 180, 230 } },
			{ "41", "D", "서어나무", "Hornbeam", {  50, 130, 180 } },
			{ "42", "D", "때죽나무", "Snowbell", { 160, 200, 230 } },
			{ "43", "D", "호두나무", "Walnut", {  90, 160, 220 } },
			{ "44", "D", "백합나무", "Tulip Tree", { 110, 190, 240 } },
			{ "45", "D", "포플러", "Poplar", { 170, 220, 255 } },
			{ "46", "D", "벚나무", "Cherry", { 120,  80, 200 } },
			{ "47", "D", "느티나무", "Zelkova", {  80,  50, 160 } },
			{ "48", "D", "층층나무", "Dogwood", { 100,  70, 180 } },
			{ "49", "D", "아까시나무", "Black Locust", { 150, 120, 210 } },
			{ "30", "D", "기타활엽수", "Other Deciduous", { 180, 160, 230 } },
			{ "61", "E", "가시나무", "Evergreen Oak", {   0,  80,  60 } },
			{ "62", "E", "구실잣밤나무", "Castanopsis", {   0, 100,  70 } },
			{ "63", "E", "녹나무", "Camphor Tree", {  20, 120,  80 } },
			{ "64", "E", "굴거리나무", "Daphniphyllum", {  40, 130,  90 } },
			{ "65", "E", "황칠나무", "Dendropanax", {  60, 140, 100 } },
			{ "66", "E", "사스레피나무", "Eurya", {  80, 150, 110 } },
			{ "67", "E", "후박나무", "Machilus", { 100, 160, 120 } },
			{ "68", "E", "새덕이", "Neolitsea", { 120, 170, 130 } },
			{ "60", "E", "기타상록활엽수", "Other Evergreen", { 140, 180, 140 } },
			{ "77", "M", "침활혼효림", "Mixed Forest", { 128, 128,   0 } },
			{ "78", "N", "죽림", "Bamboo", {   0, 200, 100 } },
			{ "81", "N", "미립목지", "Unstocked", { 211, 211, 211 } },
			{ "82", "N", "제지", "Bare Land", { 169, 169, 169 } },
			{ "83", "N", "관목덤불", "Shrub", { 144, 238, 144 } },
			{ "91", "N", "주거지", "Residential", { 192, 192, 192 } },
			{ "92", "N", "초지", "Grassland", { 200, 230, 200 } },
			{ "93", "N", "경작지", "Farmland", { 220, 220, 180 } },
			{ "94", "N", "수체", "Water", { 255, 200,   0 } },
			{ "95", "N", "과수원", "Orchard", { 180, 255, 180 } },
			{ "99", "N", "비산림", "Non-Forest", { 230, 230, 230 } },
		};

		return e;
	}

	// Lookups

	static bool has(const std::string& code)
	{
		return byCode().count(code) > 0;
	}

	static std::string group(const std::string& code)
	{
		auto it = byCode().find(code);

		return (it != byCode().end()) ? it->second->group : std::string();
	}

	static bool isTrainable(const std::string& code)
	{
		std::string g = group(code);

		return (g == "C") || (g == "D") || (g == "E");
	}

	static bool isMixed(const std::string& code)
	{
		return group(code) == "M";
	}

	static bool isNonForest(const std::string& code)
	{
		return group(code) == "N";
	}

	static std::string name(const std::string& code, bool ko = false)
	{
		auto it = byCode().find(code);

		if (it != byCode().end())
		{
			return ko ? it->second->nameKo : it->second->nameEn;
		}

		for (auto& g : groups())
		{
			if (code == g.code) return ko ? g.nameKo : g.nameEn;
		}

		return std::string();
	}

	static std::string groupName(const std::string& groupCode, bool ko = false)
	{
		return name(groupCode, ko);
	}

	static cv::Scalar color(const std::string& code)
	{
		auto it = byCode().find(code);

		if (it != byCode().end()) return it->second->color;

		for (auto& g : groups())
		{
			if (code == g.code) return g.color;
		}

		return cv::Scalar(200, 200, 200);
	}

	// Lighter / desaturated version for polygon fills (blend 60% toward white)
	static cv::Scalar fillColor(const std::string& code, double blend = 0.6)
	{
		cv::Scalar c = color(code);

		return cv::Scalar(c[0] + (255 - c[0]) * blend, c[1] + (255 - c[1]) * blend, c[2] + (255 - c[2]) * blend);
	}

	static std::vector<std::string> codesInGroup(const std::string& groupCode)
	{
		std::vector<std::string> out;

		for (auto& e : entries())
		{
			if (groupCode == e.group) out.push_back(e.code);
		}

		return out;
	}

private:

	static const std::map<std::string, const Entry*>& byCode()
	{
		static const std::map<std::string, const Entry*> m = []()
		{
			std::map<std::string, const Entry*> t;

			for (auto& e : entries())
			{
				t[e.code] = &e;
			}

			return t;
		}();

		return m;
	}
};