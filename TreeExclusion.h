#pragma once

// ----------------------------------------------------------------------------------------------------------------------------
//  TreeExclusion.h
//  Manual corrections to a LiDAR tree list. The tree_info CSV itself is never modified.
//
//  File : <course>_tree_info_excluded.csv next to the tree_info CSV (or config "tree_exclude")
//    TreeID,action,note
//    action = remove       : dropped by TreeLoader::load and FeatureExtractor::loadCSV
//             auto_forest  : kept (handled later in export)
//    TreeID is the tree_info TreeID (no course offset).
// ----------------------------------------------------------------------------------------------------------------------------

#include "CourseDataSetLoader.h"

#include <map>
#include <unordered_set>

class TreeExclusion
{
public:

	struct Entry
	{
		std::string action;
		std::string note;
	};

	static constexpr const char* kRemove = "remove";
	static constexpr const char* kAutoForest = "auto_forest";
	static constexpr const char* kNone = "<none>";			// TreeLoader::load: do not apply any file
	static constexpr const char* kAutoNotePrefix = "auto_";	// note of rows written by an automatic filter (dropped by dropAutoRows)

	// explicitPath (config tree_exclude) if set, otherwise "<tree_info stem>_excluded.csv"
	static std::string resolvePath(const std::string& treeInfoPath, const std::string& explicitPath = "")
	{
		if (!explicitPath.empty())
		{
			return explicitPath;
		}

		size_t dot = treeInfoPath.find_last_of('.');
		size_t sep = treeInfoPath.find_last_of("/\\");

		bool hasExt = (dot != std::string::npos) && ((sep == std::string::npos) || (dot > sep));

		return (hasExt ? treeInfoPath.substr(0, dot) : treeInfoPath) + "_excluded.csv";
	}

	// Empty map if the file does not exist. Header and malformed lines are skipped.
	static std::map<long long, Entry> load(const std::string& path)
	{
		std::map<long long, Entry> result;
		std::ifstream file(path);

		if (!file.is_open())
		{
			return result;
		}

		std::string line;

		while (std::getline(file, line))
		{
			if (!line.empty() && line.back() == '\r')
			{
				line.pop_back();
			}

			size_t c1 = line.find(',');

			if (c1 == std::string::npos)
			{
				continue;
			}

			long long id = -1;

			try { id = std::stoll(TSICommon::trim(line.substr(0, c1))); }
			catch (...) { continue; }

			size_t c2 = line.find(',', c1 + 1);

			Entry e;
			e.action = TSICommon::trim(line.substr(c1 + 1, (c2 == std::string::npos) ? std::string::npos : c2 - c1 - 1));
			e.note = (c2 == std::string::npos) ? "" : line.substr(c2 + 1);

			if (e.action.empty())
			{
				continue;
			}

			result[id] = e;
		}

		return result;
	}

	static void save(const std::string& path, const std::map<long long, Entry>& entries)
	{
		std::ofstream file(path, std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("Failed to write: " + path);
		}

		file << "TreeID,action,note\r\n";

		for (auto& kv : entries)
		{
			file << kv.first << "," << kv.second.action << "," << kv.second.note << "\r\n";
		}
	}

	// Exclusions are manual only: drop rows whose note marks an automatic filter (note starts with "auto_").
	// The file is rewritten only if such rows exist. Returns the number of dropped rows.
	static int dropAutoRows(const std::string& path, int* remaining = nullptr)
	{
		std::map<long long, Entry> entries = load(path);
		int dropped = 0;

		for (auto it = entries.begin() ; it != entries.end() ; )
		{
			if (TSICommon::trim(it->second.note).rfind(kAutoNotePrefix, 0) == 0)
			{
				it = entries.erase(it);
				++dropped;
			}
			else
			{
				++it;
			}
		}

		if (dropped > 0)
		{
			save(path, entries);
		}

		if (remaining)
		{
			*remaining = (int)entries.size();
		}

		return dropped;
	}

	static std::unordered_set<long long> removedIds(const std::map<long long, Entry>& entries)
	{
		std::unordered_set<long long> ids;

		for (auto& kv : entries)
		{
			if (kv.second.action == kRemove)
			{
				ids.insert(kv.first);
			}
		}

		return ids;
	}

	// Registry for FeatureExtractor::loadCSV, keyed by TreeID + course offset
	static void registerConfig(const CourseDataSetConfig& cfg)
	{
		auto& reg = registry();
		reg.clear();

		for (auto& c : cfg.courses)
		{
			std::string path = resolvePath(c.treeInfoPath, c.treeExcludePath);
			std::unordered_set<long long> removed = removedIds(load(path));

			for (auto id : removed)
			{
				reg.insert(c.treeIdOffset + id);
			}

			if (!removed.empty())
			{
				std::cout << "  tree_info_excluded: " << c.courseName << " remove " << removed.size() << " (" << path << ")\n";
			}
		}
	}

	static bool hasRegistry()
	{
		return !registry().empty();
	}

	static bool isRemoved(long long globalTreeId)
	{
		return registry().count(globalTreeId) > 0;
	}

private:

	static std::unordered_set<long long>& registry()
	{
		static std::unordered_set<long long> reg;

		return reg;
	}
};
