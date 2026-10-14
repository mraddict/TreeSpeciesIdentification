#pragma once

// ----------------------------------------------------------------------------------------------------------------------------
//  TreeEditor.h
//  Interactive keep/delete editor for exported trees, built on ImageViewer (pan / zoom / reset).
//
//  Mouse:  left drag = pan, wheel = zoom, left click = toggle nearest tree
//          Shift + left drag = rectangle select
//  Keys:   L = lasso mode (click vertices, Enter = select, Esc = cancel)
//          Delete = mark selected as deleted, U = restore selected, Ctrl+Z = undo last Delete/U
//          A = clear selection, S = save excluded_ids.csv, R = reset view, Q = quit (twice if unsaved)
//
//  excluded_ids.csv : one column "TreeID". Read on start; IDs not present in trees.json
//  (already removed by a later export) are kept when saving.
// ----------------------------------------------------------------------------------------------------------------------------

#include "ImageViewer.h"
#include "AerialLoader.h"
#include "TreeLoader.h"
#include "CoordTransformer.h"

#include <unordered_set>

class TreeEditor
{
public:

	// TreeID column (header / non-numeric lines ignored). Empty set if the file does not exist.
	static std::unordered_set<long long> loadExcludedIds(const std::string& path)
	{
		std::unordered_set<long long> ids;
		std::ifstream file(path);

		if (!file.is_open())
		{
			return ids;
		}

		std::string line;

		while (std::getline(file, line))
		{
			std::string v = TSICommon::trim(line);

			if (v.empty() || !(std::isdigit((unsigned char)v[0]) || (v[0] == '-')))
			{
				continue;
			}

			try { ids.insert(std::stoll(v)); }
			catch (...) {}
		}

		std::cout << "  Loaded " << ids.size() << " excluded IDs: " << path << "\n";

		return ids;
	}

	static void saveExcludedIds(const std::string& path, const std::unordered_set<long long>& ids)
	{
		std::vector<long long> sorted(ids.begin(), ids.end());
		std::sort(sorted.begin(), sorted.end());

		std::ofstream file(path, std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("Failed to write: " + path);
		}

		file << "TreeID\r\n";

		for (auto id : sorted)
		{
			file << id << "\r\n";
		}

		std::cout << "  Saved " << sorted.size() << " excluded IDs: " << path << "\n";
	}

	// aerial : image + GeoTransform (may be a downscaled load), trees : coordinates in the aerial CRS
	static void run(const AerialPhoto& aerial, const std::vector<TreePoint>& trees, const std::string& excludedPath, const std::string& title)
	{
		State st;
		st.excludedPath = excludedPath;

		double gt[6] = { aerial.originX, aerial.pixelW, aerial.rotationX, aerial.originY, aerial.rotationY, aerial.pixelH };
		double igt[6];

		if (!CoordTransformer::computeInverseGT(gt, igt))
		{
			throw std::runtime_error("Failed to compute inverse GeoTransform");
		}

		double pixPerMeter = std::abs(igt[1]);

		std::unordered_set<long long> excluded = loadExcludedIds(excludedPath);
		std::unordered_set<long long> shown;

		for (auto& tp : trees)
		{
			Tree t;
			t.id = tp.treeId;
			t.px = cv::Point2f((float)(igt[0] + igt[1] * tp.x + igt[2] * tp.y), (float)(igt[3] + igt[4] * tp.x + igt[5] * tp.y));
			t.rPx = (float)std::max(1.0, tp.crownD * 0.5 * pixPerMeter);
			t.excluded = (excluded.count(tp.treeId) > 0);

			st.trees.push_back(t);
			shown.insert(tp.treeId);
		}

		for (auto id : excluded)
		{
			if (!shown.count(id))
			{
				st.hiddenExcluded.insert(id);
			}
		}

		ImageViewer::Options opt;
		opt.title = title;
		opt.printControls = false;
		opt.hint = "Click:toggle  Shift+drag:rect  L:lasso  Del:delete  U:restore  Ctrl+Z:undo  A:clear  S:save  R:reset  Q:quit";
		opt.showGeoCoord = false;

		opt.drawOverlay = [&st](cv::Mat& canvas, const ImageViewer::View& v) { drawOverlay(st, canvas, v); };
		opt.onMouse = [&st](int event, int x, int y, int flags, const ImageViewer::View& v) { return onMouse(st, event, x, y, flags, v); };
		opt.onKey = [&st](int key) { return onKey(st, key); };

		std::cout << "=== Tree editor ===\n"
			<< "  Trees: " << st.trees.size() << " (deleted: " << countExcluded(st) << ", deleted but not in trees.json: " << st.hiddenExcluded.size() << ")\n"
			<< "  Left drag     : pan,  wheel: zoom,  R: reset view\n"
			<< "  Left click    : toggle nearest tree\n"
			<< "  Shift + drag  : rectangle select\n"
			<< "  L             : lasso (click vertices, Enter = select, Esc = cancel)\n"
			<< "  Delete / U    : delete / restore selected,  Ctrl+Z: undo,  A: clear selection\n"
			<< "  S             : save " << excludedPath << "\n"
			<< "  Q             : quit (press twice if there are unsaved changes)\n";

		ImageViewer::show(aerial.image, opt);

		if (st.dirty)
		{
			std::cout << "  ! Quit with unsaved changes\n";
		}
	}

private:

	struct Tree
	{
		long long id = -1;
		cv::Point2f px;						// image pixel (of the loaded, possibly downscaled, image)
		float rPx = 1.0f;					// crown radius in image pixels
		bool excluded = false;
		bool selected = false;
	};

	struct State
	{
		std::vector<Tree> trees;
		std::unordered_set<long long> hiddenExcluded;		// excluded IDs absent from trees.json, kept on save
		std::string excludedPath;

		std::vector<std::vector<std::pair<int, bool>>> undo;	// per Delete/U: (tree index, previous excluded)
		bool dirty = false;
		bool quitArmed = false;

		// click / rectangle
		bool rectActive = false;
		cv::Point downPt;
		cv::Point mousePt;

		// lasso (image pixels)
		bool lassoMode = false;
		std::vector<cv::Point2f> lasso;

		std::string message;
	};

	static constexpr int kDeleteKey = 0x2E0000;		// waitKeyEx code on Windows
	static constexpr int kCtrlZ = 26;
	static constexpr int kEnter = 13;
	static constexpr int kEsc = 27;
	static constexpr int kClickSlop = 4;					// px: larger movement is a drag (pan), not a click
	static constexpr int kPickRadius = 12;				// px on screen, minimum pick distance

	static cv::Point2f toScreen(const cv::Point2f& p, const ImageViewer::View& v)
	{
		return cv::Point2f((float)(v.panX + p.x * v.zoom), (float)(v.panY + p.y * v.zoom));
	}

	static cv::Point2f toImage(int x, int y, const ImageViewer::View& v)
	{
		return cv::Point2f((float)((x - v.panX) / v.zoom), (float)((y - v.panY) / v.zoom));
	}

	static int countExcluded(const State& st)
	{
		int n = 0;

		for (auto& t : st.trees)
		{
			n += t.excluded ? 1 : 0;
		}

		return n;
	}

	static int countSelected(const State& st)
	{
		int n = 0;

		for (auto& t : st.trees)
		{
			n += t.selected ? 1 : 0;
		}

		return n;
	}

	static void setExcludedOnSelection(State& st, bool excluded)
	{
		std::vector<std::pair<int, bool>> op;

		for (int i = 0 ; i < (int)st.trees.size() ; ++i)
		{
			Tree& t = st.trees[i];

			if (t.selected && (t.excluded != excluded))
			{
				op.push_back({ i, t.excluded });
				t.excluded = excluded;
			}
		}

		if (op.empty())
		{
			st.message = "Nothing to change";

			return;
		}

		st.message = std::string(excluded ? "Deleted " : "Restored ") + std::to_string(op.size());
		st.undo.push_back(std::move(op));
		st.dirty = true;
	}

	static void undoLast(State& st)
	{
		if (st.undo.empty())
		{
			st.message = "Nothing to undo";

			return;
		}

		for (auto& e : st.undo.back())
		{
			st.trees[e.first].excluded = e.second;
		}

		st.message = "Undo (" + std::to_string(st.undo.back().size()) + ")";
		st.undo.pop_back();
		st.dirty = true;
	}

	static void save(State& st)
	{
		std::unordered_set<long long> ids = st.hiddenExcluded;

		for (auto& t : st.trees)
		{
			if (t.excluded)
			{
				ids.insert(t.id);
			}
		}

		saveExcludedIds(st.excludedPath, ids);

		st.dirty = false;
		st.message = "Saved " + std::to_string(ids.size()) + " IDs";
	}

	static void toggleNearest(State& st, int x, int y, const ImageViewer::View& v)
	{
		int best = -1;
		double bestD = 0.0;

		for (int i = 0 ; i < (int)st.trees.size() ; ++i)
		{
			cv::Point2f s = toScreen(st.trees[i].px, v);
			double d = std::hypot(s.x - x, s.y - y);
			double pick = std::max((double)kPickRadius, st.trees[i].rPx * v.zoom);

			if ((d <= pick) && ((best < 0) || (d < bestD)))
			{
				best = i;
				bestD = d;
			}
		}

		if (best >= 0)
		{
			st.trees[best].selected = !st.trees[best].selected;
			st.message = "TreeID " + std::to_string(st.trees[best].id) + (st.trees[best].selected ? " selected" : " unselected");
		}
	}

	static void selectInRect(State& st, const cv::Point& a, const cv::Point& b, const ImageViewer::View& v)
	{
		cv::Point2f p0 = toImage(std::min(a.x, b.x), std::min(a.y, b.y), v);
		cv::Point2f p1 = toImage(std::max(a.x, b.x), std::max(a.y, b.y), v);
		int n = 0;

		for (auto& t : st.trees)
		{
			if ((t.px.x >= p0.x) && (t.px.x <= p1.x) && (t.px.y >= p0.y) && (t.px.y <= p1.y))
			{
				n += t.selected ? 0 : 1;
				t.selected = true;
			}
		}

		st.message = "Rectangle: +" + std::to_string(n) + " selected";
	}

	static void selectInLasso(State& st)
	{
		if (st.lasso.size() < 3)
		{
			st.message = "Lasso needs at least 3 points";

			return;
		}

		int n = 0;

		for (auto& t : st.trees)
		{
			if (cv::pointPolygonTest(st.lasso, t.px, false) >= 0)
			{
				n += t.selected ? 0 : 1;
				t.selected = true;
			}
		}

		st.message = "Lasso: +" + std::to_string(n) + " selected";
	}

	static bool onMouse(State& st, int event, int x, int y, int flags, const ImageViewer::View& v)
	{
		st.mousePt = cv::Point(x, y);

		// Shift + drag: rectangle (consumed so the viewer does not pan)
		if ((event == cv::EVENT_LBUTTONDOWN) && (flags & cv::EVENT_FLAG_SHIFTKEY) && !st.lassoMode)
		{
			st.rectActive = true;
			st.downPt = cv::Point(x, y);

			return true;
		}

		if (st.rectActive)
		{
			if (event == cv::EVENT_LBUTTONUP)
			{
				st.rectActive = false;
				selectInRect(st, st.downPt, cv::Point(x, y), v);
			}

			return true;
		}

		if (event == cv::EVENT_LBUTTONDOWN)
		{
			st.downPt = cv::Point(x, y);
		}

		// Click (not a pan drag): toggle nearest tree, or add a lasso vertex
		if ((event == cv::EVENT_LBUTTONUP) && (std::abs(x - st.downPt.x) <= kClickSlop) && (std::abs(y - st.downPt.y) <= kClickSlop))
		{
			if (st.lassoMode)
			{
				st.lasso.push_back(toImage(x, y, v));
			}
			else
			{
				toggleNearest(st, x, y, v);
			}
		}

		return false;		// let the viewer pan / track the mouse
	}

	static bool onKey(State& st, int key)
	{
		int k = key & 0xFF;

		if ((k == 'q') || (k == 'Q'))
		{
			if (st.dirty && !st.quitArmed)
			{
				st.quitArmed = true;
				st.message = "Unsaved changes! S = save, Q again = quit without saving";

				return true;
			}

			return false;		// viewer quits
		}

		st.quitArmed = false;

		if (key == kDeleteKey)
		{
			setExcludedOnSelection(st, true);

			return true;
		}

		if (k == kCtrlZ)
		{
			undoLast(st);

			return true;
		}

		if (k == kEsc)
		{
			st.lassoMode = false;
			st.lasso.clear();
			st.rectActive = false;
			st.message = "Cancelled";

			return true;		// Esc never quits the editor
		}

		if (k == kEnter)
		{
			if (st.lassoMode)
			{
				selectInLasso(st);
				st.lassoMode = false;
				st.lasso.clear();
			}

			return true;
		}

		switch (k)
		{
			case 'l':
			case 'L':
				st.lassoMode = true;
				st.lasso.clear();
				st.message = "Lasso: click vertices, Enter = select, Esc = cancel";
				return true;

			case 'u':
			case 'U':
				setExcludedOnSelection(st, false);
				return true;

			case 'a':
			case 'A':
				for (auto& t : st.trees)
				{
					t.selected = false;
				}
				st.message = "Selection cleared";
				return true;

			case 's':
			case 'S':
				save(st);
				return true;
		}

		return false;			// R and others: viewer default
	}

	static void drawOverlay(State& st, cv::Mat& canvas, const ImageViewer::View& v)
	{
		const cv::Scalar kKeep(0, 220, 0);
		const cv::Scalar kDeleted(0, 0, 255);
		const cv::Scalar kSelected(0, 255, 255);

		int W = canvas.cols;
		int H = canvas.rows;

		for (auto& t : st.trees)
		{
			cv::Point2f s = toScreen(t.px, v);
			int r = std::max(1, (int)(t.rPx * v.zoom));

			if ((s.x < -r) || (s.y < -r) || (s.x > W + r) || (s.y > H + r))
			{
				continue;
			}

			cv::Point c((int)s.x, (int)s.y);
			cv::Scalar col = t.excluded ? kDeleted : kKeep;

			cv::circle(canvas, c, r, col, 1, cv::LINE_AA);
			cv::circle(canvas, c, 2, col, -1, cv::LINE_AA);

			if (t.excluded)
			{
				int d = std::max(3, (int)(r * 0.7));

				cv::line(canvas, c + cv::Point(-d, -d), c + cv::Point(d, d), kDeleted, 2, cv::LINE_AA);
				cv::line(canvas, c + cv::Point(-d, d), c + cv::Point(d, -d), kDeleted, 2, cv::LINE_AA);
			}

			if (t.selected)
			{
				cv::circle(canvas, c, r + 3, kSelected, 2, cv::LINE_AA);
			}
		}

		// Rectangle in progress
		if (st.rectActive)
		{
			cv::rectangle(canvas, st.downPt, st.mousePt, kSelected, 1, cv::LINE_AA);
		}

		// Lasso in progress
		if (st.lassoMode && !st.lasso.empty())
		{
			std::vector<cv::Point> pts;

			for (auto& p : st.lasso)
			{
				cv::Point2f s = toScreen(p, v);
				pts.push_back(cv::Point((int)s.x, (int)s.y));
			}

			cv::polylines(canvas, pts, false, kSelected, 1, cv::LINE_AA);
			cv::line(canvas, pts.back(), st.mousePt, kSelected, 1, cv::LINE_AA);
			cv::line(canvas, st.mousePt, pts.front(), kSelected, 1, cv::LINE_AA);

			for (auto& p : pts)
			{
				cv::circle(canvas, p, 3, kSelected, -1, cv::LINE_AA);
			}
		}

		// Counts panel (below the viewer's zoom / pixel lines)
		int total = (int)st.trees.size();
		int deleted = countExcluded(st);
		int selected = countSelected(st);

		std::vector<std::string> lines =
		{
			"Total " + std::to_string(total) + "   Keep " + std::to_string(total - deleted) + "   Deleted " + std::to_string(deleted) + "   Selected " + std::to_string(selected),
		};

		if (!st.hiddenExcluded.empty())
		{
			lines.push_back("+ " + std::to_string(st.hiddenExcluded.size()) + " deleted IDs not in trees.json (kept on save)");
		}

		if (st.lassoMode)
		{
			lines.push_back("LASSO: " + std::to_string(st.lasso.size()) + " points  (Enter = select, Esc = cancel)");
		}

		lines.push_back(st.dirty ? "* unsaved changes (S = save)" : "saved");

		if (!st.message.empty())
		{
			lines.push_back(st.message);
		}

		int x = 10;
		int y = 104;
		int lineH = 22;
		cv::Rect box(x - 4, y - 18, 560, (int)lines.size() * lineH + 8);
		box &= cv::Rect(0, 0, W, H);

		if ((box.width > 0) && (box.height > 0))
		{
			cv::Mat roi = canvas(box);
			cv::Mat dark(roi.size(), roi.type(), cv::Scalar(20, 20, 20));
			cv::addWeighted(dark, 0.7, roi, 0.3, 0, roi);
		}

		for (size_t i = 0 ; i < lines.size() ; ++i)
		{
			cv::Scalar col = (i == 0) ? cv::Scalar(255, 255, 255) : cv::Scalar(200, 220, 255);
			cv::putText(canvas, lines[i], { x, y + (int)i * lineH }, cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv::LINE_AA);
		}
	}
};
