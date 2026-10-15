#pragma once

// ----------------------------------------------------------------------------------------------------------------------------
//  TreeEditor.h
//  False-detection editor for a tree_info CSV, built on ImageViewer (pan / zoom / reset).
//  Saves TreeExclusion files (<course>_tree_info_excluded.csv); the tree_info CSV is never modified.
//
//  Mouse:  left drag = pan, wheel = zoom, left click = toggle nearest tree (15 px)
//          Shift + left drag = rectangle select (adds)
//  Keys:   L = lasso (click vertices, Enter = select, Esc = cancel)
//          A = clear selection, I = invert selection
//          Delete = remove, U = keep, F = auto_forest, Ctrl+Z = undo
//          S = save, R = reset view, Q / Esc = quit (press twice if unsaved)
// ----------------------------------------------------------------------------------------------------------------------------

#include "ImageViewer.h"
#include "AerialLoader.h"
#include "TreeLoader.h"
#include "TreeExclusion.h"
#include "CoordTransformer.h"

#include <deque>

class TreeEditor
{
public:

	// aerial : image + GeoTransform (may be a downscaled load), trees : coordinates in the aerial CRS, TreeID without offset
	static void run(const AerialPhoto& aerial, const std::vector<TreePoint>& trees, const std::string& exclusionPath, const std::string& title)
	{
		State st;
		st.path = exclusionPath;
		st.fileEntries = TreeExclusion::load(exclusionPath);

		double gt[6] = { aerial.originX, aerial.pixelW, aerial.rotationX, aerial.originY, aerial.rotationY, aerial.pixelH };
		double igt[6];

		if (!CoordTransformer::computeInverseGT(gt, igt))
		{
			throw std::runtime_error("Failed to compute inverse GeoTransform");
		}

		double pixPerMeter = std::abs(igt[1]);
		int restored = 0;

		for (auto& tp : trees)
		{
			Tree t;
			t.id = tp.treeId;
			t.px = cv::Point2f((float)(igt[0] + igt[1] * tp.x + igt[2] * tp.y), (float)(igt[3] + igt[4] * tp.x + igt[5] * tp.y));
			t.rPx = (float)std::max(0.5, tp.crownD * 0.5 * pixPerMeter);

			auto it = st.fileEntries.find(tp.treeId);

			if (it != st.fileEntries.end())
			{
				t.mark = fromAction(it->second.action);
				restored += (t.mark != Mark::Keep) ? 1 : 0;
			}

			st.trees.push_back(t);
		}

		st.saved.resize(st.trees.size());

		for (size_t i = 0 ; i < st.trees.size() ; ++i)
		{
			st.saved[i] = st.trees[i].mark;
		}

		ImageViewer::Options opt;
		opt.title = title;
		opt.printControls = false;
		opt.hint = "Click  Shift+drag  L:lasso  A:clear  I:invert  Del:remove  U:keep  F:auto_forest  Ctrl+Z:undo  S:save  R:reset  Q:quit";

		opt.drawOverlay = [&st](cv::Mat& canvas, const ImageViewer::View& v) { drawOverlay(st, canvas, v); };
		opt.onMouse = [&st](int event, int x, int y, int flags, const ImageViewer::View& v) { return onMouse(st, event, x, y, flags, v); };
		opt.onKey = [&st](int key) { return onKey(st, key); };

		std::cout << "=== Tree editor ===\n"
			<< "  Trees: " << st.trees.size() << ", restored from file: " << restored << " (" << exclusionPath << ")\n"
			<< "  Left drag     : pan,  wheel: zoom,  R: reset view\n"
			<< "  Left click    : toggle nearest tree (within 15 px)\n"
			<< "  Shift + drag  : rectangle select (adds)\n"
			<< "  L             : lasso (click vertices, Enter = select, Esc = cancel)\n"
			<< "  A / I         : clear / invert selection\n"
			<< "  Delete / U / F: remove / keep / auto_forest,  Ctrl+Z: undo\n"
			<< "  S             : save\n"
			<< "  Q / Esc       : quit (press twice if there are unsaved changes)\n";

		ImageViewer::show(aerial.image, opt);

		int unsaved = countUnsaved(st);

		if (unsaved > 0)
		{
			std::cout << "  ! Quit without saving " << unsaved << " changes\n";
		}
	}

private:

	enum class Mark
	{
		Keep,
		Remove,
		AutoForest
	};

	struct Tree
	{
		long long id = -1;
		cv::Point2f px;						// image pixel (of the loaded, possibly downscaled, image)
		float rPx = 1.0f;					// crown radius in image pixels
		Mark mark = Mark::Keep;
		bool selected = false;
	};

	struct State
	{
		std::vector<Tree> trees;
		std::vector<Mark> saved;									// marks at the last load / save
		std::map<long long, TreeExclusion::Entry> fileEntries;	// notes + IDs not in tree_info are kept on save
		std::string path;

		std::deque<std::vector<std::pair<int, Mark>>> undo;	// per edit: (tree index, previous mark)
		bool quitArmed = false;

		bool rectActive = false;
		cv::Point downPt;
		cv::Point mousePt;

		bool lassoMode = false;
		std::vector<cv::Point2f> lasso;					// image pixels

		std::string message;
	};

	static constexpr int kDeleteKey = 0x2E0000;		// waitKeyEx code on Windows
	static constexpr int kCtrlZ = 26;
	static constexpr int kEnter = 13;
	static constexpr int kEsc = 27;
	static constexpr int kClickSlop = 4;					// px: larger movement is a pan, not a click
	static constexpr int kPickRadius = 15;				// px on screen
	static constexpr int kMinCrownPx = 3;				// below this screen radius only the center dot is drawn
	static constexpr size_t kUndoLimit = 1000;

	static Mark fromAction(const std::string& action)
	{
		if (action == TreeExclusion::kRemove)
		{
			return Mark::Remove;
		}

		if (action == TreeExclusion::kAutoForest)
		{
			return Mark::AutoForest;
		}

		return Mark::Keep;
	}

	static const char* toAction(Mark m)
	{
		return (m == Mark::Remove) ? TreeExclusion::kRemove : TreeExclusion::kAutoForest;
	}

	static cv::Point2f toScreen(const cv::Point2f& p, const ImageViewer::View& v)
	{
		return cv::Point2f((float)(v.panX + p.x * v.zoom), (float)(v.panY + p.y * v.zoom));
	}

	static cv::Point2f toImage(int x, int y, const ImageViewer::View& v)
	{
		return cv::Point2f((float)((x - v.panX) / v.zoom), (float)((y - v.panY) / v.zoom));
	}

	static int countUnsaved(const State& st)
	{
		int n = 0;

		for (size_t i = 0 ; i < st.trees.size() ; ++i)
		{
			n += (st.trees[i].mark != st.saved[i]) ? 1 : 0;
		}

		return n;
	}

	static void applyMark(State& st, Mark mark)
	{
		std::vector<std::pair<int, Mark>> op;

		for (int i = 0 ; i < (int)st.trees.size() ; ++i)
		{
			Tree& t = st.trees[i];

			if (t.selected && (t.mark != mark))
			{
				op.push_back({ i, t.mark });
				t.mark = mark;
			}
		}

		if (op.empty())
		{
			st.message = "Nothing to change";

			return;
		}

		const char* name = (mark == Mark::Keep) ? "keep" : toAction(mark);
		st.message = std::to_string(op.size()) + " -> " + name;

		st.undo.push_back(std::move(op));

		if (st.undo.size() > kUndoLimit)
		{
			st.undo.pop_front();
		}
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
			st.trees[e.first].mark = e.second;
		}

		st.message = "Undo " + std::to_string(st.undo.back().size());
		st.undo.pop_back();
	}

	static void save(State& st)
	{
		std::map<long long, TreeExclusion::Entry> out;
		std::unordered_set<long long> shown;

		for (auto& t : st.trees)
		{
			shown.insert(t.id);

			if (t.mark == Mark::Keep)
			{
				continue;
			}

			auto it = st.fileEntries.find(t.id);

			TreeExclusion::Entry e;
			e.action = toAction(t.mark);
			e.note = (it != st.fileEntries.end()) ? it->second.note : "";

			out[t.id] = e;
		}

		// entries for TreeIDs not in the tree_info CSV are kept as they were
		for (auto& kv : st.fileEntries)
		{
			if (!shown.count(kv.first))
			{
				out[kv.first] = kv.second;
			}
		}

		TreeExclusion::save(st.path, out);

		st.fileEntries = out;

		for (size_t i = 0 ; i < st.trees.size() ; ++i)
		{
			st.saved[i] = st.trees[i].mark;
		}

		st.message = "Saved " + std::to_string(out.size()) + " rows";
		std::cout << "  Saved " << out.size() << " rows: " << st.path << "\n";
	}

	// true = quit now
	static bool requestQuit(State& st)
	{
		int unsaved = countUnsaved(st);

		if ((unsaved > 0) && !st.quitArmed)
		{
			st.quitArmed = true;
			st.message = "Unsaved changes: " + std::to_string(unsaved) + " (S = save, Q/Esc again = quit)";

			std::cout << "  ! 저장하지 않은 변경 " << unsaved << "건 - S 로 저장하거나, 한 번 더 누르면 저장하지 않고 종료합니다.\n";

			return false;
		}

		return true;
	}

	static void toggleNearest(State& st, int x, int y, const ImageViewer::View& v)
	{
		int best = -1;
		double bestD = kPickRadius;

		for (int i = 0 ; i < (int)st.trees.size() ; ++i)
		{
			cv::Point2f s = toScreen(st.trees[i].px, v);
			double d = std::hypot(s.x - x, s.y - y);

			if (d <= bestD)
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
			if (!t.selected && (t.px.x >= p0.x) && (t.px.x <= p1.x) && (t.px.y >= p0.y) && (t.px.y <= p1.y))
			{
				t.selected = true;
				++n;
			}
		}

		st.message = "Rectangle: +" + std::to_string(n);
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
			if (!t.selected && (cv::pointPolygonTest(st.lasso, t.px, false) >= 0))
			{
				t.selected = true;
				++n;
			}
		}

		st.message = "Lasso: +" + std::to_string(n);
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

		// Click (not a pan drag): lasso vertex or toggle nearest
		if ((event == cv::EVENT_LBUTTONUP) && (std::abs(x - st.downPt.x) <= kClickSlop) && (std::abs(y - st.downPt.y) <= kClickSlop))
		{
			st.quitArmed = false;

			if (st.lassoMode)
			{
				st.lasso.push_back(toImage(x, y, v));
			}
			else
			{
				toggleNearest(st, x, y, v);
			}
		}

		return false;		// the viewer keeps panning / tracking
	}

	static bool onKey(State& st, int key)
	{
		int k = key & 0xFF;

		if ((k == kEsc) && (st.lassoMode || st.rectActive))
		{
			st.lassoMode = false;
			st.lasso.clear();
			st.rectActive = false;
			st.message = "Cancelled";

			return true;
		}

		if ((k == kEsc) || (k == 'q') || (k == 'Q'))
		{
			return !requestQuit(st);		// false -> viewer quits
		}

		st.quitArmed = false;

		if (key == kDeleteKey)
		{
			applyMark(st, Mark::Remove);

			return true;
		}

		if (k == kCtrlZ)
		{
			undoLast(st);

			return true;
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

			case 'a':
			case 'A':
				for (auto& t : st.trees)
				{
					t.selected = false;
				}
				st.message = "Selection cleared";
				return true;

			case 'i':
			case 'I':
				for (auto& t : st.trees)
				{
					t.selected = !t.selected;
				}
				st.message = "Selection inverted";
				return true;

			case 'u':
			case 'U':
				applyMark(st, Mark::Keep);
				return true;

			case 'f':
			case 'F':
				applyMark(st, Mark::AutoForest);
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
		const cv::Scalar kKeepColor(0, 220, 0);
		const cv::Scalar kRemoveColor(0, 0, 255);
		const cv::Scalar kForestColor(255, 140, 0);
		const cv::Scalar kSelectColor(0, 255, 255);

		int W = canvas.cols;
		int H = canvas.rows;
		int counts[3] = { 0, 0, 0 };
		int selected = 0;

		for (auto& t : st.trees)
		{
			counts[(int)t.mark] += 1;
			selected += t.selected ? 1 : 0;

			cv::Point2f s = toScreen(t.px, v);
			int r = (int)(t.rPx * v.zoom);

			if ((s.x < -r - 4) || (s.y < -r - 4) || (s.x > W + r + 4) || (s.y > H + r + 4))
			{
				continue;
			}

			cv::Point c((int)s.x, (int)s.y);
			cv::Scalar col = (t.mark == Mark::Remove) ? kRemoveColor : (t.mark == Mark::AutoForest) ? kForestColor : kKeepColor;

			if (r < kMinCrownPx)
			{
				cv::circle(canvas, c, 1, col, -1);

				if (t.selected)
				{
					cv::circle(canvas, c, 4, kSelectColor, 1);
				}

				continue;
			}

			cv::circle(canvas, c, r, col, 1, cv::LINE_AA);
			cv::circle(canvas, c, 2, col, -1, cv::LINE_AA);

			if (t.mark == Mark::Remove)
			{
				int d = std::max(3, (int)(r * 0.7));

				cv::line(canvas, c + cv::Point(-d, -d), c + cv::Point(d, d), kRemoveColor, 2, cv::LINE_AA);
				cv::line(canvas, c + cv::Point(-d, d), c + cv::Point(d, -d), kRemoveColor, 2, cv::LINE_AA);
			}

			if (t.selected)
			{
				cv::circle(canvas, c, r + 3, kSelectColor, 2, cv::LINE_AA);
			}
		}

		if (st.rectActive)
		{
			cv::rectangle(canvas, st.downPt, st.mousePt, kSelectColor, 1, cv::LINE_AA);
		}

		if (st.lassoMode && !st.lasso.empty())
		{
			std::vector<cv::Point> pts;

			for (auto& p : st.lasso)
			{
				cv::Point2f s = toScreen(p, v);
				pts.push_back(cv::Point((int)s.x, (int)s.y));
			}

			cv::polylines(canvas, pts, false, kSelectColor, 1, cv::LINE_AA);
			cv::line(canvas, pts.back(), st.mousePt, kSelectColor, 1, cv::LINE_AA);
			cv::line(canvas, st.mousePt, pts.front(), kSelectColor, 1, cv::LINE_AA);

			for (auto& p : pts)
			{
				cv::circle(canvas, p, 3, kSelectColor, -1, cv::LINE_AA);
			}
		}

		// Panel below the viewer's Zoom / Px / BGR lines
		std::string mode = st.lassoMode ? "LASSO (" + std::to_string(st.lasso.size()) + " pts, Enter / Esc)" : st.rectActive ? "RECT" : "NORMAL";
		int unsaved = countUnsaved(st);

		std::vector<std::string> lines =
		{
			"Total " + std::to_string(st.trees.size()) + "  Keep " + std::to_string(counts[(int)Mark::Keep]) + "  Removed " + std::to_string(counts[(int)Mark::Remove])
				+ "  AutoForest " + std::to_string(counts[(int)Mark::AutoForest]) + "  Selected " + std::to_string(selected),
			"Mode: " + mode,
			(unsaved > 0) ? "* unsaved changes: " + std::to_string(unsaved) : "saved",
		};

		if (!st.message.empty())
		{
			lines.push_back(st.message);
		}

		int x = 10;
		int y = 104;
		int lineH = 22;
		cv::Rect box(x - 4, y - 18, 640, (int)lines.size() * lineH + 8);
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
