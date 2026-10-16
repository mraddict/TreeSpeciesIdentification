#pragma once

// ----------------------------------------------------------------------------------------------------------------------------
//  TreeEditor.h
//  False-detection editor for a tree_info CSV, built on ImageViewer (pan / zoom / reset).
//  Saves TreeExclusion files (<course>_tree_info_excluded.csv); the tree_info CSV is never modified.
//
//  Mouse:  left drag = pan, wheel = zoom, left click = toggle nearest tree (15 px), Ctrl + click = unselect nearest
//          Shift + left drag = rectangle select (adds), Ctrl + Shift + drag = rectangle unselect
//  Keys:   L = lasso (click vertices, Enter = select, Backspace = unselect, Esc = cancel)
//          A = clear selection, I = invert selection
//          Delete = remove, U = keep, F = auto_forest, G = no_image, Ctrl+Z = undo
//          C = toggle clearSelectionAfterAction (default on: Delete / U / F / G clear the selection)
//          S = save, R = reset view, Q / Esc = quit (press twice if unsaved)
//  B     = select removed trees inside the forest map (InsideForest == Y; scope = rect / lasso area, otherwise all)
//  Review: X = show removed trees only / all
//          N / P = next / previous removed tree (order of the file at start); centred, zoomed and selected,
//                  drawn from the full-resolution file when aerialPath is given. U restores it.
//  Assist: V = add "bright, low saturation" kept trees to the selection (never removes; Delete does)
//              scope = last rectangle / lasso area (cleared by A), otherwise the visible screen
//              crown mean HSV: V >= brightMin and S <= satMax.  [ / ] = brightMin -/+ 5, { / } = satMax -/+ 5
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
	// aerialPath : original file for full-resolution close-ups in review (N / P); empty = downscaled image only
	// insideForest : TreeID (no offset) -> InsideForest from features_dino.csv, for B; nullptr = unknown
	static void run(const AerialPhoto& aerial, const std::vector<TreePoint>& trees, const std::string& exclusionPath, const std::string& title,
		const std::string& aerialPath = "", const std::unordered_map<long long, bool>* insideForest = nullptr)
	{
		State st;
		st.path = exclusionPath;
		st.fileEntries = TreeExclusion::load(exclusionPath);
		st.aerialPath = aerialPath;
		st.loadScale = (aerial.loadScale > 0.0) ? aerial.loadScale : 1.0;
		st.image = &aerial.image;

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
			t.height = (float)tp.height;
			t.crownD = (float)tp.crownD;

			if (insideForest)
			{
				auto f = insideForest->find(tp.treeId);
				t.forest = (f == insideForest->end()) ? -1 : (f->second ? 1 : 0);
			}

			auto it = st.fileEntries.find(tp.treeId);

			if (it != st.fileEntries.end())
			{
				t.mark = fromAction(it->second.action);
				restored += (t.mark != Mark::Keep) ? 1 : 0;
			}

			st.trees.push_back(t);
		}

		st.saved.resize(st.trees.size());
		st.crownColor.assign(st.trees.size(), cv::Vec2f(-1.0f, -1.0f));

		for (size_t i = 0 ; i < st.trees.size() ; ++i)
		{
			st.saved[i] = st.trees[i].mark;

			if (st.trees[i].mark == Mark::Remove)
			{
				st.reviewList.push_back((int)i);
			}
		}

		ImageViewer::Options opt;
		opt.title = title;
		opt.printControls = false;
		opt.hint = "Click  Ctrl+click:unselect  Shift+drag  L:lasso  A:clear  Del:remove  U:keep  F:auto_forest  G:no_image  B:removed in forest  C:auto-clear  Ctrl+Z:undo  S:save  X  N/P  V  R  Q";
		st.hasForestFlags = (insideForest != nullptr) && !insideForest->empty();

		st.windowW = opt.windowW;
		st.windowH = opt.windowH;

		opt.drawOverlay = [&st](cv::Mat& canvas, const ImageViewer::View& v) { drawOverlay(st, canvas, v); };
		opt.onMouse = [&st](int event, int x, int y, int flags, const ImageViewer::View& v) { return onMouse(st, event, x, y, flags, v); };
		opt.onKey = [&st](int key) { return onKey(st, key); };
		opt.overrideView = [&st](ImageViewer::View& v) { return focusView(st, v); };

		std::cout << "=== Tree editor ===\n"
			<< "  Trees: " << st.trees.size() << ", restored from file: " << restored << " (" << exclusionPath << ")\n"
			<< "  Left drag     : pan,  wheel: zoom,  R: reset view\n"
			<< "  Left click    : toggle nearest tree (within 15 px),  Ctrl + click: unselect nearest\n"
			<< "  Shift + drag  : rectangle select (adds),  Ctrl + Shift + drag: rectangle unselect\n"
			<< "  L             : lasso (click vertices, Enter = select, Backspace = unselect, Esc = cancel)\n"
			<< "  A / I         : clear / invert selection\n"
			<< "  Delete/U/F/G  : remove / keep / auto_forest / no_image,  Ctrl+Z: undo\n"
			<< "  C             : clear selection after Delete/U/F/G on/off (now " << (st.clearSelectionAfterAction ? "on" : "off") << ")\n"
			<< "  B             : select removed trees inside the forest map (rect/lasso area, otherwise all)" << (st.hasForestFlags ? "" : " - no InsideForest flags") << "\n"
			<< "  S             : save\n"
			<< "  X             : show removed trees only / all\n"
			<< "  N / P         : next / previous removed tree (" << st.reviewList.size() << "), zoomed" << (aerialPath.empty() ? "" : " at full resolution") << ", U = restore\n"
			<< "  V             : add bright, low-saturation kept trees in the rect/lasso area (or screen) to the selection - Delete to remove\n"
			<< "  [ / ]  { / }  : bright min -/+ 5,  saturation max -/+ 5 (HSV 0-255)\n"
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
		AutoForest,
		NoImage,
		Count
	};

	struct Tree
	{
		long long id = -1;
		cv::Point2f px;						// image pixel (of the loaded, possibly downscaled, image)
		float rPx = 1.0f;					// crown radius in image pixels
		float height = 0.0f;				// m
		float crownD = 0.0f;				// m
		int forest = -1;					// InsideForest: 1 = Y, 0 = N, -1 = unknown
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
		bool rectUnselect = false;							// Ctrl was held when the rectangle started
		cv::Point downPt;
		cv::Point mousePt;

		bool lassoMode = false;
		std::vector<cv::Point2f> lasso;					// image pixels

		// review
		bool removedOnly = false;							// X: draw / pick removed trees only
		std::vector<int> reviewList;						// tree indices marked remove at start (N / P order)
		int reviewPos = -1;
		bool focusPending = false;							// centre the view on reviewList[reviewPos] next frame
		int windowW = 1280;
		int windowH = 960;

		std::string aerialPath;							// original file for close-ups
		double loadScale = 1.0;								// loaded image / file
		cv::Mat hires;										// full-resolution window around the current review tree
		cv::Rect2f hiresRect;								// its extent in loaded-image pixels

		// V: bright / low-saturation selection assist
		const cv::Mat* image = nullptr;					// loaded (downscaled) aerial image, BGR
		std::vector<cv::Vec2f> crownColor;					// per tree: mean HSV (V, S) in the crown circle, -1 = not computed
		int brightMin = kDefaultBrightMin;
		int satMax = kDefaultSatMax;
		std::vector<cv::Point2f> region;					// last rectangle / lasso area (image px), V scope
		std::string regionKind;
		ImageViewer::View lastView{ 1.0, 0.0, 0.0 };

		bool clearSelectionAfterAction = true;			// Delete / U / F / G clear the selection (C toggles)
		bool hasForestFlags = false;						// InsideForest known (B)

		std::string message;
	};

	static constexpr int kDeleteKey = 0x2E0000;		// waitKeyEx code on Windows
	static constexpr int kCtrlZ = 26;
	static constexpr int kEnter = 13;
	static constexpr int kBackspace = 8;
	static constexpr int kEsc = 27;
	static constexpr int kClickSlop = 4;					// px: larger movement is a pan, not a click
	static constexpr int kPickRadius = 15;				// px on screen
	static constexpr int kMinCrownPx = 3;				// below this screen radius only the center dot is drawn
	static constexpr size_t kUndoLimit = 1000;
	static constexpr double kFocusZoom = 8.0;				// screen px per loaded-image px for N / P (1/4 load: 2x the original)
	static constexpr int kHiresHalf = 768;				// full-resolution window half size (file px) read for N / P
	static constexpr int kDefaultBrightMin = 170;		// V: crown mean HSV value >= this (0-255)
	static constexpr int kDefaultSatMax = 60;			// V: crown mean HSV saturation <= this (0-255)
	static constexpr int kThresholdStep = 5;

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

		if (action == TreeExclusion::kNoImage)
		{
			return Mark::NoImage;
		}

		return Mark::Keep;
	}

	static const char* toAction(Mark m)
	{
		switch (m)
		{
			case Mark::Remove:		return TreeExclusion::kRemove;
			case Mark::AutoForest:	return TreeExclusion::kAutoForest;
			case Mark::NoImage:		return TreeExclusion::kNoImage;
			default:				return "keep";
		}
	}

	static cv::Point2f toScreen(const cv::Point2f& p, const ImageViewer::View& v)
	{
		return cv::Point2f((float)(v.panX + p.x * v.zoom), (float)(v.panY + p.y * v.zoom));
	}

	static cv::Point2f toImage(int x, int y, const ImageViewer::View& v)
	{
		return cv::Point2f((float)((x - v.panX) / v.zoom), (float)((y - v.panY) / v.zoom));
	}

	// X mode hides (and makes unpickable) everything but removed trees
	static bool isVisible(const State& st, const Tree& t)
	{
		return !st.removedOnly || (t.mark == Mark::Remove);
	}

	// N / P: move through reviewList, select only that tree, centre on it next frame
	static void stepReview(State& st, int dir)
	{
		if (st.reviewList.empty())
		{
			st.message = "No removed trees to review";

			return;
		}

		int n = (int)st.reviewList.size();
		st.reviewPos = (st.reviewPos < 0) ? ((dir > 0) ? 0 : n - 1) : (st.reviewPos + dir + n) % n;

		for (auto& t : st.trees)
		{
			t.selected = false;
		}

		Tree& t = st.trees[st.reviewList[st.reviewPos]];
		t.selected = true;
		st.focusPending = true;

		loadHires(st, t);

		st.message = "Review " + std::to_string(st.reviewPos + 1) + "/" + std::to_string(n) + (st.hires.empty() ? "  (downscaled image)" : "  (full resolution)");
	}

	static void loadHires(State& st, const Tree& t)
	{
		st.hires.release();

		if (st.aerialPath.empty())
		{
			return;
		}

		double inv = 1.0 / st.loadScale;
		int fx = (int)std::lround(t.px.x * inv);
		int fy = (int)std::lround(t.px.y * inv);
		cv::Rect r;

		st.hires = AerialLoader::readWindow(st.aerialPath, cv::Rect(fx - kHiresHalf, fy - kHiresHalf, kHiresHalf * 2, kHiresHalf * 2), &r);
		st.hiresRect = cv::Rect2f((float)(r.x * st.loadScale), (float)(r.y * st.loadScale), (float)(r.width * st.loadScale), (float)(r.height * st.loadScale));
	}

	static bool focusView(State& st, ImageViewer::View& v)
	{
		if (!st.focusPending || (st.reviewPos < 0))
		{
			return false;
		}

		st.focusPending = false;

		const Tree& t = st.trees[st.reviewList[st.reviewPos]];
		v.zoom = kFocusZoom;
		v.panX = st.windowW * 0.5 - t.px.x * v.zoom;
		v.panY = st.windowH * 0.5 - t.px.y * v.zoom;

		return true;
	}

	// Full-resolution window drawn over the downscaled image (only the part on screen)
	static void drawHires(const State& st, cv::Mat& canvas, const ImageViewer::View& v)
	{
		if (st.hires.empty())
		{
			return;
		}

		cv::Point2f s0 = toScreen(st.hiresRect.tl(), v);
		cv::Point2f s1 = toScreen(st.hiresRect.br(), v);
		cv::Rect dst((int)std::floor(s0.x), (int)std::floor(s0.y), (int)std::ceil(s1.x - s0.x), (int)std::ceil(s1.y - s0.y));
		cv::Rect vis = dst & cv::Rect(0, 0, canvas.cols, canvas.rows);

		if ((vis.width <= 0) || (vis.height <= 0) || (dst.width <= 0) || (dst.height <= 0))
		{
			return;
		}

		// source sub-rect of hires matching the visible part
		double sx = (double)st.hires.cols / dst.width;
		double sy = (double)st.hires.rows / dst.height;
		cv::Rect src((int)((vis.x - dst.x) * sx), (int)((vis.y - dst.y) * sy), std::max(1, (int)std::ceil(vis.width * sx)), std::max(1, (int)std::ceil(vis.height * sy)));
		src &= cv::Rect(0, 0, st.hires.cols, st.hires.rows);

		if ((src.width <= 0) || (src.height <= 0))
		{
			return;
		}

		cv::Mat part;
		cv::resize(st.hires(src), part, vis.size(), 0, 0, (sx < 1.0) ? cv::INTER_LINEAR : cv::INTER_AREA);
		part.copyTo(canvas(vis));
	}

	// Mean HSV (V, S) inside the crown circle on the loaded image (cached)
	static cv::Vec2f crownColorOf(State& st, int i)
	{
		cv::Vec2f& c = st.crownColor[i];

		if (c[0] >= 0.0f)
		{
			return c;
		}

		const Tree& t = st.trees[i];
		const cv::Mat& img = *st.image;
		int r = std::max(2, (int)std::lround(t.rPx));
		cv::Rect box = cv::Rect((int)t.px.x - r, (int)t.px.y - r, 2 * r + 1, 2 * r + 1) & cv::Rect(0, 0, img.cols, img.rows);

		c = cv::Vec2f(0.0f, 255.0f);			// outside the image: never "bright"

		if (box.area() > 0)
		{
			cv::Mat hsv;
			cv::cvtColor(img(box), hsv, cv::COLOR_BGR2HSV);

			cv::Mat mask = cv::Mat::zeros(box.size(), CV_8U);
			cv::circle(mask, cv::Point((int)t.px.x - box.x, (int)t.px.y - box.y), r, cv::Scalar(255), -1);

			cv::Scalar m = cv::mean(hsv, mask);
			c = cv::Vec2f((float)m[2], (float)m[1]);
		}

		return c;
	}

	// V: add kept, visible, unselected trees with a bright, low-saturation crown inside the scope to the selection
	static void selectBright(State& st)
	{
		std::vector<cv::Point2f> scope = st.region;
		std::string scopeName = st.regionKind;

		if (scope.empty())
		{
			cv::Point2f a = toImage(0, 0, st.lastView);
			cv::Point2f b = toImage(st.windowW, st.windowH, st.lastView);

			scope = { a, cv::Point2f(b.x, a.y), b, cv::Point2f(a.x, b.y) };
			scopeName = "screen";
		}

		int candidates = 0;
		int added = 0;

		for (int i = 0 ; i < (int)st.trees.size() ; ++i)
		{
			Tree& t = st.trees[i];

			if ((t.mark != Mark::Keep) || t.selected || !isVisible(st, t) || (cv::pointPolygonTest(scope, t.px, false) < 0))
			{
				continue;
			}

			++candidates;

			cv::Vec2f c = crownColorOf(st, i);

			if ((c[0] >= st.brightMin) && (c[1] <= st.satMax))
			{
				t.selected = true;
				++added;
			}
		}

		st.message = "V (" + scopeName + "): +" + std::to_string(added) + " selected of " + std::to_string(candidates) + " kept trees - Delete to remove";
	}

	static void setRegion(State& st, const std::vector<cv::Point2f>& poly, const char* kind)
	{
		st.region = poly;
		st.regionKind = kind;
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

		// the next selection starts empty instead of merging with this one
		int cleared = st.clearSelectionAfterAction ? clearSelection(st) : 0;
		std::string clearedText = (cleared > 0) ? "  (selection cleared: " + std::to_string(cleared) + ")" : "";

		if (op.empty())
		{
			st.message = "Nothing to change" + clearedText;

			return;
		}

		st.message = std::to_string(op.size()) + " -> " + toAction(mark) + clearedText;

		st.undo.push_back(std::move(op));

		if (st.undo.size() > kUndoLimit)
		{
			st.undo.pop_front();
		}
	}

	// Returns the number of trees that were selected
	static int clearSelection(State& st)
	{
		int n = 0;

		for (auto& t : st.trees)
		{
			n += t.selected ? 1 : 0;
			t.selected = false;
		}

		return n;
	}

	// B: removed trees inside the forest map (InsideForest == Y) -> selection; scope = rect / lasso area, otherwise the whole course
	static void selectRemovedInForest(State& st)
	{
		if (!st.hasForestFlags)
		{
			st.message = "B: no InsideForest flags (out/features_dino.csv missing)";

			return;
		}

		int added = 0;
		int candidates = 0;

		for (auto& t : st.trees)
		{
			if ((t.mark != Mark::Remove) || (t.forest != 1) || !isVisible(st, t))
			{
				continue;
			}

			if (!st.region.empty() && (cv::pointPolygonTest(st.region, t.px, false) < 0))
			{
				continue;
			}

			++candidates;

			if (!t.selected)
			{
				t.selected = true;
				++added;
			}
		}

		st.message = "B (" + (st.region.empty() ? std::string("all") : st.regionKind) + "): removed + InsideForest Y " + std::to_string(candidates) + ", +" + std::to_string(added) + " selected - G = no_image";
	}

	// Ctrl + click: unselect the nearest selected tree
	static void unselectNearest(State& st, int x, int y, const ImageViewer::View& v)
	{
		int best = -1;
		double bestD = kPickRadius;

		for (int i = 0 ; i < (int)st.trees.size() ; ++i)
		{
			if (!st.trees[i].selected)
			{
				continue;
			}

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
			st.trees[best].selected = false;
			st.message = "TreeID " + std::to_string(st.trees[best].id) + " unselected";
		}
		else
		{
			st.message = "No selected tree within " + std::to_string(kPickRadius) + " px";
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
			if (!isVisible(st, st.trees[i]))
			{
				continue;
			}

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

	// unselect = false: add visible trees in the area to the selection (and make it the V / B scope); true: remove them from it
	static void selectInRect(State& st, const cv::Point& a, const cv::Point& b, const ImageViewer::View& v, bool unselect)
	{
		cv::Point2f p0 = toImage(std::min(a.x, b.x), std::min(a.y, b.y), v);
		cv::Point2f p1 = toImage(std::max(a.x, b.x), std::max(a.y, b.y), v);
		std::vector<cv::Point2f> poly = { p0, cv::Point2f(p1.x, p0.y), p1, cv::Point2f(p0.x, p1.y) };

		int n = setSelectionInArea(st, poly, !unselect);

		if (!unselect)
		{
			setRegion(st, poly, "rect");
		}

		st.message = std::string("Rectangle: ") + (unselect ? "-" : "+") + std::to_string(n);
	}

	static void selectInLasso(State& st, bool unselect)
	{
		if (st.lasso.size() < 3)
		{
			st.message = "Lasso needs at least 3 points";

			return;
		}

		int n = setSelectionInArea(st, st.lasso, !unselect);

		if (!unselect)
		{
			setRegion(st, st.lasso, "lasso");
		}

		st.message = std::string("Lasso: ") + (unselect ? "-" : "+") + std::to_string(n);
	}

	// Sets 'selected' of visible trees inside poly; returns how many changed
	static int setSelectionInArea(State& st, const std::vector<cv::Point2f>& poly, bool select)
	{
		int n = 0;

		for (auto& t : st.trees)
		{
			if ((t.selected != select) && isVisible(st, t) && (cv::pointPolygonTest(poly, t.px, false) >= 0))
			{
				t.selected = select;
				++n;
			}
		}

		return n;
	}

	static bool onMouse(State& st, int event, int x, int y, int flags, const ImageViewer::View& v)
	{
		st.mousePt = cv::Point(x, y);

		// Shift + drag: rectangle select, Ctrl + Shift + drag: rectangle unselect (consumed so the viewer does not pan)
		if ((event == cv::EVENT_LBUTTONDOWN) && (flags & cv::EVENT_FLAG_SHIFTKEY) && !st.lassoMode)
		{
			st.rectActive = true;
			st.rectUnselect = (flags & cv::EVENT_FLAG_CTRLKEY) != 0;
			st.downPt = cv::Point(x, y);

			return true;
		}

		if (st.rectActive)
		{
			if (event == cv::EVENT_LBUTTONUP)
			{
				st.rectActive = false;
				selectInRect(st, st.downPt, cv::Point(x, y), v, st.rectUnselect);
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
			else if (flags & cv::EVENT_FLAG_CTRLKEY)
			{
				unselectNearest(st, x, y, v);
			}
			else
			{
				toggleNearest(st, x, y, v);
			}
		}

		return false;		// the viewer keeps panning / tracking
	}

	// Hangul input mode sends compatibility jamo (U+3131..U+3163) instead of letters: map back to the 2-beolsik QWERTY key,
	// otherwise e.g. G (U+314E) would be read as 'N' after & 0xFF
	static int fromHangulJamo(int key)
	{
		static const std::map<int, char> m =
		{
			{ 0x3131, 'r' }, { 0x3132, 'R' }, { 0x3134, 's' }, { 0x3137, 'e' }, { 0x3138, 'E' }, { 0x3139, 'f' }, { 0x3141, 'a' },
			{ 0x3142, 'q' }, { 0x3143, 'Q' }, { 0x3145, 't' }, { 0x3146, 'T' }, { 0x3147, 'd' }, { 0x3148, 'w' }, { 0x3149, 'W' },
			{ 0x314A, 'c' }, { 0x314B, 'z' }, { 0x314C, 'x' }, { 0x314D, 'v' }, { 0x314E, 'g' }, { 0x314F, 'k' }, { 0x3150, 'o' },
			{ 0x3151, 'i' }, { 0x3152, 'O' }, { 0x3153, 'j' }, { 0x3154, 'p' }, { 0x3155, 'u' }, { 0x3156, 'P' }, { 0x3157, 'h' },
			{ 0x315B, 'y' }, { 0x315C, 'n' }, { 0x3160, 'b' }, { 0x3161, 'm' }, { 0x3163, 'l' },
		};

		auto it = m.find(key);

		return (it != m.end()) ? it->second : key;
	}

	static bool onKey(State& st, int key)
	{
		key = fromHangulJamo(key);
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

		// lasso: Enter = select inside, Backspace = unselect inside
		if (((k == kEnter) || (k == kBackspace)) && st.lassoMode)
		{
			selectInLasso(st, k == kBackspace);
			st.lassoMode = false;
			st.lasso.clear();

			return true;
		}

		if (k == kEnter)
		{
			return true;
		}

		switch (k)
		{
			case 'l':
			case 'L':
				st.lassoMode = true;
				st.lasso.clear();
				st.message = "Lasso: click vertices, Enter = select, Backspace = unselect, Esc = cancel";
				return true;

			case 'a':
			case 'A':
			{
				int n = clearSelection(st);
				st.region.clear();
				st.regionKind.clear();
				st.message = "Selection cleared (" + std::to_string(n) + ")";
				return true;
			}

			case 'g':
			case 'G':
				applyMark(st, Mark::NoImage);
				return true;

			case 'b':
			case 'B':
				selectRemovedInForest(st);
				return true;

			case 'c':
			case 'C':
				st.clearSelectionAfterAction = !st.clearSelectionAfterAction;
				st.message = std::string("clearSelectionAfterAction = ") + (st.clearSelectionAfterAction ? "true" : "false");
				return true;

			case 'v':
			case 'V':
				selectBright(st);
				return true;

			case '[':
			case ']':
				st.brightMin = std::min(255, std::max(0, st.brightMin + ((k == ']') ? kThresholdStep : -kThresholdStep)));
				st.message = "Bright min " + std::to_string(st.brightMin);
				return true;

			case '{':
			case '}':
				st.satMax = std::min(255, std::max(0, st.satMax + ((k == '}') ? kThresholdStep : -kThresholdStep)));
				st.message = "Saturation max " + std::to_string(st.satMax);
				return true;

			case 'i':
			case 'I':
				for (auto& t : st.trees)
				{
					t.selected = isVisible(st, t) && !t.selected;
				}
				st.message = "Selection inverted";
				return true;

			case 'x':
			case 'X':
				st.removedOnly = !st.removedOnly;

				// hidden trees must not stay selected (Delete / U / F act on the selection)
				for (auto& t : st.trees)
				{
					t.selected = t.selected && isVisible(st, t);
				}
				st.message = st.removedOnly ? "Showing removed trees only" : "Showing all trees";
				return true;

			case 'n':
			case 'N':
				stepReview(st, +1);
				return true;

			case 'p':
			case 'P':
				stepReview(st, -1);
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
		const cv::Scalar kNoImageColor(170, 170, 170);
		const cv::Scalar kSelectColor(0, 255, 255);

		int W = canvas.cols;
		int H = canvas.rows;
		int counts[(int)Mark::Count] = {};
		int selCounts[(int)Mark::Count] = {};		// selected trees by mark
		int selected = 0;

		st.lastView = v;

		drawHires(st, canvas, v);

		for (auto& t : st.trees)
		{
			counts[(int)t.mark] += 1;
			selected += t.selected ? 1 : 0;
			selCounts[(int)t.mark] += t.selected ? 1 : 0;

			if (!isVisible(st, t))
			{
				continue;
			}

			cv::Point2f s = toScreen(t.px, v);
			int r = (int)(t.rPx * v.zoom);

			if ((s.x < -r - 4) || (s.y < -r - 4) || (s.x > W + r + 4) || (s.y > H + r + 4))
			{
				continue;
			}

			cv::Point c((int)s.x, (int)s.y);
			cv::Scalar col = (t.mark == Mark::Remove) ? kRemoveColor : (t.mark == Mark::AutoForest) ? kForestColor : (t.mark == Mark::NoImage) ? kNoImageColor : kKeepColor;

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

		// V scope (last rectangle / lasso area)
		if (!st.region.empty())
		{
			std::vector<cv::Point> pts;

			for (auto& p : st.region)
			{
				cv::Point2f s = toScreen(p, v);
				pts.push_back(cv::Point((int)s.x, (int)s.y));
			}

			cv::polylines(canvas, pts, true, cv::Scalar(200, 200, 120), 1, cv::LINE_AA);
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
				+ "  AutoForest " + std::to_string(counts[(int)Mark::AutoForest]) + "  NoImage " + std::to_string(counts[(int)Mark::NoImage]) + "  Selected " + std::to_string(selected),
			"Selected: removed " + std::to_string(selCounts[(int)Mark::Remove]) + " / kept " + std::to_string(selCounts[(int)Mark::Keep])
				+ ((selCounts[(int)Mark::AutoForest] > 0) ? " / auto_forest " + std::to_string(selCounts[(int)Mark::AutoForest]) : std::string())
				+ ((selCounts[(int)Mark::NoImage] > 0) ? " / no_image " + std::to_string(selCounts[(int)Mark::NoImage]) : std::string())
				+ "   (U -> keep " + std::to_string(selected - selCounts[(int)Mark::Keep]) + ", Del -> remove " + std::to_string(selected - selCounts[(int)Mark::Remove])
				+ ", G -> no_image " + std::to_string(selected - selCounts[(int)Mark::NoImage]) + ")",
			"Mode: " + mode + (st.removedOnly ? "   [X] removed only" : "") + "   clearSelectionAfterAction = " + (st.clearSelectionAfterAction ? "true" : "false") + " (C)",
			(unsaved > 0) ? "* unsaved changes: " + std::to_string(unsaved) : "saved",
			"V select: bright >= " + std::to_string(st.brightMin) + "  sat <= " + std::to_string(st.satMax)
				+ "   ([ ] bright, { } sat)   scope: " + (st.region.empty() ? std::string("screen") : st.regionKind),
		};

		if (st.reviewPos >= 0)
		{
			const Tree& t = st.trees[st.reviewList[st.reviewPos]];
			const char* state = toAction(t.mark);
			char buf[160];

			sprintf_s(buf, "Review %d/%d  TreeID %lld  H %.1f m  CrownD %.1f m  [%s]", st.reviewPos + 1, (int)st.reviewList.size(), t.id, t.height, t.crownD, state);
			lines.insert(lines.begin() + 1, buf);
		}

		if (!st.message.empty())
		{
			lines.push_back(st.message);
		}

		int x = 10;
		int y = 104;
		int lineH = 22;
		cv::Rect box(x - 4, y - 18, 1060, (int)lines.size() * lineH + 8);
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
