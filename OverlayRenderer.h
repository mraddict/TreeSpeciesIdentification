#pragma once

// ─────────────────────────────────────────────────────────
//  OverlayRenderer.h
//  Aerial photo + Forest polygon + Tree location overlay
//
//  Renders all three data layers onto the aerial photo
//  and displays using ImageViewer.
//
//  Dependencies:
//    - AerialLoader.h   (AerialPhoto)
//    - ForestLoader.h   (ForestLayer, ForestFeature)
//    - TreeLoader.h     (TreeData, TreePoint)
//    - ImageViewer.h    (interactive pan/zoom viewer)
//    - GDAL/OGR, OpenCV
// ─────────────────────────────────────────────────────────

#include "AerialLoader.h"
#include "ForestLoader.h"
#include "TreeLoader.h"
#include "ImageViewer.h"
#include "CoordTransformer.h"

#include "Hierarchicalclassifier.h"

// ─────────────────────────────────────────────
//  Overlay display options
// ─────────────────────────────────────────────

struct OverlayOptions
{
	// Coordinate system
	int aerialEpsg = 0;									// Aerial EPSG (0 = read from file)
	int forestEpsg = 0;									// Forest SHP EPSG (0 = same as aerial)
	int treeEpsg = 0;										// Tree CSV EPSG (0 = same as aerial)

	// Forest polygon rendering
	bool drawPolygons = true;
	double polyAlpha = 0.25;										// polygon fill transparency (0~1)
	int polyLineWidth = 2;											// polygon border thickness
	bool colorBySpecies = true;									// color polygons by KOFTR_GROU
	cv::Scalar polyBorderColor = { 128, 128, 128 };	// border color (used when polyAlpha=0)

	// Tree point rendering
	bool drawTrees = true;	
	bool drawCrown = true;							// draw crown circle
	bool drawCenter = true;							// draw center dot
	int centerRadius = 3;								// center dot radius (px)
	cv::Scalar centerColor = {0, 255, 255};	// yellow (BGR)
	double crownAlpha = 0.3;						// crown circle transparency
	double crownScale = 1.0;						// crown circle scale (1.0=actual size)
	cv::Scalar crownColor = {0, 200, 0};		// green (BGR)
	int crownOutlineWidth = 0;						// outline width (0=none)

	// Tree label display
	bool showTreeId = true;							// show TreeID text
	bool showHeight = false;							// show height text
	double minZoomFromLabel = 5.0;			// min zoom to show labels (relative)

	// Height-based tree coloring
	bool colorByHeight = false;						// color trees by height
	double heightColorMin = 0.0;					// min height for color scale
	double heightColorMax = 25.0;				// max height for color scale

	// ─── Classification result rendering ───
	enum ResultColorMode
	{
		RESULT_BY_SPECIES,						// color by predicted species code
		RESULT_BY_GROUP,						// color by predicted group (C/D/E/N)
		RESULT_CORRECTNESS					// green=correct, yellow=group ok, red=wrong
	};

	ResultColorMode resultColorMode = RESULT_BY_SPECIES;

	// Legend
	bool drawLegend = true;
	int legendX = 30;
	int legendY = 30;
	std::string resultTitle = "Classification results";

	// Viewer window
	int windowW = 1200;
	int windowH = 960;
	std::string title = "TSI Overlay Viewer";
};

class OverlayRenderer
{
public:

	// Render all layers onto aerial photo and show in viewer
	static void showOverlay(const AerialPhoto& aerialImg, const ForestLayer& forest, const TreeData& trees, const OverlayOptions& opts = OverlayOptions())
	{
		cv::Mat canvas = buildOverlay(aerialImg, forest, trees, opts);

		double scaleX = aerialImg.image.cols / (aerialImg.geoMaxX - aerialImg.geoMinX);

		ImageViewer::Options viewOpt;
		viewOpt.title = opts.title;
		viewOpt.windowW = opts.windowW;
		viewOpt.windowH = opts.windowH;
		viewOpt.showGeoCoord = true;
		viewOpt.geoMinX = aerialImg.geoMinX;
		viewOpt.geoMinY = aerialImg.geoMinY;
		viewOpt.geoMaxY = aerialImg.geoMaxY;
		viewOpt.meterPerPixel = 1.0 / scaleX;

		ImageViewer::show(canvas, viewOpt);
	}

	// Render overlay to file (no viewer)
	static void saveOverlay(const AerialPhoto& aerialImg, const ForestLayer& forest, const TreeData& trees, const std::string& outputPath, const OverlayOptions& opts = OverlayOptions())
	{
		cv::Mat canvas = buildOverlay(aerialImg, forest, trees, opts);

		cv::imwrite(outputPath, canvas);
		std::cout << "Overlay saved: " << outputPath << " (" << canvas.cols << "x" << canvas.rows << ")\n";
	}

	// Aerial + polygon borders + trees colored by prediction
	static void saveResultOverlay(
		const AerialPhoto& aerialImg, const ForestLayer& forest, const TreeData& trees,
		const std::vector<HierarchicalResult>& results,
		const std::string& outputPath,
		const OverlayOptions& opts = OverlayOptions())
	{
		std::cout << "=== Building result overlay ===\n";

		if (aerialImg.image.empty())
		{
			throw std::runtime_error("Aerial image is empty.");
		}

		cv::Mat canvas = aerialImg.image.clone();
		int imgW = canvas.cols;
		int imgH = canvas.rows;

		double scaleX = imgW / (aerialImg.geoMaxX - aerialImg.geoMinX);
		double scaleY = imgH / (aerialImg.geoMaxY - aerialImg.geoMinY);
		double pixPerMeter = scaleX;

		auto geoToPx = [&](double gx, double gy) -> cv::Point
		{
			int px = (int)(gx - aerialImg.geoMinX) * scaleX;
			int py = (int)(aerialImg.geoMaxY - gy) * scaleY;

			return cv::Point(px, py);
		};

		// Draw polygon borders
		if (opts.drawPolygons)
		{
			drawForestPolygons(canvas, imgW, imgH, forest, geoToPx, opts);
		}

		// Build result lookup
		std::map<long long, const HierarchicalResult*> resultLookup;

		for (auto& r : results)
		{
			resultLookup[r.treeId] = &r;
		}

		// Collect species info for legend
		std::map<std::string, cv::Scalar> legendColors;
		std::map<std::string, std::string> legendNames;
		std::map<std::string, int> legendCorrect;
		std::map<std::string, int> legendTotal;

		for (auto& r : results)
		{
			std::string key = (opts.resultColorMode == OverlayOptions::RESULT_BY_GROUP) ? r.predictedGroup : r.predictedCode;

			if (legendColors.find(key) == legendColors.end())
			{
				legendColors[key] = getSpeciesColor(key);
			}

			if (legendNames.find(key) == legendNames.end())
			{
				legendNames[key] = (opts.resultColorMode == OverlayOptions::RESULT_BY_GROUP) ? r.predictedGroupName : r.predictedName;
			}

			if (!r.trueCode.empty())
			{
				legendTotal[key]++;

				bool ok = (opts.resultColorMode == OverlayOptions::RESULT_BY_GROUP) ? r.groupCorrect : r.speciesCorrect;

				if (ok)
				{
					legendCorrect[key]++;
				}
			}
		}

		// Draw trees
		cv::Mat overlay = canvas.clone();
		int drawn = 0;
		int skipped = 0;

		for (auto& tp : trees.tps)
		{
			auto rit = resultLookup.find(tp.treeId);

			if (rit == resultLookup.end())
			{
				++skipped;

				continue;
			}

			const HierarchicalResult* r = rit->second;
			cv::Point center = geoToPx(tp.x, tp.y);

			if ((center.x < -50) || (center.x >= imgW + 50) || (center.y < -50) || (center.y >= imgH + 50))
			{
				++skipped;

				continue;
			}

			// Crown radius (reuse crownScale)
			int crownPxR = (int)(tp.crownD * 0.5 * pixPerMeter * opts.crownScale);

			if (crownPxR < 2)
			{
				crownPxR = 2;
			}

			// Color
			cv::Scalar color;

			if (opts.resultColorMode == OverlayOptions::RESULT_CORRECTNESS)
			{
				if (r->trueCode.empty())
				{
					color = cv::Scalar(200, 200, 0);
				}
				else if (r->speciesCorrect)
				{
					color = cv::Scalar(0, 200, 0);
				}
				else if (r->groupCorrect)
				{
					color = cv::Scalar(0, 200, 200);
				}
				else
				{
					color = cv::Scalar(0, 0, 200);
				}
			}
			else
			{
				std::string key = (opts.resultColorMode == OverlayOptions::RESULT_BY_GROUP) ? r->predictedGroup : r->predictedCode;
				auto cit = legendColors.find(key);

				color = (cit != legendColors.end()) ? cit->second : SPECIES_DEFAULT_COLOR;
			}

			// Filled circle on overlay
			cv::circle(overlay, center, crownPxR, color, -1);
			// Outline on canvas
			if (opts.crownOutlineWidth > 0)
			{
				cv::circle(canvas, center, crownPxR, color, opts.crownOutlineWidth);
			}

			++drawn;
		}

		// Blend (reuse crownAlpha)
		if (opts.crownAlpha > 0.01)
		{
			cv::addWeighted(overlay, opts.crownAlpha, canvas, 1.0 - opts.crownAlpha, 0, canvas);
		}

		std::cout << "  Trees drawn: " << drawn << ", skipped: " << skipped << "\n";

		// Legend
		if (opts.drawLegend && (opts.resultColorMode != OverlayOptions::RESULT_CORRECTNESS))
		{
			drawResultLegend(canvas, legendColors, legendNames, legendCorrect, legendTotal, opts);
		}
		else if (opts.drawLegend)
		{
			drawCorrectnessLegend(canvas, results, opts);
		}
	
		cv::imwrite(outputPath, canvas);

		std::cout << "  Saved: " << outputPath << " (" << imgW << "x" << imgH << ")\n";
	}

private:

	// ─── Core rendering ───
	static cv::Mat buildOverlay(const AerialPhoto& aerialImg, const ForestLayer& forest, const TreeData& trees, const OverlayOptions& opts)
	{
		if (aerialImg.image.empty())
		{
			throw std::runtime_error("Aerial image is empty.");
		}
	
		std::cout << "=== Building overlay ===\n";

		// Determine EPSG codes
		int aerialEpsg = (opts.aerialEpsg > 0) ? opts.aerialEpsg : aerialImg.epsg;
		int forestEpsg = (opts.forestEpsg > 0) ? opts.forestEpsg : forest.epsg;
		int treeEpsg = (opts.treeEpsg > 0) ? opts.treeEpsg : 0;

		std::cout << "  Aerial EPSG: " << aerialEpsg << "\n";

		if (forestEpsg > 0)
		{
			std::cout << "  Forest EPSG: " << forestEpsg << "\n";
		}

		if (treeEpsg > 0)
		{
			std::cout << "  Tree   EPSG: " << treeEpsg << "\n";
		}

		// Create coordinate transformers if EPSG differs
		OGRCoordinateTransformation* forestCT = nullptr;
		OGRCoordinateTransformation* treeCT = nullptr;

		if ((forestEpsg > 0) && (aerialEpsg > 0) && (forestEpsg != aerialEpsg))
		{
			forestCT = CoordTransformer::createTransform(forestEpsg, aerialEpsg);

			if (forestCT)
			{
				std::cout << "  Forest transform: EPSG:" << forestEpsg << " -> EPSG:" << aerialEpsg << "\n";
			}
			else
			{
				std::cerr << "  WARNING: Failed to create forest coordinate transformer!\n";
			}
		}

		if ((treeEpsg > 0) && (aerialEpsg > 0) && (treeEpsg != aerialEpsg))
		{
			treeCT = CoordTransformer::createTransform(treeEpsg, aerialEpsg);

			if (treeCT)
			{
				std::cout << "  Tree   transform: EPSG:" << treeEpsg << " -> EPSG:" << aerialEpsg << "\n";
			}
			else
			{
				std::cerr << "  WARNING: Failed to create tree coordinate transformer!\n";
			}
		}

		// Geo-to-pixel using inverse GeoTransform (handles rotation)
		// GeoTransform:
		//    Xgeo = gt[0] + px*gt[1] + py*gt[2]
		//    Ygeo = gt[3] + px*gt[4] + py*gt[5]}

		double gt[6] = { aerialImg.originX, aerialImg.pixelW, aerialImg.rotationX, aerialImg.originY, aerialImg.rotationY, aerialImg.pixelH };
		double igt[6];

		if (!computeInverseGT(gt, igt))
		{
			// Fallback: simple linear (no rotation)
			double scaleX = aerialImg.image.cols / (aerialImg.geoMaxX - aerialImg.geoMinX);
			double scaleY = aerialImg.image.rows / (aerialImg.geoMaxY - aerialImg.geoMinY);

			igt[0] = -aerialImg.geoMinX * scaleX;
			igt[1] = scaleX;
			igt[2] = 0.0;
			igt[3] = aerialImg.geoMaxY * scaleY;
			igt[4] = 0.0;
			igt[5] = -scaleY;

			std::cout << "  WARNING: GeoTransform inversion failed, using simple linear.\n";
		}
	
		//  Inverse:
		//    px = igt[0] + Xgeo*igt[1] + Ygeo*igt[2]
		//    py = igt[3] + Xgeo*igt[4] + Ygeo*igt[5]

		auto geoToPx = [&](double gx, double gy) -> cv::Point
		{
			int px = (int)(igt[0] + igt[1] * gx + igt[2] * gy);
			int py = (int)(igt[3] + igt[4] * gx + igt[5] * gy);

			return { px , py };
		};

		auto geoToPxF = [&](double gx, double gy) -> cv::Point2f
		{
			float px = (float)(igt[0] + igt[1] * gx + igt[2] * gy);
			float py = (float)(igt[3] + igt[4] * gx + igt[5] * gy);

			return { px , py };
		};

		// forest geo -> pixel (with optional CRS transform)
		auto forestGeoToPx = [&](double gx, double gy) -> cv::Point
		{
			if (forestCT)
			{
				double z = 0;
				forestCT->Transform(1, &gx, &gy, &z);
			}

			return geoToPx(gx, gy);
		};

		// tree geo -> pixel (with optional CRS transform)
		auto treeGeoToPxF = [&](double gx, double gy) -> cv::Point2f
		{
			if (treeCT)
			{
				double z = 0;
				treeCT->Transform(1, &gx, &gy, &z);
			}

			return geoToPxF(gx, gy);
		};

		// Debug: print coordinate mapping
		printCoordDebug(aerialImg, forest, trees, geoToPx, forestGeoToPx, treeGeoToPxF);

		// 
		cv::Mat canvas = aerialImg.image.clone();
		int imgW = canvas.cols;
		int imgH = canvas.rows;

		// 1. Draw forest polygons
		if (opts.drawPolygons && !forest.features.empty())
		{
			std::cout << "  Drawing " << forest.features.size() << " forest polygons...\n";

			drawForestPolygons(canvas, imgW, imgH, forest, forestGeoToPx, opts);
		}

		// 2. Draw tree points
		if (opts.drawTrees && !trees.tps.empty())
		{
			std::cout << "  Drawing " << trees.tps.size() << " trees...\n";

			double pixPerMeter = std::abs(igt[1]);
			drawTreePoints(canvas, imgW, imgH, trees, treeGeoToPxF, pixPerMeter, opts);
		}
	
		// Cleanup transformers
		if (forestCT)
		{
			OGRCoordinateTransformation::DestroyCT(forestCT);
		}

		if (treeCT)
		{
			OGRCoordinateTransformation::DestroyCT(treeCT);
		}

		std::cout << "  Overlay complete.\n";

		return canvas;
}

	// ─── Inverse GeoTransform ───
	static bool computeInverseGT(const double gt[6], double igt[6])
	{
		double det = gt[1] * gt[5] - gt[2] * gt[4];

		if (std::abs(det) < 1e-15)
		{
			return false;
		}

		double invDet = 1.0 / det;

		igt[1] = gt[5] * invDet;
		igt[2] = -gt[2] * invDet;
		igt[4] = -gt[4] * invDet;
		igt[5] = gt[1] * invDet;

		igt[0] = -(igt[1] * gt[0] + igt[2] * gt[3]);
		igt[3] = -(igt[4] * gt[0] + igt[5] * gt[3]);

		return true;
	}

	// ─── Debug: coordinate mapping verification ───
	static void printCoordDebug(const AerialPhoto& aerialImg, const ForestLayer& forest, const TreeData& trees, 
		std::function<cv::Point(double, double)> aerialGeoToPx,
		std::function<cv::Point(double, double)> forestGeoToPx,
		std::function<cv::Point2f(double, double)> treeGeoToPx)
	{
		std::cout << std::fixed << std::setprecision(3);
		std::cout << "  [Coordinate debug]\n";

		// Aerial corners
		cv::Point tlAPx = aerialGeoToPx(aerialImg.geoMinX, aerialImg.geoMaxY);
		cv::Point brAPx = aerialGeoToPx(aerialImg.geoMaxX, aerialImg.geoMinY);
	
		std::cout << "    Aerial  TL geo(" << aerialImg.geoMinX << ", " << aerialImg.geoMaxY << ") -> px(" << tlAPx.x << ", " << tlAPx.y << ")\n";
		std::cout << "    Aerial  BR geo(" << aerialImg.geoMaxX << ", " << aerialImg.geoMinY << ") -> px(" << brAPx.x << ", " << brAPx.y << ")\n";
		std::cout << "    Image size: " << aerialImg.image.cols << " x " << aerialImg.image.rows << "\n";

		// Forest bbox -> pixel
		cv::Point tlFPx = forestGeoToPx(forest.extent.MinX, forest.extent.MaxY);
		cv::Point brFPx = forestGeoToPx(forest.extent.MaxX, forest.extent.MinY);

		std::cout << "    Forest  TL geo(" << forest.extent.MinX << ", " << forest.extent.MaxY << ") -> px(" << tlFPx.x << ", " << tlFPx.y << ")";

		bool tlFIn = ((tlFPx.x >= 0) && (tlFPx.x < aerialImg.image.cols) && (tlFPx.y >= 0) && (tlFPx.y < aerialImg.image.rows));
	
		if (!tlFIn)
		{
			std::cout << " OUT";
		}

		std::cout << "\n";
		std::cout << "    Forest  BR geo(" << forest.extent.MaxX << ", " << forest.extent.MinY << ") -> px(" << brFPx.x << ", " << brFPx.y << ")";

		bool brFIn = ((brFPx.x >= 0) && (brFPx.x < aerialImg.image.cols) && (brFPx.y >= 0) && (brFPx.y < aerialImg.image.rows));

		if (!brFIn)
		{
			std::cout << " OUT";
		}

		std::cout << "\n";

		// Tree bbox -> pixel
		if (!trees.tps.empty())
		{
			cv::Point tlTPx = treeGeoToPx(trees.extent.MinX, trees.extent.MaxY);
			cv::Point brTPx = treeGeoToPx(trees.extent.MaxX, trees.extent.MinY);

			std::cout << "    Tree    TL geo(" << trees.extent.MinX << ", " << trees.extent.MaxY << ") -> px(" << tlTPx.x << ", " << tlTPx.y << ")";

			bool tlTIn = ((tlTPx.x >= 0) && (tlTPx.x < aerialImg.image.cols) && (tlTPx.y >= 0) && (tlTPx.y < aerialImg.image.rows));

			if (!tlTIn)
			{
				std::cout << " OUT";
			}

			std::cout << "\n";
			std::cout << "    Tree    BR geo(" << trees.extent.MaxX << ", " << trees.extent.MinY << ") -> px(" << brTPx.x << ", " << brTPx.y << ")";

			bool brTIn = ((brTPx.x >= 0) && (brTPx.x < aerialImg.image.cols) && (brTPx.y >= 0) && (brTPx.y < aerialImg.image.rows));

			if (!brTIn)
			{
				std::cout << " OUT";
			}

			std::cout << "\n";
		}

		std::cout << std::setprecision(6);
	}

	// ─── Draw forest polygons ───
	static void drawForestPolygons(cv::Mat& canvas, int imgW, int imgH, const ForestLayer& forest, std::function<cv::Point(double, double)> geoToPx, const OverlayOptions& opts)
	{
		// Single overlay for ALL polygons - blend once at end
		cv::Mat overlay;

		if (opts.polyAlpha > 0.01)
		{
			overlay = canvas.clone();
		}

		int drawn = 0;
		int clipped = 0;

		for (auto& ft : forest.features)
		{
			if (!ft.geometry)
			{
				continue;
			}

			// Quick bbox reject: skip polygons entirely outside image
			OGREnvelope env;
			ft.geometry->getEnvelope(&env);

			cv::Point tlEnv = geoToPx(env.MinX, env.MaxY);
			cv::Point brEnv = geoToPx(env.MaxX, env.MinY);

			int eMinX = std::min(tlEnv.x, brEnv.x);
			int eMaxX = std::max(tlEnv.x, brEnv.x);
			int eMinY = std::min(tlEnv.y, brEnv.y);
			int eMaxY = std::max(tlEnv.y, brEnv.y);

			if ((eMaxX < 0) || (eMinX >= imgW) || (eMaxY < 0) || (eMinY >= imgH))
			{
				++clipped;

				continue;
			}

			// Get species code for coloring
			std::string speciesCode;
			auto itCode = ft.attributes.find(SPECIES_CODE);

			if (itCode != ft.attributes.end())
			{
				speciesCode = itCode->second;
			}

			cv::Scalar color = (opts.colorBySpecies && !speciesCode.empty()) ? getPolygonFillColor(speciesCode) : cv::Scalar(100, 200, 100);

			// Extract polygon points
			OGRwkbGeometryType flatType = wkbFlatten(ft.geometry->getGeometryType());

			if (flatType == wkbPolygon)
			{
				drawSinglePolygon(canvas, overlay, ft.geometry->toPolygon(), geoToPx, color, opts);

				++drawn;
			}
			else if (flatType == wkbMultiPolygon)
			{
				OGRMultiPolygon* multiPoly = ft.geometry->toMultiPolygon();

				if (!multiPoly)
				{
					continue;
				}

				for (int gi = 0 ; gi < multiPoly->getNumGeometries(); ++gi)
				{
					OGRPolygon* poly = (OGRPolygon*)multiPoly->getGeometryRef(gi);
					drawSinglePolygon(canvas, overlay, poly, geoToPx, color, opts);
				}

				++drawn;
			}
		}

		// Blend all polygons at once
		if (!overlay.empty() && (opts.polyAlpha > 0.01))
		{
			cv::addWeighted(overlay, opts.polyAlpha, canvas, 1.0 - opts.polyAlpha, 0, canvas);
		}

		std::cout << "    Polygons visible: " << drawn << ", outside image: " << clipped << "\n";
	}

	static void drawSinglePolygon(cv::Mat& canvas, cv::Mat& overlay, const OGRPolygon* poly, std::function<cv::Point(double, double)>& geoToPx, const cv::Scalar& color, const OverlayOptions& opts)
	{
		if (!poly)
		{
			return;
		}

		const OGRLinearRing* ring = poly->getExteriorRing();

		if (!ring)
		{
			return;
		}

		int nPts = ring->getNumPoints();

		if (nPts < 3)
		{
			return;
		}

		std::vector<cv::Point> pts(nPts);

		for (int i = 0 ; i < nPts ; ++i)
		{
			pts[i] = geoToPx(ring->getX(i), ring->getY(i));
		}

		// Fill on overlay (blended once at end)
		if (!overlay.empty())
		{
			std::vector<std::vector<cv::Point>> polyPts = {pts};
			cv::fillPoly(overlay, polyPts, color);
		}

		// Border
		if (opts.polyLineWidth > 0)
		{
			// When polyAlpha=0 (borders only mode), use polyBorderColor
			cv::Scalar borderColor = (opts.polyAlpha < 0.01) ? opts.polyBorderColor : color;

			std::vector<std::vector<cv::Point>> polyPts = {pts};
			cv::polylines(canvas, polyPts, true, borderColor, opts.polyLineWidth, cv::LINE_AA);
		}
	}

	static void drawTreePoints(cv::Mat& canvas, int imgW, int imgH, const TreeData& trees, std::function<cv::Point2f(double, double)> geoToPxF, double pixPerMeter, const OverlayOptions& opts)
	{
		int drawn = 0;
		int outside = 0;

		// Crown overlay: use separate layer for alpha blending
		cv::Mat crownLayer;

		if (opts.drawCrown && (opts.crownAlpha > 0.01))
		{
			crownLayer = canvas.clone();
		}

		for (auto& t : trees.tps)
		{
			cv::Point2f center = geoToPxF(t.x, t.y);
			int cx = (int)center.x;
			int cy = (int)center.y;

			// Margin for partially visible crowns
			int margin = (t.crownD > 0) ? (int)(t.crownD * 0.5 * pixPerMeter) + 5 : 5;\

			// Out of bounds check
			if ((cx < -margin) || (cx >= imgW + margin) || (cy < -margin) || (cy >= imgH + margin))
			{
				++outside;

				continue;
			}

/*
			if (((t.treeId + 0) % 10) != 0)
			{
				continue;
			}
*/

			// Crown circle
			if (opts.drawCrown && (t.crownD > 0))
			{
				int crownPxR = (int)(t.crownD * 0.5 * pixPerMeter);

				if (crownPxR < 1)
				{
					crownPxR = 1;
				}

				cv::Scalar cColor = opts.crownColor; //opts.colorByHeight ? heightToColor(t.height, opts.heightColorMin, opts.heightColorMax) : opts.crownColor;

				if (!crownLayer.empty())
				{
					//cv::circle(crownLayer, {cx, cy}, crownPxR, cColor, -1, cv::LINE_AA);
					cv::circle(crownLayer, {cx, cy}, crownPxR, cColor * 0.7, 2, cv::LINE_AA);

					// treeId 텍스트
					cv::putText(crownLayer, std::to_string(t.treeId), cv::Point(cx + crownPxR + 1, cy), cv::FONT_HERSHEY_PLAIN, 0.6, cColor, 1);

				}
				else
				{
					cv::circle(canvas, {cx, cy}, crownPxR, cColor, 1, cv::LINE_AA);
				}
			}

			// Center dot
			if (opts.drawCenter)
			{
				cv::Scalar dotColor = opts.centerColor; //opts.colorByHeight ? heightToColor(t.height, opts.heightColorMin, opts.heightColorMax) : opts.centerColor;

				if ((cx >= 0) && (cx < imgW) && (cy >= 0) && (cy < imgH))
				{
					//cv::circle(canvas, { cx, cy }, opts.centerRadius, dotColor, -1, cv::LINE_AA);
				}
			}

			++drawn;
		}

		// Single blend for all crown
		if (!crownLayer.empty())
		{
			cv::addWeighted(crownLayer, opts.crownAlpha, canvas, 1.0 - opts.crownAlpha, 0, canvas);
		}

		std::cout << "    Trees visible: " << drawn << ", outside image: " << outside << "\n";
	}

	// ─── Result legend ───
	static void drawResultLegend(
		cv::Mat& canvas,
		const std::map<std::string, cv::Scalar>& colors,
		const std::map<std::string, std::string>& names,
		const std::map<std::string, int>& correct,
		const std::map<std::string, int>& total,
		const OverlayOptions& opts)
	{
		std::vector<std::pair<std::string, int>> sorted;
		
		for (auto& kv : total)
		{
			sorted.push_back(kv);
		}

		std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b) { return a.second > b.second; });
		
		if (sorted.empty())
		{
			for (auto& kv : colors)
			{
				sorted.push_back({ kv.first, 0 });
			}
		}

		int x = opts.legendX;
		int y = opts.legendY;
		int lineH = 28;
		int boxW = 320;
		int boxH = 40 + (int)sorted.size() * lineH;

		// Background
		cv::Rect lr(x, y, boxW, boxH);
		lr &= cv::Rect(0, 0, canvas.cols, canvas.rows);

		if (lr.width > 0 && lr.height > 0)
		{
			cv::Mat roi = canvas(lr);
			cv::Mat dark(roi.size(), roi.type(), cv::Scalar(30, 30, 30));
			cv::addWeighted(dark, 0.7, roi, 0.3, 0, roi);
		}

		cv::putText(canvas, opts.resultTitle, cv::Point(x + 10, y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

		int row = 0;

		for (auto& kv : sorted)
		{
			int ey = y + 38 + row * lineH;
			auto cit = colors.find(kv.first);
			cv::Scalar c = (cit != colors.end()) ? cit->second : SPECIES_DEFAULT_COLOR;
			cv::rectangle(canvas, cv::Point(x + 10, ey - 8), cv::Point(x + 24, ey + 6), c, -1);

			auto nit = names.find(kv.first);
			std::string label = kv.first;

			if (nit != names.end() && !nit->second.empty())
			{
				label = kv.first + " " + nit->second;
			}

			auto tit = total.find(kv.first);
			auto crit = correct.find(kv.first);

			if (tit != total.end() && tit->second > 0)
			{
				int cc = (crit != correct.end()) ? crit->second : 0;
				label += " (" + std::to_string((int)(100.0 * cc / tit->second)) + "%)";
			}

			cv::putText(canvas, label, cv::Point(x + 30, ey + 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(220, 220, 220), 1);

			++row;
		}
	}

	// ─── Correctness legend ───
	static void drawCorrectnessLegend(
		cv::Mat& canvas,
		const std::vector<HierarchicalResult>& results,
		const OverlayOptions& opts)
	{
		int x = opts.legendX;
		int y = opts.legendY;

		// Count
		int correct = 0, groupOk = 0, wrong = 0, noTruth = 0;

		for (auto& r : results)
		{
			if (r.trueCode.empty())
			{
				++noTruth;
				continue;
			}

			if (r.speciesCorrect)
			{
				++correct;
			}
			else if (r.groupCorrect)
			{
				++groupOk;
			}
			else
			{
				++wrong;
			}
		}

		int boxW = 300;
		int boxH = 160;
		cv::Rect lr(x, y, boxW, boxH);
		lr &= cv::Rect(0, 0, canvas.cols, canvas.rows);

		if (lr.width > 0 && lr.height > 0)
		{
			cv::Mat roi = canvas(lr);
			cv::Mat dark(roi.size(), roi.type(), cv::Scalar(30, 30, 30));
			cv::addWeighted(dark, 0.7, roi, 0.3, 0, roi);
		}

		cv::putText(canvas, opts.resultTitle, cv::Point(x + 10, y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);

		struct Entry
		{
			cv::Scalar color;
			std::string label;
			int count;
		};

		Entry entries[] =
		{
			{{0,200,0},     "Correct species", correct},
			{{0,200,200},   "Right group, wrong species", groupOk},
			{{0,0,200},     "Wrong group", wrong},
			{{200,200,0},   "No ground truth", noTruth},
		};

		for (int i = 0; i < 4; ++i)
		{
			int ey = y + 46 + i * 28;
			cv::rectangle(canvas, cv::Point(x + 10, ey - 8), cv::Point(x + 24, ey + 6), entries[i].color, -1);
			
			std::string label = entries[i].label + " (" + std::to_string(entries[i].count) + ")";
			cv::putText(canvas, label, cv::Point(x + 30, ey + 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(220, 220, 220), 1);
		}
	}

/*
	// ─── Fixed color for classification results ───
	  // Tree points: saturated, vivid colors
	static cv::Scalar getResultColor(const std::string& code) {
		// Conifer: green shades (vivid)
		if (code == "14" || code == "C")  return cv::Scalar(0, 180, 0);
		if (code == "11")                 return cv::Scalar(0, 220, 80);
		if (code == "12")                 return cv::Scalar(0, 140, 60);
		if (code == "13")                 return cv::Scalar(80, 200, 0);
		// Deciduous: warm shades (vivid)
		if (code == "31" || code == "D")  return cv::Scalar(0, 140, 220);
		if (code == "34")                 return cv::Scalar(0, 100, 200);
		if (code == "30")                 return cv::Scalar(60, 60, 200);
		if (code == "39")                 return cv::Scalar(0, 180, 255);
		if (code == "45")                 return cv::Scalar(200, 0, 200);
		// Non-forest: cool shades (vivid)
		if (code == "77" || code == "N")  return cv::Scalar(200, 160, 60);
		if (code == "81")                 return cv::Scalar(200, 200, 200);
		if (code == "82")                 return cv::Scalar(128, 128, 128);
		// Evergreen
		if (code == "E")                  return cv::Scalar(180, 180, 0);
		// Default
		std::hash<std::string> hasher;
		size_t h = hasher(code);
		return cv::Scalar((h * 97) % 200 + 55, (h * 53) % 200 + 55, (h * 31) % 200 + 55);
	}

	// Forest polygon fill: pastel/muted version of the same hue
	// Distinct from tree point colors so they don't blend together
	static cv::Scalar getPolygonFillColor(const std::string& code) {
		// Conifer: blue-purple tones (contrast with green tree points)
		if (code == "14")  return cv::Scalar(180, 140, 100);    // muted blue
		if (code == "11")  return cv::Scalar(180, 160, 120);    // lighter blue
		if (code == "12")  return cv::Scalar(160, 120, 80);     // dark blue
		if (code == "13")  return cv::Scalar(160, 160, 100);    // teal
		if (code == "10")  return cv::Scalar(170, 150, 110);    // blue-gray
		// Deciduous: cool tones (contrast with warm tree points)
		if (code == "31")  return cv::Scalar(140, 180, 140);    // pale green
		if (code == "34")  return cv::Scalar(130, 170, 130);    // sage
		if (code == "30")  return cv::Scalar(120, 160, 120);    // muted green
		if (code == "39")  return cv::Scalar(140, 190, 150);    // light sage
		if (code == "45")  return cv::Scalar(160, 140, 180);    // lavender
		// Non-forest: warm earth tones (contrast with cool tree points)
		if (code == "77")  return cv::Scalar(120, 140, 180);    // tan
		if (code == "81")  return cv::Scalar(160, 170, 190);    // beige
		if (code == "82")  return cv::Scalar(150, 155, 160);    // warm gray
		// Evergreen
		if (code == "61")  return cv::Scalar(170, 160, 100);
		// Default
		std::hash<std::string> hasher;
		size_t h = hasher(code);
		return cv::Scalar((h * 41) % 80 + 140, (h * 67) % 80 + 140, (h * 23) % 80 + 140);
	}
*/
};

inline std::vector<std::vector<Point2D>> extractRings(const OGRGeometry* geom)
{
	std::vector<std::vector<Point2D>> rings;

	if (!geom)
	{
		return rings;
	}

	auto addRing = [&](const OGRLinearRing* r)
	{
		if (!r)
		{
			return;
		}

		std::vector<Point2D> ring;

		for (int i = 0; i < r->getNumPoints(); ++i)
		{
			ring.push_back({ r->getX(i), r->getY(i) });
		}

		rings.push_back(std::move(ring));
	};

	auto processPoly = [&](const OGRPolygon* poly)
	{
		if (!poly)
		{
			return;
		}

		addRing(poly->getExteriorRing());

		for (int h = 0; h < poly->getNumInteriorRings(); ++h)
		{
			addRing(poly->getInteriorRing(h));
		}
	};

	OGRwkbGeometryType t = wkbFlatten(geom->getGeometryType());

	if (t == wkbPolygon)
	{
		processPoly((const OGRPolygon*)geom);
	}
	else if (t == wkbMultiPolygon)
	{
		auto* mp = (const OGRMultiPolygon*)geom;

		for (int i = 0; i < mp->getNumGeometries(); ++i)
		{
			processPoly((const OGRPolygon*)mp->getGeometryRef(i));
		}
	}

	return rings;
}

// 링 하나를 Point2D 벡터로 변환
inline std::vector<Point2D> ringToPoints(const OGRLinearRing* r)
{
	std::vector<Point2D> pts;

	if (!r)
	{
		return pts;
	}

	pts.reserve(r->getNumPoints());

	for (int i = 0; i < r->getNumPoints(); ++i)
	{
		pts.push_back({ r->getX(i), r->getY(i) });
	}

	return pts;
}
