#pragma once

#include "OverlayRenderer.h"
#include "Hierarchicalclassifier.h"

struct ResultOverlayOptions
{
	// Polygon borders
	bool drawPolygonBorders = true;
	int borderWidth = 1;
	cv::Scalar borderColor = { 128, 128, 128 };

	// Tree rendering
	int treeRadius = 0;						// 0 = auto from crown diameter
	double crownScale = 0.5;			// crown circle scale (0.5 = half of actual)
	double crownAlpha = 0.4;			// fill transparency
	int outlineWidth = 1;						// circle outline

	// Coloring mode
	bool colorByGroup = false;			// true=group color, false=species color
	bool showCorrectness = false;	// true=green/red for correct/wrong

	// Legend
	bool drawLegend = true;
	int legendX = 30;
	int legendY = 30;

	// Output
	std::string title = "Classificaiton Results";
};

class ResultOverlayRenderer
{
public:

	// 式式式 Save result overlay to file 式式式
	static void save(
		const AerialPhoto& aerialImg, 
		const ForestLayer& forest, 
		const std::vector<HierarchicalResult>& results, 
		const ExtractionResult& features, 
		const std::string& outputPath, 
		const ResultOverlayOptions& opts = ResultOverlayOptions())
	{
		std::cout << "=== Building result overlay ===\n";

		if (aerialImg.image.empty())
		{
			throw std::runtime_error("Aerial image is empty.");
		}

		cv::Mat canvas = aerialImg.image.clone();
		int imgW = canvas.cols;
		int imgH = canvas.rows;

		// Geo-to-pixel using aerial bounds
		double scaleX = imgW / (aerialImg.geoMaxX - aerialImg.geoMinX);
		double scaleY = imgH / (aerialImg.geoMaxY - aerialImg.geoMinY);

		auto geoToPx = [&](double gx, double gy) -> cv::Point
		{
			int px = (int)(gx - aerialImg.geoMinX) * scaleX;
			int py = (int)(aerialImg.geoMaxY - gy) * scaleY;

			return cv::Point(px, py);
		};

		// Draw polygon borders
		if (opts.drawPolygonBorders)
		{
			//drawPolygonBorders(canvas, imgW, imgH, forest, geoToPx, opts);
		}
	}
};