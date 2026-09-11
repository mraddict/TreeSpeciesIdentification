#pragma once

#include "TSICommon.h"

class ImageViewer
{
public:

	struct Options
	{
		std::string title = "Viewer";
		int windowW = 1280;
		int windowH = 960;
		double zoomStep = 1.15;		// 휠 한 칸당 배율
		double minZoom = 0.05;			// 최소 줌 배율
		double maxZoom = 50.0;			// 최대 줌 배율

		// 지리 좌표 표시 (geoMinX/Y, mpp 설정 시 마우스 위치에 지리 좌표 표시)
		bool showGeoCoord = false;
		double geoMinX = 0.0;
		double geoMinY = 0.0;
		double geoMaxY = 0.0;
		double meterPerPixel = 1.0;
	};

	static void show(const cv::Mat& img, const Options& opt = {})
	{
		if (img.empty())
		{
			throw std::runtime_error("이미지가 비어 있습니다.");
		}

		ViewerState s;

		s.img = img;
		s.opt = opt;
		s.zoom = std::min((double)opt.windowW / img.cols, (double)opt.windowH / img.rows);

		// 초기 뷰: 이미지 전체가 창에 맞게
		s.panX = (opt.windowW - img.cols * s.zoom) * 0.5;
		s.panY = (opt.windowH - img.rows * s.zoom) * 0.5;
		s.initZoom = s.zoom;
		s.initPanX = s.panX;
		s.initPanY = s.panY;

		cv::namedWindow(opt.title, cv::WINDOW_NORMAL);
		cv::resizeWindow(opt.title, opt.windowW, opt.windowH);
		cv::setMouseCallback(opt.title, onMouse, &s);

		std::cout << "=== 뷰어 조작법 ===\n"
			<< "  마우스 휠     : 줌 인/아웃 (마우스 위치 기준)\n"
			<< "  좌클릭 드래그 : 패닝\n"
			<< "  R             : 초기 뷰 리셋\n"
			<< "  S             : 현재 화면 저장\n"
			<< "  ESC / Q       : 종료\n"
			<< "  이미지 크기   : " << img.cols << " x " << img.rows << " px\n";

		while (true)
		{
			cv::Mat canvas(opt.windowH, opt.windowW, CV_8UC3, cv::Scalar(30, 30, 30));

			drawImage(canvas, s);
			drawInfo(canvas, s);

			cv::imshow(opt.title, canvas);

			int key = cv::waitKey(16) & 0xFF;

			if ((key == 27) || (key == 'q') || (key == 'Q'))
			{
				break;
			}

			if ((key == 'r') || (key == 'R'))
			{
				s.zoom = s.initZoom;
				s.panX = s.initPanX;
				s.panY = s.initPanY;
			}

			if ((key == 's') || (key == 'S'))
			{
				std::string path = "snapshot_" + std::to_string(s.snapCount++) + ".png";
				cv::imwrite(path, canvas);

				std::cout << "저장: " << path << "\n";
			}
		}

		cv::destroyWindow(opt.title);
	}

private:

	struct ViewerState
	{
		cv::Mat img;
		Options opt;
		double zoom = 1.0;
		double panX = 0.0;			// 이미지 좌상단의 캔버스 내 X 픽셀
		double panY = 0.0;
		double initZoom, initPanX, initPanY;
		bool dragging = false;
		int dragStartX = 0;
		int dragStartY = 0;
		double dragPanX = 0.0;
		double dragPanY = 0.0;
		int mouseX = 0;
		int mouseY = 0;
		int snapCount = 0;
	};

	// 현재 줌/팬 상태로 이미지를 캔버스에 그림
	static void drawImage(cv::Mat& canvas, const ViewerState& s)
	{
		int W = canvas.cols;
		int H = canvas.rows;

		// 이미지의 캔버스 내 렌더링 영역
		int dstX = (int)s.panX;
		int dstY = (int)s.panY;
		int dstW = (int)(s.img.cols * s.zoom);
		int dstH = (int)(s.img.rows * s.zoom);

		// 클리핑
		int clipX = std::max(0, -dstX);
		int clipY = std::max(0, -dstY);
		int clipW = std::min(dstW - clipX, W - std::max(0, dstX));
		int clipH = std::min(dstH - clipY, H - std::max(0, dstY));

		if ((clipW <= 0) || (clipH <= 0))
		{
			return;
		}

		// 원본 이미지에서 대응 영역
		double invZ = 1.0 / s.zoom;
		int srcX = (int)(clipX * invZ);
		int srcY = (int)(clipY * invZ);
		int srcW = (int)(clipW * invZ);
		int srcH = (int)(clipH * invZ);

		srcW = std::min(srcW, s.img.cols - srcX);
		srcH = std::min(srcH, s.img.rows - srcY);

		if ((srcW <= 0) || (srcH <= 0))
		{
			return;
		}

		cv::Mat srcROI = s.img(cv::Rect(srcX, srcY, srcW, srcH));
		cv::Mat dstROI = canvas(cv::Rect(std::max(0, dstX), std::max(0, dstY), clipW, clipH));

		// 줌 비율에 따라 보간 방식 선택
		int interp = (s.zoom >= 1.0) ? cv::INTER_NEAREST : cv::INTER_LINEAR;
		cv::resize(srcROI, dstROI, {clipW, clipH}, 0, 0, interp);
	}

	// 정보 오버레이 표시
	static void drawInfo(cv::Mat& canvas, const ViewerState& s)
	{
		// 줌 배율
		std::string zoomStr = "Zoom: " + formatZoom(s.zoom / s.initZoom) + "x";
		cv::putText(canvas, zoomStr, {10, 24}, cv::FONT_HERSHEY_SIMPLEX, 0.65, {200, 200, 200}, 1, cv::LINE_AA);

		// 마우스 위치 → 이미지 픽셀 좌표
		int imgPxX = (int)((s.mouseX - s.panX) / s.zoom);
		int imgPxY = (int)((s.mouseY - s.panY) / s.zoom);

		if ((imgPxX >= 0) && (imgPxX < s.img.cols) && (imgPxY >= 0) && (imgPxY < s.img.rows))
		{
			std::string pxStr = "Px: (" + std::to_string(imgPxX) + ", " + std::to_string(imgPxY) + ")";
			cv::putText(canvas, pxStr, {10, 48}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {180, 180, 180}, 1, cv::LINE_AA);

			// 지리 좌표 표시 (옵션)
			if (s.opt.showGeoCoord)
			{
				double gx = s.opt.geoMinX + imgPxX * s.opt.meterPerPixel;
				double gy = s.opt.geoMaxY - imgPxY * s.opt.meterPerPixel;

				char buf[128];
				std::snprintf(buf, sizeof(buf), "Geo: (%.1f, %.1f)", gx, gy); cv::putText(canvas, buf, { 10, 70 }, cv::FONT_HERSHEY_SIMPLEX, 0.55, { 180, 220, 180 }, 1, cv::LINE_AA);
			}

			// 픽셀 색상 표시
			cv::Vec3b color = s.img.at<cv::Vec3b>(imgPxY, imgPxX);
			char colBuf[64];
			std::snprintf(colBuf, sizeof(colBuf), "BGR: (%d, %d, %d)", color[0], color[1], color[2]);
			
			int lineY = s.opt.showGeoCoord ? 92 : 70;
			cv::putText(canvas, colBuf, { 10, lineY }, cv::FONT_HERSHEY_SIMPLEX, 0.55, { 180, 180, 220 }, 1, cv::LINE_AA);
		}

		// 조작법 힌트
		cv::putText(canvas, "R:reset  S:save  ESC:quit", { 10, canvas.rows - 10 }, cv::FONT_HERSHEY_SIMPLEX, 0.45, { 120, 120, 120 }, 1, cv::LINE_AA);
	}

	static void onMouse(int event, int x, int y, int flags, void* userData)
	{
		ViewerState* s = (ViewerState*)userData;
		s->mouseX = x;
		s->mouseY = y;

		//std::cout << "event=" << event << " flags=" << flags << "\n";

		// 휠 줌: 마우스 위치를 중심으로 줌
		if (event == cv::EVENT_MOUSEWHEEL)
		{
			double factor = (cv::getMouseWheelDelta(flags) > 0) ? s->opt.zoomStep : 1.0 / s->opt.zoomStep;
			applyZoom(s, x, y, factor);

			return;
		}

		// 휠 줌 방식 2: EVENT_MOUSEMOVE + flags (Windows)
		if (event == cv::EVENT_MOUSEMOVE)
		{
			if (flags & cv::EVENT_FLAG_SHIFTKEY)
			{

			}
		}

		// 드래그 시작
		if (event == cv::EVENT_LBUTTONDOWN)
		{
			s->dragging = true;
			s->dragStartX = x;
			s->dragStartY = y;
			s->dragPanX = s->panX;
			s->dragPanY = s->panY;
		}

		// 드래그 중
		if ((event == cv::EVENT_MOUSEMOVE) && s->dragging)
		{
			s->panX = s->dragPanX + (x - s->dragStartX);
			s->panY = s->dragPanY + (y - s->dragStartY);
		}

		// 드래그 종료
		if (event == cv::EVENT_LBUTTONUP)
		{
			s->dragging = false;
		}
	}

	static std::string formatZoom(double z)
	{
		char buf[32];

		if (z >= 10.0)
		{
			std::snprintf(buf, sizeof(buf), "%.0f", z);
		}
		else if (z >= 1)
		{
			std::snprintf(buf, sizeof(buf), "%.1f", z);
		}
		else
		{
			std::snprintf(buf, sizeof(buf), "%.2f", z);
		}

		return buf;
	}

	static void applyZoom(ViewerState* s, int x, int y, double factor)
	{
		double newZoom = std::min(std::max(s->zoom * factor, s->opt.minZoom), s->opt.maxZoom);

		s->panX = x - (x - s->panX) * (newZoom / s->zoom);
		s->panY = y - (y - s->panY) * (newZoom / s->zoom);

		s->zoom = newZoom;
	}

};
