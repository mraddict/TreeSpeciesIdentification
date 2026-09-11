#pragma once

#include "TSICommon.h"

class CoordTransformer
{
public:

	CoordTransformer(int srcEPSG, int dstEPSG)
	{
		// 1. 소스 좌표계 설정 (임상도: EPSG 5179)
		// 2. 타겟 좌표계 설정 (GPS 위경도: EPSG 4326)

		OGRSpatialReference src;
		OGRSpatialReference dst;

		src.importFromEPSG(srcEPSG);
		dst.importFromEPSG(dstEPSG);

		src.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
		dst.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

		ct_ = OGRCreateCoordinateTransformation(&src, &dst);

		if (!ct_)
		{
			throw std::runtime_error("Failed create CoordTransformer");
		}
	}

	~CoordTransformer()
	{
		OGRCoordinateTransformation::DestroyCT(ct_);
	}

	bool transform(double& x, double& y) const
	{
		int res = ct_->Transform(1, &x, &y);

		if (res)
		{
			// 결과: x는 경도(Longitude), y는 위도(Latitude)가 됩니다.
			//printf("GPS: 위도 %.6f, 경도 %.6f\n", y, x);

			return true;
		}

		return false;
	}

	bool transform(size_t count, double* px, double* py) const
	{
		int res = ct_->Transform(count, px, py);

		if (res)
		{
			return true;
		}

		return false;
	}

	std::vector<Point2D> transformRing(const std::vector<Point2D>& ring) const
	{
		std::vector<Point2D> out = ring;

		for (auto& p : out)
		{
			transform(p.x, p.y);


			return out;
		}
	}

	// ─── Create OGR coordinate transformer ───
	static OGRCoordinateTransformation* createTransform(int srcEpsg, int dstEpsg)
	{
		OGRSpatialReference srcSRS;
		srcSRS.importFromEPSG(srcEpsg);
		srcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

		OGRSpatialReference dstSRS;
		dstSRS.importFromEPSG(dstEpsg);
		dstSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

		return OGRCreateCoordinateTransformation(&srcSRS, &dstSRS);
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

	static void convert()
	{
		// ── LiDAR 범위 (EPSG:5186) ──
		double lidarMinX = 216538.526, lidarMaxX = 218006.234;
		double lidarMinY = 496833.418, lidarMaxY = 498188.266;

		// ── 임상도 범위 (EPSG:5179) ── (로딩 디버그 출력값 입력)
		double forestMinX = 971102.0, forestMaxX = 975556.0;
		double forestMinY = 1894650.0, forestMaxY = 1900200.0;

		// ── EPSG:5186 → EPSG:5179 좌표 변환기 ──

		CoordTransformer ct(5186, 5179);

		// LiDAR 4 코너를 변환해서 5179 기준 범위 계산
		double xs[4] = { lidarMinX, lidarMaxX, lidarMinX, lidarMaxX };
		double ys[4] = { lidarMinY, lidarMinY, lidarMaxY, lidarMaxY };

		ct.transform(4, xs, ys);

		double convMinX = *std::min_element(xs, xs + 4);
		double convMaxX = *std::max_element(xs, xs + 4);
		double convMinY = *std::min_element(ys, ys + 4);
		double convMaxY = *std::max_element(ys, ys + 4);

		// ── 교집합 범위 계산 ──
		double overlapMinX = std::max(convMinX, forestMinX);
		double overlapMinY = std::max(convMinY, forestMinY);
		double overlapMaxX = std::min(convMaxX, forestMaxX);
		double overlapMaxY = std::min(convMaxY, forestMaxY);

		bool hasOverlap = (overlapMinX < overlapMaxX) && (overlapMinY < overlapMaxY);

		// ── 면적 계산 ──
		auto area = [](double x1, double y1, double x2, double y2)
			{
				return (x2 - x1) * (y2 - y1) / 1e6;
			};

		double lidarArea = area(convMinX, convMinY, convMaxX, convMaxY);
		double forestArea = area(forestMinX, forestMinY, forestMaxX, forestMaxY);
		double overlapArea = hasOverlap ? area(overlapMinX, overlapMinY, overlapMaxX, overlapMaxY) : 0.0;

		// ── 출력 ──
		std::cout << "=== LiDAR 범위 (EPSG:5186 원본) ===\n"
			<< "  X: " << lidarMinX << " ~ " << lidarMaxX << "\n"
			<< "  Y: " << lidarMinY << " ~ " << lidarMaxY << "\n\n";

		std::cout << "=== LiDAR 범위 (EPSG:5179 변환 후) ===\n"
			<< "  X: " << convMinX << " ~ " << convMaxX << "\n"
			<< "  Y: " << convMinY << " ~ " << convMaxY << "\n\n";

		std::cout << "=== 임상도 범위 (EPSG:5179) ===\n"
			<< "  X: " << forestMinX << " ~ " << forestMaxX << "\n"
			<< "  Y: " << forestMinY << " ~ " << forestMaxY << "\n\n";

		if (hasOverlap) {
			std::cout << "=== 겹치는 영역 ===\n"
				<< "  X: " << overlapMinX << " ~ " << overlapMaxX << "\n"
				<< "  Y: " << overlapMinY << " ~ " << overlapMaxY << "\n"
				<< "  LiDAR 면적    : " << lidarArea << " ㎢\n"
				<< "  임상도 면적   : " << forestArea << " ㎢\n"
				<< "  겹치는 면적   : " << overlapArea << " ㎢\n"
				<< "  LiDAR 대비    : "
				<< (overlapArea / lidarArea * 100.0) << " %\n"
				<< "  임상도 대비   : "
				<< (overlapArea / forestArea * 100.0) << " %\n";
		}
		else {
			std::cout << "겹치는 영역 없음\n"
				<< "  두 데이터의 범위가 전혀 겹치지 않습니다.\n";
		}
	}

private:

	OGRCoordinateTransformation* ct_ = nullptr;
};
