#pragma once
// ─────────────────────────────────────────────────────────
//  AerialReproject.h
//  항공사진 GeoTIFF 좌표계 변환 유틸리티
//
//  주 용도: 항공사진(EPSG:32652 UTM 52N)을
//           임상도 좌표계(EPSG:5179 Korea Unified CS)로 변환
//
//  의존성:
//    - GDAL (libgdal-dev) : gdalwarp 기반 좌표 변환
//
//  빌드 예시:
//    g++ -std=c++17 -o aerial_reproject main_aerial_reproject.cpp \
//        $(gdal-config --cflags --libs)
// ─────────────────────────────────────────────────────────
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <algorithm>

#include <gdal_priv.h>
#include <gdal_alg.h>
#include <gdalwarper.h>
#include <cpl_conv.h>
#include <cpl_string.h>
#include <ogr_spatialref.h>

// ─────────────────────────────────────────────
//  좌표 변환 옵션
// ─────────────────────────────────────────────
struct ReprojectOptions {
    int         srcEpsg     = 0;        // 원본 EPSG (0이면 파일에서 자동 감지)
    int         dstEpsg     = 5179;     // 대상 EPSG (기본: 한국 통합좌표계)
    double      dstResX     = 0.0;      // 대상 해상도 X (0이면 자동 계산)
    double      dstResY     = 0.0;      // 대상 해상도 Y (0이면 자동 계산)
    std::string resampleAlg = "bilinear"; // 리샘플링: near, bilinear, cubic, cubicspline, lanczos
    std::string format      = "GTiff";  // 출력 포맷
    bool        compress    = true;     // 출력 압축 (LZW)
    int         numThreads  = 4;        // 워프 병렬 스레드 수
    double      noDataValue = 0.0;      // NoData 값
    bool        hasNoData   = false;    // NoData 사용 여부
    double      memoryLimit = 512.0;    // 워프 메모리 제한 (MB)
};

// ─────────────────────────────────────────────
//  변환 결과 리포트
// ─────────────────────────────────────────────
struct ReprojectReport {
    // 원본 정보
    int     srcEpsg     = 0;
    int     srcWidth    = 0, srcHeight  = 0;
    double  srcMinX     = 0, srcMaxX    = 0;
    double  srcMinY     = 0, srcMaxY    = 0;
    double  srcPixelW   = 0, srcPixelH  = 0;

    // 변환 후 정보
    int     dstEpsg     = 0;
    int     dstWidth    = 0, dstHeight  = 0;
    double  dstMinX     = 0, dstMaxX    = 0;
    double  dstMinY     = 0, dstMaxY    = 0;
    double  dstPixelW   = 0, dstPixelH  = 0;

    // 검증
    bool    success     = false;
    std::string message;
};

// ─────────────────────────────────────────────
//  AerialReproject 클래스
// ─────────────────────────────────────────────
class AerialReproject {
public:

    // ─── 좌표계 변환 실행 ───
    static ReprojectReport reproject(
        const std::string& srcPath,
        const std::string& dstPath,
        const ReprojectOptions& opts = ReprojectOptions())
    {
        GDALAllRegister();
        ReprojectReport report;

        // ── 1. 원본 파일 열기 ──
        GDALDataset* srcDs = (GDALDataset*)GDALOpen(srcPath.c_str(), GA_ReadOnly);
        if (!srcDs)
            throw std::runtime_error("원본 파일 열기 실패: " + srcPath);

        report.srcWidth  = srcDs->GetRasterXSize();
        report.srcHeight = srcDs->GetRasterYSize();
        int nBands       = srcDs->GetRasterCount();

        // 원본 GeoTransform
        double srcGT[6];
        srcDs->GetGeoTransform(srcGT);
        report.srcPixelW = srcGT[1];
        report.srcPixelH = srcGT[5];
        report.srcMinX   = srcGT[0];
        report.srcMaxY   = srcGT[3];
        report.srcMaxX   = srcGT[0] + report.srcWidth  * srcGT[1];
        report.srcMinY   = srcGT[3] + report.srcHeight * srcGT[5];
        if (report.srcMinY > report.srcMaxY)
            std::swap(report.srcMinY, report.srcMaxY);

        // ── 2. 좌표계 설정 ──
        OGRSpatialReference srcSRS, dstSRS;

        // 원본 좌표계
        if (opts.srcEpsg > 0) {
            srcSRS.importFromEPSG(opts.srcEpsg);
            report.srcEpsg = opts.srcEpsg;
        } else {
            const char* srcWkt = srcDs->GetProjectionRef();
            if (srcWkt && srcWkt[0] != '\0') {
                srcSRS.importFromWkt(srcWkt);
                const char* auth = srcSRS.GetAuthorityCode(nullptr);
                report.srcEpsg = auth ? std::atoi(auth) : 0;
            } else {
                GDALClose(srcDs);
                throw std::runtime_error("원본 파일에 좌표계 정보가 없습니다. srcEpsg를 지정하세요.");
            }
        }

        // 대상 좌표계
        dstSRS.importFromEPSG(opts.dstEpsg);
        report.dstEpsg = opts.dstEpsg;

        // WKT 문자열 생성
        char* srcWkt = nullptr;
        char* dstWkt = nullptr;
        srcSRS.exportToWkt(&srcWkt);
        dstSRS.exportToWkt(&dstWkt);

        std::cout << "=== 좌표계 변환 시작 ===\n"
                  << "  원본: EPSG:" << report.srcEpsg
                  << " (" << report.srcWidth << "x" << report.srcHeight
                  << ", " << nBands << "밴드)\n"
                  << "  대상: EPSG:" << report.dstEpsg << "\n";

        // ── 3. 대상 영역 및 해상도 계산 ──
        double dstGT[6];
        int dstWidth, dstHeight;
        computeTargetExtent(
            srcDs, srcWkt, dstWkt,
            opts.dstResX, opts.dstResY,
            dstGT, dstWidth, dstHeight);

        report.dstWidth  = dstWidth;
        report.dstHeight = dstHeight;
        report.dstPixelW = dstGT[1];
        report.dstPixelH = dstGT[5];
        report.dstMinX   = dstGT[0];
        report.dstMaxY   = dstGT[3];
        report.dstMaxX   = dstGT[0] + dstWidth  * dstGT[1];
        report.dstMinY   = dstGT[3] + dstHeight * dstGT[5];
        if (report.dstMinY > report.dstMaxY)
            std::swap(report.dstMinY, report.dstMaxY);

        std::cout << "  변환 후 크기: " << dstWidth << "x" << dstHeight << "\n"
                  << "  변환 후 해상도: " << dstGT[1] << " m/pixel\n"
                  << "  변환 후 범위: X[" << std::fixed << std::setprecision(3)
                  << report.dstMinX << " ~ " << report.dstMaxX << "] Y["
                  << report.dstMinY << " ~ " << report.dstMaxY << "]\n";

        // ── 4. 출력 파일 생성 ──
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName(
            opts.format.c_str());
        if (!driver) {
            CPLFree(srcWkt); CPLFree(dstWkt); GDALClose(srcDs);
            throw std::runtime_error("GDAL 드라이버를 찾을 수 없습니다: " + opts.format);
        }

        // 생성 옵션
        char** createOpts = nullptr;
        if (opts.compress && opts.format == "GTiff") {
            createOpts = CSLSetNameValue(createOpts, "COMPRESS", "LZW");
            createOpts = CSLSetNameValue(createOpts, "TILED", "YES");
            createOpts = CSLSetNameValue(createOpts, "BLOCKXSIZE", "512");
            createOpts = CSLSetNameValue(createOpts, "BLOCKYSIZE", "512");
            createOpts = CSLSetNameValue(createOpts, "BIGTIFF", "IF_SAFER");
        }

        GDALDataType srcType = srcDs->GetRasterBand(1)->GetRasterDataType();
        GDALDataset* dstDs = driver->Create(
            dstPath.c_str(), dstWidth, dstHeight, nBands,
            srcType, createOpts);
        CSLDestroy(createOpts);

        if (!dstDs) {
            CPLFree(srcWkt); CPLFree(dstWkt); GDALClose(srcDs);
            throw std::runtime_error("출력 파일 생성 실패: " + dstPath);
        }

        dstDs->SetGeoTransform(dstGT);
        dstDs->SetProjection(dstWkt);

        // NoData 설정
        if (opts.hasNoData) {
            for (int b = 1; b <= nBands; ++b)
                dstDs->GetRasterBand(b)->SetNoDataValue(opts.noDataValue);
        }

        // ── 5. 워프 (좌표 변환 + 리샘플링) 실행 ──
        std::cout << "  리샘플링 방법: " << opts.resampleAlg << "\n"
                  << "  워프 실행 중...\n";

        GDALResampleAlg resAlg = parseResampleAlg(opts.resampleAlg);

        // 워프 옵션 설정
        GDALWarpOptions* warpOpts = GDALCreateWarpOptions();
        warpOpts->hSrcDS       = srcDs;
        warpOpts->hDstDS       = dstDs;
        warpOpts->nBandCount   = nBands;
        warpOpts->panSrcBands  = (int*)CPLMalloc(sizeof(int) * nBands);
        warpOpts->panDstBands  = (int*)CPLMalloc(sizeof(int) * nBands);
        for (int b = 0; b < nBands; ++b) {
            warpOpts->panSrcBands[b] = b + 1;
            warpOpts->panDstBands[b] = b + 1;
        }
        warpOpts->eResampleAlg = resAlg;
        warpOpts->dfWarpMemoryLimit = opts.memoryLimit * 1024 * 1024;
        warpOpts->pfnProgress   = GDALTermProgress;

        // 멀티스레드 설정
        if (opts.numThreads > 1) {
            std::string threadStr = "NUM_THREADS=" + std::to_string(opts.numThreads);
            warpOpts->papszWarpOptions = CSLSetNameValue(
                warpOpts->papszWarpOptions,
                "NUM_THREADS", std::to_string(opts.numThreads).c_str());
        }

        // 좌표 변환기 생성
        warpOpts->pTransformerArg = GDALCreateGenImgProjTransformer(
            srcDs, srcWkt, dstDs, dstWkt, FALSE, 0.0, 1);
        if (!warpOpts->pTransformerArg) {
            GDALDestroyWarpOptions(warpOpts);
            CPLFree(srcWkt); CPLFree(dstWkt);
            GDALClose(dstDs); GDALClose(srcDs);
            throw std::runtime_error("좌표 변환기 생성 실패");
        }
        warpOpts->pfnTransformer = GDALGenImgProjTransform;

        // 워프 실행
        GDALWarpOperation warpOp;
        CPLErr err = warpOp.Initialize(warpOpts);
        if (err == CE_None) {
            err = warpOp.ChunkAndWarpImage(0, 0, dstWidth, dstHeight);
        }

        // 정리
        GDALDestroyGenImgProjTransformer(warpOpts->pTransformerArg);
        GDALDestroyWarpOptions(warpOpts);
        CPLFree(srcWkt);
        CPLFree(dstWkt);

        GDALClose(dstDs);
        GDALClose(srcDs);

        if (err != CE_None) {
            report.success = false;
            report.message = "워프 실행 중 오류가 발생했습니다.";
            throw std::runtime_error(report.message);
        }

        report.success = true;
        report.message = "변환 완료";
        std::cout << "  → 변환 완료: " << dstPath << "\n";

        return report;
    }

    // ─── 변환 결과 검증 ───
    static void verify(
        const std::string& dstPath,
        int expectedEpsg,
        double refMinX, double refMaxX,
        double refMinY, double refMaxY)
    {
        GDALAllRegister();

        GDALDataset* ds = (GDALDataset*)GDALOpen(dstPath.c_str(), GA_ReadOnly);
        if (!ds)
            throw std::runtime_error("변환 결과 파일 열기 실패: " + dstPath);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n=== 변환 결과 검증 ===\n";

        // EPSG 확인
        OGRSpatialReference srs;
        const char* wkt = ds->GetProjectionRef();
        if (wkt) srs.importFromWkt(wkt);
        const char* auth = srs.GetAuthorityCode(nullptr);
        int epsg = auth ? std::atoi(auth) : 0;

        if (epsg == expectedEpsg) {
            std::cout << "  ✓ 좌표계 일치: EPSG:" << epsg << "\n";
        } else {
            std::cout << "  ✗ 좌표계 불일치! 기대: EPSG:" << expectedEpsg
                      << " / 실제: EPSG:" << epsg << "\n";
        }

        // 범위 확인
        double gt[6];
        ds->GetGeoTransform(gt);
        int w = ds->GetRasterXSize();
        int h = ds->GetRasterYSize();

        double dstMinX = gt[0];
        double dstMaxY = gt[3];
        double dstMaxX = gt[0] + w * gt[1];
        double dstMinY = gt[3] + h * gt[5];
        if (dstMinY > dstMaxY) std::swap(dstMinY, dstMaxY);

        std::cout << "  변환 후 범위:\n"
                  << "    X: [" << dstMinX << " ~ " << dstMaxX << "]\n"
                  << "    Y: [" << dstMinY << " ~ " << dstMaxY << "]\n";
        std::cout << "  임상도 범위:\n"
                  << "    X: [" << refMinX << " ~ " << refMaxX << "]\n"
                  << "    Y: [" << refMinY << " ~ " << refMaxY << "]\n";

        // 겹침 영역 계산
        double overlapMinX = std::max(dstMinX, refMinX);
        double overlapMaxX = std::min(dstMaxX, refMaxX);
        double overlapMinY = std::max(dstMinY, refMinY);
        double overlapMaxY = std::min(dstMaxY, refMaxY);

        if (overlapMinX < overlapMaxX && overlapMinY < overlapMaxY) {
            double overlapArea = (overlapMaxX - overlapMinX) *
                                 (overlapMaxY - overlapMinY);
            double aerialArea  = (dstMaxX - dstMinX) * (dstMaxY - dstMinY);
            double forestArea  = (refMaxX - refMinX) * (refMaxY - refMinY);

            std::cout << "  ✓ 범위 겹침 확인\n"
                      << "    겹침 영역: X[" << overlapMinX << " ~ "
                      << overlapMaxX << "] Y[" << overlapMinY << " ~ "
                      << overlapMaxY << "]\n"
                      << "    항공사진 대비: "
                      << std::setprecision(1)
                      << (overlapArea / aerialArea * 100.0) << "%\n"
                      << "    임상도 대비  : "
                      << (overlapArea / forestArea * 100.0) << "%\n";
        } else {
            std::cout << "  ✗ 범위 겹침 없음! 변환이 올바른지 확인하세요.\n";
        }

        // 해상도
        std::cout << std::setprecision(4)
                  << "  해상도: " << gt[1] << " m/pixel\n"
                  << "  크기  : " << w << " x " << h << " pixels\n";

        // 파일 크기
        GDALClose(ds);

        std::cout << "  → 검증 완료\n";
    }

    // ─── 리포트 출력 ───
    static void printReport(const ReprojectReport& r) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\n╔══════════════════════════════════════════════╗\n";
        std::cout << "║       좌표계 변환 결과 리포트                ║\n";
        std::cout << "╠══════════════════════════════════════════════╣\n";

        std::cout << "║ [원본]                                       ║\n";
        std::cout << "╟──────────────────────────────────────────────╢\n";
        std::cout << "  EPSG          : " << r.srcEpsg << "\n";
        std::cout << "  크기          : " << r.srcWidth << " x " << r.srcHeight << "\n";
        std::cout << "  해상도        : " << r.srcPixelW << " m/pixel\n";
        std::cout << "  범위 X        : [" << r.srcMinX << " ~ " << r.srcMaxX << "]\n";
        std::cout << "  범위 Y        : [" << r.srcMinY << " ~ " << r.srcMaxY << "]\n";

        std::cout << "╟──────────────────────────────────────────────╢\n";
        std::cout << "║ [변환 결과]                                  ║\n";
        std::cout << "╟──────────────────────────────────────────────╢\n";
        std::cout << "  EPSG          : " << r.dstEpsg << "\n";
        std::cout << "  크기          : " << r.dstWidth << " x " << r.dstHeight << "\n";
        std::cout << "  해상도        : " << r.dstPixelW << " m/pixel\n";
        std::cout << "  범위 X        : [" << r.dstMinX << " ~ " << r.dstMaxX << "]\n";
        std::cout << "  범위 Y        : [" << r.dstMinY << " ~ " << r.dstMaxY << "]\n";

        std::cout << "╟──────────────────────────────────────────────╢\n";
        std::cout << "  상태          : "
                  << (r.success ? "✓ 성공" : "✗ 실패") << "\n";
        if (!r.message.empty())
            std::cout << "  메시지        : " << r.message << "\n";

        std::cout << "╚══════════════════════════════════════════════╝\n";
    }

private:

    // 대상 영역 및 해상도 자동 계산
    static void computeTargetExtent(
        GDALDataset* srcDs,
        const char* srcWkt, const char* dstWkt,
        double reqResX, double reqResY,
        double* dstGT, int& dstWidth, int& dstHeight)
    {
        // 원본 4 코너 좌표를 대상 좌표계로 변환
        double srcGT[6];
        srcDs->GetGeoTransform(srcGT);
        int srcW = srcDs->GetRasterXSize();
        int srcH = srcDs->GetRasterYSize();

        // 원본 코너 좌표 (+ 중간점 포함해 왜곡 보정)
        const int nPts = 9;
        double xs[nPts] = {
            srcGT[0],                                    // 좌상
            srcGT[0] + srcW * 0.5 * srcGT[1],           // 상단 중앙
            srcGT[0] + srcW * srcGT[1],                  // 우상
            srcGT[0],                                    // 좌측 중앙
            srcGT[0] + srcW * 0.5 * srcGT[1],           // 중앙
            srcGT[0] + srcW * srcGT[1],                  // 우측 중앙
            srcGT[0],                                    // 좌하
            srcGT[0] + srcW * 0.5 * srcGT[1],           // 하단 중앙
            srcGT[0] + srcW * srcGT[1],                  // 우하
        };
        double ys[nPts] = {
            srcGT[3],
            srcGT[3],
            srcGT[3],
            srcGT[3] + srcH * 0.5 * srcGT[5],
            srcGT[3] + srcH * 0.5 * srcGT[5],
            srcGT[3] + srcH * 0.5 * srcGT[5],
            srcGT[3] + srcH * srcGT[5],
            srcGT[3] + srcH * srcGT[5],
            srcGT[3] + srcH * srcGT[5],
        };
        double zs[nPts] = {};
        int successes[nPts] = {};

        // 좌표 변환
        OGRSpatialReference oSrcSRS, oDstSRS;
        oSrcSRS.importFromWkt(srcWkt);
        oDstSRS.importFromWkt(dstWkt);

        // GDAL 3.x: 축 순서 명시적 설정 (전통적 lon/lat → X/Y)
        oSrcSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
        oDstSRS.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

        OGRCoordinateTransformation* ct =
            OGRCreateCoordinateTransformation(&oSrcSRS, &oDstSRS);
        if (!ct)
            throw std::runtime_error(
                "좌표 변환기 생성 실패 (EPSG:" +
                std::to_string(0) + " → EPSG:" + std::to_string(0) + ")");

        ct->Transform(nPts, xs, ys, zs, successes);
        OGRCoordinateTransformation::DestroyCT(ct);

        // 변환된 범위 계산
        double minX = xs[0], maxX = xs[0];
        double minY = ys[0], maxY = ys[0];
        for (int i = 1; i < nPts; ++i) {
            if (!successes[i]) continue;
            minX = std::min(minX, xs[i]);
            maxX = std::max(maxX, xs[i]);
            minY = std::min(minY, ys[i]);
            maxY = std::max(maxY, ys[i]);
        }

        // 해상도 결정
        double resX = reqResX;
        double resY = reqResY;
        if (resX <= 0.0 || resY <= 0.0) {
            // 원본 해상도를 대상 좌표계에서 추정
            // 중앙 근처 1픽셀의 변환 후 크기 사용
            double cx1 = srcGT[0] + (srcW/2)     * srcGT[1];
            double cy1 = srcGT[3] + (srcH/2)     * srcGT[5];
            double cx2 = srcGT[0] + (srcW/2 + 1) * srcGT[1];
            double cy2 = srcGT[3] + (srcH/2 + 1) * srcGT[5];

            double txs[2] = {cx1, cx2};
            double tys[2] = {cy1, cy2};
            double tzs[2] = {};
            int tsuc[2] = {};

            OGRSpatialReference s1, s2;
            s1.importFromWkt(srcWkt);
            s2.importFromWkt(dstWkt);
            s1.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
            s2.SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);

            OGRCoordinateTransformation* ct2 =
                OGRCreateCoordinateTransformation(&s1, &s2);
            if (ct2) {
                ct2->Transform(2, txs, tys, tzs, tsuc);
                OGRCoordinateTransformation::DestroyCT(ct2);
                resX = std::abs(txs[1] - txs[0]);
                resY = std::abs(tys[1] - tys[0]);
                // 가장 가까운 깔끔한 값으로 반올림 (0.05m 단위)
                resX = std::round(resX * 20.0) / 20.0;
                resY = std::round(resY * 20.0) / 20.0;
                if (resX < 0.01) resX = 0.05;
                if (resY < 0.01) resY = 0.05;
            } else {
                resX = std::abs(srcGT[1]);
                resY = std::abs(srcGT[5]);
            }
        }

        // GeoTransform 구성
        dstGT[0] = minX;          // 좌상단 X
        dstGT[1] = resX;          // 픽셀 크기 X
        dstGT[2] = 0.0;           // 회전 X
        dstGT[3] = maxY;          // 좌상단 Y
        dstGT[4] = 0.0;           // 회전 Y
        dstGT[5] = -resY;         // 픽셀 크기 Y (음수)

        dstWidth  = (int)std::ceil((maxX - minX) / resX);
        dstHeight = (int)std::ceil((maxY - minY) / resY);

        std::cout << "  자동 계산 해상도: " << resX << " x " << resY << " m\n";
    }

    // 리샘플링 알고리즘 파싱
    static GDALResampleAlg parseResampleAlg(const std::string& name) {
        if (name == "near" || name == "nearest")
            return GRA_NearestNeighbour;
        if (name == "bilinear")
            return GRA_Bilinear;
        if (name == "cubic")
            return GRA_Cubic;
        if (name == "cubicspline")
            return GRA_CubicSpline;
        if (name == "lanczos")
            return GRA_Lanczos;
        if (name == "average")
            return GRA_Average;
        if (name == "mode")
            return GRA_Mode;

        std::cerr << "  ⚠ 알 수 없는 리샘플링 '" << name
                  << "' → bilinear 사용\n";
        return GRA_Bilinear;
    }
};
