# TreeSpeciesIdentification

스크린골프 3D 코스 제작용 수종/에셋 판별 파이프라인. C++17, VS2022, Windows 11.

## 빌드
msbuild TreeSpeciesIdentification.sln /p:Configuration=Release /p:Platform=x64 /v:m
- 의존성: GDAL/OpenCV는 vcpkg(x64-windows), ONNX Runtime은 NuGet(Microsoft.ML.OnnxRuntime)
- 출력: x64\Release\TreeSpeciesIdentification.exe

## 실행 (작업 디렉터리 = 프로젝트 루트)
TreeSpeciesIdentification.exe                     기존 파이프라인
TreeSpeciesIdentification.exe experiment extract <config> out\ <onnx>
TreeSpeciesIdentification.exe experiment eval|sweep|lidar out\
- 코스 설정: ..\..\Resources\courses_dataset.json
- 모델: ..\..\Resources\models\dinov2_vits14.onnx
- 로그: results\ 및 out\logs\

## 구조
- TreeLoader/ForestLoader/AerialLoader: 입력 로딩, 좌표계 EPSG:5179 통일
- TreeLabeler: 임상도 폴리곤으로 라벨 부여, 수관 크롭 저장
- FeatureExtractor: Handcrafted(25) / Dino(392) 특징, DinoEmbedder(ONNX)
- SpeciesCodes.h: 임상도 코드 테이블의 단일 출처 — 코드/그룹/색상은 여기만 수정
- CourseCrossValidator: leave-one-course-out 평가 (코스 = treeId / 100000)
- SpeciesClassifier/HierarchicalClassifier: OpenCV RF

## 규칙
- 헤더 only 스타일, 탭 들여쓰기, Allman 중괄호, 한글 주석 허용
- 소스 인코딩: UTF-8 BOM, CRLF
- 평가 수치는 항상 LOCO 기준으로 보고 (랜덤 k-fold는 비교용)
- 코드 80번대·77은 학습 제외 (FeatureExtractor::categorize)