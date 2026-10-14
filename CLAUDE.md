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
TreeSpeciesIdentification.exe --experiment export <config> out\ <course>
- export: out\<course>\trees.json 생성 (id, x, y(EPSG:5179), height, crownD, embedding[384], suggested=null, confidence=0)
  - 임베딩은 out\features_dino.csv 에서 읽음 (extract 선행 필요). id는 tree_info CSV의 TreeID (코스 오프셋 미적용)
  - 코스 내부 기준: config 코스 항목의 "courseBoundary" SHP. 없으면 경고 후 InsideForest == N 으로 대체
  - out\<course>\excluded_ids.csv 가 있으면 그 TreeID 는 제외 (로그: Excluded by editor)
TreeSpeciesIdentification.exe --experiment render <config> out\ <course> [--skip-forest]
- render: out\<course>\render_overview.jpg (1/4), render_tile_1~3.jpg (원본 2048px, trees.json 밀집 영역)
TreeSpeciesIdentification.exe --experiment edit <config> out\ <course>
- edit: trees.json 나무 유지/삭제 편집기 (TreeEditor.h, ImageViewer 기반). 항공사진은 GDAL로 1/4 축소 로드
  - 색: 녹색=유지, 빨강+X=삭제, 노란 테두리=선택. 좌상단에 전체/유지/삭제/선택 수
  - 좌드래그: 팬, 휠: 줌, R: 뷰 리셋
  - 좌클릭: 가장 가까운 나무 선택 토글 / Shift+드래그: 사각형 선택
  - L: 올가미 시작 → 클릭으로 꼭짓점 추가 → Enter 선택 / Esc 취소 (Esc 로는 종료 안 됨)
  - Delete: 선택 삭제, U: 선택 되살리기, Ctrl+Z: 마지막 Delete/U 취소 (선택 변경은 취소 대상 아님), A: 선택 해제
  - S: out\<course>\excluded_ids.csv 저장 (TreeID 한 열). 시작 시 읽어서 삭제 상태 복원
  - Q: 종료 (저장 안 된 변경이 있으면 Q 두 번)
  - trees.json 에 없는 excluded ID(이미 export 에서 빠진 나무)는 저장 시 그대로 보존됨. 되살리려면 CSV 에서 직접 삭제
  - 흐름: edit → S → export 재실행 → trees.json 에서 제외
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
- 소스 인코딩: CP949(BOM 없음), CRLF — CP949에 없는 문자(예: '═', 이모지) 사용 금지
- 평가 수치는 항상 LOCO 기준으로 보고 (랜덤 k-fold는 비교용)
- 학습 대상은 숲 안의 C/D/E 그룹(10~21, 30~49, 60~68)만. 77(M), 78·81~83·91~99(N), 코드표에 없는 코드는 제외 (FeatureExtractor::categorize, SpeciesCodes::isTrainable)
  - 숲 안이지만 코드가 없거나 reliable이 아닌 나무는 train/predict 어디에도 안 들어가고 Excluded 로그에도 안 잡힘
  - 숲 밖 나무는 코드와 무관하게 전부 predict