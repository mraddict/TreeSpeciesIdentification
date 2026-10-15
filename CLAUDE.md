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
TreeSpeciesIdentification.exe --experiment render <config> out\ <course> [--skip-forest]
- render: out\<course>\render_overview.jpg (1/4), render_tile_1~3.jpg (원본 2048px, trees.json 밀집 영역)
TreeSpeciesIdentification.exe --experiment edit <config> <course>
- edit: tree_info 의 모든 나무를 항공사진 위에 표시하고 오검출을 표시하는 편집기 (TreeEditor.h, ImageViewer 기반)
  - 항공사진은 GDAL 로 1/4 축소본만 로드. 원본 tree_info CSV 는 절대 수정하지 않음
  - 색: 녹색=유지, 빨강+X=remove, 파랑=auto_forest, 노란 테두리=선택. 줌이 작으면 점만 표시
  - 좌상단: 전체/유지/remove/auto_forest/선택 수, 모드(NORMAL/RECT/LASSO), 저장 안 된 변경 수, 줌
  - 좌드래그: 팬, 휠: 줌, R: 뷰 리셋
  - 좌클릭: 화면 15px 이내 가장 가까운 나무 선택 토글 / Shift+드래그: 사각형 선택(추가)
  - L: 올가미 시작 → 클릭으로 꼭짓점 추가 → Enter 확정(선택 추가) / Esc 취소
  - A: 선택 해제, I: 선택 반전
  - Delete: 선택 → remove, U: 선택 → 유지, F: 선택 → auto_forest, Ctrl+Z: 마지막 편집 취소 (최대 1000단계, 선택 변경은 대상 아님)
  - S: 저장. Q/Esc: 종료 — 저장 안 된 변경이 있으면 콘솔에 "저장하지 않은 변경 N건" 경고 후 한 번 더 눌러야 종료
- 제외 파일: tree_info 옆 <course>_tree_info_excluded.csv (config 코스 항목 "tree_exclude" 로 경로 지정 가능)
  - 헤더 TreeID,action,note. TreeID 는 tree_info 원본 ID(코스 오프셋 미적용). action = remove | auto_forest, note 는 사유용
  - 편집기 시작 시 읽어서 상태 복원. tree_info 에 없는 ID 행과 note 는 저장 시 보존
  - remove: TreeLoader::load 가 로드 단계에서 제외 (로그 "Excluded by tree_info_excluded: N") → 라벨링·특징·export·render 전부 적용
  - remove: FeatureExtractor::loadCSV 도 config 기준(TreeExclusion::registerConfig, 오프셋 적용 ID)으로 행 제외 → features_*.csv 재추출 불필요
  - auto_forest: TreeLoader 는 유지 (export 단계에서 따로 처리 예정)
- 코스 설정: ..\..\Resources\courses_dataset.json (--config 없을 때 기본값, experiment eval/sweep/lidar 도 이 파일 사용)
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