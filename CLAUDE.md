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
  - 좌상단: 전체/유지/remove/auto_forest/선택 수, "Selected: removed N / kept M (U -> keep a, Del -> remove b)" (U·Delete 가 실제로 바꿀 그루 수), 모드(NORMAL/RECT/LASSO), 저장 안 된 변경 수, 줌
  - 좌드래그: 팬, 휠: 줌, R: 뷰 리셋
  - 선택은 상태와 무관 (유지·remove·auto_forest 모두 선택됨, X 모드에서 숨긴 나무만 제외)
  - 좌클릭: 화면 15px 이내 가장 가까운 나무 선택 토글 / Shift+드래그: 사각형 선택(추가)
  - L: 올가미 시작 → 클릭으로 꼭짓점 추가 → Enter 확정(선택 추가) / Esc 취소
  - A: 선택 해제, I: 선택 반전
  - Delete: 선택 → remove, U: 선택 → 유지 (remove 된 나무 되살리기), F: 선택 → auto_forest, Ctrl+Z: 마지막 편집 취소 (최대 1000단계, 선택 변경은 대상 아님)
  - 되살리기: 제외된 나무(빨강 X) 클릭/Shift+드래그/올가미로 선택 → U → 녹색. Ctrl+Z 로 취소, S 로 저장해야 파일에 반영
  - 검토: X: remove 된 나무만 표시/전체 토글(숨긴 나무는 선택 불가), N/P: 시작 시점 remove 목록을 순서대로 이동(중앙·줌 8, 그 나무만 선택, 원본 해상도 1536px 창을 GDAL 로 읽어 겹쳐 그림), 좌상단에 TreeID/수고/수관직경/현재 상태. 그 상태에서 U = 그 나무만 되살림
  - 선택 보조: V: 마지막 사각형/올가미 영역(A 로 해제, 없으면 화면 범위) 안의 유지 나무 중 수관 평균 HSV 가 밝고(V >= bright) 채도 낮은(S <= sat) 것을 선택에 추가만 함 (제외는 Delete). [ / ]: bright -/+5, { / }: sat -/+5 (기본 170 / 60), 좌상단 표시
  - 제외는 수동만: edit 시작 시 note 가 auto_ 로 시작하는 행은 삭제 (로그 "auto_* rows reverted: N, remaining rows: M")
  - S: 저장. Q/Esc: 종료 — 저장 안 된 변경이 있으면 콘솔에 "저장하지 않은 변경 N건" 경고 후 한 번 더 눌러야 종료
TreeSpeciesIdentification.exe --experiment review <config> <course>
- review: tree_info_excluded 의 remove 가 맞는지 검토용 산출물 (출력 out\<course>\, auto_forest 는 유지로 집계)
  - review_overview.jpg (1/4, remove=빨강 원+X, 유지=얇은 녹색 원), review_tile_01~06.jpg (원본 1536px, 1536px 그리드에서 remove 많은 칸의 remove 중심, 좌상단에 중심 좌표 EPSG:5179 + 타일 내 remove/유지 수)
  - review_excluded_sheet_NN.jpg (remove 전부, TreeID 순, 96x96 20열 400개/장), review_kept_sheet.jpg (유지 무작위 400, seed 42). 썸네일은 features_dino.csv ImagePath 의 크롭, 없으면 항공사진에서 잘라냄(TreeID 노란색)
  - review_summary.txt: 수고/수관직경 분포, InsideForest 비율, 그리드 상위 10칸(remove/전체, 90% 이상이면 block)
  - remove 행이 없으면(파일 없음 포함) 경고 후 전부 유지로 그림 (타일·제외 시트 없음)
TreeSpeciesIdentification.exe --experiment prepare <config> <course|all>
- prepare: export → review 를 순서대로 실행 (출력 out\<course>\). all = config 에서 enabled 인 코스 전부
  - tree_info_excluded.csv 가 없으면 "제외 파일 없음 - edit 로 먼저 검토 권장" 경고만 하고 진행
  - features_dino.csv 에 그 코스 행이 없으면 export 생략 + "extract needed" (extract 는 실행 안 함), review 는 진행
  - 코스별 오류는 note 에 남기고 다음 코스 계속
  - all: 마지막에 코스별 표 (total=tree_info 전체 / removed / forest(Y)=유지 나무 중 InsideForest==Y / export=trees.json 수) 를 콘솔과 out\prepare_summary.txt 에
- 제외 파일: tree_info 옆 <course>_tree_info_excluded.csv (config 코스 항목 "tree_exclude" 로 경로 지정 가능)
  - 헤더 TreeID,action,note. TreeID 는 tree_info 원본 ID(코스 오프셋 미적용). action = remove | auto_forest, note 는 사유용
  - 편집기 시작 시 읽어서 상태 복원. tree_info 에 없는 ID 행과 note 는 저장 시 보존
  - remove: TreeLoader::load 가 로드 단계에서 제외 (로그 "Excluded by tree_info_excluded: N") → 라벨링·특징·export·render 전부 적용
  - remove: FeatureExtractor::loadCSV 도 config 기준(TreeExclusion::registerConfig, 오프셋 적용 ID)으로 행 제외 → features_*.csv 재추출 불필요
  - auto_forest: TreeLoader 는 유지 (export 단계에서 따로 처리 예정)
- 코스 설정: ..\..\Resources\courses_dataset.json (--config 없을 때 기본값, experiment eval/sweep/lidar 도 이 파일 사용)
  - Hermon(HermonCC) 은 enabled=false 유지 (prepare all 등 모든 작업에서 제외). 항공사진·tree_info·임상도 SHP 는 있으나 features_*.csv 에 없음 → 다시 포함하려면 extract 필요
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