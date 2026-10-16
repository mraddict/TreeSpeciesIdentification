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
- export: out\<course>\trees.json 생성 — remove 를 뺀 모든 나무 (config "courseBoundary" SHP 가 있으면 그 안만)
  - 공통 필드: id(tree_info TreeID, 코스 오프셋 미적용), x, y(EPSG:5179), height, crownD, embedding, mode, suggested, source, confidence
  - review: mode="review", embedding[384](out\features_dino.csv, 없으면 null), suggested=null, source=null, confidence=0
  - 나머지(파일에 없음 = auto_forest 기본값, 명시 auto_forest, no_image): mode="auto_forest", embedding=null,
    suggested=임상도 최근접 수종 폴리곤(C/D/E/M)의 SpeciesCodes::asset(code), source="forest:<code>",
    confidence 폴리곤 안 0.9 / 50m 이내 0.6 / 그 밖 0.3 / 폴리곤 없음 0(suggested=null, source="forest:none")
  - SpeciesCodes::asset(code) 는 영문 이름 snake_case (예: 11 → pine, 77 → mixed_forest) 임시 키. 실제 에셋 카탈로그가 정해지면 교체
TreeSpeciesIdentification.exe --experiment render <config> out\ <course> [--skip-forest]
- render: out\<course>\render_overview.jpg (1/4), render_tile_1~3.jpg (원본 2048px, trees.json 밀집 영역)
TreeSpeciesIdentification.exe --experiment edit <config> <course>
- edit: tree_info 의 모든 나무를 항공사진 위에 표시하고 action 을 지정하는 편집기 (TreeEditor.h, ImageViewer 기반)
  - 항공사진은 GDAL 로 1/4 축소본만 로드. 원본 tree_info CSV 는 절대 수정하지 않음
  - 색: 얇은 녹색=auto_forest(기본값, 파일에 없음), 굵은 밝은 녹색=review, 빨강+X=remove, 회색=no_image, 노란 테두리=선택. 줌이 작으면 점만 표시
  - 좌상단: 전체 / auto_forest / remove / no_image / review / 선택 수, "Selected: ... (Del -> no_image a, Shift+Del -> remove b, R -> review c, U -> auto d)" (각 키가 실제로 바꿀 그루 수), 모드 + clearSelectionAfterAction 값, 저장 안 된 변경 수, 줌
  - 좌드래그: 팬, 휠: 줌, Home: 뷰 리셋 (R 은 review)
  - 선택은 상태와 무관 (네 상태 모두 선택됨, X 모드에서 숨긴 나무만 제외)
  - 좌클릭: 화면 15px 이내 가장 가까운 나무 선택 토글 / Ctrl+클릭: 가장 가까운 선택된 나무 하나만 해제
  - Shift+드래그: 사각형 선택(추가) / Ctrl+Shift+드래그: 사각형 안 선택 해제
  - L: 올가미 시작 → 클릭으로 꼭짓점 추가 → Enter 확정(선택 추가) / Backspace(안쪽 선택 해제) / Esc 취소
  - A: 선택 전부 해제(+ V/B 범위 해제), I: 선택 반전
  - Delete: 선택 → no_image, Shift+Delete: 선택 → remove, R: 선택 → review, U: 선택 → auto_forest(기본값, 저장 시 파일에서 행 삭제), Ctrl+Z: 마지막 편집 취소 (최대 1000단계, 선택 변경은 대상 아님). F 는 없음
  - clearSelectionAfterAction = true (기본): Delete/Shift+Delete/R/U 적용 후 선택 자동 해제 → 다음 올가미 선택과 합쳐지지 않음. C 키로 켜고 끔
  - B: remove 상태이면서 InsideForest==Y(out\features_dino.csv) 인 나무를 선택에 추가. 범위 = 마지막 사각형/올가미 영역(A 로 해제), 없으면 코스 전체
  - 한글 입력 상태에서 눌린 자모(ㅁ, ㅎ 등)는 2벌식 위치의 영문 키로 처리
  - 검토: X: remove 된 나무만 표시/전체 토글(숨긴 나무는 선택 불가), N/P: 시작 시점 remove 목록을 순서대로 이동(중앙·줌 8, 그 나무만 선택, 원본 해상도 1536px 창을 GDAL 로 읽어 겹쳐 그림), 좌상단에 TreeID/수고/수관직경/현재 상태
  - 선택 보조: V: 마지막 사각형/올가미 영역(없으면 화면) 안의 auto_forest 나무 중 수관 평균 HSV 가 밝고(V >= bright) 채도 낮은(S <= sat) 것을 선택에 추가만 함. [ / ]: bright -/+5, { / }: sat -/+5 (기본 170 / 60)
  - edit 시작 시 note 가 auto_ 로 시작하는 행은 삭제 (로그 "auto_* rows reverted: N, remaining rows: M")
  - S: 저장. Q/Esc: 종료 — 저장 안 된 변경이 있으면 콘솔에 "저장하지 않은 변경 N건" 경고 후 한 번 더 눌러야 종료
TreeSpeciesIdentification.exe --experiment review <config> <course>
- review: tree_info_excluded 의 remove 가 맞는지 검토용 산출물 (출력 out\<course>\, remove 외는 유지로 집계)
  - review_overview.jpg (1/4, remove=빨강 원+X, 유지=얇은 녹색 원), review_tile_01~06.jpg (원본 1536px, 1536px 그리드에서 remove 많은 칸의 remove 중심, 좌상단에 중심 좌표 EPSG:5179 + 타일 내 remove/유지 수)
  - review_excluded_sheet_NN.jpg (remove 전부, TreeID 순, 96x96 20열 400개/장), review_kept_sheet.jpg (유지 무작위 400, seed 42). 썸네일은 features_dino.csv ImagePath 의 크롭, 없으면 항공사진에서 잘라냄(TreeID 노란색)
  - review_summary.txt: 수고/수관직경 분포, InsideForest 비율, 그리드 상위 10칸(remove/전체, 90% 이상이면 block)
  - remove 행이 없으면(파일 없음 포함) 경고 후 전부 유지로 그림 (타일·제외 시트 없음)
TreeSpeciesIdentification.exe --experiment prepare <config> <course|all>
- prepare: export → review 를 순서대로 실행 (출력 out\<course>\). all = config 에서 enabled 인 코스 전부
  - tree_info_excluded.csv 가 없으면 "제외 파일 없음 - edit 로 먼저 검토 권장" 경고만 하고 진행
  - features_dino.csv 에 그 코스 행이 없으면 export 생략 + "extract needed" (extract 는 실행 안 함), review 는 진행
  - 코스별 오류는 note 에 남기고 다음 코스 계속
  - all: 마지막에 코스별 표 (total / removed / no_image / review / forest(Y)=유지 나무 중 InsideForest==Y 이고 이미지 사용 가능 / export=trees.json 전체) 를 콘솔과 out\prepare_summary.txt 에
TreeSpeciesIdentification.exe --experiment migrate-excluded <config> <course|all>
- migrate-excluded: 옛 remove 행을 remove(구조물, note "structure") / no_image(note "low-quality-aerial") 로 나눔
  - 규칙: features_handcrafted.csv 의 V_mean >= tv 이고 S_mean <= ts 면 구조물. tv/ts 는 Hwarang 행(.bak 우선)의 TreeID 35518~35890(비닐하우스) / 36166~37367(숲) 으로 0~255 격자 탐색(균형 정확도 최대, 동률은 중심)
  - 2026-10-16 실행 결과: tv=91, ts=87, 균형 정확도 66.4% (비닐하우스 24/67, 숲 198/204). 수관 색만으로는 잘 안 갈림 — 청색 지붕은 채도가 높아 no_image 로, 도로/맨땅/저품질 녹색 채움 일부는 remove 로 감. ExG(2G-R-B) 는 56.8% 로 더 나쁨
  - features 행이 없는 remove 는 no_image ("low-quality-aerial; no feature row"). no_image/review/auto_forest 행은 그대로
  - 백업: <tree_info stem>_excluded.bak.csv (이미 있으면 덮어쓰지 않음). 다시 돌리면 현재 파일의 remove 행만 재분류 (no_image 로 바뀐 행은 되돌리지 않음)
  - 검증 시트: out\<course>\review_remove_sheet.jpg (remove 첫 400), review_noimage_sheet_01.jpg (no_image 첫 400). 크롭이 없으면 원본 항공사진에서 GDAL 창 읽기
- 제외 파일: tree_info 옆 <course>_tree_info_excluded.csv (config 코스 항목 "tree_exclude" 로 경로 지정 가능)
  - 헤더 TreeID,action,note. TreeID 는 tree_info 원본 ID(코스 오프셋 미적용). note 는 사유용
  - 파일에 없는 나무 = auto_forest (기본값): 디자이너가 보지 않고 임상도 최근접 폴리곤으로 에셋 자동 지정
  - action: remove(나무 아님, TreeLoader 에서 제거) / no_image(나무지만 항공사진 저품질: 유지, 학습·평가·전파 제외, 에셋은 임상도 자동) / auto_forest(기본값과 같음, 보통 안 씀) / review(디자이너가 직접 판단, export mode "review")
  - 편집기 시작 시 읽어서 상태 복원. tree_info 에 없는 ID 행과 note 는 저장 시 보존. 명시 auto_forest 행은 기본값으로 읽혀 저장 시 빠짐
  - remove: TreeLoader::load 가 로드 단계에서 제외 (로그 "Excluded by tree_info_excluded: N") → 라벨링·특징·export·render 전부 적용
  - remove: FeatureExtractor::loadCSV 도 config 기준(TreeExclusion::registerConfig, 오프셋 적용 ID)으로 행 제외 → features_*.csv 재추출 불필요
  - no_image: TreeLabeler/FeatureExtractor::loadCSV 에서 imageUsable=false → categorize 가 train·predict 어디에도 안 넣음 (로그 "Excluded: image unusable (no_image) N")
  - 로그: TreeLoader "tree_info_excluded rows: remove N, no_image N, review N, auto_forest (explicit) N", registerConfig 도 코스별 같은 형식
  - review/prepare: no_image·review 는 유지로 집계, forest(Y) 는 no_image 제외
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