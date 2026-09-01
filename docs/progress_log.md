# 진행 기록

## 1단계: RAG 핵심 검색

- pypdf로 `data/manuals/hyosung_motor_manual_ko.pdf`를 페이지 단위로 로딩,
  유효 페이지 14개 확보 (빈 페이지 제외).
- 페이지 경계를 넘지 않는 600자/100자 오버랩 청킹으로 청크 49개 생성.
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 임베딩 +
  NumPy 내적 기반 코사인 유사도 검색, 상위 3개 반환.
- 임계값 0.25 기준 `insufficient_evidence` 판정, `data/index/`에 캐시.
- 지정된 4개 질문(진동/소음, 그리스 보충, 과열, 서울 날씨)에 대한 스모크 테스트 통과.

### 목차 점선(dot leader) 문제와 해결

pypdf가 목차 페이지(2페이지)의 점선("……")을 그대로 텍스트로 추출했고, 그 결과
해당 청크의 임베딩이 왜곡되어 목차 페이지가 실제 본문(예: 그리스 주입 방법이 있는
8페이지)보다 높은 유사도로 검색되는 문제가 발견되었다. "베어링 그리스는 어떻게
보충해야 하나요?" 질문에서 상위 3개 결과가 모두 목차 청크였다.

`src/pdf_loader.py`의 `clean_text`에 점선(3개 이상 연속된 "."과 공백) 제거
정규식을 추가해 해결했다. 회귀 테스트(`test_clean_text_removes_toc_dot_leaders`)를
추가해 재발을 방지했다.

## 2단계: 검색 품질 보완 (하이브리드 검색)

1단계 결과에서도 "베어링 그리스는 어떻게 보충해야 하나요?" 질문의 정답 페이지(8페이지,
"4.2.1 그리스 주입 방법")가 3위에 겨우 포함되었고, 1~2위는 간접적으로만 관련된
페이지(11페이지 고장진단표, 14페이지 A/S 연락처)였다.

`src/keyword_rules.py`에 한국어 키워드 확장 규칙(보충→주입/급유/윤활,
그리스→윤활/급유 등)을 추가하고, `src/embedding_index.py`에서
`final_score = semantic_score + keyword_bonus`로 전체 청크를 재정렬하도록 변경했다.
키워드 보너스는 항목당 0.03, 최대 0.15로 제한해 의미 검색 점수를 과도하게
압도하지 않게 했다.

개선 후 그리스 질문의 상위 3개 결과가 모두 8페이지(그리스 주입 방법, 베어링의 윤활,
정지 후 재가동시 주입 방법)로 바뀌어 1~3위를 모두 정답 페이지가 차지했다.

텍스트 정제/청킹/점수 로직 변경이 기존 캐시에 반영되도록 `INDEX_VERSION` 상수를
추가하고 캐시 메타데이터에 저장해, 값이 바뀌면 `data/index/`의 기존 캐시가 자동으로
무효화되고 재생성되도록 구현했다.

## 3단계: Agent 구성과 Streamlit 화면

- `src/supervisor_agent.py`: 모터 업무 키워드 유무로 업무 범위(in_scope)를 판정하고,
  점검표/작업지시서 키워드로 의도(manual_query/maintenance_request/out_of_scope)와
  요청 산출물을 규칙 기반으로 분류. 특정 문장을 하드코딩하지 않고 키워드 매칭만 사용.
- `src/maintenance_agent.py`: RAG 근거만으로 점검 체크리스트(안전정책 고정 항목 +
  근거 기반 항목)와 작업지시서 초안을 결정적 템플릿으로 생성. 근거 부족 시 문서를
  생성하지 않고, "확인 필요" 표현만 사용하며 특정 부품 교체를 지시하지 않음.
- `src/orchestrator.py`: Supervisor → RAG → Maintenance 순서로 실행하고 결과를
  하나의 dict(`query`, `equipment_id`, `supervisor`, `rag`, `maintenance`,
  `final_message`, `safety_notice`)로 반환. out_of_scope/insufficient_evidence일
  때 후속 Agent 호출을 생략.
- `app.py`: Streamlit 화면에서 Supervisor 판단 / RAG 근거 / Maintenance 결과 /
  최종 안내를 네 영역으로 분리해 표시. `st.cache_resource`로 RagAgent를 세션 내
  1회만 로딩하도록 구성. 작업지시서 초안은 .txt로 다운로드 가능.

## 테스트 결과

- 신규 테스트: `tests/test_supervisor.py`, `tests/test_maintenance_agent.py`,
  `tests/test_orchestrator.py` 추가, 기존 `tests/test_pdf_loader.py`,
  `tests/test_chunker.py`는 변경 없이 유지.
- `evaluation/test_cases.csv`에 8개 평가 질문(범위 안/밖, manual_query,
  maintenance_request 단일/복합 산출물 포함) 정리.
- 최종 검증 명령 3종(`pytest`, `rag_smoke_test.py`, `compileall`)을 모두 실행해
  통과를 확인함 (세부 통과 수는 README/최종 보고 참고).
