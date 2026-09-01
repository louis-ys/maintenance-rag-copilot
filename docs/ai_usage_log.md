# AI 활용 내역

이 프로젝트에서 사용한 AI 도구와 각 도구의 역할, 그리고 최종 의사결정 주체를 기록한다.

## ChatGPT

- 프로젝트 전체 구조 설계 (Supervisor / RAG / Maintenance Agent 3단 구조, Orchestrator 흐름)
- 업무 범위(in_scope/out_of_scope) 조정 및 안전장치(부품 교체 미지시, 근거 부족 시 문서
  미생성 등) 정책 수립
- Claude Code가 작성한 코드에 대한 검토 및 개선 방향 제안

## Claude Code

- `src/`, `app.py`, `tests/` 등 실제 코드 초안 작성
- PDF 로더, 청킹, 임베딩 검색, 하이브리드 검색(키워드 보너스), Supervisor/Maintenance
  Agent, Orchestrator, Streamlit 화면 구현
- pytest 테스트 작성 및 실행, 발견된 문제(목차 점선 노이즈, 그리스 질문 순위 등) 수정
- `python -m compileall`, `pytest`, 스모크 테스트 스크립트 직접 실행 및 결과 확인

## 사용자 (프로젝트 담당자)

- 요구사항 정의 및 단계별 지시 (1단계 RAG → 최종 MVP)
- Git 커밋 등 되돌리기 어려운 작업의 최종 승인
- 각 단계 산출물의 실행 결과 검증 및 최종 의사결정
- 심사 제출 여부 등 프로젝트 관련 모든 최종 판단
