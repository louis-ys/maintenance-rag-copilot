<div align="right">

[English](./README.md) | **한국어**

</div>

# 저압전동기 유지보수 AI Copilot

**Python · Streamlit · RAG · Sentence Transformers · NumPy · Solo Project**

저압전동기 유지보수 매뉴얼을 근거로 질문에 답하고, 근거가 충분한 경우에만 점검 체크리스트와 작업지시서 초안을 생성하는 **evidence-grounded maintenance assistant**입니다.

이 프로젝트는 외부 LLM API 없이 동작합니다. Sentence Transformer 임베딩 검색, 한국어 키워드 보정, 규칙 기반 Supervisor, 근거 제한형 Maintenance Agent를 조합해 **검색 근거 → 업무 분류 → 산출물 생성** 흐름을 구현했습니다.

> **Portfolio prototype**  
> 이 저장소는 독립적인 포트폴리오 프로젝트이며 효성중공업과 제휴·승인된 제품이 아닙니다. 원본 제조사 매뉴얼은 공개 저장소에 재배포하지 않습니다.

---

## What I Built

- PDF 페이지 단위 로딩 및 텍스트 정제
- 600자 / 100자 overlap 기반 페이지 경계 보존 청킹
- `paraphrase-multilingual-MiniLM-L12-v2` 임베딩 검색
- NumPy 기반 cosine similarity
- 의미 점수 + 한국어 keyword bonus 하이브리드 검색
- 상위 근거 3개와 source page 표시
- `insufficient_evidence` 기반 근거 부족 차단
- Supervisor Agent의 업무 범위 / 의도 / 산출물 분류
- 시연용 가상 설비 JSON 조회와 설비 선택 단계
- 근거 기반 점검 체크리스트 / 작업지시서 초안 생성
- out-of-scope / evidence-insufficient 상황의 후속 Agent 실행 차단
- Streamlit UI와 작업지시서 `.txt` 다운로드
- 단위 테스트, 스모크 테스트, 평가 질문 세트

---

## Architecture

```text
User Question
    ↓
Supervisor Agent
    ├─ out_of_scope → Stop
    └─ in_scope
         ↓
Equipment Selection
(maintenance request only)
         ↓
RAG Agent
PDF → page text → chunks → embeddings
         ↓
Hybrid Retrieval
semantic cosine score + keyword bonus
         ↓
Evidence Check
    ├─ insufficient → Stop
    └─ sufficient
         ↓
Maintenance Agent
checklist / work-order draft
         ↓
Orchestrator → Streamlit UI
```

### Supervisor

`src/supervisor_agent.py`

질문을 다음 범주로 분리합니다.

- `manual_query`
- `maintenance_request`
- `out_of_scope`

업무 범위 밖 질문은 RAG 검색 전에 종료합니다.

### RAG

`src/rag_agent.py`, `src/embedding_index.py`

```text
final_score = semantic_score + keyword_bonus
```

의미 임베딩 점수에 한국어 유지보수 관련 키워드 확장 보너스를 제한적으로 더해 검색 순위를 보정합니다. 검색 결과에는 근거 청크뿐 아니라 해당 source page 텍스트도 함께 연결합니다.

### Maintenance Agent

`src/maintenance_agent.py`

검색된 근거 범위 안에서만 점검 체크리스트와 작업지시서 초안을 구성합니다.

- 근거 없는 고장 원인 확정 금지
- 특정 부품 교체 명령 금지
- 근거 부족 시 문서 생성 중단
- 안전 절차 확인 항목 포함
- 최종 판단은 담당자 검토를 전제로 함

### Equipment Repository

`data/equipment/motors.json`은 실제 사업장 정보가 아닌 **시연용 가상 설비 데이터**입니다. 설비 조회는 임베딩 검색이 아니라 일반 JSON 조회로 처리하며, 매뉴얼 근거와 설비 정보 출처를 구분합니다.

---

## Source Manual

개발 기준 문서는 효성중공업에서 공개한 한국어 저압전동기 취급설명서입니다.

공개 포트폴리오 저장소에는 제조사 PDF를 포함하지 않습니다.

실행하려면 공식 사이트에서 매뉴얼을 내려받아 다음 경로에 저장합니다.

```text
data/manuals/hyosung_motor_manual_ko.pdf
```

설정 방법: [`data/manuals/README.md`](data/manuals/README.md)

---

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| Retrieval | Sentence Transformers |
| Similarity | NumPy cosine similarity |
| PDF | pypdf |
| Data | JSON / pandas |
| Testing | pytest |
| Version Control | Git / GitHub |

외부 LLM API, LangChain, LangGraph, FAISS, Chroma는 사용하지 않았습니다.

---

## Run Locally

### 1. Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Add the source manual

공식 제조사 사이트에서 저압전동기 한국어 매뉴얼을 내려받아:

```text
data/manuals/hyosung_motor_manual_ko.pdf
```

로 저장합니다.

### 3. Run tests

```bash
.venv\Scripts\python.exe -m pytest tests -v
```

### 4. RAG smoke test

```bash
.venv\Scripts\python.exe scripts/rag_smoke_test.py
```

### 5. Streamlit

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

---

## Example Scenarios

**Manual query**

> 베어링 그리스는 어떻게 보충해야 하나요?

Supervisor → RAG 검색 → 근거 페이지 표시. Maintenance Agent는 호출하지 않습니다.

**Maintenance request**

> 모터에서 심한 진동과 금속성 소음이 납니다. 점검 체크리스트를 만들어 주세요.

Supervisor → 설비 선택 → RAG → 근거 확인 → Maintenance Agent → 체크리스트 초안.

**Out of scope**

> 오늘 서울 날씨는 어떤가요?

Supervisor 단계에서 종료합니다.

---

## Problem Solving

### PDF 목차 노이즈

PDF 목차의 점선(dot leader)이 임베딩 검색을 왜곡해 실제 본문보다 목차가 상위에 노출되는 문제를 확인했습니다.

`src/pdf_loader.py`의 정제 로직에서 연속 점선을 제거하고 회귀 테스트를 추가했습니다.

### Korean retrieval quality

단순 의미 검색만 사용했을 때 그리스 관련 질문에서 직접적인 정답 페이지보다 간접 관련 페이지가 위에 노출되는 문제가 있었습니다.

`src/keyword_rules.py`의 관련어 확장과 제한된 keyword bonus를 추가해 의미 검색을 유지하면서 도메인 단어 매칭을 보완했습니다.

### Cache invalidation

PDF 또는 청킹 설정뿐 아니라 검색 로직 변경이 기존 임베딩 캐시에 반영되도록 `INDEX_VERSION`을 포함한 캐시 무효화 기준을 적용했습니다.

---

## Repository Structure

```text
app.py
src/
  models.py
  pdf_loader.py
  chunker.py
  keyword_rules.py
  embedding_index.py
  rag_agent.py
  equipment_repository.py
  supervisor_agent.py
  maintenance_agent.py
  orchestrator.py
scripts/
  rag_smoke_test.py
tests/
evaluation/
docs/
data/
  equipment/motors.json
  manuals/README.md
```

---

## Limitations

- 작은 단일 매뉴얼을 대상으로 한 포트폴리오 프로토타입입니다.
- 키워드 확장 규칙은 사전에 정의된 유지보수 표현에 한정됩니다.
- 유사도 임계값은 고정 설정이며 모든 문서/도메인에 일반화된 값이 아닙니다.
- Maintenance Agent는 생성형 LLM이 아니라 근거 기반 규칙/템플릿 방식입니다.
- 실제 정비 판단, 안전 절차, 부품 교체 결정은 현장 담당자와 제조사 지침을 우선해야 합니다.

---

## Development Notes

단계별 문제 해결과 검증 기록은 [`docs/progress_log.md`](docs/progress_log.md)에서 확인할 수 있습니다.
