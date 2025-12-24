# AEGIS-RAG

멀티테넌트 RAG(Retrieval-Augmented Generation) 시스템입니다.

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [핵심 기능](#핵심-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [구현 상세](#구현-상세)
- [설치 및 실행](#설치-및-실행)
- [성능 및 보안](#성능-및-보안)

---

## 프로젝트 개요

### 배경

기업용 멀티 테넌트 RAG 시스템을 구축할 때 다음과 같은 요구사항이 자주 등장합니다.

- A사 데이터가 B사에 노출되면 안 됩니다
- 누가 얼마나 썼는지 정확히 추적해서 청구해야 합니다
- LLM이 없는 내용을 지어내면 안 됩니다
- 사용자는 즉시 답을 원합니다

이 프로젝트는 이러한 요구사항을 충족하는 RAG 시스템의 아키텍처와 구현을 제시합니다.

### 기술 스택

| 계층 | 기술 | 선택 이유 |
|------|------|----------|
| API | FastAPI | 비동기 지원, 자동 문서화 |
| DB | PostgreSQL + pgvector | 벡터/메타데이터 통합 관리, RLS 지원 |
| 큐 | Redis + Celery | 백그라운드 작업 처리 |
| 임베딩 | Qwen/Qwen3-Embedding-0.6B  | 로컬 실행, 데이터 유출 방지 |
| LLM | Ollama(gpt-oss:20b) | 로컬 실행, 데이터 유출 방지 |
| ORM | SQLAlchemy 2.0 | 타입 안전성, 비동기 지원 |

---

## 핵심 기능

### 1. 데이터베이스 레벨 격리

**구현**: PostgreSQL Row Level Security (RLS)

```sql
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON documents 
  USING (tenant_id = current_setting('app.tenant_id')::UUID);
```

인증 미들웨어가 API 키로 테넌트를 식별하고 세션 변수를 설정합니다. 이후 모든 쿼리는 자동으로 해당 테넌트 데이터만 접근합니다.

### 2. 환각 제어

**구현**: 유사도 임계값 기반 답변 거부

```python
if max(similarities) < 0.7:
    return {"refused": True, "reason": "관련 문서를 찾을 수 없습니다"}
```

LLM이 신뢰할 수 없는 정보를 생성하지 않도록, 검색된 문서의 최대 유사도가 0.7 미만이면 답변을 거부합니다. 추후 유사도 점수를 조정할 예정입니다.

### 3. 문서 버전 관리

**구현**: 버전 이력 추적 및 시점 복원

```sql
ALTER TABLE documents 
    ADD COLUMN version INTEGER DEFAULT 1,
    ADD COLUMN is_latest BOOLEAN DEFAULT true,
    ADD COLUMN previous_version_id UUID,
    ADD COLUMN replaced_at TIMESTAMP;
```

같은 파일 재업로드 시 이전 버전을 보존합니다. 특정 시점의 문서로 쿼리를 재현할 수 있습니다.

```python
# 2023년 12월 31일 시점의 문서로 쿼리
results = await retrieval_service.search(
    query="매출은?",
    as_of_date=datetime(2023, 12, 31)
)
```

**효과**: 감사 추적 완전성, 규제 준수, 실수 복구 가능

### 4. 스트리밍 응답

**구현**: Server-Sent Events (SSE)

```python
@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    async def generate():
        chunks = await retrieval_service.search(request.query)
        yield {"event": "retrieval", "data": {...}}
        
        async for token in llm_service.generate_stream(prompt):
            yield {"event": "token", "data": {"content": token}}
        
        yield {"event": "done", "data": {"cost": ..., "sources": ...}}
    
    return EventSourceResponse(generate())
```

긴 답변도 즉시 시작됩니다. 10초 대기가 1초로 체감됩니다.

### 5. 멀티모달 지원

**구현**: OCR + 테이블 추출 + 이미지 임베딩

```sql
CREATE TABLE document_assets (
    id UUID PRIMARY KEY,
    asset_type VARCHAR(20),  -- 'image', 'table', 'chart'
    extracted_text TEXT,
    structured_data JSONB,
    text_embedding vector(768),
    image_embedding vector(512)
);
```

예시 코드
```python
class MultimodalProcessor:
    async def process_document(self, pdf_path: str):
        images = convert_from_path(pdf_path, dpi=300)
        
        for page_num, image in enumerate(images, 1):
            # 텍스트 추출
            text = pytesseract.image_to_string(image)
            
            # 테이블 감지
            tables = self.table_detector.extract_tables(image)
            for table in tables:
                await self.save_asset({
                    'asset_type': 'table',
                    'structured_data': self.table_to_json(table),
                    'text_embedding': self.embed_text(table)
                })
            
            # 차트/이미지 감지
            figures = self.detect_figures(image)
            for fig in figures:
                caption = self.generate_caption(fig)  # BLIP
                image_emb = self.clip_model.encode(fig)  # CLIP
                await self.save_asset({
                    'asset_type': 'chart',
                    'caption': caption,
                    'image_embedding': image_emb
                })
```

**효과**: 차트 기반 질문 가능 ("2023년 매출 그래프 보여줘")

### 6. 하이브리드 검색

**구현**: BM25 키워드 검색 + 벡터 검색

```python
class HybridRetrievalService:
    async def search(self, query: str, alpha: float = 0.5):
        # 벡터 검색 (의미적 유사도)
        vector_results = await self.vector_search(query, top_k=20)
        
        # BM25 검색 (키워드 매칭)
        bm25_results = await self.bm25_search(query, top_k=20)
        
        # Reciprocal Rank Fusion
        return self.reciprocal_rank_fusion(
            vector_results, bm25_results, alpha=alpha
        )
```

벡터만으로는 정확한 키워드를 놓칠 수 있습니다. BM25와 결합하면 정확도가 올라갑니다.

### 7. 비용 추적 및 감사 로깅

**구현**: 쿼리별 상세 기록

```python
await save_query_log({
    'tenant_id': tenant_id,
    'query_text': query,
    'response_text': response,
    'prompt_tokens': 1234,
    'completion_tokens': 87,
    'cost_usd': 0.0023,
    'latency_ms': 1847,
    'retrieved_chunk_ids': [...],
    'confidence_score': 0.89
})
```

**효과**: 
- 테넌트별 사용량 집계 → 정확한 청구
- 비싼 쿼리 식별 → 최적화 포인트 발견
- 전체 실행 과정 재현 → 디버깅/감사

---

## 시스템 아키텍처

### 전체 흐름

```
┌─────────┐
│ Client  │
└────┬────┘
     │ HTTPS + API Key
     ▼
┌─────────────────────────────────┐
│     FastAPI Server              │
│  ┌──────────────────────────┐  │
│  │ AuthMiddleware           │  │─── tenant_id 추출
│  │ RateLimitMiddleware      │  │─── 10 req/s 제한
│  └──────────────────────────┘  │
│                                 │
│  POST /documents                │
│     ├─ 파일 검증 (SHA-256)      │
│     ├─ DB INSERT (pending)      │
│     └─ Celery Task 큐잉        │
│                                 │
│  POST /query                    │
│     ├─ 쿼리 임베딩 생성         │
│     ├─ 벡터 검색 (pgvector)     │
│     ├─ 신뢰도 검증 (>0.7)       │
│     ├─ LLM 생성 (Ollama)        │
│     └─ 감사 로그 저장           │
└─────────────────────────────────┘
            │
            ▼
     ┌──────────────┐
     │ Celery Queue │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │ Worker       │
     │  ├─ PDF 파싱 │
     │  ├─ 텍스트 청킹 │
     │  ├─ 멀티모달 처리 (OCR, 테이블 등) │
     │  └─ 임베딩 생성 │
     └──────┬───────┘
            │
            ▼
┌─────────────────────────────────┐
│  PostgreSQL + pgvector          │
│  ┌───────────────────────────┐  │
│  │ Row Level Security (RLS) │  │
│  └───────────────────────────┘  │
│                                 │
│  • tenants (API 키, 쿼터)       │
│  • documents (파일, 버전)       │
│  • chunks (텍스트 조각)         │
│  • embeddings (벡터)    │
│  • document_assets (이미지, 표) │
│  • queries (감사 로그)          │
│  • usage_records (비용 집계)    │
└─────────────────────────────────┘
```

### 기술 선택 근거

#### PostgreSQL + pgvector vs 전문 벡터 DB

**선택**: PostgreSQL + pgvector

**이유**:
- 벡터와 메타데이터가 한 DB에 → 조인 간단
- ACID 트랜잭션 → 데이터 일관성 보장
- RLS 지원 → 코드 실수로부터 보호
- 익숙한 도구 (pg_dump, 복제 등)

---

## 구현 상세

### 데이터베이스 스키마

```sql
-- 테넌트
CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    api_key_hash VARCHAR(255) UNIQUE NOT NULL,
    monthly_quota_usd DECIMAL(10,2) DEFAULT 100.00,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 문서 (버전 관리 포함)
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    filename VARCHAR(500) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    version INTEGER DEFAULT 1,
    is_latest BOOLEAN DEFAULT true,
    previous_version_id UUID REFERENCES documents(id),
    replaced_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, content_hash)
);

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_documents ON documents
    USING (tenant_id = current_setting('app.tenant_id')::UUID);

-- 텍스트 청크
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE(document_id, chunk_index)
);

-- Full-Text Search 인덱스
CREATE INDEX chunks_fts_idx ON chunks USING GIN(content_tsv);

-- 벡터 임베딩
CREATE TABLE embeddings (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    vector vector(768),
    model_version VARCHAR(50) NOT NULL,
    UNIQUE(chunk_id)
);

CREATE INDEX embeddings_vector_idx ON embeddings 
    USING ivfflat (vector vector_cosine_ops) WITH (lists = 100);

-- 멀티모달 자산
CREATE TABLE document_assets (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    asset_type VARCHAR(20) NOT NULL,  -- 'image', 'table', 'chart'
    page_number INTEGER,
    extracted_text TEXT,
    structured_data JSONB,
    caption TEXT,
    text_embedding vector(768),
    image_embedding vector(512),
    image_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE document_assets ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_assets ON document_assets
    USING (tenant_id = current_setting('app.tenant_id')::UUID);

-- 쿼리 감사 로그
CREATE TABLE queries (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    query_text TEXT NOT NULL,
    response_text TEXT,
    confidence_score DECIMAL(5,4),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd DECIMAL(10,6),
    latency_ms INTEGER,
    refused BOOLEAN DEFAULT false,
    retrieved_chunk_ids UUID[],
    document_versions JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 사용량 집계
CREATE TABLE usage_records (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id),
    operation_type VARCHAR(50) NOT NULL,
    tokens INTEGER NOT NULL,
    cost_usd DECIMAL(10,6) NOT NULL,
    period_start DATE NOT NULL
);

CREATE INDEX usage_tenant_period_idx ON usage_records(tenant_id, period_start);
```

### 모듈 구조 (레이어드 아키텍처)

```
aegis/
├── presentation/            # Presentation Layer (API, UI)
│   ├── api/
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py      # 문서 업로드/조회 엔드포인트
│   │   │   ├── query.py          # 쿼리 엔드포인트
│   │   │   ├── usage.py          # 사용량 조회 엔드포인트
│   │   │   └── admin.py          # 관리자 엔드포인트
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py           # 인증 미들웨어 (Strategy Pattern)
│   │   │   └── rate_limit.py     # Rate Limiting (Decorator Pattern)
│   │   ├── dependencies.py       # FastAPI Dependency Injection
│   │   ├── schemas.py            # Pydantic DTO (Data Transfer Objects)
│   │   └── main.py               # FastAPI 앱 초기화
│   └── cli/                      # CLI 인터페이스 (선택적)
│       └── commands.py
│
├── application/             # Application Layer (Use Cases)
│   ├── use_cases/
│   │   ├── __init__.py
│   │   ├── upload_document.py    # 문서 업로드 유스케이스
│   │   ├── process_document.py   # 문서 처리 유스케이스
│   │   ├── query_documents.py    # 문서 쿼리 유스케이스
│   │   ├── track_usage.py        # 사용량 추적 유스케이스
│   │   └── manage_tenant.py      # 테넌트 관리 유스케이스
│   ├── dto/                      # Data Transfer Objects
│   │   ├── __init__.py
│   │   ├── document_dto.py
│   │   ├── query_dto.py
│   │   └── usage_dto.py
│   └── interfaces/               # Application Service Interfaces
│       ├── __init__.py
│       ├── embedding_interface.py
│       ├── llm_interface.py
│       └── storage_interface.py
│
├── domain/                  # Domain Layer (Business Logic)
│   ├── entities/            # Domain Entities (Pure Business Objects)
│   │   ├── __init__.py
│   │   ├── tenant.py            # Tenant 엔티티
│   │   ├── document.py          # Document 엔티티 (버전 관리 로직)
│   │   ├── chunk.py             # Chunk 엔티티
│   │   ├── embedding.py         # Embedding 엔티티
│   │   ├── document_asset.py    # DocumentAsset 엔티티
│   │   ├── query.py             # Query 엔티티
│   │   └── usage_record.py      # UsageRecord 엔티티
│   ├── value_objects/       # Value Objects (Immutable)
│   │   ├── __init__.py
│   │   ├── api_key.py           # API Key Value Object
│   │   ├── tenant_id.py         # TenantId Value Object
│   │   ├── document_version.py  # DocumentVersion Value Object
│   │   └── cost.py              # Cost Value Object
│   ├── services/            # Domain Services (Business Logic)
│   │   ├── __init__.py
│   │   ├── document_version_service.py  # 버전 관리 로직
│   │   ├── confidence_service.py        # 신뢰도 계산
│   │   ├── cost_calculator.py           # 비용 계산
│   │   └── text_chunker.py              # 텍스트 청킹 전략
│   ├── repositories/        # Repository Interfaces (Abstract)
│   │   ├── __init__.py
│   │   ├── tenant_repository.py
│   │   ├── document_repository.py
│   │   ├── chunk_repository.py
│   │   ├── embedding_repository.py
│   │   ├── document_asset_repository.py
│   │   ├── query_repository.py
│   │   └── usage_record_repository.py
│   ├── events/              # Domain Events (Event-Driven)
│   │   ├── __init__.py
│   │   ├── document_uploaded.py
│   │   ├── document_processed.py
│   │   └── query_executed.py
│   └── exceptions/          # Domain Exceptions
│       ├── __init__.py
│       ├── document_exceptions.py
│       ├── tenant_exceptions.py
│       └── query_exceptions.py
│
├── infrastructure/          # Infrastructure Layer (External Dependencies)
│   ├── persistence/         # Database (Repository Pattern 구현)
│   │   ├── sqlalchemy/
│   │   │   ├── __init__.py
│   │   │   ├── session.py           # DB Session Factory
│   │   │   ├── models.py            # SQLAlchemy ORM Models
│   │   │   ├── unit_of_work.py      # Unit of Work Pattern
│   │   │   ├── repositories/        # Repository 구현체
│   │   │   │   ├── __init__.py
│   │   │   │   ├── tenant_repo_impl.py
│   │   │   │   ├── document_repo_impl.py
│   │   │   │   ├── chunk_repo_impl.py
│   │   │   │   ├── embedding_repo_impl.py
│   │   │   │   ├── document_asset_repo_impl.py
│   │   │   │   ├── query_repo_impl.py
│   │   │   │   └── usage_record_repo_impl.py
│   │   │   └── migrations/          # Alembic
│   │   │       └── versions/
│   │   └── redis/
│   │       ├── __init__.py
│   │       └── cache.py             # Redis 캐시
│   ├── embeddings/          # Embedding Services (Strategy Pattern)
│   │   ├── __init__.py
│   │   ├── base.py                  # Embedding Strategy Interface
│   │   ├── sentence_transformer.py  # SentenceTransformer 구현
│   │   ├── clip_encoder.py          # CLIP 구현
│   │   └── factory.py               # Factory Pattern (임베딩 생성)
│   ├── llm/                 # LLM Services (Strategy Pattern)
│   │   ├── __init__.py
│   │   ├── base.py                  # LLM Strategy Interface
│   │   ├── ollama_client.py         # Ollama 구현
│   │   └── factory.py               # Factory Pattern (LLM 생성)
│   ├── retrieval/           # Retrieval Strategies (Strategy Pattern)
│   │   ├── __init__.py
│   │   ├── base.py                  # Retrieval Strategy Interface
│   │   ├── vector_retrieval.py      # 벡터 검색
│   │   ├── bm25_retrieval.py        # BM25 검색
│   │   ├── hybrid_retrieval.py      # 하이브리드 검색
│   │   └── factory.py               # Factory Pattern (검색 전략 생성)
│   ├── processors/          # Document Processors (Chain of Responsibility)
│   │   ├── __init__.py
│   │   ├── base.py                  # Processor Interface
│   │   ├── pdf_parser.py            # PDF 파싱
│   │   ├── table_extractor.py       # 테이블 추출
│   │   ├── image_captioner.py       # 이미지 캡셔닝
│   │   └── processor_chain.py       # Chain of Responsibility Pattern
│   ├── messaging/           # Message Queue
│   │   ├── __init__.py
│   │   ├── celery_app.py            # Celery 설정
│   │   └── tasks.py                 # Celery Tasks
│   └── monitoring/          # Logging & Monitoring
│       ├── __init__.py
│       ├── logger.py                # Structured Logging
│       └── metrics.py               # Prometheus Metrics
│
└── shared/                  # Shared Kernel (공통 유틸리티)
    ├── __init__.py
    ├── config.py            # 설정 관리 (Singleton Pattern)
    ├── constants.py         # 상수
    ├── utils.py             # 유틸리티 함수
    └── types.py             # 공통 타입 정의
```

### 적용된 디자인 패턴

#### 1. Repository Pattern
- **목적**: 데이터 접근 로직을 도메인 로직으로부터 분리
- **위치**: `domain/repositories/` (인터페이스), `infrastructure/persistence/sqlalchemy/repositories/` (구현)
- **효과**: DB 변경 시 도메인 레이어 영향 없음

#### 2. Unit of Work Pattern
- **목적**: 트랜잭션 경계 관리, 여러 Repository 작업을 하나의 트랜잭션으로 묶음
- **위치**: `infrastructure/persistence/sqlalchemy/unit_of_work.py`
- **효과**: 데이터 일관성 보장, 복잡한 트랜잭션 관리 단순화

```python
async with UnitOfWork() as uow:
    document = await uow.documents.create(...)
    chunks = await uow.chunks.create_batch(...)
    await uow.commit()
```

#### 3. Strategy Pattern
- **목적**: 알고리즘을 런타임에 교체 가능하도록
- **적용 영역**:
  - 임베딩 전략 (`infrastructure/embeddings/`)
  - LLM 전략 (`infrastructure/llm/`)
  - 검색 전략 (`infrastructure/retrieval/`)
- **효과**: 새로운 임베딩 모델이나 LLM 추가가 용이

```python
# 임베딩 전략 교체
embedding_service = EmbeddingFactory.create("sentence-transformer")
embedding_service = EmbeddingFactory.create("clip")  # 런타임 변경
```

#### 4. Factory Pattern
- **목적**: 객체 생성 로직 캡슐화
- **위치**: 각 Strategy 디렉토리의 `factory.py`
- **효과**: 의존성 주입 간소화, 객체 생성 로직 중앙화

#### 5. Dependency Injection
- **목적**: 의존성을 외부에서 주입하여 결합도 감소
- **위치**: `presentation/api/dependencies.py`
- **효과**: 테스트 용이, 모의 객체 주입 가능

```python
@router.post("/query")
async def query(
    request: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    generation_service: GenerationService = Depends(get_generation_service)
):
    ...
```

#### 6. Chain of Responsibility Pattern
- **목적**: 요청을 여러 핸들러 체인으로 전달
- **위치**: `infrastructure/processors/processor_chain.py`
- **효과**: 문서 처리 파이프라인 유연하게 구성

```python
processor_chain = (
    PDFParser()
    .set_next(TableExtractor())
    .set_next(ImageCaptioner())
)
result = await processor_chain.process(document)
```

#### 7. Value Object Pattern
- **목적**: 불변 값 객체로 도메인 개념 표현
- **위치**: `domain/value_objects/`
- **효과**: 타입 안정성, 비즈니스 규칙 캡슐화

```python
api_key = ApiKey.generate()  # 자동 검증
tenant_id = TenantId(uuid.uuid4())  # 불변
```

#### 8. Event-Driven Architecture
- **목적**: 도메인 이벤트로 느슨한 결합
- **위치**: `domain/events/`
- **효과**: 시스템 확장성 향상, 감사 로깅 자동화

```python
await event_bus.publish(DocumentUploaded(
    document_id=doc.id,
    tenant_id=tenant.id,
    timestamp=datetime.now()
))
```

### API 엔드포인트

#### POST /documents

문서 업로드

```bash
curl -X POST http://localhost:8000/documents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@report.pdf"
```

**응답**:
```json
{
  "id": "550e8400-...",
  "filename": "report.pdf",
  "status": "processing",
  "version": 1
}
```

#### GET /documents/{filename}/versions

버전 이력 조회

```bash
curl http://localhost:8000/documents/report.pdf/versions \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**응답**:
```json
[
  {
    "version": 2,
    "is_latest": true,
    "created_at": "2024-12-22T10:30:00Z"
  },
  {
    "version": 1,
    "is_latest": false,
    "replaced_at": "2024-12-22T10:30:00Z"
  }
]
```

#### POST /query

일반 쿼리

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "2023년 3분기 매출은?",
    "top_k": 5,
    "min_confidence": 0.7,
    "as_of_date": "2023-12-31"
  }'
```

**성공 응답**:
```json
{
  "query_id": "770e8400-...",
  "answer": "2023년 3분기 매출은 42억원입니다.",
  "confidence": 0.89,
  "refused": false,
  "sources": [
    {
      "chunk_id": "880e8400-...",
      "similarity": 0.91,
      "content": "Q3 FY2023 매출은 42억원..."
    }
  ],
  "cost_usd": 0.0023,
  "latency_ms": 1847
}
```

**거부 응답**:
```json
{
  "query_id": "770e8400-...",
  "refused": true,
  "reason": "관련 문서를 찾을 수 없습니다 (최대 유사도: 0.62)"
}
```

#### POST /query/stream

스트리밍 쿼리

```javascript
const eventSource = new EventSource('/query/stream');

eventSource.addEventListener('retrieval', (e) => {
    console.log(`Found ${JSON.parse(e.data).chunks_found} chunks`);
});

eventSource.addEventListener('token', (e) => {
    appendToAnswer(JSON.parse(e.data).content);
});

eventSource.addEventListener('done', (e) => {
    displayCost(JSON.parse(e.data).cost_usd);
    eventSource.close();
});
```

#### GET /usage

사용량 조회

```bash
curl "http://localhost:8000/usage?period_start=2024-12-01&period_end=2024-12-31" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**응답**:
```json
{
  "total_cost_usd": 12.47,
  "monthly_quota_usd": 100.00,
  "usage_pct": 12.47,
  "breakdown": {
    "embedding_cost_usd": 2.35,
    "query_cost_usd": 10.12
  }
}
```

#### GET /audit/{query_id}

감사 로그 조회

```bash
curl http://localhost:8000/audit/770e8400-... \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**응답**:
```json
{
  "query_id": "770e8400-...",
  "created_at": "2024-12-22T14:30:00Z",
  "input": {"query": "매출은?", "top_k": 5},
  "retrieval": {
    "chunks_searched": 1247,
    "chunks_retrieved": 5,
    "max_similarity": 0.91
  },
  "generation": {
    "model": "mistral:7b",
    "prompt_tokens": 1234,
    "completion_tokens": 87,
    "latency_ms": 1758
  },
  "cost_usd": 0.0023
}
```

---

## 설치 및 실행

### 요구사항

- Python 3.12+
- Docker & Docker Compose
- Ollama
- GPU (권장, CPU는 느림)

### 설치

### 사용 예시

## 성능 및 보안

### 성능 벤치마크

| 지표 | 목표 | 실제 |
|------|------|----------------|
| 쿼리 응답 (p95) | < 3초 |  |
| 벡터 검색 (10만) | < 100ms |  |
| 동시 쿼리 | 100 req/s |  |
| 문서 처리 | 100 페이지/분 |  |

### 보안

- **데이터 격리**: PostgreSQL RLS (DB 레벨 강제)
- **인증**: API 키 (SHA-256 해시 저장)
- **전송**: HTTPS 필수
- **Rate Limiting**: 테넌트별 10 req/s

### 테스트

```bash
pytest                    # 전체
pytest tests/unit         # 단위 테스트
pytest --cov=aegis        # 커버리지
```

---

## 라이선스

MIT License
