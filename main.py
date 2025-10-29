from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import boto3
import json
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram 
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="My API", version="1.0.0")

# 앱에 메트릭 수집기 적용 및 /metrics 엔드포인트 노출
Instrumentator().instrument(app).expose(app)

# Counter: 오직 증가만 하는 값 (예: 생성된 총 아이템 수, 발생한 에러 수)
ITEMS_CREATED_COUNTER = Counter(
    "items_created_total", 
    "Total number of items created",
    ["item_type"]  # 'item_type'별로 구분해서 집계 (라벨)
)

# Gauge: 오르내릴 수 있는 값 (예: 현재 활성 사용자 수, 현재 DB 커넥션 수)
ACTIVE_CONNECTIONS_GAUGE = Gauge(
    "active_database_connections",
    "Current number of active database connections"
)

# Histogram: 분포(시간, 크기)를 측정 (예: 특정 함수의 처리 시간)
DB_QUERY_DURATION_SECONDS = Histogram(
    "db_query_duration_seconds",
    "Histogram for database query durations",
    ["query_type"] # 'query_type'별로 구분
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLAlchemy 설정
Base = declarative_base()

# 데이터베이스 연결 정보를 저장할 전역 변수
DATABASE_URL = None
engine = None
SessionLocal = None

class Item(Base):
    __tablename__ = "items"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic 모델
class ItemCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

def get_db_credentials():
    """AWS Secrets Manager에서 DB 연결 정보 가져오기"""
    secret_name = os.getenv("DB_SECRET_NAME", "myapp/database")
    region_name = os.getenv("AWS_REGION", "ap-northeast-2")
    
    try:
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )
        
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(get_secret_value_response['SecretString'])
        
        return secret
    except Exception as e:
        logger.error(f"Error retrieving secret: {e}")
        # 로컬 개발 환경을 위한 fallback
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "myapp"),
            "username": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "password")
        }

def init_database():
    """데이터베이스 초기화"""
    global DATABASE_URL, engine, SessionLocal
    
    try:
        db_creds = get_db_credentials()
        
        # PostgreSQL 연결 URL 생성
        DATABASE_URL = f"postgresql://{db_creds['username']}:{db_creds['password']}@{db_creds['host']}:{db_creds['port']}/{db_creds['database']}"
        
        engine = create_engine(
            DATABASE_URL,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800
        )

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # 테이블 생성
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

def get_db():
    """데이터베이스 세션 의존성"""
    if SessionLocal is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    #Gauge 사용 (커넥션 생성 시 +1)
    ACTIVE_CONNECTIONS_GAUGE.inc() 
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        #커넥션 반환 시 -1)
        ACTIVE_CONNECTIONS_GAUGE.dec()

@app.on_event("startup")
async def startup_event():
    """앱 시작 시 실행"""
    init_database()
    logger.info("Application startup complete")

@app.get("/")
async def root():
    """헬스 체크 엔드포인트"""
    return {
        "message": "API is running",
        "version": "1.1.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """상세 헬스 체크"""
    db_status = "disconnected"
    try:
        if engine is not None:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            db_status = "connected"
    except Exception as e:
        logger.error(f"Health check failed: {e}")
    
    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/items", response_model=List[ItemResponse])
async def get_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """모든 아이템 조회"""
    items = db.query(Item).offset(skip).limit(limit).all()
    return items

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int, db: Session = Depends(get_db)):
    """특정 아이템 조회"""
    item = db.query(Item).filter(Item.id == item_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@app.post("/items", response_model=ItemResponse, status_code=201)
async def create_item(item: ItemCreate, db: Session = Depends(get_db)):
    """새 아이템 생성"""
    # Counter 사용(아이템 생성 시 +1)
    # normal이라는 라벨 값으로 카운터 1 증가
    ITEMS_CREATED_COUNTER.labels(item_type="normal").inc()
    
    # Histogram 사용(DB 작업 시간 측정)
    start_time = time.time() # 시작 시간 기록
    
    db_item = Item(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    
    duration = time.time() - start_time # 소요 시간 계산
    # create_item_query 라벨로 소요 시간(duration)을 기록
    DB_QUERY_DURATION_SECONDS.labels(query_type="create_item_query").observe(duration)
    
    return db_item

@app.put("/items/{item_id}", response_model=ItemResponse)
async def update_item(item_id: int, item: ItemCreate, db: Session = Depends(get_db)):
    """아이템 수정"""
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    db_item.name = item.name
    db_item.description = item.description
    db.commit()
    db.refresh(db_item)
    return db_item

@app.delete("/items/{item_id}")
async def delete_item(item_id: int, db: Session = Depends(get_db)):
    """아이템 삭제"""
    db_item = db.query(Item).filter(Item.id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    
    db.delete(db_item)
    db.commit()
    return {"message": "Item deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)