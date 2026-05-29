from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import endpoints
from app.database.sqlite_db import init_db



def create_app() -> FastAPI:
    # FastAPI 앱 초기화
    app = FastAPI(
        title="DMU_adv_AI API",
        description="미국 주식 데이터 분석 및 AI 예측 시스템",
        version="1.0.0"
    )

    # SQLite 테이블 초기화 (stock_prices, member, member_stock)
    init_db()

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
       # allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API 라우터 등록
    app.include_router(endpoints.router, prefix="/api/v1", tags=["v1"])

    @app.get("/")
    async def root():
        return {
            "project": "DMU_adv_AI",
            "status": "Running",
            "message": "미국 우량주 10종목 기반 AI 예측 시스템이 작동 중입니다."
        }

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
   