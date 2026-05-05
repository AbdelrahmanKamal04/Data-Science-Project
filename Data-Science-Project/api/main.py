import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.config import Config
from api.engine import FraudDetectionEngine
from api.schemas import PredictionRequest, PredictionResponse

logging.basicConfig(
    level=logging.DEBUG if Config.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Initializing Fraud Detection API...")
    try:
        engine = FraudDetectionEngine()
        logger.info("Engine ready. Accepting predictions.")
    except Exception as e:
        logger.critical(f"Failed to initialize engine: {e}")
        raise RuntimeError("Model initialization failed") from e
    yield
    logger.info("Shutting down API...")
    engine = None

app = FastAPI(
    title=Config.APP_NAME,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy" if engine is not None else "unhealthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = engine.predict(request.model_dump())
        return PredictionResponse(**result)
    except ValueError as ve:
        logger.warning(f"Input validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal prediction error")