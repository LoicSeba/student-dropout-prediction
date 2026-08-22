import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_pipeline.joblib"

TARGET_LABELS = {0: "Graduate", 1: "Dropout", 2: "Enrolled"}

ml_pipeline = {} 

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run `python src/train.py` first."
        )
    ml_pipeline["model"] = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model pipeline from {MODEL_PATH}")
    yield
    ml_pipeline.clear()

app = FastAPI(title="Student Dropout Prediction API", lifespan=lifespan)

class StudentInput(BaseModel):
    """Pydantic model for student data"""

    model_config = {"populate_by_name": True}

    marital_status: int = Field(alias="Marital status")
    application_mode: int = Field(alias="Application mode")
    application_order: int = Field(alias="Application order")
    course: str = Field(alias="Course")
    daytime_evening_attendance: int = Field(alias="Daytime/evening attendance\t")
    previous_qualification: int = Field(alias="Previous qualification")
    previous_qualification_grade: float = Field(alias="Previous qualification (grade)")
    nacionality: int = Field(alias="Nacionality")
    mothers_qualification: int = Field(alias="Mother's qualification")
    fathers_qualification: int = Field(alias="Father's qualification")
    mothers_occupation: int = Field(alias="Mother's occupation")
    fathers_occupation: int = Field(alias="Father's occupation")
    admission_grade: float = Field(alias="Admission grade")
    displaced: int = Field(alias="Displaced")
    educational_special_needs: int = Field(alias="Educational special needs")
    debtor: int = Field(alias="Debtor")
    tuition_fees_up_to_date: int = Field(alias="Tuition fees up to date")
    gender: int = Field(alias="Gender")
    scholarship_holder: int = Field(alias="Scholarship holder")
    age_at_enrollment: int = Field(alias="Age at enrollment")
    international: int = Field(alias="International")
    curricular_units_1st_sem_credited: int = Field(alias="Curricular units 1st sem (credited)")
    curricular_units_1st_sem_enrolled: int = Field(alias="Curricular units 1st sem (enrolled)")
    curricular_units_1st_sem_evaluations: int = Field(alias="Curricular units 1st sem (evaluations)")
    curricular_units_1st_sem_approved: int = Field(alias="Curricular units 1st sem (approved)")
    curricular_units_1st_sem_grade: float = Field(alias="Curricular units 1st sem (grade)")
    curricular_units_1st_sem_without_evaluations: int = Field(alias="Curricular units 1st sem (without evaluations)")
    curricular_units_2nd_sem_credited: int = Field(alias="Curricular units 2nd sem (credited)")
    curricular_units_2nd_sem_enrolled: int = Field(alias="Curricular units 2nd sem (enrolled)")
    curricular_units_2nd_sem_evaluations: int = Field(alias="Curricular units 2nd sem (evaluations)")
    curricular_units_2nd_sem_approved: int = Field(alias="Curricular units 2nd sem (approved)")
    curricular_units_2nd_sem_grade: float = Field(alias="Curricular units 2nd sem (grade)")
    curricular_units_2nd_sem_without_evaluations: int = Field(alias="Curricular units 2nd sem (without evaluations)")
    unemployment_rate: float = Field(alias="Unemployment rate")
    inflation_rate: float = Field(alias="Inflation rate")
    gdp: float = Field(alias="GDP")

class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]

@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput):
    pipeline = ml_pipeline.get("model")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([student.model_dump(by_alias=True)])

    try:
        pred_class = pipeline.predict(row)[0]
        pred_proba = pipeline.predict_proba(row)[0]
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    return PredictionResponse(
        prediction=TARGET_LABELS[int(pred_class)],
        probabilities={
            TARGET_LABELS[i]: round(float(p), 4) for i, p in enumerate(pred_proba)
        },
    )
