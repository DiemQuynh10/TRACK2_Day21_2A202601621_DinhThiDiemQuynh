from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Doc ten bucket tu bien moi truong (duoc dat trong systemd service)
GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """Tai file model.pkl tu GCS ve may khi server khoi dong."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_KEY)
    blob.download_to_filename(MODEL_PATH)
    print(f"Da tai model tu gs://{GCS_BUCKET}/{GCS_MODEL_KEY} ve {MODEL_PATH}")


# Goi ham nay khi module duoc import (chay khi server khoi dong)
download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


LABELS = {0: "thap", 1: "trung_binh", 2: "cao"}


@app.get("/health")
def health():
    """Endpoint kiem tra suc khoe server. GitHub Actions dung endpoint nay de xac nhan deploy thanh cong."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan.

    Dau vao: JSON {"features": [f1, f2, ..., f12]}
    Dau ra:  JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}
    """
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")

    prediction = int(model.predict([req.features])[0])

    return {"prediction": prediction, "label": LABELS[prediction]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
