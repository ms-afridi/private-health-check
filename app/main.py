from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import hospital_side, model_utils, patient_side
from app.schemas import (
    DecryptRequest,
    DecryptResponse,
    EncryptResponse,
    HospitalPredictRequest,
    HospitalPredictResponse,
    PatientValues,
    PlainResponse,
    SessionResponse,
)

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

app = FastAPI(
    title="Private Health Check",
    description="Heart-disease risk prediction on encrypted data (CKKS homomorphic encryption).",
    version="1.0.0",
)


def _check_values(values: dict) -> None:
    missing = [f for f in model_utils.FEATURES if f not in values]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing values: {missing}")


# ---------- General ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    """Slider ranges, model metrics and approximation settings for the page."""
    return model_utils.get_public_config()


@app.get("/api/approximation")
def approximation():
    """Data for the 'real sigmoid vs polynomial' chart."""
    return model_utils.approximation_report()


# ---------- Patient side ----------

@app.post("/patient/session", response_model=SessionResponse)
def patient_session():
    """
    Patient device creates its keys.
    Only the PUBLIC context is handed to the hospital module.
    The secret key stays inside patient_side.
    """
    s = patient_side.create_session()
    hospital_side.register_public_context(s["session_id"], s["public_context"])
    return SessionResponse(
        session_id=s["session_id"],
        keygen_seconds=s["keygen_seconds"],
        public_context_bytes=s["public_context_bytes"],
        poly_modulus_degree=s["poly_modulus_degree"],
        coeff_mod_bit_sizes=s["coeff_mod_bit_sizes"],
        scale_bits=s["scale_bits"],
        hospital_has_secret_key=False,
    )


@app.post("/patient/encrypt", response_model=EncryptResponse)
def patient_encrypt(req: PatientValues):
    _check_values(req.values)
    try:
        return patient_side.encrypt(req.session_id, req.values)
    except KeyError as e:
        raise HTTPException(status_code=410, detail=str(e))


@app.post("/patient/decrypt", response_model=DecryptResponse)
def patient_decrypt(req: DecryptRequest):
    try:
        out = patient_side.decrypt(req.session_id, req.result_ciphertext)
    except KeyError as e:
        raise HTTPException(status_code=410, detail=str(e))
    p = out["probability"]
    return DecryptResponse(
        probability=p,
        label=model_utils.risk_label(p),
        decrypt_seconds=out["decrypt_seconds"],
    )


# ---------- Hospital side ----------

@app.post("/hospital/predict", response_model=HospitalPredictResponse)
def hospital_predict(req: HospitalPredictRequest):
    """Receives only a session id and a ciphertext. No patient values."""
    try:
        return hospital_side.predict_encrypted(req.session_id, req.ciphertext)
    except KeyError as e:
        raise HTTPException(status_code=410, detail=str(e))


# ---------- Comparison ----------

@app.post("/predict_plain", response_model=PlainResponse)
def predict_plain(req: PatientValues):
    """Normal unencrypted prediction, used only to prove the encrypted one is correct.
    (Reuses PatientValues; the session_id is ignored.)"""
    _check_values(req.values)
    return model_utils.plain_predict(req.values)


# ---------- Web page ----------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")