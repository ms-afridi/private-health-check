from typing import Dict, List

from pydantic import BaseModel, Field


# ---------- Patient side ----------

class SessionResponse(BaseModel):
    """Result of key generation. Note: no secret key and no big key blobs here."""
    session_id: str
    keygen_seconds: float
    public_context_bytes: int
    poly_modulus_degree: int
    coeff_mod_bit_sizes: List[int]
    scale_bits: int
    hospital_has_secret_key: bool  # always False; shown on the page as proof


class PatientValues(BaseModel):
    """The slider values, e.g. {'age': 54, 'trestbps': 130, ...}"""
    session_id: str
    values: Dict[str, float]


class EncryptResponse(BaseModel):
    ciphertext: str                 # base64, this is what travels to the hospital
    ciphertext_bytes: int
    plaintext_bytes: int
    preview: List[int]              # first bytes, for the "wall of random numbers"
    encrypt_seconds: float


class DecryptRequest(BaseModel):
    session_id: str
    result_ciphertext: str


class DecryptResponse(BaseModel):
    probability: float
    label: str
    decrypt_seconds: float


# ---------- Hospital side ----------

class HospitalPredictRequest(BaseModel):
    """The hospital receives ONLY a session id and a ciphertext. No patient values."""
    session_id: str
    ciphertext: str = Field(..., description="base64 CKKS ciphertext")


class HospitalPredictResponse(BaseModel):
    result_ciphertext: str
    result_bytes: int
    compute_seconds: float
    operations: List[str]
    can_read_data: bool


# ---------- Comparison (plain model) ----------

class PlainResponse(BaseModel):
    logit: float
    probability: float
    probability_poly: float
    label: str