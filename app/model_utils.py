import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

# Degree of the polynomial that replaces the sigmoid.
# Encryption only supports + and x, so sigmoid(z) is approximated by
#   0.5 + a1*z + a3*z^3   (degree 3)
# Higher degree = more accurate but needs more multiplications (slower).
POLY_DEGREE = 3


def _load(name: str) -> dict:
    return json.loads((MODEL_DIR / name).read_text())


_W = _load("weights.json")
_S = _load("scaler.json")
_M = _load("metrics.json")

FEATURES: list = _W["features"]
WEIGHTS = np.array(_W["weights"], dtype=float)
BIAS: float = float(_W["bias"])
MEAN = np.array(_S["mean"], dtype=float)
STD = np.array(_S["std"], dtype=float)
RANGES: dict = _S["ranges"]
METRICS: dict = _M

# Interval on which the sigmoid approximation is fitted.
# Based on the largest model score seen in the data, plus a safety margin.
APPROX_LIMIT: float = float(max(abs(_M["logit_min"]), abs(_M["logit_max"])) + 1.0)


# ---------- Sigmoid approximation ----------

def _fit_odd_polynomial(limit: float, degree: int) -> list:
    """
    sigmoid(z) - 0.5 is an odd function, so we only need odd powers.
    Least-squares fit on [-limit, limit].
    Returns coefficients [a0, a1, a3, ...] for 0.5 + a1*z + a3*z^3 + ...
    """
    z = np.linspace(-limit, limit, 2001)
    target = 1.0 / (1.0 + np.exp(-z)) - 0.5
    powers = list(range(1, degree + 1, 2))  # [1, 3] for degree 3
    A = np.stack([z ** p for p in powers], axis=1)
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    return [0.5] + [float(c) for c in coef]


POLY_COEFFS: list = _fit_odd_polynomial(APPROX_LIMIT, POLY_DEGREE)
# POLY_COEFFS = [0.5, a1, a3]  (degree 3)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def sigmoid_poly(z):
    """Polynomial stand-in for the sigmoid (the version encryption can compute)."""
    z = np.asarray(z, dtype=float)
    out = POLY_COEFFS[0] + 0.0 * z
    for coef, power in zip(POLY_COEFFS[1:], range(1, POLY_DEGREE + 1, 2)):
        out = out + coef * z ** power
    return out


def approximation_report(points: int = 81) -> dict:
    """Data for the 'real sigmoid vs approximation' chart on the page."""
    z = np.linspace(-APPROX_LIMIT, APPROX_LIMIT, points)
    real = sigmoid(z)
    approx = sigmoid_poly(z)
    return {
        "degree": POLY_DEGREE,
        "interval": [-APPROX_LIMIT, APPROX_LIMIT],
        "coefficients": POLY_COEFFS,
        "z": z.round(4).tolist(),
        "sigmoid": real.round(5).tolist(),
        "approx": approx.round(5).tolist(),
        "max_abs_error": float(np.max(np.abs(real - approx))),
    }


# ---------- Patient data helpers ----------

def to_vector(raw: dict) -> np.ndarray:
    """Turn {'age': 54, ...} into a vector in the model's feature order."""
    return np.array([float(raw[f]) for f in FEATURES], dtype=float)


def standardize(raw: dict) -> np.ndarray:
    """(x - mean) / std. The scaler is public, so the patient side can do this locally."""
    return (to_vector(raw) - MEAN) / STD


def risk_label(p: float) -> str:
    if p < 0.33:
        return "Low"
    if p < 0.66:
        return "Medium"
    return "High"


# ---------- Plain (unencrypted) prediction, used only for comparison ----------

def plain_predict(raw: dict) -> dict:
    x = standardize(raw)
    logit = float(WEIGHTS @ x + BIAS)
    return {
        "logit": logit,
        "probability": float(sigmoid(logit)),            # true sigmoid
        "probability_poly": float(sigmoid_poly(logit)),  # what encrypted version should match
        "label": risk_label(float(sigmoid(logit))),
    }


def get_public_config() -> dict:
    """Everything the web page needs to build sliders and show model info."""
    return {
        "features": FEATURES,
        "ranges": RANGES,
        "metrics": METRICS,
        "poly_degree": POLY_DEGREE,
        "poly_coefficients": POLY_COEFFS,
        "approx_interval": [-APPROX_LIMIT, APPROX_LIMIT],
    }