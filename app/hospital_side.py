import base64
import time
from collections import OrderedDict

import tenseal as ts

from app.model_utils import BIAS, POLY_COEFFS, WEIGHTS

MAX_SESSIONS = 50
_PUBLIC_CONTEXTS: "OrderedDict[str, ts.Context]" = OrderedDict()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def register_public_context(session_id: str, public_context_b64: str) -> dict:
    """
    The hospital receives the patient's public context once per session.
    We refuse it if it somehow contains a secret key.
    """
    ctx = ts.context_from(_unb64(public_context_b64))

    if ctx.is_private():
        raise ValueError("Refusing context: it contains a secret key.")

    _PUBLIC_CONTEXTS[session_id] = ctx
    while len(_PUBLIC_CONTEXTS) > MAX_SESSIONS:
        _PUBLIC_CONTEXTS.popitem(last=False)

    return {"has_secret_key": False, "registered": True}


def predict_encrypted(session_id: str, ciphertext_b64: str) -> dict:
    """
    Computes   sigmoid_poly( w . x + b )   directly on the ciphertext.

    Multiplicative depth used (we have 3 levels):
      1) dot product with the plain weights      -> 1 level
      2) degree-3 polynomial (z^2, then z^3)     -> 2 levels
    """
    if session_id not in _PUBLIC_CONTEXTS:
        raise KeyError("Hospital has no public context for this session.")
    _PUBLIC_CONTEXTS.move_to_end(session_id)
    ctx = _PUBLIC_CONTEXTS[session_id]

    t0 = time.perf_counter()

    enc_x = ts.ckks_vector_from(ctx, _unb64(ciphertext_b64))

    # Step A: encrypted logit = w . x + b  (weights are the hospital's own, in plain)
    enc_logit = enc_x.dot(WEIGHTS.tolist())
    enc_logit += BIAS

    # Step B: polynomial replacement of the sigmoid: 0.5 + a1*z + 0*z^2 + a3*z^3
    a0, a1, a3 = POLY_COEFFS
    enc_prob = enc_logit.polyval([a0, a1, 0.0, a3])

    blob = enc_prob.serialize()

    return {
        "result_ciphertext": _b64(blob),
        "result_bytes": len(blob),
        "compute_seconds": round(time.perf_counter() - t0, 4),
        "operations": [
            "encrypted dot product with model weights",
            "add encrypted bias",
            "evaluate degree-3 polynomial (sigmoid approximation)",
        ],
        "can_read_data": False,
    }