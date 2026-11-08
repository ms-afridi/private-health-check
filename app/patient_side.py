import base64
import time
import uuid
from collections import OrderedDict

import tenseal as ts

from app.model_utils import standardize

# ---------- CKKS parameters (the trade-off: security / precision / speed) ----------
# poly_modulus_degree 8192 -> 128-bit security as long as the total modulus
# stays at or below 218 bits.
# Our computation needs 3 multiplications in a row (dot product, then the
# degree-3 polynomial), so we use three 30-bit "levels":
#   total = 50 + 30 + 30 + 30 + 50 = 190 bits  (< 218, so security holds)
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [50, 30, 30, 30, 50]
GLOBAL_SCALE = 2 ** 30

# Keep only the most recent sessions in memory (demo safety).
MAX_SESSIONS = 50
_SESSIONS: "OrderedDict[str, dict]" = OrderedDict()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def create_session() -> dict:
    """
    Step 1: the patient device creates its keys.
    Returns the PUBLIC context only. The secret key stays in _SESSIONS.
    """
    t0 = time.perf_counter()

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()  # needed so the hospital can sum inside a ciphertext

    # Serialize WITHOUT the secret key -> this is what the hospital receives.
    public_bytes = ctx.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=True,
    )

    session_id = uuid.uuid4().hex
    _SESSIONS[session_id] = {"ctx": ctx}  # full context incl. secret key: stays here
    while len(_SESSIONS) > MAX_SESSIONS:
        _SESSIONS.popitem(last=False)

    return {
        "session_id": session_id,
        "public_context": _b64(public_bytes),
        "public_context_bytes": len(public_bytes),
        "keygen_seconds": round(time.perf_counter() - t0, 3),
        "poly_modulus_degree": POLY_MODULUS_DEGREE,
        "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
        "scale_bits": 30,
    }


def encrypt(session_id: str, raw: dict) -> dict:
    """
    Step 2: standardize the patient's values locally, then encrypt them.
    """
    ctx = _get_ctx(session_id)
    t0 = time.perf_counter()

    x = standardize(raw).tolist()
    enc = ts.ckks_vector(ctx, x)
    blob = enc.serialize()

    # A peek at what the hospital will see: the first raw bytes as numbers.
    preview = list(blob[:48])

    return {
        "ciphertext": _b64(blob),
        "ciphertext_bytes": len(blob),
        "plaintext_bytes": len(x) * 8,
        "preview": preview,
        "encrypt_seconds": round(time.perf_counter() - t0, 4),
    }


def decrypt(session_id: str, result_ciphertext: str) -> dict:
    """
    Step 4: only the patient side can unlock the hospital's answer.
    """
    ctx = _get_ctx(session_id)
    t0 = time.perf_counter()

    enc = ts.ckks_vector_from(ctx, _unb64(result_ciphertext))
    value = float(enc.decrypt()[0])

    return {
        "probability": value,
        "decrypt_seconds": round(time.perf_counter() - t0, 4),
    }


def _get_ctx(session_id: str):
    if session_id not in _SESSIONS:
        raise KeyError("Unknown or expired session. Please create new keys.")
    _SESSIONS.move_to_end(session_id)
    return _SESSIONS[session_id]["ctx"]