import pytest
import tenseal as ts

from app import hospital_side, model_utils, patient_side
from app.patient_side import (
    COEFF_MOD_BIT_SIZES,
    GLOBAL_SCALE,
    POLY_MODULUS_DEGREE,
    _b64,
)

TOLERANCE = 1e-3  # CKKS is approximate; we expect about 1e-5 in practice


def _sample_patients():
    """Default values, all-minimum, all-maximum and the midpoint of every slider."""
    r = model_utils.RANGES
    feats = model_utils.FEATURES
    return {
        "defaults": {f: r[f]["default"] for f in feats},
        "all_min": {f: r[f]["min"] for f in feats},
        "all_max": {f: r[f]["max"] for f in feats},
        "midpoint": {f: (r[f]["min"] + r[f]["max"]) / 2 for f in feats},
    }


@pytest.fixture(scope="module")
def session():
    """Create keys once and hand the public context to the hospital module."""
    s = patient_side.create_session()
    hospital_side.register_public_context(s["session_id"], s["public_context"])
    return s["session_id"]


def _run_encrypted(session_id, raw):
    enc = patient_side.encrypt(session_id, raw)
    hosp = hospital_side.predict_encrypted(session_id, enc["ciphertext"])
    dec = patient_side.decrypt(session_id, hosp["result_ciphertext"])
    return dec["probability"]


# ---------- Correctness ----------

@pytest.mark.parametrize("name", ["defaults", "all_min", "all_max", "midpoint"])
def test_encrypted_matches_plain_polynomial(session, name):
    raw = _sample_patients()[name]
    expected = model_utils.plain_predict(raw)["probability_poly"]
    got = _run_encrypted(session, raw)
    assert abs(got - expected) < TOLERANCE, (
        f"{name}: encrypted {got:.6f} vs plain polynomial {expected:.6f}"
    )


def test_polynomial_approximation_error_is_bounded():
    report = model_utils.approximation_report()
    assert report["max_abs_error"] < 0.1


def test_polynomial_close_to_true_sigmoid_on_default_patient(session):
    raw = _sample_patients()["defaults"]
    plain = model_utils.plain_predict(raw)
    assert abs(plain["probability"] - plain["probability_poly"]) < 0.1


# ---------- Trust boundary ----------

def test_hospital_module_does_not_import_patient_module():
    assert not hasattr(hospital_side, "patient_side")


def test_hospital_context_has_no_secret_key(session):
    ctx = hospital_side._PUBLIC_CONTEXTS[session]
    assert not ctx.is_private()


def test_hospital_rejects_context_with_secret_key():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = GLOBAL_SCALE
    with_secret = ctx.serialize(save_secret_key=True)
    with pytest.raises(ValueError):
        hospital_side.register_public_context("bad-session", _b64(with_secret))


def test_unknown_session_is_rejected():
    with pytest.raises(KeyError):
        hospital_side.predict_encrypted("does-not-exist", "AAAA")