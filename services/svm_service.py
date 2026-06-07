import joblib
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "svm"

# ─── Load model & vectorizer saat startup (singleton) ────────────────────────
_svm_model = joblib.load(MODEL_DIR / "svm_model.pkl")
_tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
_label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")


def _sigmoid(x: float) -> float:
    """Konversi decision function score ke pseudo-probability [0,1] via sigmoid."""
    return float(1.0 / (1.0 + np.exp(-x)))


def predict_svm(text_after_stopword: str) -> dict:
    """
    Menerima teks setelah preprocessing (cleaning + stopword removal).
    LinearSVC tidak punya predict_proba, jadi confidence dihitung
    dari decision_function → sigmoid.

    Return: { label: str, confidence: float }
    """
    vec = _tfidf.transform([text_after_stopword])

    # Predict label
    pred_encoded = _svm_model.predict(vec)[0]
    raw_label = _label_encoder.inverse_transform([pred_encoded])[0]
    
    # Map to Indonesian as specified in PLAN.md
    label = "positif" if raw_label == "positive" else "negatif"

    # Confidence via decision_function → sigmoid
    decision = _svm_model.decision_function(vec)[0]

    # Binary case: scalar, multi-class: array
    if np.ndim(decision) == 0:
        # Nilai positif → "positive", negatif → "negative"
        # Confidence = sigmoid dari abs(decision) atau dari arah yang benar
        confidence = _sigmoid(float(decision)) if raw_label == "positive" else _sigmoid(-float(decision))
    else:
        # Multi-class: ambil max score
        confidence = _sigmoid(float(np.max(decision)))

    return {
        "label": label,
        "confidence": round(confidence, 4),
    }
