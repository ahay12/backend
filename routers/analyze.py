import io
import pandas as pd
from collections import defaultdict
from fastapi import APIRouter, UploadFile, File, HTTPException

from schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeResponse,
    BatchResultItem,
    BatchSummary,
    BatchAccuracy,
    ModelResult,
    PreprocessingDetail,
)
from services.preprocessing import preprocess_for_svm, preprocess_for_indobert
from services.svm_service import predict_svm
from services.indobert_service import predict_indobert

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analisis sentimen satu teks",
    description="Menerima satu teks review dan mengembalikan prediksi sentimen dari model SVM dan IndoBERT beserta detail preprocessing.",
)
async def analyze(request: AnalyzeRequest):
    original = request.text

    # --- Preprocessing ---
    after_cleaning, after_stopword = preprocess_for_svm(original)
    indobert_input = preprocess_for_indobert(original)  # cleaning saja

    # --- Inference ---
    svm_result = predict_svm(after_stopword)
    indobert_result = predict_indobert(indobert_input)

    return AnalyzeResponse(
        preprocessing=PreprocessingDetail(
            original=original,
            after_cleaning=after_cleaning,
            after_stopword=after_stopword,
        ),
        svm=ModelResult(**svm_result),
        indobert=ModelResult(**indobert_result),
    )


@router.post(
    "/analyze-batch",
    response_model=BatchAnalyzeResponse,
    summary="Analisis sentimen batch dari file CSV",
    description=(
        "Upload file CSV dengan kolom `content` (wajib) dan `sentiment` (opsional). "
        "Jika kolom `sentiment` ada, accuracy akan dihitung."
    ),
)
async def analyze_batch(file: UploadFile = File(...)):
    # ─── Validasi tipe file ───────────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File harus berformat CSV.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal membaca CSV: {str(e)}")

    if "content" not in df.columns:
        raise HTTPException(
            status_code=422,
            detail="Kolom `content` tidak ditemukan dalam CSV. Pastikan header kolom bernama `content`.",
        )

    has_labels = "sentiment" in df.columns
    results: list[BatchResultItem] = []
    svm_counts: dict[str, int] = defaultdict(int)
    indobert_counts: dict[str, int] = defaultdict(int)

    svm_correct = 0
    indobert_correct = 0
    total_labeled = 0

    svm_cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    indobert_cm = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for _, row in df.iterrows():
        text = str(row["content"]) if pd.notna(row["content"]) else ""
        actual = str(row["sentiment"]).strip() if has_labels and pd.notna(row.get("sentiment")) else None

        # Truncate to 512 chars
        text = text[:512]

        # Preprocessing
        _, after_stopword = preprocess_for_svm(text)
        indobert_input = preprocess_for_indobert(text)

        # Inference
        svm_res = predict_svm(after_stopword)
        indobert_res = predict_indobert(indobert_input)

        svm_label = svm_res["label"]
        indobert_label = indobert_res["label"]

        svm_counts[svm_label] += 1
        indobert_counts[indobert_label] += 1

        if actual:
            total_labeled += 1
            if svm_label == actual:
                svm_correct += 1
            if indobert_label == actual:
                indobert_correct += 1
            
            # Update confusion matrix
            is_actual_pos = (actual == "positif")
            
            is_svm_pos = (svm_label == "positif")
            if is_actual_pos and is_svm_pos: svm_cm["tp"] += 1
            elif not is_actual_pos and not is_svm_pos: svm_cm["tn"] += 1
            elif not is_actual_pos and is_svm_pos: svm_cm["fp"] += 1
            elif is_actual_pos and not is_svm_pos: svm_cm["fn"] += 1

            is_indo_pos = (indobert_label == "positif")
            if is_actual_pos and is_indo_pos: indobert_cm["tp"] += 1
            elif not is_actual_pos and not is_indo_pos: indobert_cm["tn"] += 1
            elif not is_actual_pos and is_indo_pos: indobert_cm["fp"] += 1
            elif is_actual_pos and not is_indo_pos: indobert_cm["fn"] += 1

        svm_reason = None
        if svm_res.get("word_weights") and len(svm_res["word_weights"]) > 0:
            top = svm_res["word_weights"][0]
            svm_reason = f"Kata '{top['word']}' sangat memengaruhi ({'+' if top['weight']>0 else ''}{top['weight']:.2f})"
            
        indo_reason = None
        if indobert_res.get("tokens") and len(indobert_res["tokens"]) > 0:
            tokens = indobert_res["tokens"]
            indo_reason = f"Tokens: {' '.join(tokens[:5])}{'...' if len(tokens)>5 else ''}"

        results.append(
            BatchResultItem(
                content=text,
                svm=svm_label,
                confidence_svm=svm_res.get("confidence"),
                svm_reason=svm_reason,
                indobert=indobert_label,
                confidence_indobert=indobert_res.get("confidence"),
                indobert_reason=indo_reason,
                actual=actual,
            )
        )

    accuracy = None
    confusion_matrix_result = None
    if has_labels and total_labeled > 0:
        accuracy = BatchAccuracy(
            svm=round(svm_correct / total_labeled, 4),
            indobert=round(indobert_correct / total_labeled, 4),
        )
        confusion_matrix_result = {
            "svm": svm_cm,
            "indobert": indobert_cm
        }

    return BatchAnalyzeResponse(
        total=len(results),
        has_labels=has_labels,
        results=results,
        summary=BatchSummary(
            svm=dict(svm_counts),
            indobert=dict(indobert_counts),
        ),
        accuracy=accuracy,
        confusion_matrix=confusion_matrix_result,
    )
