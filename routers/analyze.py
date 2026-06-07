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

        results.append(
            BatchResultItem(
                content=text,
                svm=svm_label,
                indobert=indobert_label,
                actual=actual,
            )
        )

    accuracy = None
    if has_labels and total_labeled > 0:
        accuracy = BatchAccuracy(
            svm=round(svm_correct / total_labeled, 4),
            indobert=round(indobert_correct / total_labeled, 4),
        )

    return BatchAnalyzeResponse(
        total=len(results),
        has_labels=has_labels,
        results=results,
        summary=BatchSummary(
            svm=dict(svm_counts),
            indobert=dict(indobert_counts),
        ),
        accuracy=accuracy,
    )
