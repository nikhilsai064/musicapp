all_predictions = []

for idx, row in tqdm(test_enriched_df.iterrows(), total=len(test_enriched_df)):
    supplier_code = supplier_mapping.get(row["supplier_catalog_key"], None)

    initial_candidates = hybrid_retrieve(
        query_text=row["embedding_text"],
        query_embedding=row["embedding"],
        supplier_code=supplier_code,
        top_k=50
    )

    pred_major, pred_minor, major_conf, minor_conf = predict_hierarchy_from_candidates(initial_candidates)

    final_candidates = hybrid_retrieve(
        query_text=row["embedding_text"],
        query_embedding=row["embedding"],
        supplier_code=supplier_code,
        pred_major_class=pred_major,
        pred_minor_class=pred_minor,
        top_k=20
    )

    rerank_result = gpt_rerank(
        product_text=row["embedding_text"],
        candidates=final_candidates,
        top_n=10
    )

    pred_code = normalize_hcpcs(rerank_result.get("best_hcpcs_code", ""))

    candidate_lookup = {
        c["hcpcs_code"]: c for c in final_candidates
    }

    predicted_candidate = candidate_lookup.get(
        pred_code,
        final_candidates[0] if final_candidates else {}
    )

    rule_result = rule_validation(row, predicted_candidate)

    rerank_conf = float(rerank_result.get("confidence", 0) or 0)

    if rerank_conf > 1:
        rerank_conf = rerank_conf / 100

    hybrid_score = float(predicted_candidate.get("hybrid_score", 0))

    final_confidence = (
        0.50 * rerank_conf +
        0.25 * hybrid_score +
        0.15 * major_conf +
        0.10 * minor_conf
    )

    if supplier_code == pred_code:
        final_confidence += 0.10

    if not rule_result["rule_passed"]:
        final_confidence -= 0.20

    final_confidence = max(0, min(1, final_confidence))

    retrieved_codes = [c["hcpcs_code"] for c in final_candidates]
    top_3_codes = rerank_result.get("top_3_codes", retrieved_codes[:3])

    prediction_record = row.drop(labels=["embedding"]).to_dict()

    prediction_record.update({
        "supplier_history_code": supplier_code,
        "pred_major_class": pred_major,
        "major_confidence": major_conf,
        "pred_minor_class": pred_minor,
        "minor_confidence": minor_conf,
        "retrieved_top_20_codes": retrieved_codes,
        "reranked_top_3_codes": top_3_codes,
        "pred_hcpcs_code": pred_code,
        "rerank_reason": rerank_result.get("reason", ""),
        "rerank_confidence": rerank_conf,
        "final_confidence": final_confidence,
        "confidence_bucket": confidence_bucket(final_confidence),
        "rule_passed": rule_result["rule_passed"],
        "rule_violations": rule_result["rule_violations"],
        "needs_human_review": final_confidence < 0.85,
        "top_1_correct": pred_code == row["true_hcpcs_code"],
        "top_3_correct": row["true_hcpcs_code"] in top_3_codes,
        "retrieval_recall_at_5": row["true_hcpcs_code"] in retrieved_codes[:5],
        "retrieval_recall_at_10": row["true_hcpcs_code"] in retrieved_codes[:10],
        "retrieval_recall_at_20": row["true_hcpcs_code"] in retrieved_codes[:20],
    })

    all_predictions.append(prediction_record)

predictions_df = pd.DataFrame(all_predictions)
predictions_df.head()


21
metrics = {
    "total_records": len(predictions_df),

    "retrieval_recall_at_5": predictions_df["retrieval_recall_at_5"].mean(),
    "retrieval_recall_at_10": predictions_df["retrieval_recall_at_10"].mean(),
    "retrieval_recall_at_20": predictions_df["retrieval_recall_at_20"].mean(),

    "top_1_accuracy": predictions_df["top_1_correct"].mean(),
    "top_3_accuracy": predictions_df["top_3_correct"].mean(),

    "high_confidence_coverage": (predictions_df["confidence_bucket"] == "HIGH").mean(),
    "human_review_rate": predictions_df["needs_human_review"].mean(),

    "high_confidence_accuracy": predictions_df[
        predictions_df["confidence_bucket"] == "HIGH"
    ]["top_1_correct"].mean()
}

metrics_df = pd.DataFrame([metrics])

print("===== HCPCS Metrics Summary =====")
display(metrics_df.T)


22
confidence_metrics = (
    predictions_df
    .groupby("confidence_bucket")
    .agg(
        total=("pred_hcpcs_code", "count"),
        top_1_accuracy=("top_1_correct", "mean"),
        top_3_accuracy=("top_3_correct", "mean"),
        avg_confidence=("final_confidence", "mean")
    )
    .reset_index()
)

display(confidence_metrics)

23
error_df = predictions_df[predictions_df["top_1_correct"] == False].copy()

error_columns = [
    "item_name",
    "supplier_name",
    "catalog_number",
    "true_hcpcs_code",
    "pred_hcpcs_code",
    "supplier_history_code",
    "pred_major_class",
    "pred_minor_class",
    "retrieved_top_20_codes",
    "reranked_top_3_codes",
    "final_confidence",
    "confidence_bucket",
    "rule_passed",
    "rule_violations",
    "rerank_reason"
]

error_df = error_df[error_columns]

display(error_df.head(20))

24
predictions_df.to_csv("hcpcs_predictions_with_tracking.csv", index=False)
error_df.to_csv("hcpcs_error_analysis.csv", index=False)
metrics_df.to_json("hcpcs_metrics_summary.json", orient="records", indent=2)
confidence_metrics.to_csv("hcpcs_confidence_metrics.csv", index=False)

print("Saved files:")
print("hcpcs_predictions_with_tracking.csv")
print("hcpcs_error_analysis.csv")
print("hcpcs_metrics_summary.json")
print("hcpcs_confidence_metrics.csv")

