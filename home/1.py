product_records = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    result = generate_product_embedding_text(row)

    record = row.to_dict()
    record.update(result)

    if not record.get("embedding_text"):
        record["embedding_text"] = f"""
Item Name: {record.get("item_name", "")}
Supplier: {record.get("supplier_name", "")}
Catalog: {record.get("catalog_number", "")}
Description: {record.get("item_name", "")}
Keywords: {record.get("keywords", "")}
""".strip()

    product_records.append(record)

test_enriched_df = pd.DataFrame(product_records)
test_enriched_df.head()


16
product_embeddings = []

for text in tqdm(test_enriched_df["embedding_text"]):
    emb = get_embedding(text)
    product_embeddings.append(emb)

test_enriched_df["embedding"] = product_embeddings
test_enriched_df = test_enriched_df[test_enriched_df["embedding"].notna()].reset_index(drop=True)


17
def predict_hierarchy_from_candidates(candidates):
    if not candidates:
        return "", "", 0.0, 0.0

    major_scores = {}
    minor_scores = {}

    for c in candidates[:20]:
        major = c.get("major_class", "")
        minor = c.get("minor_class", "")
        score = c.get("hybrid_score", 0)

        major_scores[major] = major_scores.get(major, 0) + score
        minor_scores[minor] = minor_scores.get(minor, 0) + score

    pred_major = max(major_scores, key=major_scores.get) if major_scores else ""
    pred_minor = max(minor_scores, key=minor_scores.get) if minor_scores else ""

    major_conf = major_scores.get(pred_major, 0) / (sum(major_scores.values()) + 1e-9)
    minor_conf = minor_scores.get(pred_minor, 0) / (sum(minor_scores.values()) + 1e-9)

    return pred_major, pred_minor, major_conf, minor_conf

18

def gpt_rerank(product_text, candidates, top_n=10, retries=3):
    candidate_text = ""

    for i, c in enumerate(candidates[:top_n], start=1):
        candidate_text += f"""
Candidate {i}
HCPCS Code: {c.get("hcpcs_code", "")}
Major Class: {c.get("major_class", "")}
Minor Class: {c.get("minor_class", "")}
Short Description: {c.get("short_description", "")}
Long Description: {c.get("long_description", "")}
Semantic Text: {c.get("embedding_text", "")}
Hybrid Score: {c.get("hybrid_score", 0)}
"""

    prompt = f"""
You are an expert HCPCS coding reranker.

Given one product and candidate HCPCS codes, choose the best matching HCPCS code.

Rules:
- Select only from the provided candidates.
- Pay close attention to product type, material, sterility, size, dosage, units, and usage.
- Do not invent a new HCPCS code.
- Return valid JSON only.

Product:
{product_text}

Candidates:
{candidate_text}

Return JSON:
{{
  "best_hcpcs_code": "",
  "confidence": 0.0,
  "reason": "",
  "top_3_codes": []
}}
"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an HCPCS coding reranker. Return JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            content = response.choices[0].message.content
            result = safe_json_loads(content)

            return result

        except Exception as e:
            print(f"Rerank error attempt {attempt + 1}: {e}")
            time.sleep(2)

    fallback = candidates[0] if candidates else {}
    return {
        "best_hcpcs_code": fallback.get("hcpcs_code", ""),
        "confidence": fallback.get("hybrid_score", 0),
        "reason": "Fallback to highest hybrid score",
        "top_3_codes": [c.get("hcpcs_code", "") for c in candidates[:3]]
    }

19
def rule_validation(product_row, predicted_candidate):
    violations = []

    product_text = clean_text(product_row.get("embedding_text", ""))
    candidate_text = clean_text(predicted_candidate.get("embedding_text", ""))

    if "sterile" in product_text and "non sterile" in candidate_text:
        violations.append("Sterility mismatch")

    if "non sterile" in product_text and "sterile" in candidate_text and "non sterile" not in candidate_text:
        violations.append("Sterility mismatch")

    product_type = clean_text(product_row.get("product_type", ""))
    candidate_type = clean_text(predicted_candidate.get("product_type", ""))

    if product_type and candidate_type and product_type not in candidate_type and candidate_type not in product_type:
        violations.append("Product type mismatch")

    return {
        "rule_passed": len(violations) == 0,
        "rule_violations": violations
    }
    def confidence_bucket(score):
    if score >= 0.85:
        return "HIGH"
    elif score >= 0.65:
        return "MEDIUM"
    else:
        return "LOW"
