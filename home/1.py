def generate_product_embedding_text(row, retries=3):
    prompt = f"""
You are an expert medical product semantic retrieval optimizer.

Convert this product record into structured semantic retrieval text for HCPCS code matching.

Rules:
- Preserve important medical product meaning.
- Emphasize product category, material, sterility, usage, size, dosage, units, packaging, and medical purpose.
- Normalize abbreviations only when obvious.
- Add medically relevant synonyms only when obvious.
- Do not infer unsupported facts.
- Output valid JSON only.

Input:
Item Name: {row.get("item_name", "")}
Supplier Name: {row.get("supplier_name", "")}
Catalog Number: {row.get("catalog_number", "")}

Return JSON format:
{{
  "product_type": "",
  "material": "",
  "sterility": "",
  "usage": "",
  "size": "",
  "units": "",
  "dosage": "",
  "packaging": "",
  "keywords": "",
  "embedding_text": ""
}}
"""

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            content = response.choices[0].message.content
            return safe_json_loads(content)

        except Exception as e:
            print(f"Product text generation error attempt {attempt + 1}: {e}")
            time.sleep(2)

    return {
        "product_type": "",
        "material": "",
        "sterility": "",
        "usage": "",
        "size": "",
        "units": "",
        "dosage": "",
        "packaging": "",
        "keywords": "",
        "embedding_text": ""
    }
