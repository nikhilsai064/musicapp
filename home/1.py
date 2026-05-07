def generate_master_embedding_text(row, retries=3):
    prompt = f"""
You are an expert HCPCS semantic retrieval optimizer.

Convert this HCPCS record into structured semantic retrieval text for vector embeddings.

Rules:
- Preserve important medical and billing meaning.
- Emphasize product category, material, sterility, usage, size, dosage, units, packaging, and medical purpose.
- Expand abbreviations only when obvious.
- Add medically relevant synonyms only when obvious.
- Do not invent unsupported details.
- Output valid JSON only.

Input:
HCPCS Code: {row.get("hcpcs_code", "")}
Major Class: {row.get("major_class", "")}
Minor Class: {row.get("minor_class", "")}
Short Description: {row.get("short_description", "")}
Long Description: {row.get("long_description", "")}

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
            print(f"Master text generation error attempt {attempt + 1}: {e}")
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
