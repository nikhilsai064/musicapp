master_records = []

for _, row in tqdm(master_df.iterrows(), total=len(master_df)):
    result = generate_master_embedding_text(row)

    record = row.to_dict()
    record.update(result)

    if not record.get("embedding_text"):
        record["embedding_text"] = f"""
Category: {record.get("major_class", "")}
Subcategory: {record.get("minor_class", "")}
HCPCS Code: {record.get("hcpcs_code", "")}
Description: {record.get("short_description", "")} {record.get("long_description", "")}
Keywords: {record.get("keywords", "")}
""".strip()

    master_records.append(record)

master_enriched_df = pd.DataFrame(master_records)
master_enriched_df.head()



7
embeddings = []

for text in tqdm(master_enriched_df["embedding_text"]):
    emb = get_embedding(text)
    embeddings.append(emb)

master_enriched_df["embedding"] = embeddings

master_enriched_df = master_enriched_df[master_enriched_df["embedding"].notna()].reset_index(drop=True)

master_enriched_df.to_json(
    "hcpcs_master_enriched_embeddings.json",
    orient="records",
    indent=2
)

print("Saved: hcpcs_master_enriched_embeddings.json")
