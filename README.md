
## Epic: HCPCS Code Predictor System

---

# Story 1: HCPCS Master Data Preparation and Embedding Generation

### Summary

Prepare enriched HCPCS master dataset with structured semantic retrieval text and embeddings.

### Description

Build the offline HCPCS master data preparation pipeline that:

* Loads HCPCS master data
* Cleans and normalizes descriptions
* Generates semantic embedding text
* Creates embeddings using Azure OpenAI
* Saves enriched master JSON for downstream retrieval

### Acceptance Criteria

```text
1. HCPCS master CSV loads successfully
2. major_class and minor_class are attached correctly
3. Semantic embedding_text is generated for every HCPCS code
4. text-embedding-3-large embeddings are generated successfully
5. Output JSON contains:
   - hcpcs_code
   - hierarchy fields
   - semantic attributes
   - embedding_text
   - embedding vector
6. Enriched master JSON is saved successfully
```

### Technical Tasks

```text
- Create master data loader
- Build text cleaning utility
- Generate semantic retrieval text
- Generate embeddings
- Save processed JSON
```

### Output Files

```text
hcpcs_master_enriched_embeddings.json
```

---

# Story 2: Vector Index and Hybrid Retrieval Engine

### Summary

Build FAISS vector index and BM25 hybrid retrieval system for HCPCS candidate generation.

### Description

Implement retrieval infrastructure using:

* FAISS vector search
* BM25 keyword search
* Hybrid score combination
* Supplier history boost
* Hierarchy filtering

The system should retrieve Top-K HCPCS candidates for incoming products.

### Acceptance Criteria

```text
1. FAISS index builds successfully
2. BM25 index builds successfully
3. Hybrid retrieval returns Top-K candidates
4. Supplier history boost works correctly
5. Retrieval metadata mapping works correctly
6. Recall@20 metric is measurable
7. Retrieval outputs are reproducible
```

### Technical Tasks

```text
- Create FAISS index
- Create metadata store
- Build BM25 index
- Implement vector retrieval
- Implement BM25 retrieval
- Implement hybrid scoring
- Implement hierarchy filtering
```

### Output Files

```text
hcpcs_faiss.index
hcpcs_metadata.json
hcpcs_bm25.pkl
```

---

# Story 3: HCPCS Prediction Pipeline and Reranking

### Summary

Build end-to-end HCPCS prediction pipeline using hybrid retrieval, GPT reranking, and confidence scoring.

### Description

Implement prediction pipeline that:

* Processes test product data
* Generates semantic retrieval text
* Creates product embeddings
* Retrieves candidate HCPCS codes
* Reranks candidates using GPT-4.1-mini
* Applies rule validation
* Generates confidence scores
* Produces final HCPCS predictions

### Acceptance Criteria

```text
1. Test data loads successfully
2. Product semantic text is generated
3. Product embeddings are generated
4. Top-K candidate retrieval works
5. GPT reranker selects best HCPCS code
6. Rule validation executes successfully
7. Confidence score is generated
8. Final prediction output is saved
9. Tracking columns are appended to predictions
```

### Technical Tasks

```text
- Create test data preprocessing
- Build product embedding pipeline
- Implement supplier lookup
- Implement hybrid retrieval
- Implement GPT reranking
- Implement rule validation
- Implement confidence engine
- Save prediction outputs
```

### Output Files

```text
hcpcs_predictions_with_tracking.csv
```

---

# Story 4: Evaluation Metrics and Error Analysis Framework

### Summary

Build evaluation and error analysis framework for stage-wise HCPCS prediction accuracy tracking.

### Description

Implement metrics and analysis framework to evaluate:

* Retrieval accuracy
* Ranking accuracy
* Confidence bucket performance
* Error categories
* Human review rate

Generate detailed tracking and error analysis outputs.

### Acceptance Criteria

```text
1. Recall@5, Recall@10, Recall@20 are calculated
2. Top-1 and Top-3 accuracy are calculated
3. Confidence bucket metrics are generated
4. Human review rate is calculated
5. Error analysis report is generated
6. Retrieval failures are identifiable
7. Reranking failures are identifiable
8. Metrics outputs are saved successfully
```

### Technical Tasks

```text
- Build metrics calculation module
- Build confidence bucket evaluation
- Build retrieval failure analysis
- Build reranking failure analysis
- Build error reporting framework
- Save metrics outputs
```

### Output Files

```text
hcpcs_metrics_summary.json
hcpcs_confidence_metrics.csv
hcpcs_error_analysis.csv
```
