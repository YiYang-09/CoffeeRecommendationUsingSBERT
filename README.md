# ☕ Beyond Keywords: A Hybrid Neural Semantic Approach for Coffee Recommendation

> A hybrid coffee recommendation engine that combines **rule-based constraint filtering** with **SBERT semantic search** — improving semantic similarity over a TF-IDF baseline by **+32% to +221%** across three query styles.

## 📌 Overview

Keyword search breaks down the moment a user describes what they *actually* want: searching `"fruity"` misses every coffee described as *"notes of stone fruit and berry"*. This project builds a recommendation engine that understands a natural-language query in two layers:

1. **Hard constraints** — price budget, roast level, origin, and roaster are extracted from the query with regex + entity matching, then applied as exact filters.
2. **Soft (semantic) matching** — the remaining flavor intent is embedded with **Sentence-BERT** (`all-MiniLM-L6-v2`) and matched against 2,095 coffee reviews via cosine similarity. A **TF-IDF + cosine similarity** pipeline serves as the baseline.

Recommendations from both systems are judged with **BERTScore** as an external evaluation metric across three query styles: integrated queries (constraints + flavor), short noisy queries, and rich descriptive queries.

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Language & Data | Python, pandas, NumPy |
| Constraint Extraction | regex, fuzzy entity matching |
| Baseline Matcher | scikit-learn (TF-IDF, cosine similarity) |
| Semantic Matcher | sentence-transformers (`all-MiniLM-L6-v2`), PyTorch |
| Evaluation | `bert-score` (F1 between query and recommended reviews) |

## 🔄 How It Works

```
 Natural-language query
  "Light roast coffee from Ethiopia under 20 dollars with bright citrus and floral notes"
        │
        ▼
 ┌─────────────────────────────┐
 │  1. Constraint Extraction   │  regex + entity matching
 │     price ≤ $20 · roast =   │
 │     Light · origin = Ethiopia│
 └─────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────┐
 │  2. Hard Filtering          │  boolean masking over 2,095 coffees
 └─────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │  3. Semantic Matching on flavor descriptions│
 │     Baseline:  TF-IDF + cosine similarity   │
 │     Proposed:  SBERT embeddings + cos sim   │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────┐
 │  4. Ranking & Top-N Output  │  sort by similarity, then rating
 └─────────────────────────────┘
```

**Design detail:** the two stages are complementary — rules guarantee the *non-negotiables* (budget, roast, origin) are respected exactly, while embeddings handle the *fuzzy part* (flavor language), where synonyms like *citrus* ↔ *lemon verbena* ↔ *grapefruit zest* matter more than exact word overlap.

## 📊 Results

Three query scenarios, each evaluated for both systems (Top-5):

| Query Style | Example | Metric | TF-IDF | SBERT |
|---|---|---|---|---|
| Integrated (constraints + flavor) | *"Light roast coffee from Ethiopia under 20 dollars with bright citrus and floral notes"* | Avg. similarity | 0.1363 | **0.4382** (+221.6%) |
| | | Avg. BERTScore | 0.8385 | 0.8352 |
| Short / noisy | *"floral jasmin"* | Avg. similarity | 0.3833 | **0.5077** (+32.5%) |
| | | Avg. BERTScore | 0.7810 | 0.7876 |
| Rich descriptive | *"very floral like a garden with some citrus fruit mixed in"* | Avg. similarity | 0.2243 | **0.6447** (+187.5%) |
| | | Avg. BERTScore | 0.8190 | 0.8207 |

**Takeaways**

- TF-IDF collapses on queries whose vocabulary doesn't literally overlap with the reviews (0.14 similarity for the integrated query); SBERT stays robust because it matches *meaning*, not tokens.
- BERTScore (an external judge, independent of each system's internal score) confirms the SBERT results read as more relevant — and is roughly tied on the two harder scenarios, which is itself informative: similarity score inflation alone doesn't imply better recommendations, which is why an external metric is used at all.
- The full written report is available in [`Beyond Keywords — A Hybrid Neural Semantic Approach for Coffee Recommendation.pdf`](Beyond%20Keywords%20A%20%20Hybrid%20Neural%20Semantic%20Approach%20for%20Coffee%20Recommendation.pdf).

## 🚀 Getting Started

```bash
git clone https://github.com/YiYang-09/CoffeeRecommendationUsingSBERT.git
cd CoffeeRecommendationUsingSBERT
pip install pandas scikit-learn sentence-transformers torch bert-score
```

Then open `main.ipynb` and run top to bottom. Notes:

- The dataset (`data/coffee_analysis.csv`, 2,095 coffees: name, roaster, roast, origin, price/100g, rating, sensory description) is included.
- On first run, the SBERT section encodes all 2,095 descriptions and saves them to `coffee_embeddings.pt` (~3 MB). This file is a generated artifact and is not tracked in git — it is rebuilt automatically by the notebook.
- Hardware: CPU is sufficient; encoding the corpus takes about a minute.

## 📁 Repository Structure

```
├── main.ipynb                 # Full pipeline: preprocessing, both engines, evaluation
├── data/
│   └── coffee_analysis.csv    # Cleaned dataset (2,095 coffees, 12 features)
├── coffee_embeddings.pt       # (generated) SBERT embeddings — rebuilt on first run
└── Beyond Keywords ....pdf    # Full project report
```
