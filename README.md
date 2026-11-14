# UniGuide Courses

A **hybrid recommender system** for university online courses, built on the
[Course Reviews on Coursera](https://www.kaggle.com/) dataset (Kaggle, Muhammad, 2023).

It combines **collaborative filtering (CF)** and **content-based filtering (CB)** to recommend
courses that are relevant, more diverse and more novel than a plain popularity list, including
when the user or the course is new (cold-start).

> Status: **work in progress** (first version of the README, to be updated at each milestone).

---

## 1. Goals

1. Analyse the dataset (users, courses, interactions, sparsity, popularity distribution).
2. Implement a popularity baseline and two base approaches (content-based and collaborative).
3. Design a justified hybrid strategy (weighted combination with switching).
4. Compare the hybrid with each approach alone and with the baseline, using accuracy metrics
   and beyond-accuracy metrics (coverage, diversity, novelty).
5. Test cold-start explicitly (new users and new courses).
6. Provide a simple Streamlit interface showing recommendations with explanations.
7. Document the use of generative AI transparently.

## 2. Methods

| Method | Principle |
|---|---|
| Popularity (baseline) | Weighted rating (IMDb formula), no personalisation |
| Content-Based (CB) | TF-IDF on course text, rating-weighted user profile, cosine similarity |
| Collaborative (CF) | Item-Item kNN (cosine) and truncated SVD |
| CB-Embeddings (optional) | Sentence Transformers instead of / in addition to TF-IDF |
| **Hybrid** | `Score(u,i) = alpha * CB(u,i) + (1 - alpha) * CF(u,i)`, alpha depends on the user history size (switching) |
| Hybrid-RRF (optional) | Reciprocal Rank Fusion of the rankings |

## 3. Project structure

```
uniguide-courses/
├── README.md
├── requirements.txt
├── config.yaml                  # seeds, paths, thresholds (shared by A and B)
├── data/
│   ├── raw/                     # Kaggle CSVs (NOT versioned)
│   ├── interim/                 # intermediate files (NOT versioned)
│   └── processed/               # cleaned data and splits (parquet)
├── docs/
│   ├── contracts.md             # roles, interfaces, workflow
│   ├── decisions.md             # justified technical decisions
│   ├── ai_usage_log.md          # shared AI usage journal
│   └── report/                  # report sources (10 pages max)
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_splits.ipynb
│   ├── 03_baseline_and_evaluation.ipynb
│   ├── 04_content_based.ipynb
│   ├── 05_collaborative.ipynb
│   ├── 06_hybrid.ipynb
│   ├── 07_experiments.ipynb
│   └── main.ipynb               # end-to-end run
├── src/uniguide/
│   ├── config.py                # loads config.yaml
│   ├── data/                    # loading, cleaning, filtering, text, splits, stats
│   ├── models/                  # base, popularity, content_based, embeddings,
│   │                            # collaborative, hybrid
│   ├── evaluation/              # metrics, beyond_accuracy, protocol, experiments
│   └── explain/                 # explainer
├── app/
│   └── streamlit_app.py
├── results/                     # figures, tables, precomputed results (for the demo)
├── ai_logs/                     # full exports of AI conversations (person_a/, person_b/)
└── tests/                       # small-example tests (metrics, splits, no leakage)
```

## 4. Installation

Requirements: Python 3.10+.

```bash
git clone <repository-url>
cd uniguide-courses
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Data

The dataset is **not** stored in the repository.

1. Search Kaggle for **"Course Reviews on Coursera"** and download the two files.
2. Place them in `data/raw/`:
   - `Coursera_reviews.csv` (reviews: `reviews`, `reviewers`, `date_reviews`, `rating`, `course_id`)
   - `Coursera_courses.csv` (courses: `name`, `institution`, `course_url`, `course_id`)
3. Both files are joined on `course_id`.

## 6. Usage

```bash
# 1. Build the cleaned dataset and the train/validation/test splits
python -m uniguide.data.build

# 2. Run the full pipeline (or open notebooks/main.ipynb)
jupyter notebook notebooks/main.ipynb

# 3. Launch the demo interface
streamlit run app/streamlit_app.py
```

Notebooks `01` to `07` follow the project steps and can be run in order.

## 7. Evaluation protocol (summary)

- **Relevance**: an interaction is relevant if `rating >= 4`.
- **Split**: per user, the most recent 20 % of interactions form the test set; a part of the
  training set is used for validation (tuning alpha, number of neighbours).
- **Cold-start users**: training history truncated to 0, 1, 3 or 5 interactions.
- **Cold-start courses**: about 10 % of courses removed from collaborative training.
- **Metrics**: Precision@K, Recall@K, NDCG@K, HR@K (K = 5 and 10); catalogue coverage,
  intra-list diversity, novelty; optionally MAE / RMSE.
- **Rigour**: fixed seeds, several seeds with mean and standard deviation when possible.

## 8. Team

| Person | Main responsibility |
|---|---|
| A | Data, content-based, beyond-accuracy metrics, explanations, interface |
| B | Collaborative, hybrid, evaluation framework, experiments |

Details, interfaces and working rules: see [`docs/contracts.md`](docs/contracts.md).

## 9. Use of generative AI

All AI conversations are exported without cuts to `ai_logs/`, and a shared journal is kept in
[`docs/ai_usage_log.md`](docs/ai_usage_log.md) (what was reused, what was modified and why).

## 10. Known limitations

- No official course descriptions: the text is built from name, institution and reviews.
- No reliable user identifier: reviewers are identified by display name (pseudo-identifier).
- Ratings are strongly biased towards positive values.
- The data is subsampled to run on a personal computer.
- Results must not be generalised beyond this dataset.

## 11. Roadmap

- [ ] Step 0: repository, environment, dataset, AI journal
- [ ] Step 1: data loading, cleaning, statistics
- [ ] Step 2: train / validation / test splits and cold-start groups
- [ ] Step 3: popularity baseline and evaluation framework
- [ ] Step 4: CB and CF
- [ ] Step 5: hybrid
- [ ] Step 6: extended metrics, cold-start, sensitivity, ablation
- [ ] Step 7: explanations and Streamlit interface
- [ ] Step 8: report, cleanup, rehearsal

## 12. Credits

Dataset: *Course Reviews on Coursera*, Kaggle (Muhammad, 2023).
