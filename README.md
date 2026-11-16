# UniGuide Courses

A **hybrid recommender system** for university online courses, built on the
*Course Reviews on Coursera* dataset (Kaggle, Muhammad, 2023).

It combines **collaborative filtering (CF)** and **content-based filtering (CB)** to recommend
courses to a student, with an explanation for every recommendation, and it is evaluated on
accuracy, catalogue coverage, diversity, novelty and **cold-start** (new students and new courses).

- Python package: `src/uniguide/`
- Notebooks: `notebooks/` (one per project step, plus an end-to-end `main.ipynb`)
- Demo interface: `app/streamlit_app.py`
- Documentation: `docs/contracts.md` (roles and interfaces), `docs/decisions.md` (every technical choice and its evidence)

---

## 1. What the system does

| Method | Principle |
|---|---|
| Popularity (baseline) | IMDb weighted rating `WR = v/(v+m)·R + m/(v+m)·C`, same list for everybody |
| Content-Based (CB) | TF-IDF on course text, user profile = rating-weighted mean of liked courses (rating ≥ 4), cosine similarity |
| CB with embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) instead of TF-IDF; fusion of both |
| Collaborative (CF) | Item-Item kNN (cosine, ranked by sum of similarity × rating) and truncated SVD |
| **Hybrid** | `Score(u,i) = α·CB(u,i) + (1−α)·CF(u,i)`, min-max normalised per user, switching rule on history size, CB alone for courses CF cannot score |
| Hybrid RRF | Reciprocal Rank Fusion of the CB and CF rankings (variant) |

Rules built into the hybrid: a user with **no history** gets the popularity ranking; a **new course**
(no ratings) is scored by content only.

Each recommendation is explained with real data: the contribution of each component to the score,
the liked courses that support it ("Because you liked …") and its top TF-IDF keywords.

## 2. Project structure

```
UniGuide-Courses/
├── README.md
├── config.yaml                    # seed, paths, thresholds and final model settings
├── pyproject.toml / requirements.txt
├── data/
│   ├── raw/                       # Kaggle CSV files (NOT versioned)
│   ├── interim/                   # embedding cache (NOT versioned)
│   └── processed/                 # cleaned data, splits, cold-start groups (parquet / json)
├── docs/
│   ├── contracts.md               # roles, data and model interfaces, workflow
│   ├── decisions.md               # justified technical decisions and open issues
│   ├── ai_usage_log.md            # shared journal of generative-AI use
│   └── report/                    # report sources (10 pages max)
├── notebooks/
│   ├── 01_data_analysis.ipynb     # statistics, sparsity, popularity, ratings
│   ├── 02_splits.ipynb            # train / validation / test, cold-start groups
│   ├── 03_baseline_and_evaluation.ipynb
│   ├── 04_content_based.ipynb
│   ├── 04b_embeddings.ipynb
│   ├── 05_collaborative.ipynb
│   ├── 06_hybrid.ipynb            # alpha, switching rule, RRF
│   ├── 07_experiments.ipynb       # comparison, cold-start, sensitivity, ablation
│   ├── 08_beyond_accuracy.ipynb   # coverage, diversity, novelty, comparison table
│   └── main.ipynb                 # end-to-end run with the settings of config.yaml
├── src/uniguide/
│   ├── config.py                  # reads config.yaml
│   ├── pipeline.py                # builds the final models + explainer from config.yaml
│   ├── data/                      # loading, cleaning, filtering, text, splits, stats
│   ├── models/                    # base, popularity, content_based, embeddings,
│   │                              # collaborative, hybrid
│   ├── evaluation/                # metrics, protocol, beyond_accuracy, experiments
│   └── explain/                   # explainer
├── app/streamlit_app.py           # demo interface
├── results/{tables,figures}/      # all result files used by the report and the demo
├── ai_logs/                       # full exports of the AI conversations (person_a/, person_b/)
└── tests/                         # unit tests (metrics, splits, models, explainer)
```

## 3. Installation

Python 3.10 or newer.

```bash
git clone <repository-url>
cd UniGuide-Courses
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                     # makes `import uniguide` work everywhere
```

`sentence-transformers` (PyTorch) is only needed for the embedding models. If its installation fails on
your Python version, compute the embeddings once elsewhere (e.g. Google Colab) and copy the cache file
`data/interim/embeddings_*.npz`: the cache is recognised as long as the course texts are identical.
The default final model uses TF-IDF and does not need it.

## 4. Data

The dataset is **not** stored in the repository.

1. Download *Course Reviews on Coursera* from Kaggle.
2. Put both files in `data/raw/`:
   - `Coursera_reviews.csv` (`reviews`, `reviewers`, `date_reviews`, `rating`, `course_id`)
   - `Coursera_courses.csv` (`name`, `institution`, `course_url`, `course_id`)

## 5. Usage

```bash
# 1. Cleaned interactions, then train / validation / test splits and cold-start groups
python -m uniguide.data.build
python -m uniguide.data.splits

# 2. Tests
pytest tests/

# 3. Notebooks 01 to 08 in order, or the end-to-end run
jupyter notebook notebooks/main.ipynb

# 4. Demo interface
streamlit run app/streamlit_app.py
```

The final hyperparameters are read from the `model:` section of `config.yaml` by `uniguide.pipeline`,
so notebooks and the app always use the same models. Tuning is done in notebooks 03 to 06 on the
validation set; copy the selected values into `config.yaml`.

The app fits the models at start-up (a few seconds) and has four tabs: **Recommendations** (top-K with
explanations and a held-out check), **Compare methods** (popularity, CB, CF, hybrid side by side),
**New user (cold-start)** (simulate a student with 0, 1, 2… rated courses) and **Results**.

## 6. Evaluation protocol

- **Relevance:** a rating ≥ 4 is relevant (92.8 % of ratings are ≥ 4, so the threshold is permissive).
- **Split:** per user, chronological: 70 % train, 10 % validation, 20 % test (the latest). The split is done
  first; a test is run that checks no (user, course) pair is shared and that the order is chronological.
- **Tuning and final run:** hyperparameters are tuned by fitting on train and evaluating on validation;
  final models are fitted on train + validation and evaluated once on test.
- **Users evaluated:** those with at least one relevant test course (14,039 of 14,756), identical for all methods.
- **Main comparison:** all methods rank the same 380 courses (the "warm" catalogue), K = 5 and 10.
- **Cold-start users:** groups of about 445 users whose history is cut to 0, 1, 3 or 5 interactions.
- **Cold-start courses:** 42 courses (10 %) removed from every training set; ranked among themselves.
- **Metrics:** Precision, Recall, NDCG, HitRate @K; catalogue coverage, intra-list diversity (two similarity
  lenses), novelty; MAE / RMSE for rating prediction (kNN only).
- **Rigour:** fixed seed 42; standard deviation and 95 % confidence interval over users; several seeds for
  the sparsity experiment.

## 7. Dataset in numbers

| | |
|---|---|
| Raw reviews / pseudo-users / courses | 1,454,711 / 287,808 / 604 |
| After cleaning and filtering (≥ 5 reviews per user, ≥ 50 per course) | 138,756 interactions, 14,792 users, 422 courses |
| Sparsity | 97.8 % |
| Interactions per user | mean 9.4, median 7, max 88 |
| Interactions per course | mean 329, median 168, max 3,499 |
| Share of interactions on the top 10 % of courses | 44 % |
| Rating distribution | 1: 1.5 %, 2: 1.6 %, 3: 4.1 %, 4: 16.6 %, 5: 76.2 % |
| Period | 2015-08-07 to 2020-10-10 |
| Train / validation / test rows | 81,530 / 16,983 / 24,629 |

## 8. Results

All figures come from `results/tables/` (generated by `notebooks/main.ipynb` and notebooks 03 to 08).
`docs/decisions.md` explains each choice and states which numbers must be refreshed.

Reference run on the test set, K = 10, 14,039 users (item-kNN with k = 200, α = 0.2, no switching;
**to be regenerated with `main.ipynb` after the collaborative-ranking fix, see `docs/decisions.md` §9**):

| Method | NDCG | HitRate | Coverage | Diversity | Novelty |
|---|---|---|---|---|---|
| Popularity | 0.109 | 0.279 | 0.061 | 0.929 | 3.44 |
| Content-based | 0.097 | 0.204 | 0.966 | 0.884 | 5.93 |
| Collaborative (kNN) | 0.189 | 0.370 | 0.882 | 0.928 | 3.53 |
| **Hybrid** | **0.189** | **0.370** | 0.903 | 0.922 | 3.68 |
| Hybrid RRF | 0.154 | 0.308 | not computed | not computed | not computed |

Main findings:

1. CF and the hybrid are far better than popularity (+0.08 NDCG, +0.09 HitRate).
2. On students with a history, the hybrid is **not** better than CF alone (difference +0.0003 ± 0.0011 NDCG).
3. The content-based model is the only method that can recommend **new courses**: on the 42 held-out courses it
   reaches NDCG 0.369 / HitRate 0.646 where CF and popularity score 0.
4. Item-kNN already beats popularity with a single training interaction, so a strong content weight for short
   histories is not supported by this dataset.

## 9. Use of generative AI

All conversations are exported without cuts to `ai_logs/`. `docs/ai_usage_log.md` lists what was generated,
what was reused or modified, the AI errors found and the choices made by the team.
`docs/decisions.md` §10 summarises the AI errors detected so far.

## 10. Known limitations

- No official course description: course text = name + institution + a sample of training reviews.
- No reliable user id: reviewers are identified by display name; homonyms are merged, generic names removed.
- Ratings are strongly biased towards positive values, so almost every held-out interaction is "relevant".
- The explanations' keywords come from reviews and can contain instructor names or review jargon.
- Results hold for this dataset only and must not be generalised.

## 11. Team

| Person | Responsibility |
|---|---|
| A | Data, splits, popularity, content-based and embeddings, beyond-accuracy metrics, explanations, interface |
| B | Evaluation framework, collaborative filtering, hybrid, experiments |

Interfaces and working rules: [`docs/contracts.md`](docs/contracts.md).

## 12. Status

- [x] Data cleaning, filtering, statistics, splits and cold-start groups
- [x] Popularity, content-based (TF-IDF, embeddings, fusion), collaborative, hybrid, RRF
- [x] Evaluation framework, beyond-accuracy metrics, cold-start, sensitivity, ablation code
- [x] Explanations and Streamlit interface
- [ ] Apply the collaborative-ranking fix, update `config.yaml`, re-run notebooks 05 to 08 and `main.ipynb`
- [ ] `docs/ai_usage_log.md` and export of the AI conversations to `ai_logs/`
- [ ] Report (10 pages) and demonstration rehearsal

## 13. Credits

Dataset: *Course Reviews on Coursera*, Kaggle (Muhammad, 2023).