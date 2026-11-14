# Project Contracts (A and B)

This document is the **single source of truth** for how the two team members work together.
It is fixed at Step 0. Any change must be agreed by both people and recorded in the
[change log](#14-change-log).

---

## 1. Principles

- Each person **owns** distinct blocks, but **both must understand and explain all the code**
  (oral defence).
- Modules share the **same interface** so they can be developed in parallel and combined.
- Everything is **reproducible**: fixed seeds, shared config, no hard-coded paths.
- Code, comments, prints, README and all `.md` files are written in **English**.

## 2. Roles and ownership

### Person A: Data, content, interface

| Area | Deliverables | Location |
|---|---|---|
| Data | Loading, join, cleaning, filtering, sampling, course text construction | `src/uniguide/data/` |
| Analysis | Statistics and plots (sparsity, popularity long tail, ratings, reviews per user, time) | `notebooks/01_*` |
| Splits | Train / validation / test, cold-start user and course groups | `src/uniguide/data/splits.py` |
| Baseline | Popularity (weighted rating, IMDb formula) | `src/uniguide/models/popularity.py` |
| Content-Based | TF-IDF, user profiles, cosine similarity; optional embeddings | `models/content_based.py`, `models/embeddings.py` |
| Beyond accuracy | Coverage, intra-list diversity, novelty | `src/uniguide/evaluation/beyond_accuracy.py` |
| Explanations | Component contributions, "Because you liked...", TF-IDF keywords | `src/uniguide/explain/` |
| Interface | Streamlit application | `app/streamlit_app.py` |
| Report | Context, dataset and preparation, content approach, explanations and interface | `docs/report/` |

### Person B: Collaborative, hybrid, evaluation

| Area | Deliverables | Location |
|---|---|---|
| Accuracy metrics | Precision@K, Recall@K, NDCG@K, HR@K, optional MAE/RMSE | `src/uniguide/evaluation/metrics.py` |
| Protocol | Common evaluation loop used by all methods | `src/uniguide/evaluation/protocol.py` |
| Collaborative | Item-Item kNN (manual) and SVD | `src/uniguide/models/collaborative.py` |
| Hybrid | Normalisation, weighted combination, switching, alpha tuning, optional RRF | `src/uniguide/models/hybrid.py` |
| Experiments | Main comparison, cold-start, sensitivity, ablation | `src/uniguide/evaluation/experiments.py`, `notebooks/07_*` |
| Report | Collaborative approach, hybrid, protocol and results, limits | `docs/report/` |

### Shared tasks

| Task | How |
|---|---|
| Hybrid strategy choice | Decided together, justified in writing in `docs/decisions.md` |
| Results interpretation | Joint discussion of tables and curves |
| Cross code review | Each person reviews and comments the other's code before merge |
| AI journal | Maintained jointly in `docs/ai_usage_log.md` |
| README and final notebook | Assembled together; full run on a clean machine |
| Defence preparation | Cross questions: each answers about the other's code |
| Report: AI usage section | Written together |

## 3. Dependency order and parallel work

B's evaluation code needs A's splits, and A's CB needs B's metrics to be measured. To avoid
blocking each other:

1. **Step 0**: both agree on this document and create a **toy dataset**
   (`tests/fixtures/toy_*.parquet`, about 50 users, 20 courses) following the schemas below.
2. A delivers the real cleaned data and splits (Steps 1-2). Until then B develops metrics and
   CF on the toy dataset.
3. B delivers the metrics (Step 3) so A can evaluate popularity and CB.
4. Both models (CB, CF) are ready before the hybrid (Step 5).

Priority order if late: data, baseline + evaluation, CB, CF, hybrid, cold-start, extended
metrics, interface, embeddings / RRF / LLM.

## 4. Data contract

All processed files are stored in `data/processed/` as **parquet**.

### 4.1 `interactions.parquet` (cleaned data, before splitting)

| Column | Type | Description |
|---|---|---|
| `user_id` | int | Pseudo-identifier built from the reviewer display name |
| `course_id` | str | Course identifier (same as in the raw files) |
| `rating` | int | Rating from 1 to 5 |
| `timestamp` | datetime64 | Parsed from `date_reviews` |
| `is_relevant` | bool | `rating >= RELEVANCE_THRESHOLD` (default 4) |

### 4.2 `courses.parquet`

| Column | Type | Description |
|---|---|---|
| `course_id` | str | Join key |
| `name` | str | Course name |
| `institution` | str | Institution |
| `url` | str | Course URL |
| `text` | str | Cleaned text: name + institution + at most N reviews |
| `n_reviews` | int | Number of reviews (after filtering) |

### 4.3 Splits

| File | Content |
|---|---|
| `train.parquet` | Same columns as `interactions.parquet` |
| `val.parquet` | Validation subset taken from the **past** of each user (never from test) |
| `test.parquet` | Most recent interactions per user |
| `cold_users.json` | `{"0": [user_ids], "1": [...], "3": [...], "5": [...]}`: users whose train history is truncated to that size |
| `cold_courses.json` | List of `course_id` removed from collaborative training |

Rules:

- The split is performed **first**, before any model fitting, similarity or statistic that
  could use test data (popularity, TF-IDF vocabulary, user profiles included).
- Test ratings never enter any recommendation score.
- A split-integrity test (`tests/test_splits.py`) checks that no `(user_id, course_id)` pair
  appears in both train and test and that test timestamps are not older than train timestamps
  for the same user.
- Sizes before and after each filter are logged and reported.

## 5. Configuration contract

All seeds, paths and thresholds live in `config.yaml` and are read through
`uniguide.config`. Nobody hard-codes them.

```yaml
seed: 42
paths:
  raw: data/raw
  processed: data/processed
  results: results
data:
  min_user_reviews: 5
  min_course_reviews: 50
  max_reviews_per_course_text: 50
  n_users_sample: 30000
  relevance_threshold: 4
split:
  test_ratio: 0.2
  val_ratio: 0.1
  cold_user_sizes: [0, 1, 3, 5]
  cold_course_ratio: 0.1
evaluation:
  k_values: [5, 10]
  n_seeds: 3
```

Values above are starting points and will be adjusted and justified in `docs/decisions.md`.

## 6. Model interface contract

Every recommender (popularity, CB, CF, hybrid, RRF, embeddings) inherits from
`BaseRecommender` in `src/uniguide/models/base.py`.

```python
class BaseRecommender(ABC):
    name: str

    def fit(self, train: pd.DataFrame, courses: pd.DataFrame) -> "BaseRecommender":
        """Learn from training data only. Must return self."""

    def score(self, user_id: int, item_ids: list[str]) -> np.ndarray:
        """One raw score per item_id, same order as item_ids (np.nan if not scorable)."""

    def recommend(self, user_id: int, k: int = 10,
                  exclude_seen: bool = True) -> pd.DataFrame:
        """DataFrame with columns ['course_id', 'score'], sorted by score descending."""
```

Behaviour rules:

- **Unknown user or user with 0 interactions**: fall back to popularity scores
  (never raise an exception).
- **Unknown item** (not in the model): `score` returns `np.nan` for that item. The hybrid
  handles this case (see section 7).
- `exclude_seen=True` removes every course present in the user's **training** history.
- Models must be **deterministic** given the seed.
- No model reads `test.parquet`.

## 7. Score normalisation and hybrid contract

- Scores of each component are normalised with **min-max per user over the candidate set**
  into `[0, 1]`. If all scores are equal, the normalised score is `0.0`.
- Hybrid: `Score(u,i) = alpha * CB(u,i) + (1 - alpha) * CF(u,i)`.
- `alpha` is chosen **on validation** (grid from 0 to 1) and may depend on the number of
  training interactions of the user (switching): few interactions means more weight on CB,
  0 interactions means popularity, many interactions means more weight on CF.
- **New course without CF score** (`np.nan`): weights are renormalised so the score relies on
  CB only.
- The final alpha rules (thresholds and values) are decided jointly and written in
  `docs/decisions.md`.

## 8. Evaluation contract

Functions in `evaluation/metrics.py` (owned by B) use these signatures, so A can call them:

```python
precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float
recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float
ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float
hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float
```

Beyond-accuracy functions (owned by A) in `evaluation/beyond_accuracy.py`:

```python
catalog_coverage(all_recommendations: list[list[str]], catalog: set[str]) -> float
intra_list_diversity(recommended: list[str], item_similarity) -> float
novelty(recommended: list[str], popularity: dict[str, float]) -> float
```

Common protocol:

- `relevant` for a user = courses in **test** with `is_relevant == True`.
- All methods are evaluated on **the same users**, the **same K** (5 and 10) and the same
  candidate rules. Users with no relevant item in test are excluded and counted.
- Results are saved as CSV in `results/` with columns
  `method, experiment, seed, k, metric, value`.
- Several seeds when possible, reported as mean and standard deviation.

## 9. Coding conventions

- Python 3.10+, formatting with `black`, imports sorted, type hints on public functions.
- English for identifiers, comments, docstrings, prints, notebook text and `.md` files.
- Naming: `snake_case` for functions and variables, `PascalCase` for classes.
- DataFrame column names are exactly those in section 4. Do not rename them in modules.
- Each public function has a short docstring (purpose, arguments, return value).
- Notebooks call functions from `src/`; no business logic copied between notebooks.
- Clear outputs before committing notebooks, except `main.ipynb` and notebooks with
  figures required by the report.

## 10. Git workflow

- `main`: always runnable, only receives reviewed merges.
- One branch per person: `feature/a-<topic>` and `feature/b-<topic>`.
- Merge via pull request, **reviewed by the other person**; the reviewer reads, runs and
  comments the code.
- Commit messages: `type: short description` where type is
  `data | model | eval | app | docs | fix | test | refactor`.
- Never commit data files, `.venv`, large result files or API keys (`.gitignore`).

## 11. Communication and rhythm

- Short sync **every two days**: progress, blockers, integration (15 minutes).
- Module walkthrough sessions: each person presents their module to the other once it works.
- Blocking questions about interfaces are written in the repository (issue or this file),
  not only said orally.
- Weekly checklist: Roadmap in `README.md` updated.

## 12. Use of generative AI (rules)

- One account or conversation per person; export conversations **regularly and without cuts**
  to `ai_logs/person_a/` and `ai_logs/person_b/`.
- Journal entry for each significant use in `docs/ai_usage_log.md`:

  | Date | Person | Tool | Request | Reused as is | Modified (and why) | Errors found |
  |---|---|---|---|---|---|---|

- Never integrate code that is not understood: read it, run it, test it on a small example.
- Pay special attention to **splitting** and **metric computation** (most frequent errors).
- Keep track of: AI-generated vs hand-written code, AI errors corrected, and technical choices
  made by us (threshold, filtering, hybrid strategy, hyperparameters).

## 13. Definition of done

A task is done when:

- [ ] the code runs from a clean environment with the shared config;
- [ ] public functions follow the interfaces above and have docstrings;
- [ ] a small test or sanity check exists (especially metrics and splits);
- [ ] the other person has reviewed it and can explain it;
- [ ] the journal and `docs/decisions.md` are updated when needed;
- [ ] results (if any) are saved in `results/` with seed and data size noted.

## 14. Change log

| Date | Change | Agreed by |
|---|---|---|
| (Step 0) | First version of the contracts | A, B |
