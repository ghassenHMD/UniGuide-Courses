# Technical Decisions

Every non-trivial choice of the project, with the alternatives considered and the evidence
(file in `results/tables/` or notebook). Decisions are taken by the team, not by the AI tools.

**Status:** `Final` = settled and verified. `Confirm` = depends on the collaborative-ranking fix
(§6, §9) and must be checked after re-running notebooks 05 to 08 and `main.ipynb`.
**Owner:** A, B or Joint.

---

## 1. Data and preprocessing

| ID | Decision | Why / evidence | Owner | Status |
|---|---|---|---|---|
| D01 | Use both Kaggle files, joined on `course_id`; keep only reviews of courses present in the courses file. | 604 courses have reviews, 623 are listed. | A | Final |
| D02 | The user id is a **pseudo-identifier**: reviewer display name without the `By ` prefix, mapped to an integer. | The dataset has no user id. Homonyms are merged: this is a documented limit. | A | Final |
| D03 | Exclude generic reviewers (`Deleted A` and every name starting with "deleted"). | `By Deleted A` alone has 5,412 reviews. | A | Final |
| D04 | Drop duplicate (user, course) pairs, keeping the most recent review. | 1,449,143 rows fall to 505,177: many pseudo-users are several people with the same name reviewing the same course. Strong sign of merged homonyms. | A | Final |
| D05 | Drop pseudo-users with more than 100 reviews (`max_user_reviews`). | A person does not review 400 of 623 courses (`David M`: 435 reviews). Removes 225 rows, 2 pseudo-users. | A | Final |
| D06 | Keep users with ≥ 5 interactions and courses with ≥ 50, **iteratively** until stable. | Removing courses lowers user counts and vice versa. Result: 138,756 interactions, 14,792 users, 422 courses. | A | Final |
| D07 | The planned sample of 30,000 users is a **no-op**. | Only 14,792 users remain after filtering; the matrix is small enough for a laptop. | A | Final |
| D08 | Relevance threshold: **rating ≥ 4**. | Imposed by the project statement. Caveat: 92.8 % of ratings are ≥ 4 (76.2 % are 5), so almost every held-out interaction is relevant and the task is closer to "what will the user rate next" than to "what will they like". | Joint | Final |

## 2. Splits and evaluation protocol

| ID | Decision | Why / evidence | Owner | Status |
|---|---|---|---|---|
| D09 | **Per-user chronological split** 70 / 10 / 20 (train / validation / test). Ties on the daily date are broken by `course_id`. Users need ≥ 3 interactions after removing cold courses. | Test is the future of each user. Sizes: train 81,530, validation 16,983, test 24,629 rows. | A | Final |
| D10 | The split is the **first** step; `check_no_leakage` verifies no pair is shared and the order is chronological; unit tests cover it. | Leakage is the most frequent error of this kind of project. | A | Final |
| D11 | **Tune on validation, report on test.** Tuning: fit on train, evaluate on validation (3,000-user sample, fixed seed, for the slow grids). Final: fit on train + validation, evaluate once on test. | Test ratings never choose anything. | Joint | Final |
| D12 | **Course text is built from training reviews only** (name + institution + ≤ 50 reviews sampled with a fixed seed, lowercased, letters only, English stop words and tokens ≤ 2 letters removed). Median text: 524 words. | A text built from all reviews would leak test content into the content model. | A | Final |
| D13 | **Cold-start courses:** 10 % of courses (42) removed from train, validation and test; their interactions go to `cold_course_test.parquet` (15,434 rows, 9,514 users). Their text is name + institution only. | A real new course has no reviews. | A | Final |
| D14 | **Cold-start users:** four disjoint groups of 445 users (5 % each), drawn among users with ≥ 5 pre-test interactions. History is cut to the N most recent interactions (N = 0, 1, 3, 5) at experiment time; files are not truncated. | One fit per experiment; the main comparison is untouched. Limit: these users are slightly more active than average. | A | Final |
| D15 | **Evaluated users:** those with ≥ 1 relevant test course: 14,039 of 14,756 (717 excluded and counted). Identical for every method and K. | Metrics are undefined without a relevant item. | B | Final |
| D16 | **Main comparison ranks the 380 warm courses for every method.** | Cold courses never appear in test; a model allowed to recommend them (CB) would be penalised unfairly. | Joint | Final |
| D17 | **Cold-start course experiment ranks the 42 new courses among themselves** (`RestrictedRecommender`), with `1/42` per slot as chance reference. | Ranking them among the 380 warm courses would hide the content signal behind popular courses. A random baseline on an earlier run gave NDCG@10 0.121 / HitRate@10 0.334. | B | Final |
| D18 | Metrics: Precision, Recall, NDCG, HitRate at K = 5 and 10; mean, standard deviation and 95 % CI over users. | Project statement; CI makes small differences readable. | B | Final |

## 3. Popularity baseline (A)

| ID | Decision | Why / evidence | Status |
|---|---|---|---|
| D19 | IMDb weighted rating `WR = v/(v+m)·R + m/(v+m)·C`, fitted on training ratings. `m` is a quantile of the course rating counts. | Imposed by the project statement. | Final |
| D20 | A course with no training rating scores NaN (not the global mean). | Popularity cannot rank a new course; this is exactly what the cold-course experiment shows. | Final |
| D21 | `m_quantile = 0.99` in `config.yaml`. | `tuning_popularity.csv` (validation NDCG@10): q = 0.9 → 0.0920, **0.99 → 0.0999**. **The grid stops at 0.99 and is still rising.** A check on the full validation set gave 0.995 → 0.1016, 0.999 → 0.1030, 1.0 → 0.1030. Large `m` makes WR almost a ranking by number of reviews. | Confirm |
| D22 | Report a stricter second baseline, **most reviewed courses** (`PopularityRecommender(mode="count")`, if added). | On the validation set, ranking by raw review count gave NDCG@10 **0.1101**, better than any WR. A hybrid should beat the best naive baseline. | Confirm |

## 4. Content-based and embeddings (A)

| ID | Decision | Why / evidence | Status |
|---|---|---|---|
| D23 | TF-IDF, unigrams + bigrams, `min_df=2`, `max_df=0.5`, sublinear tf, 20,000 features. | `max_df=0.5` removes words shared by most courses ("course", "great"). `tuning_content_based.csv`: NDCG@10 0.1002 (unigrams) to 0.1054 (best); differences are inside the noise (±0.004 on 3,000 users), so the defaults are kept. | Final |
| D24 | User profile = mean of the TF-IDF vectors of **liked** courses (rating ≥ 4), weighted by the rating (`rating_offset = 0`); score = cosine. | Project statement. `rating_offset = 3` brought no gain (0.1050). | Final |
| D25 | Users with no usable profile fall back to the popularity ranking; never an exception. | Interface contract (`docs/contracts.md` §6). | Final |
| D26 | Embeddings: `all-MiniLM-L6-v2`, text cut in 100-word chunks (≤ 15) averaged, plus a separate embedding of "name, institution", mixed with `name_weight`. Cached on disk (does not depend on `name_weight`). | The model reads ~256 tokens only; reviews are generic, the name carries the topic. Neighbours of *Machine Learning* are sensible (Duke, Michigan, Washington ML courses). | Final |
| D27 | `name_weight = 0.5`. | `tuning_embeddings.csv` (validation): 0.0 → NDCG 0.0852; **0.5 → 0.1047, HitRate 0.2115**; 1.0 → 0.1075, HitRate 0.2012. NDCG of 0.5 and 1.0 is a tie; 0.5 has the best HitRate. | Final |
| D28 | Fusion TF-IDF + embeddings: 50/50 weighted sum of min-max scores. | Best content-only model on validation (NDCG 0.1165) and on test (0.108 vs 0.097). | Final |
| D29 | **The final hybrid uses TF-IDF** as its content model (`model.content: tfidf`). | Inside the hybrid, the content weight is small, so the content variants are indistinguishable on accuracy (validation 0.2068 for both TF-IDF and fusion at α = 0.2). TF-IDF is faster and supplies the keywords used by the explanations. Embeddings and fusion stay as compared CB variants. | Confirm |
| D30 | Embeddings are not available for the explainer: it uses a TF-IDF model for keywords and "because you liked". | Keywords must be words of the course text. | Final |

## 5. Beyond-accuracy metrics (A)

| ID | Decision | Why |
|---|---|---|
| D31 | **Coverage** = share of the 380 warm courses recommended at least once to at least one evaluated user. | Standard definition. The statement says "share of courses never recommended": that is `1 − coverage`. |
| D32 | **Diversity** = 1 − mean pairwise cosine similarity inside a list (K = 10), measured with **two lenses**: TF-IDF and embeddings. | With a TF-IDF lens a TF-IDF model looks less diverse by construction (it recommends what resembles the profile). A conclusion that holds under both lenses is robust. |
| D33 | **Novelty** = mean of `−log2(p)` where `p` is the share of training users who rated the course (floor of one user). | Self-information form of "inverse popularity"; 1 bit if half the users rated it, ~4.3 bits at 5 %. |
| D34 | Metrics are computed on the same users and the same lists as accuracy; one `recommend(k=10)` call per user, top-5 = first five. | Consistency and speed. |

## 6. Collaborative filtering (B)

| ID | Decision | Why / evidence | Status |
|---|---|---|---|
| D35 | Item-Item kNN (cosine on the user-item rating matrix, neighbours pruned to the top `n_neighbors`) and truncated SVD are both implemented. | Project statement. | Final |
| D36 | **Item-kNN ranks by the sum of `similarity × rating` over the user's rated neighbours.** The current repository ranks by the similarity-weighted **mean** (a rating prediction). | Weighted mean puts niche courses with one good neighbour first. Same code, same split, validation NDCG@10: weighted mean **0.0418 (k=20) / 0.0329 (k=50)**; sum **0.2002 / 0.2059**. Test (hybrid-free CF): 0.0379 → **0.1888**. Tell-tale in the repository: `tuning_cf.csv` gets worse with more neighbours. Rating prediction (MAE) keeps the weighted mean via `predict_ratings`. | **Confirm** (fix to apply, §9) |
| D37 | `cf_n_neighbors = 200`. | Corrected validation NDCG@10: k = 5 → 0.1767, 10 → 0.1864, 20 → 0.1904, 50 → 0.1956, 100 → 0.1981, **200 → 0.2006**. Still rising at the edge of the grid (380 courses): re-run with "all neighbours" before final. | Confirm |
| D38 | kNN rather than SVD as the final CF. | Corrected validation: kNN 0.2006 vs SVD best 0.1369 (5 components; 10 → 0.1304, 20 → 0.1089). On the repository's current (weighted-mean) kNN the SVD looked better: that was the bug, not a property of the methods. | Confirm |
| D39 | MAE / RMSE are reported for **kNN only**. | The current `eval_rating_prediction.csv` gives SVD an MAE of 4.34 (worse than the global mean, 0.55): the SVD reconstruction of the raw rating matrix is not on the rating scale. | Confirm |
| D40 | Unknown users fall back to the popularity ranking (count / max). | Interface contract. Note: this fallback is a count ranking, different from the IMDb baseline of §3. | Final |

## 7. Hybrid (Joint)

| ID | Decision | Why / evidence | Status |
|---|---|---|---|
| D41 | Weighted combination `α·CB + (1−α)·CF` of **min-max normalised** scores per user (constant scores → 0). | Statement. Raw scales differ (cosine vs similarity sums). | Final |
| D42 | If a component cannot score a course (NaN), the other one is used alone with weight 1. New courses are therefore scored by content only. | Project statement; lets the hybrid recommend new courses. | Final |
| D43 | Zero training interactions → popularity ranking. | Statement ("popularity at 0 interaction"). | Final |
| D44 | `alpha = 0.2`. | Corrected validation (3,000 users): α = 0 → 0.2006, 0.1 → 0.2006, **0.2 → 0.2002**, 0.3 → 0.1990, 0.5 → 0.1894, 0.9 → 0.1279, 1 → 0.1078. The first alphas are equal within the noise. Rule: the largest α with no measurable loss, so the content component stays active. The repository's current α = 0.9 is an artefact of D36. | Confirm |
| D45 | **No switching rule** (`alpha_schedule: []`) beyond the popularity fallback. | The statement suggested a high CB weight for short histories. The data contradicts it: item-kNN already beats popularity with 1 interaction (cold-start table below). The tuned schedule (`[(1,0.2),(3,0.3),(5,0.0)]`) is noisy (groups of ~445 users) and does not support high CB weights. The switching mechanism stays in the code. | Confirm |
| D46 | Reciprocal Rank Fusion is reported as a variant, not chosen. | Test NDCG@10 0.1535 vs 0.1890 for the weighted hybrid: ranks discard the score magnitudes that carry the evidence. | Confirm |

**Reference results on the test set** (14,039 users; kNN k = 200, α = 0.2, no switching). Produced by re-running the
repository's own experiment functions with the fix; `main.ipynb` must regenerate the official files.

| Method | NDCG@5 | NDCG@10 | HitRate@10 | Precision@10 | Recall@10 |
|---|---|---|---|---|---|
| Popularity | 0.0818 | 0.1092 | 0.2790 | 0.0330 | 0.1966 |
| Content-based | 0.0823 | 0.0969 | 0.2043 | 0.0216 | 0.1591 |
| Collaborative | 0.1623 | 0.1888 | 0.3703 | 0.0433 | 0.2842 |
| Hybrid | 0.1628 | 0.1890 | 0.3702 | 0.0431 | 0.2847 |
| Hybrid RRF | 0.1317 | 0.1535 | 0.3081 | 0.0337 | 0.2373 |

95 % CI on NDCG@10: ±0.003 to ±0.005. Paired differences (NDCG@10): hybrid − CF **+0.0003 ± 0.0011**;
hybrid − popularity **+0.0798 ± 0.0054**.

| Beyond accuracy (K = 10, TF-IDF lens) | Coverage | Diversity | Novelty (bits) |
|---|---|---|---|
| Popularity | 0.061 | 0.929 | 3.44 |
| Content-based | 0.966 | 0.884 | 5.93 |
| Collaborative | 0.882 | 0.928 | 3.53 |
| Hybrid | 0.903 | 0.922 | 3.68 |

| Cold-start users, NDCG@10 (training history) | 0 | 1 | 3 | 5 |
|---|---|---|---|---|
| Popularity | 0.109 | 0.107 | 0.108 | 0.117 |
| Content-based | 0.097 | 0.097 | 0.085 | 0.071 |
| Collaborative | 0.092 | 0.157 | 0.132 | 0.147 |
| Hybrid | 0.092 | 0.156 | 0.135 | 0.142 |

Groups have 423 to 436 evaluated users; at 0 interactions the differences are inside the noise.

| Cold-start courses (42 new courses, ranked among themselves, K = 10) | NDCG | HitRate | Recall |
|---|---|---|---|
| Collaborative / Popularity | 0.000 | 0.000 | 0.000 |
| Content-based = Hybrid | 0.369 | 0.646 | 0.529 |

**How to read it in the report.** The hybrid matches CF on students with a history (it does not improve it); its value is
cold-start courses, where only content works, and a small gain in coverage and novelty over CF. This is the outcome
the project statement anticipates ("hybrid not better than CF alone → analyse by group and report honestly").

## 8. Explanations and interface (A)

| ID | Decision | Why |
|---|---|---|
| D47 | Explanations are **computed from the model**, never invented: component contributions (`cb_part`, `cf_part`, popularity for users without history), the liked courses (rating ≥ 4) closest to the recommended one (collaborative weight or text similarity), and the top TF-IDF terms. A test checks that the contributions add up to the hybrid score and that the same courses are returned. | Statement §5.4. |
| D48 | No language model rewrites the explanations. | Optional in the statement and the lowest priority; templates keep them deterministic and checkable. |
| D49 | Streamlit app with four tabs (Recommendations, Compare methods, New user, Results), models fitted at start-up from `config.yaml`; a **held-out check** marks recommended courses the user really rated ≥ 4 in the test period. | Demo scenario of the statement; the check shows real hits without leaking test data into the models. |
| D50 | The app shows the pseudo-id only, never the reviewer's display name. | Privacy. |
| D51 | Known weakness: keywords come from reviews and may contain instructor names (*andrew*, *chuck*) or review jargon (*resubmit assignment*). | Accepted; improving needs the course syllabus, which the dataset lacks. |

## 9. Open issues before submission

| # | Issue | Owner | Priority |
|---|---|---|---|
| 1 | **Apply the collaborative-ranking fix** (D36): `_predict_all(ranking=...)`, `predict_ratings`, `rating_errors` uses it, plus the test. 50 tests pass with it. | B | **Blocking** |
| 2 | **Sync `config.yaml`** with the selected values: `cf_method: knn`, `cf_n_neighbors: 200` (or all), `alpha: 0.2`, `alpha_schedule: []`, `popularity_m_quantile` (D21). Today it holds `knn / 20 / 0.5 / [[2,0.9],[5,0.7]]`, while notebooks 05 and 06 print `svd / 0.9 / [[1,0.8],[3,0.9],[5,0.9]]`. | B | Blocking |
| 3 | Re-run notebooks 05, 06, 07, 08 and `main.ipynb` so every CSV, figure and the app use the same models. | Joint | Blocking |
| 4 | `comparison_k10.csv` lists only popularity and the content models: add collaborative, hybrid and RRF in notebook 08 (the app's Results tab reads this file). | A | High |
| 5 | Extend the popularity grid beyond 0.99 and decide on the "most reviewed" baseline (D21, D22). | A | Medium |
| 6 | Restrict the MAE / RMSE table to kNN (D39) and state in the report that the global mean is a strong rating baseline (MAE 0.55). | B | Medium |
| 7 | Re-run kNN with all neighbours to confirm D37. | B | Medium |
| 8 | Create `docs/ai_usage_log.md` and export all AI conversations to `ai_logs/person_a/` and `ai_logs/person_b/` without cuts. | Joint | Required |
| 9 | Report (10 pages), demo rehearsal, cross-questions on each other's code. | Joint | Required |

## 10. Generative-AI use: errors found and choices made by us

Material for the report's AI section (details go in `docs/ai_usage_log.md`).

**AI errors detected and corrected**

- A content-based attribute named `item_to_idx` collided with the name `evaluate_cold_start` uses to decide which courses
  the CF knows: it would have returned an empty result for the content model, silently. Renamed `item_index`.
- The first embedding cache design included `name_weight` in its key, so tuning three values would have re-encoded the
  corpus three times. The cache now stores only the slow part.
- A label edit in the Streamlit app introduced nested quotes (syntax error), caught by running the app headlessly.
- **The collaborative kNN ranked by predicted rating instead of by evidence (D36).** It passed every unit test, produced no
  error, and made CF five times worse than it should be. It was found by auditing the results (CF below popularity; accuracy
  decreasing with more neighbours) and confirmed by comparing both scorings on the same split.
- The hybrid's tuned alpha (0.9) and its poor result were a consequence of that error: a wrong component led to a wrong conclusion.

**Choices made by the team, not by the AI:** relevance threshold (D08), user filtering and homonym cap (D03 to D07),
temporal split and cold-start groups (D09, D13, D14), evaluation protocol (D15 to D17), `m_quantile` (D21), TF-IDF and embedding
settings (D23 to D28), definitions of the beyond-accuracy metrics (D31 to D33), alpha and the decision to drop the switching
rule (D44, D45), the final content model (D29).

**Verification habits that worked:** a leakage test on the split, metric tests on tiny examples, comparing two implementations on
the same split, and checking that the explanation scores add up to the model scores.

## 11. Limitations to state in the report

1. Pseudo-identifiers: homonyms are merged (D02, D04); the real number of students is unknown.
2. Positive-rating bias: 92.8 % of interactions are relevant (D08); results depend on this threshold.
3. Course text is built from reviews, not from official descriptions (D12).
4. 380 warm courses only: popularity is a strong baseline because the catalogue is small and concentrated
   (the top 10 % of courses hold 44 % of the interactions).
5. Cold-start groups are small (~445 users) and slightly more active than average; the course cold-start uses 42 courses.
6. The explanation keywords inherit the noise of reviews (D51).
7. Results apply to this dataset and must not be generalised.