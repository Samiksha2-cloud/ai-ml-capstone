# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

## Overview

This part builds on Part 2's classification setup (`X_train_scaled`, `X_test_scaled`, `y_clf_train`, `y_clf_test`, using the same `random_state=42` split for consistency) to explore tree-based models — from a single decision tree through to tuned ensemble methods — culminating in a serialized, production-ready pipeline.

---

## Task 1 — Decision Tree Baseline (Unconstrained)

Trained a `DecisionTreeClassifier` with no depth restriction (`max_depth=None`).

- **Train accuracy:** 0.99993
- **Test accuracy:** 0.97370
- **Gap:** ~2.62 percentage points

**Does this show overfitting?** Yes — the near-perfect training accuracy (99.99%) compared to noticeably lower test accuracy (97.37%) is a classic overfitting signature. The model essentially memorized the training set almost perfectly, including its noise, rather than learning only the patterns that generalize to new data.

**Why decision trees are high-variance models:** at each split, a decision tree makes a greedy decision — it picks whichever split best separates classes for the data currently in that node, and never revisits earlier splits once made. With no depth limit, it keeps splitting until it creates extremely narrow, specific decision regions that can fit even noise and outliers in the training data. This makes the tree's structure highly sensitive to the exact training sample it saw — a different training sample would likely produce a meaningfully different tree. That sensitivity to the specific sample is what "high variance" means.

---

## Task 2 — Controlled Decision Tree

Trained a second tree with `max_depth=5` and `min_samples_split=20`.

- **Train accuracy:** 0.97195
- **Test accuracy:** 0.97100
- **Gap:** ~0.10 percentage points

**Gap comparison:**

| Model | Train Acc | Test Acc | Gap |
|---|---|---|---|
| Unconstrained | 0.99993 | 0.97370 | 0.02623 |
| Controlled | 0.97195 | 0.97100 | 0.00095 |

The gap shrank dramatically — from ~2.6 points down to ~0.1 points — a clean demonstration of the bias-variance tradeoff.

**Role of max_depth:** limits how many levels deep the tree can grow, capping how narrow and specific its decision regions can become. This reduces variance (less overfitting) at the cost of some bias (the model can no longer capture very fine-grained patterns, even genuine ones).

**Role of min_samples_split:** prevents a node from being split further unless it contains at least 20 samples, stopping the tree from creating rules based on tiny, potentially noise-driven subsets of data.

Notably, the controlled tree's test accuracy (0.97100) is barely lower than the unconstrained tree's (0.97370) — almost nothing was lost in real-world performance, while the model became far more resistant to overfitting.

---

## Task 3 — Gini vs Entropy Comparison

Trained two trees at `max_depth=5`, one with each criterion.

- **Gini test accuracy:** 0.97100
- **Entropy test accuracy:** 0.96022

A meaningful ~1 percentage point gap in favor of Gini on this dataset — a genuine finding, since the two criteria are often assumed to be near-interchangeable.

**Gini impurity formula:**
Gini = 1 − Σ(pᵢ²), where pᵢ is the proportion of samples in a node belonging to class i.

**Entropy formula:**
Entropy = −Σ(pᵢ · log₂(pᵢ))

**What does Gini = 0 mean?** If a node contains only one class (pᵢ = 1 for that class, 0 for all others), then Gini = 1 − (1²) = 0. A Gini score of 0 means the node is perfectly pure — every sample belongs to the same class, with no remaining uncertainty at that point in the tree.

While Gini and Entropy usually produce similar trees in practice (both measure node impurity, just via different formulas), this result shows the choice can meaningfully affect outcomes on a specific dataset and depth setting, making it worth testing both empirically.

---

## Task 4 — Random Forest

Trained a `RandomForestClassifier` with `n_estimators=100, max_depth=10`.

- **Train accuracy:** 0.9894
- **Test accuracy:** 0.9796
- **ROC-AUC:** 0.9985

Better generalization than either single tree (only a ~1-point train/test gap), and a higher AUC than Part 2's Logistic Regression (0.9974).

**Top 5 features by importance:**

| Feature | Importance |
|---|---|
| y | 0.3846 |
| x | 0.2115 |
| carat | 0.1990 |
| z | 0.1209 |
| clarity | 0.0379 |

**How Random Forest computes feature importance, and why it differs from a linear regression coefficient:** feature importance is calculated by measuring how much each feature reduces impurity (Gini, by default) on average, every time it's used to split a node, averaged across all 100 trees. This differs fundamentally from a linear regression coefficient, which represents a fixed, linear, additive relationship between one feature and the target, and whose sign (positive/negative) carries direct meaning. Feature importance scores are always non-negative and reflect how *useful* a feature was for making split decisions across many trees — they say nothing about the direction of the relationship, unlike a coefficient. This also explains why Random Forest split the "size" signal fairly evenly across y, x, and carat, without the confusing sign-flip that Linear Regression showed on the `x` coefficient in Part 2 (a multicollinearity artifact that doesn't affect tree-based splitting logic the same way).

**Bagging concept:** Random Forest relies on bootstrap aggregating (bagging). Each of its 100 trees is trained on a bootstrap sample — a random sample of the training data drawn *with replacement*, the same size as the original training set, meaning some rows appear multiple times in a given tree's data while others are left out entirely. Additionally, at each split, only a random subset of features (roughly √9 ≈ 3 out of 9 features) is considered as candidates for the best split, rather than all 9. This double randomization means each of the 100 trees ends up structurally different, even though they're all trained on the same overall dataset. When their predictions are combined by majority vote, each tree's individual, idiosyncratic errors tend to cancel out, while genuine shared signal reinforces across trees — this averaging effect is what reduces variance compared to a single deep decision tree, which has no such mechanism to smooth out its overfitting tendencies.

---

## Task 4a — Gradient Boosting

Trained a `GradientBoostingClassifier` with `n_estimators=100, learning_rate=0.1, max_depth=3`.

- **Train accuracy:** 0.9807
- **Test accuracy:** 0.9785
- **ROC-AUC:** 0.9984

Nearly tied with Random Forest across all three metrics.

**Random Forest vs Gradient Boosting — how they differ:** Random Forest builds all 100 trees independently and in parallel — each tree trains separately on its own bootstrap sample, with averaging only happening at the end. Gradient Boosting instead builds trees sequentially, one at a time, where each new tree is trained specifically to correct the errors (residuals) left by the trees built before it. `learning_rate=0.1` controls how strongly each new tree's correction is weighted into the overall prediction. `max_depth=3` here is intentionally shallow, since Gradient Boosting typically relies on many small, "weak" trees plus the sequential correction process, rather than deep individual trees like Random Forest uses.

---

## Task 4b — Feature Ablation Study

Identified the 5 lowest-importance features from Task 4's Random Forest — **clarity, color, depth, cut, table** — and trained a second Random Forest (identical hyperparameters, same random_state) with those 5 features removed.

| Model | AUC |
|---|---|
| Full model (all 9 features) | 0.99847 |
| Reduced model (5 lowest-importance features removed) | 0.99179 |

A drop of about 0.67 percentage points — small in absolute terms, but consistent and real.

**Were the removed features genuinely uninformative, or contributing real signal?** Since AUC dropped meaningfully when they were removed, these 5 features — even though individually ranked lowest in importance — were still contributing real, if modest, predictive signal, not just noise. If they were truly uninformative, the reduced model's AUC would likely have stayed the same or even improved slightly.

**Production trade-off discussion:** dropping these 5 features would produce a leaner model — fewer inputs to collect and validate at prediction time, faster inference, and simpler ongoing maintenance (fewer columns to monitor for drift or missing values). Whether this trade-off is worthwhile depends on the use case's tolerance for a ~0.67 point AUC drop. For a low-stakes, high-volume application (e.g., a quick browsing-stage price estimate), this small cost might be entirely acceptable in exchange for a simpler pipeline. For a high-stakes application (e.g., final pricing/valuation decisions with real financial consequences), the drop may not be worth it — especially since two of the removed features (clarity, color) are meaningful, buyer-relevant quality indicators, so removing them also reduces the model's real-world interpretability, not just its raw score.

---

## Task 5 — Cross-Validated Comparison

Evaluated Logistic Regression (Part 2), the controlled Decision Tree, Random Forest, and Gradient Boosting using 5-fold stratified cross-validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, `scoring='roc_auc'`).

| Model | Mean AUC | Std AUC |
|---|---|---|
| Logistic Regression | 0.99754 | 0.000149 |
| Decision Tree (controlled) | 0.99555 | 0.000662 |
| Random Forest | 0.99850 | 0.000233 |
| Gradient Boosting | 0.99847 | 0.000177 |

**Why cross-validation is more reliable than a single train-test split:** a single 80/20 split gives performance on only one particular random partition of the data — if that partition happened to be slightly easier or harder than average, the reported metric could be misleadingly high or low. Cross-validation evaluates the model on 5 different train/test partitions and averages the results, giving a far more stable and representative estimate of likely performance on genuinely new data. The standard deviation across folds is also informative: a small standard deviation (like Logistic Regression's 0.000149) indicates a model whose performance doesn't swing much depending on which specific rows end up in training vs. testing — a desirable trait for production reliability, even if its raw mean AUC is a hair lower than the ensemble methods.

**Ranking:** Random Forest and Gradient Boosting are essentially tied for the best mean AUC (~0.9985), meaningfully ahead of the single Decision Tree (0.9956), with Logistic Regression close behind the ensembles (0.9975) but the most *consistent* of all four models by a wide margin.

---

## Task 6 — Hyperparameter Tuning with GridSearchCV

Built a pipeline (`SimpleImputer(strategy='median')` → `StandardScaler()` → `RandomForestClassifier(random_state=42)`) and searched the following grid:

```python
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
```

**Total configurations evaluated:** 3 × 3 × 2 = **18 combinations**, each evaluated across 5 cross-validation folds, for **90 total model fits**.

**Best parameters found:**
- `n_estimators`: 200
- `max_depth`: None
- `min_samples_leaf`: 5

**Best CV AUC:** 0.99855

It may seem surprising that `max_depth=None` (fully unconstrained depth) won, given how badly an unconstrained *single* decision tree overfit in Task 1. This makes sense in context, though: Random Forest's bagging mechanism (bootstrap sampling plus random feature subsets per split) already provides strong protection against overfitting on its own, even with deep individual trees, because averaging across 200 different trees smooths out any single tree's tendency to memorize noise. The `min_samples_leaf=5` constraint adds a small amount of extra regularization on top, but the ensemble structure itself does most of the overfitting-prevention work — unlike a lone decision tree, which has no such safety net.

**Grid Search vs Randomized Search trade-off:** Grid Search exhaustively tries every combination in the defined grid, guaranteeing it will find the best combination *within that grid* — but the number of fits grows multiplicatively as more parameters or values are added, quickly becoming computationally expensive with larger grids. Randomized Search instead samples a fixed number of random combinations from the parameter space, which is far cheaper and scales much better to large search spaces, at the cost of no longer guaranteeing the absolute best combination is found. In practice, Randomized Search often finds a combination that's nearly as good, at a fraction of the computational cost — a trade-off that becomes especially valuable once tuning many hyperparameters simultaneously.

---

## Task 6 (continued) — Manual Learning Curve

Trained the best pipeline (`grid_search.best_estimator_`) on progressively larger subsets of the training data (20%, 40%, 60%, 80%, 100%), computing training AUC and test AUC at each step.

| Training fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 0.99951 | 0.99802 |
| 0.4 | 0.99958 | 0.99834 |
| 0.6 | 0.99958 | 0.99843 |
| 0.8 | 0.99959 | 0.99850 |
| 1.0 | 0.99960 | 0.99853 |

**(i) Does training AUC decrease as the training set grows?** Not meaningfully — it stays essentially flat, inching up very slightly (0.99951 → 0.99960) rather than decreasing, which differs from the "classic" high-variance pattern where training AUC typically drops as more data is added. This suggests the tuned Random Forest (200 trees, unconstrained depth, min_samples_leaf=5) is powerful enough to fit any size of this training data extremely well without much strain.

**(ii) Does test AUC increase with more training data?** Yes, clearly — test AUC climbs steadily and consistently at every step (0.99802 → 0.99834 → 0.99843 → 0.99850 → 0.99853), a genuine, real trend even as the improvements get progressively smaller.

**(iii) Data-limited or capacity-limited?** Looking at the *rate* of improvement: the jump from 20%→40% training data is +0.00032, but the jump from 80%→100% is only +0.00003 — roughly 10x smaller. The curve is still technically rising at 100% of the data, but it is clearly flattening out, showing diminishing returns from each additional chunk of data. This suggests the model is transitioning from data-limited toward capacity-limited — more data would likely still help a little, since test AUC hasn't fully flatlined, but the gains from here would probably be small. Meaningfully improving further would likely require better or additional features, or a more powerful modeling approach, rather than simply collecting more rows of the same kind of data.

---

## Task 7 — Model Serialization

The best pipeline (Random Forest, n_estimators=200, max_depth=None, min_samples_leaf=5) was serialized using `joblib.dump(best_pipeline, 'best_model.pkl')`.

The model was then reloaded using `joblib.load('best_model.pkl')` and tested on two hand-crafted diamonds:
- **Diamond 1** (large, high quality: 1.5 carat, Ideal cut, D color, VVS2 clarity) → predicted **expensive (1)**, probability 1.0
- **Diamond 2** (small, low quality: 0.3 carat, Good cut, low color/clarity grades) → predicted **not expensive (0)**, probability 0.0

Both predictions matched real-world intuition about diamond pricing, confirming the saved model reloads and predicts correctly without errors, and is ready for deployment.

---

## Final Summary Comparison Table (Parts 2 and 3 combined)

| Model | 5-fold CV Mean AUC | 5-fold CV Std AUC | Test-set AUC |
|---|---|---|---|
| Logistic Regression (Part 2) | 0.99754 | 0.000149 | 0.99740 |
| Decision Tree (controlled) | 0.99555 | 0.000662 | — |
| Random Forest (default) | 0.99850 | 0.000233 | 0.99847 |
| Gradient Boosting | 0.99847 | 0.000177 | 0.99842 |
| **Random Forest (GridSearchCV tuned)** | **0.99855** | — | **0.99853** |

*(Note: the controlled Decision Tree's single-split test AUC was not separately computed in this analysis, hence "—"; GridSearchCV's best_score_ is already a cross-validated mean rather than a per-fold statistic, hence no std reported for that row.)*

## Final Recommendation

I recommend the **GridSearchCV-tuned Random Forest** (n_estimators=200, max_depth=None, min_samples_leaf=5) as the final model. It achieved the highest 5-fold cross-validated AUC (0.99855) and the highest test-set AUC (0.99853) of all five models evaluated, edging out both the default Random Forest and Gradient Boosting by a small but consistent margin, and clearly outperforming the single Decision Tree and Logistic Regression. While Logistic Regression showed the lowest variance across folds (std = 0.000149) and remains an attractive, highly interpretable option, the tuned Random Forest's accuracy advantage — combined with the learning curve analysis showing the model has nearly reached its performance ceiling on this data — makes it the strongest choice for deployment. It is also already serialized as `best_model.pkl` and verified to reload and predict correctly, making it immediately production-ready.

## Summary of Key Findings

- Unconstrained decision trees badly overfit (train 99.99% vs test 97.37%); constraining depth and min_samples_split nearly eliminates this gap while sacrificing almost no test accuracy.
- Gini slightly outperformed Entropy as a splitting criterion on this dataset (97.10% vs 96.02% test accuracy).
- Ensemble methods (Random Forest, Gradient Boosting) substantially outperform single trees and slightly outperform Logistic Regression on AUC, while remaining reasonably consistent across cross-validation folds.
- Feature ablation showed that even "low importance" features (clarity, color, depth, cut, table) contribute real signal — removing them cost ~0.67 AUC points, a genuine trade-off to weigh against production simplicity.
- Hyperparameter tuning via GridSearchCV (90 total fits) found a configuration that modestly outperforms the default Random Forest.
- The learning curve suggests the tuned model is approaching a performance ceiling — more data would likely yield only small further gains.
- The final tuned Random Forest was serialized, reloaded, and verified to produce sensible predictions on new data, and is recommended as the model to deploy.
