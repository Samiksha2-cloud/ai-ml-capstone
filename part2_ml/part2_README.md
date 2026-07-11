# Part 2 — Supervised Machine Learning: Regression and Classification

## Overview

This part builds two predictive models on top of the cleaned Diamonds dataset from Part 1 (`cleaned_data.csv`):
- A **regression model** predicting the continuous `price` of a diamond
- A **classification model** predicting whether a diamond is "expensive" (above median price) or not

Both models were evaluated rigorously, with attention paid to avoiding data leakage during preprocessing, and to understanding not just *whether* the models perform well, but *why*, and how reliable those results are.

---

## Task 1 — Target and Feature Definition

- **Feature matrix X:** all columns except price — carat, cut, color, clarity, depth, table, x, y, z (9 features)
- **y_reg (regression label):** `price` — the raw continuous price column
- **y_clf (classification label):** a binary label derived by binarizing price at its median: `y_clf = (price > price.median()).astype(int)`. 1 = "expensive" (above median), 0 = "not expensive" (at or below median).

**Class balance check:** `y_clf.value_counts()` returned 26,902 (class 0) vs 26,892 (class 1) — a near-perfectly balanced split, as expected since the binarization threshold was the median itself.

---

## Task 2 — Categorical Encoding

All three categorical columns — `cut`, `color`, `clarity` — were treated as **ordinal** and label-encoded, rather than one-hot encoded, because each follows a genuine, industry-standard quality grading scale rather than being an arbitrary, unordered category:

- **cut:** Fair (0) < Good (1) < Very Good (2) < Premium (3) < Ideal (4)
- **color:** J (0) < I (1) < H (2) < G (3) < F (4) < E (5) < D (6) — D is the most colorless and most valuable grade
- **clarity:** I1 (0) < SI2 (1) < SI1 (2) < VS2 (3) < VS1 (4) < VVS2 (5) < VVS1 (6) < IF (7) — IF ("internally flawless") is the best grade

Label encoding was chosen over one-hot encoding for all three columns because one-hot encoding would discard this meaningful ranking information and treat each grade as an unrelated, disconnected category. Since a genuine quality progression exists in the diamond grading system, mapping each grade to an integer that preserves that order lets a model learn a smooth, sensible trend (e.g., "higher clarity number tends to mean higher price") instead of learning nothing about the relationship between adjacent grades.

No columns in this dataset required one-hot encoding, since all three categorical variables here follow a natural ordinal scale specific to diamond grading systems, rather than being purely nominal (unordered) categories like city names or color codes would be.

---

## Task 3 — Leak-Free Train-Test Split and Scaling

Data was split using `train_test_split(X, y_reg, y_clf, test_size=0.2, random_state=42)`, producing:
- X_train: (43,035, 9)
- X_test: (10,759, 9)

An 80/20 split, with `random_state=42` ensuring the split is reproducible.

**Scaling procedure:** a `StandardScaler` was fit **only on X_train** (`scaler.fit(X_train)`), then used to transform both X_train and X_test (`scaler.transform(...)`).

**Why fitting the scaler on the full dataset would be data leakage:** if the scaler were fit on the combined train+test data, its computed mean and standard deviation would be influenced by the test set's own distribution. The model would then be trained on features that were scaled using information derived, in part, from data it's supposed to have never seen. This makes the test set no longer a true "unseen" evaluation, and any resulting performance metrics would be artificially optimistic — they wouldn't reflect how the model would truly perform on genuinely new, unseen data. Fitting the scaler on the training set alone and only *applying* (not re-fitting) it to the test set avoids this problem entirely.

---

## Task 4 — Regression: Linear Regression and Ridge

### Linear Regression

- **MSE:** 1,402,687.80
- **R²:** 0.9080

An R² of 0.908 means the model explains about 90.8% of the variance in diamond prices using the 9 available features — a strong fit for a simple linear model.

**Top 3 features by absolute coefficient value:**

| Feature | Coefficient |
|---|---|
| carat | +5116.67 |
| x | -1005.80 |
| clarity | +827.46 |

**Interpreting the coefficients:** since all features were scaled (StandardScaler), each coefficient represents the change in predicted price for a one-standard-deviation increase in that feature, holding other features constant.

- **carat (+5116.67):** the strongest driver by far — a one standard-deviation increase in carat is associated with roughly a $5,117 increase in predicted price, holding everything else constant. This matches Part 1's finding that carat had the highest correlation with price (0.92) of any feature.
- **x (−1005.80):** this negative coefficient is counter-intuitive at first, since Part 1 showed x positively correlates with price (0.88) on its own. The likely explanation is **multicollinearity**: carat, x, y, and z are all highly correlated with each other (0.95–0.98, per Part 1's heatmap), since they all essentially measure the same underlying property — the diamond's physical size. When several features carry near-duplicate information, linear regression can assign unstable or even sign-flipped coefficients to individual features while the *combined* prediction still remains accurate. This is a genuine limitation of interpreting individual coefficients in the presence of correlated features.
- **clarity (+827.46):** more straightforward — a higher clarity grade (using the ordinal encoding, where higher = better) is associated with a higher predicted price, holding size/carat constant, which matches intuition directly.

### Ridge Regression (alpha = 1.0)

- **MSE:** 1,402,697.74
- **R²:** 0.9080

**Comparison table:**

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 1,402,687.80 | 0.9080 |
| Ridge (alpha=1.0) | 1,402,697.74 | 0.9080 |

The two models perform virtually identically — the difference in MSE is negligible, and R² is the same to four decimal places.

**What alpha controls, and why Ridge changed so little here:** `alpha` controls the strength of Ridge's regularization penalty, which discourages the model from assigning very large coefficients to any single feature. A higher alpha shrinks coefficients more aggressively (useful when features are highly correlated or a model is overfitting); an alpha of 0 makes Ridge mathematically identical to plain Linear Regression. With 43,035 training rows and only 9 features, this dataset has a large amount of data relative to its feature count, so the plain Linear Regression model's coefficients were already fairly stable and not meaningfully overfitting. A mild regularization strength like alpha=1.0 therefore has very little room to change anything. The underlying multicollinearity between carat, x, y, and z (which caused x's sign flip above) is still present — a much larger alpha would likely be needed to meaningfully shrink and stabilize those specific coefficients.

---

## Task 5 — Classification: Logistic Regression

**Class balance on training set:** verified close to 50/50 (matching the overall dataset split from Task 1), so **no class imbalance correction (SMOTE or class_weight='balanced') was necessary** for this classification task.

### Confusion Matrix

```
[[5227  118]
 [ 153 5261]]
```

- True Negatives: 5227 (correctly predicted "not expensive")
- False Positives: 118 (predicted "expensive," actually wasn't)
- False Negatives: 153 (predicted "not expensive," actually was expensive)
- True Positives: 5261 (correctly predicted "expensive")

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (not expensive) | 0.97 | 0.98 | 0.97 | 5345 |
| 1 (expensive) | 0.98 | 0.97 | 0.97 | 5414 |
| **Accuracy** | | | **0.97** | 10759 |

### Precision and Recall Formulas

**Precision = TP / (TP + FP)** = 5261 / (5261 + 118) ≈ 0.978
→ Of all diamonds the model predicted as "expensive," what fraction actually were.

**Recall = TP / (TP + FN)** = 5261 / (5261 + 153) ≈ 0.972
→ Of all diamonds that were actually "expensive," what fraction the model correctly identified.

### Which metric matters more for this task?

For a diamond price-classification use case (e.g., helping a jeweler or e-commerce platform flag which diamonds fall into the "expensive" tier), a false negative (missing a genuinely expensive diamond and mislabeling it as "not expensive") is arguably more costly than a false positive (mislabeling a mid-range diamond as "expensive"). Underpricing or miscategorizing a genuinely valuable diamond can mean a real financial loss if it's sold or listed at a lower tier than it should be, whereas a false positive is a more easily corrected error (a diamond mistakenly flagged as premium just gets a second look). On this basis, **recall is slightly more important than precision** for this specific task, though both metrics are already very high (0.978 precision, 0.972 recall) and the model performs strongly on either measure.

### ROC Curve and AUC

**AUC = 0.9974**

The ROC curve rises almost vertically near the origin and hugs the top-left corner of the plot, staying far from the diagonal "random guess" reference line throughout.

**What this AUC means:** AUC represents the probability that, given one randomly chosen "expensive" diamond and one randomly chosen "not expensive" diamond, the model assigns a higher predicted probability to the actual expensive one. An AUC of 0.9974 means this holds true 99.74% of the time — near-perfect class separation. This is consistent with Part 1's finding that carat (and the strongly correlated size features) is an extremely strong driver of price, making the two price classes genuinely easy to distinguish based on the available features.

---

## Task 5b — Decision-Threshold Sensitivity

Varied the classification threshold from 0.30 to 0.70 in steps of 0.10, using the model's predicted probabilities directly rather than the default 0.5 cutoff.

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.9566 | 0.9885 | 0.9723 |
| 0.40 | 0.9699 | 0.9806 | 0.9752 |
| 0.50 | 0.9781 | 0.9717 | 0.9749 |
| 0.60 | 0.9850 | 0.9610 | 0.9729 |
| 0.70 | 0.9890 | 0.9466 | 0.9673 |

**Precision = TP / (TP + FP)**, **Recall = TP / (TP + FN)** (as defined above).

As the threshold rises from 0.30 to 0.70, precision climbs steadily (0.9566 → 0.9890) while recall falls steadily (0.9885 → 0.9466) — the classic precision-recall tradeoff, confirmed directly by these numbers.

**F1-maximizing threshold: 0.40** (F1 = 0.9752), very slightly ahead of the default 0.50 (F1 = 0.9749). All five F1 values are close together (0.967–0.975), showing the model performs robustly across a fairly wide range of thresholds.

**Given the earlier judgment that recall matters slightly more for this task** (missing a genuinely expensive diamond is costlier than a false alarm), the threshold should be **lowered** slightly — toward 0.30–0.40 rather than raised. At threshold 0.30, recall reaches 0.9885 (catching nearly every genuinely expensive diamond), at the cost of precision dropping to 0.9566 (more diamonds incorrectly flagged as expensive). This is a reasonable tradeoff given that the cost of a missed expensive diamond was judged to outweigh the cost of an occasional false alarm. That said, since the F1-optimal threshold (0.40) already sits close to the default and offers a good balance of both metrics, a threshold of 0.40 could serve as a practical middle ground rather than pushing all the way to 0.30.

---

## Task 6 — Regularization Experiment

Trained a second logistic regression with `C=0.01` (much stronger regularization than the baseline `C=1.0`) and compared performance.

| Model | Precision | Recall | AUC |
|---|---|---|---|
| Logistic Regression (C=1.0) | 0.9781 | 0.9717 | 0.9974 |
| Logistic Regression (C=0.01) | 0.9777 | 0.9712 | 0.9972 |

**What C controls:** in scikit-learn's logistic regression, `C` is the *inverse* of regularization strength — a smaller C applies a *stronger* penalty on large coefficients (pushing the model toward simplicity), while a larger C applies a weaker penalty, letting the model fit the training data more closely.

**Did reducing C help or hurt?** Reducing C from 1.0 to 0.01 (i.e., applying much stronger regularization) very slightly **worsened** performance across all three metrics — precision, recall, and AUC all dropped marginally. Given the size of this dataset (43,035 training rows for only 9 features), the baseline model likely wasn't overfitting in any meaningful way to begin with, so there was little for a strong regularization penalty to correct. Instead, the extra constraint slightly limited the model's ability to fit genuine signal already present in the data, producing the small dip in performance observed. This mirrors the earlier Ridge vs. Linear Regression finding, where mild regularization also had little effect for the same underlying reason — a large, clean dataset relative to feature count leaves little room for regularization to meaningfully help.

---

## Task 6b — Bootstrap Confidence Interval for AUC Difference

To assess whether the C=1.0 model's slightly higher AUC is a reliable difference or just noise from this particular test set, 500 bootstrap samples were drawn (with replacement) from the test set, and the AUC difference (C=1.0 minus C=0.01) was computed for each sample.

- **Mean AUC difference:** 0.000239
- **95% confidence interval:** [0.0000610, 0.000506]

**Interpretation:** the entire 95% confidence interval sits above zero — it **excludes zero**. This means that although the AUC advantage of the C=1.0 model over the C=0.01 model is numerically very small (roughly 0.024%), it was **consistent across all 500 resampled versions of the test set**. If this difference were purely due to random sampling noise, the interval would be expected to occasionally dip below zero (meaning the C=0.01 model would sometimes appear better) — but it never does across any of the 500 bootstrap iterations. So the difference, while tiny, is statistically reliable rather than coincidental.

That said, the **practical significance of this difference is minimal** given its very small magnitude — both models perform extremely similarly overall, and in practice the choice between C=1.0 and C=0.01 for this dataset would likely come down to other considerations (such as model interpretability or generalization to genuinely new data) rather than this particular AUC gap.

---

## Summary of Key Findings

- Linear Regression and Ridge Regression both achieve R² ≈ 0.908, explaining roughly 91% of price variance using 9 features. Ridge provides negligible improvement here because the large, clean dataset leaves little room for regularization to help.
- Carat is by far the strongest positive driver of price; the negative coefficient on `x` is best explained by multicollinearity among the size-related features (carat, x, y, z), not a genuine inverse relationship.
- The classification model separates "expensive" vs "not expensive" diamonds with very high accuracy (97%) and near-perfect AUC (0.9974), reflecting how strongly diamond size/carat determines price.
- No class imbalance handling was required, since the classification target was constructed via median split.
- Threshold tuning shows the model performs robustly across a range of cutoffs (F1 between 0.967–0.975), with the true F1-optimal threshold (0.40) sitting close to the conventional default (0.50).
- Stronger L2 regularization (C=0.01) slightly underperforms the baseline (C=1.0), and a 500-sample bootstrap confirms this small gap is statistically consistent rather than due to chance — though its practical impact is minimal.
