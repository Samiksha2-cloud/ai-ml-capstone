# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## Dataset Choice and Justification

I chose the **Diamonds dataset** (loaded via `seaborn.load_dataset('diamonds')`) for this project.

The dataset has **53,940 rows and 10 columns**, comfortably clearing the assignment's minimum requirement of 500 rows. It contains:

- **7 numeric columns:** carat, depth, table, price, x, y, z
- **3 categorical columns:** cut, color, clarity
- **Target variable: price** — a continuous numeric column, making it directly usable for a regression task. It can also be binarized later (e.g., `price > median` → "expensive" vs "not expensive") for a classification task in Part 2.

I picked this dataset because it's large enough to make correlations and visualizations statistically meaningful, it has a mix of genuinely ordinal categorical columns (cut, color, and clarity all have a real quality ordering), and — as this analysis will show — it has enough real-world messiness (skewed distributions, physically impossible outlier values, and a counterintuitive cut-vs-price relationship) to make for a proper, honest EDA rather than a dataset that's "too clean" to say anything interesting about.

---

## Task 1 — Load Dataset

Loaded the dataset into a pandas DataFrame using `sns.load_dataset('diamonds')`.

- Shape: **(53940, 10)** — 53,940 rows, 10 columns
- `df.head()` and `df.dtypes` confirmed the data loaded correctly, with cut/color/clarity already inferred as `category` dtype and all other columns as `float64`/`int64`.

---

## Task 2 — Null Value Analysis

Ran `df.isnull().sum()` and the percentage version across all columns.

**Result: no null values found anywhere in the dataset.** Every column showed 0 nulls and 0.0% missing values. Since there was nothing to fill, no median imputation was needed at this stage — this is reported as a genuine finding rather than skipped.

---

## Task 3 — Duplicate Detection and Removal

Used `df.duplicated().sum()` to check for exact duplicate rows, then removed them with `df.drop_duplicates()`.

- **146 duplicate rows found and removed** (53,940 → 53,794 rows)
- **Does removing duplicates change any column's null percentage?** No. After removing duplicates, null percentages remained 0% across all columns. This makes sense mathematically — null percentage is (nulls ÷ total rows) × 100, and since there were zero null values to begin with, removing rows cannot "remove" nulls that never existed. Confirmed this by re-running the null-percentage check post-deduplication and getting 0.0% everywhere, same as before.

---

## Task 4 — Data Type Correction

Originally, **no data type errors were found** in the dataset. All numeric columns were correctly typed as `float64`/`int64`, and cut, color, and clarity were already correctly typed as `category` on load — not `object` as one might expect from raw data.

To satisfy the task requirement and demonstrate the skill anyway, I intentionally converted the `cut` column from `category` back to `object`, measured memory usage, then converted it back to `category` and measured again.

**Memory usage comparison:**
- `cut` as `object` dtype: **2,974,159 bytes**
- `cut` as `category` dtype: **54,240 bytes**
- That's roughly a **55x reduction** in memory.

This happens because as `object` dtype, pandas stores the actual string ("Ideal", "Premium", etc.) separately for every single one of the 53,794 rows, even though there are only 5 unique cut values. As `category` dtype, pandas stores each unique string once and references it per row using a small integer code — much more efficient for columns with a small number of repeating values.

**Honest note:** since no genuine dtype error existed in this dataset, this conversion was done manually purely to demonstrate the technique and its memory impact, as required by the task.

---

## Task 5 — Descriptive Statistics and Skewness

Ran `df.describe()` on all numeric columns and computed `.skew()` for each.

**Skewness values (highest to lowest absolute):**
1. y — 2.446 (highest)
2. price — 1.618
3. z — 1.529
4. carat — 1.114
5. table — 0.792
6. x — 0.380
7. depth — 0.114 (near-symmetric)

**Column with highest absolute skewness: `y` (skew = 2.45)**

This is a positive/right skew, which means the bulk of the diamonds sit on the lower, normal side of the width range, while a small number of extreme high values stretch the tail out to the right, pulling the distribution's shape off-balance rather than symmetric.

Also worth noting: the minimum value for `y` (and `x`, `z`) showed as 0, which is very likely a data entry error rather than a real measurement, since a diamond cannot physically have zero width.

**Why mean-based imputation is risky for a skewed column:** the mean is calculated by summing every value and dividing by the count, so every single value — including extreme outliers — pulls on it. The median, by contrast, just finds the middle position once everything is sorted, so it doesn't care how extreme the extreme values are, only where they rank. In this dataset, `y`'s mean (5.73) and median (5.71) came out quite close to each other, but that's partly a scale effect — with 53,794 rows, the influence of one or two outliers gets diluted. In a smaller dataset, or a column with more extreme values, that gap between mean and median would be far more pronounced.

---

## Task 6 — Outlier Detection with IQR

Applied the IQR method (Q1, Q3, IQR, lower/upper bound) to two numeric columns: `y` and `carat`.

**`y` column:**
- Q1 = 4.72, Q3 = 6.54, IQR = 1.82
- Bounds: [1.99, 9.27]
- **28 outliers found**

Inspecting these 28 rows individually revealed three distinct groups:
- **6 rows with y = 0** — physically impossible (a diamond cannot have zero width), genuine data entry errors
- **2 rows with extreme values (y = 31.80 and y = 58.90)** — inconsistent with their carat size (a 0.51 carat diamond with y = 31.80mm, and a 2.00 carat diamond with y = 58.90mm, don't make physical sense compared to similarly-sized diamonds elsewhere in the dataset) — most likely decimal-point typos
- **20 rows with y between 9.31 and 10.54** — these all correspond to large-carat diamonds (3.0 to 5.01 carats), where a larger width is entirely expected and physically consistent with the diamond's weight. These are legitimate data points, not errors.

**Decision:** the 8 clear error rows (the 6 zero values and the 2 physically impossible extremes) will be corrected or removed in Part 2, since they would corrupt any model trained on this data. The remaining 20 large-diamond outliers will be retained, since they represent real, valid price-carat-size relationships that the model needs to learn from.

**`carat` column:**
- Q1 = 0.40, Q3 = 1.04, IQR = 0.64
- Bounds: [-0.56, 2.00]
- **1,873 outliers found**

Unlike `y`, these outliers are not data errors. A diamond with a large carat weight (2, 3, 4, 5 carats) is completely real — such diamonds exist, they're simply rare and expensive compared to the bulk of the dataset. Statistical "outlier" here does not mean "wrong data," it just means these values sit far from the typical middle 50% of the distribution.

**Decision:** these 1,873 outliers will be **retained**, not capped or dropped, since removing them would throw away real, valid, useful signal — Part 2's model needs to learn from the full price range, including the large/expensive diamonds.

---

## Task 7 — Visualizations

### 1. Line Plot — Diamond Prices Sorted (Low to High)
Plotted `price` sorted ascending against row index. The resulting curve stays low and relatively flat for most of the range, then rises steeply toward the end. This shape is a direct visual confirmation of price's positive skew — most diamonds are relatively inexpensive, with a smaller number of very expensive diamonds pulling the curve sharply upward at the tail end.

### 2. Bar Chart — Average Price by Cut Quality
Grouped `price` by `cut` and plotted the mean of each group. Counterintuitively, **"Fair" cut (the worst quality) shows a higher average price than "Ideal" cut (the best quality)**. This goes against the naive assumption that better cut = higher price, and is explored further in the heat map section below.

### 3. Histogram — Distribution of Diamond Width (y)
The raw histogram of `y` was heavily distorted by the extreme outlier values identified in Task 6 (y = 58.9, y = 31.8, and the six y = 0 rows). These stretched the x-axis all the way out to 60mm, compressing all 53,794 legitimate data points into just two visible bars near the left edge — making the true shape of the distribution completely invisible.

After filtering out these known data-entry errors (keeping only 0 < y < 15), the histogram revealed a much more informative shape: two rough peaks around 4.5mm and 6.5mm, with values gradually thinning out toward 9-10mm. A mild right-skew remains even after removing the extreme errors, which makes sense — diamonds tend to cluster around popular size/carat categories rather than following a single smooth bell curve.

### 4. Scatter Plot — Diamond Price vs Carat
Plotted `carat` against `price`. The relationship is clearly **positive** — as carat increases, price generally increases too. However, it is **not a straight line**; the relationship curves, with price increasing more steeply at higher carat values. There's also visible "banding" — vertical clusters of points at common carat sizes like 1.0, 1.5, and 2.0 — likely because diamonds are commonly cut to standard weight thresholds. There's also noticeable vertical spread at any given carat value, meaning carat alone doesn't fully determine price.

### 5. Box Plot — Price Distribution by Cut Quality
Split `price` by `cut` category. The median price for "Ideal" cut is the **lowest** among all five categories, despite Ideal being the best cut quality — consistent with the odd pattern seen in the bar chart. "Premium" showed the widest spread (largest box/IQR), while "Ideal" had a comparatively narrower, lower-sitting box. Every category showed a large number of outlier points stretching up toward 18,000+, meaning expensive diamonds exist within every cut category, not just the "better" ones.

**Interpretation of the cut-vs-price anomaly:** this is best explained as a **confounding variable** situation. Buyers who want a large, heavy diamond (high carat) often accept a lower cut quality, because achieving both "large" and "perfectly cut" simultaneously wastes more of the rough diamond and costs more to produce. So small, well-cut "Ideal" diamonds may be common and relatively cheap, while large, poorly-cut "Fair" diamonds are rarer but expensive purely due to their size. In other words, **carat is likely the true driver of price, and it confounds the apparent cut-price relationship** — looking at cut in isolation gives a misleading picture.

---

## Task 8 — Correlation Heat Map

Computed the Pearson correlation matrix across all numeric columns and visualized it with `sns.heatmap()`.

**Highest correlation pair: carat and x (length), r = 0.98.** Close behind were carat-y (0.95), carat-z (0.95), and the pairwise correlations among x, y, and z themselves (0.95-0.97).

**Does this suggest a causal relationship?** Not really — the more accurate explanation is that `carat` and the physical dimensions (`x`, `y`, `z`) are essentially **two different measurements of the same underlying property: the diamond's size.** Carat is a weight measurement, and weight is mathematically derived from a diamond's volume (which comes from its dimensions) multiplied by density. So this isn't a "discovery" of one variable influencing another — it's closer to a mechanical/mathematical relationship between two ways of describing the same physical thing.

**Alternative explanation worth naming:** rather than one variable causing the other, both `carat` and `x/y/z` are jointly determined by a diamond's actual physical size, which is the real underlying factor connecting them.

**A useful side note:** `price` correlates far more strongly with `carat` (0.92) than with `depth` (-0.01) or `table` (0.13). This supports the earlier confounding-variable theory from the bar chart/box plot analysis — cut has a weak direct relationship with price, while size (carat) is the dominant driver.

---

## Task 9a — Imputation Strategy Comparison (Mean vs Median)

Selected the two columns with the highest absolute skewness from Task 5: **y** and **price**.

| Column | Mean | Median |
|---|---|---|
| y | 5.7347 | 5.71 |
| price | 3933.07 | 2401.00 |

- **For `y`:** mean (5.73) and median (5.71) are quite close, because the outlier's effect gets diluted across 53,794 rows. Even so, median is the safer choice for imputation since it's completely unaffected by the extreme error values (0, 31.8, 58.9) identified earlier.
- **For `price`:** the gap is much bigger. Since price is heavily right-skewed (skew = 1.62), a small number of very expensive diamonds pull the mean noticeably above what a "typical" diamond actually costs. Filling a missing price with the mean (3933) would overestimate what a typical diamond costs — the median (2401) better reflects where most diamonds in the dataset actually sit.

**Chosen strategy:** median, for both columns, based on this skew reasoning.

Applied `fillna(df[col].median())` to both columns and confirmed with `isnull().sum()` that no nulls remain — both columns returned 0, as expected (there were no nulls in this dataset to begin with, so this step is a demonstration of the technique rather than a correction of real missing data).

---

## Task 9b — Spearman Rank Correlation

Computed the Spearman correlation matrix (`df.corr(method='spearman')`) and compared it against the Pearson matrix from Task 8 by calculating `|Spearman - Pearson|` for every column pair.

**Three column pairs with the largest absolute difference — all involving `price` and the physical dimensions:**

| Pair | Pearson | Spearman | \|Difference\| |
|---|---|---|---|
| price – y | 0.8654 | 0.9628 | 0.0974 |
| price – z | 0.8612 | 0.9573 | 0.0961 |
| price – x | 0.8845 | 0.9633 | 0.0788 |

**Interpretation:** in all three pairs, |Spearman| > |Pearson|, which indicates the relationship between price and each physical dimension is **monotonic but non-linear** — price consistently increases as x/y/z increase, but not in a strict straight-line/proportional way. This matches exactly what was visually observed in the carat-vs-price scatter plot in Task 7, where the relationship curved upward more steeply at higher carat/size values rather than following a straight line.

**Which measure will guide Part 2 feature selection, and why:** Spearman is the more informative measure here, since most of the important relationships in this dataset (price with the size-related variables) are non-linear rather than strictly linear. Relying on Pearson alone would understate how strongly these variables are actually related. That said, if Part 2/3 uses tree-based ensemble models (e.g., Random Forest, Gradient Boosting), those models naturally handle non-linear relationships regardless of which correlation measure is used for initial feature screening — so Spearman is preferred for the *feature selection/EDA stage* specifically, since it gives a more honest picture of relationship strength before modeling begins.

---

## Task 9c — Grouped Aggregation

Grouped `price` by `cut` and computed mean, standard deviation, and count for each group using `df.groupby('cut')['price'].agg(['mean', 'std', 'count'])`.

Grouped stats table:

| cut | mean | std | count |
|---|---|---|---|
| Fair | 4341.95 | 3540.12 | 1,598 |
| Good | 3919.12 | 3671.07 | 4,891 |
| Ideal | 3462.75 | 3810.93 | 21,488 |
| Premium | 4583.50 | 4348.05 | 13,748 |
| Very Good | 3981.02 | 3934.81 | 12,069 |

- **Group with the highest mean price:** Premium (4583.50)
- **Group with the highest standard deviation:** Premium (4348.05) — the same category tops both. This means Premium-cut diamonds are not only the most expensive on average, but also the most inconsistently priced, spanning a very wide range.
- Worth noting: Ideal has both the lowest mean (3462.75) *and* the highest row count (21,488 diamonds) — it's the most common cut in the dataset, and also the cheapest on average. This lines up with the earlier confounding-variable theory: Ideal cuts are likely applied more often to smaller, more affordable diamonds, since achieving a perfect cut on a large stone is harder and more wasteful.

**Is high within-group standard deviation a concern for a predictive model using this feature?** Yes. A high standard deviation within a cut category means that knowing a diamond's cut alone doesn't reliably tell you its price — prices vary widely even within the same cut group. This is a sign that `cut` by itself is a weak predictor, and other features (particularly `carat`, based on the correlation analysis above) carry much more of the actual predictive signal.

**Ratio of highest group mean to lowest group mean:** 1.32

This ratio is relatively modest — the most expensive cut category on average is only about 32% pricier than the least expensive one. This is far from a large gap (which might look more like 3x-5x if cut strongly determined price), and it's consistent with everything else found in this analysis: `cut` carries only weak predictive signal for price on its own, with `carat` (and the physical size measurements) being the dominant driver instead.

---

## File Output

The cleaned dataset (post duplicate-removal) was saved as `cleaned_data.csv` using `df.to_csv('cleaned_data.csv', index=False)`. This file is included in this folder and will be used as the starting point for Part 2 and Part 3.

## Summary of Key Findings

- Diamonds dataset: 53,940 rows → 53,794 after removing 146 duplicates. No missing values anywhere.
- `y` (width) is the most skewed column (2.45) and contains genuine data errors (zero values, and two physically impossible extreme values) that will need correcting in Part 2.
- `carat` has 1,873 statistical outliers, but these represent real, valid large diamonds and should be retained for modeling.
- Cut quality has a surprisingly weak relationship with price — this is best explained by carat acting as a confounding variable.
- `carat` and the physical dimensions (x, y, z) are almost perfectly correlated (0.95-0.98) because they all measure essentially the same underlying property: diamond size.
- Price's relationship with size-related variables is monotonic but non-linear (Spearman consistently exceeds Pearson), which should inform model choice in Part 2.
