# Hotel Booking Cancellation Analysis: A Statistical Approach
This project focuses on identifying patterns in hotel booking cancellations using Python. It covers the full data science lifecycle: from rigorous cleaning and outlier detection to survival analysis of stay lengths and functional regression modeling of lead times.

---

## 📌 Project Overview

The core of this research is to understand why and when guests cancel.

- **Key Question:** How does the "Lead Time" (days between booking and arrival) quantitatively affect the probability of cancellation?
- **Methodology:** Implemented a robust data-cleaning pipeline, exploratory data analysis (EDA) with statistical filtering, and compared multiple regression models (Linear, Log, Power, Quadratic) to find the best fit for cancellation trends.

---

## 📂 Project Structure
```
hotel-booking-cancellation-analysis/
├── README.md
├── requirements.txt
│
├── data/                  # Raw and cleaned Parquet files
├── src/                   # Modular Python scripts (regression.py, plot.py, etc.)
├── notebooks/             # Step-by-step analysis (Preprocessing, Lead Time, etc.)
│   ├── preprocess_feature.ipynb
│   ├── total_guests.ipynb
│   ├── total_nights.ipynb
│   ├── lead_time.ipynb
│   └── regression.ipynb
│
└── figures/               # Generated visualizations for README
```
---

## 🛠 Data Pre-processing & Feature Engineering (`preprocess_feature.ipynb`)

The dataset consists of **119,390 rows and 17 columns**. Below are the first few rows for reference:

|  | hotel | is_canceled | lead_time | arrival_date_year | arrival_date_month | arrival_date_week_number | arrival_date_day_of_month | stays_in_weekend_nights | stays_in_week_nights | adults | children | babies | is_repeated_guest | previous_cancellations | previous_bookings_not_canceled | reserved_room_type | assigned_room_type |
|---:|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 0 | Resort Hotel | 0 | 342 | 2015 | July | 27 | 1 | 0 | 0 | 2 | 0.0 | 0 | 0 | 0 | 0 | C | C |
| 1 | Resort Hotel | 0 | 737 | 2015 | July | 27 | 1 | 0 | 0 | 2 | 0.0 | 0 | 0 | 0 | 0 | C | C |
| 2 | Resort Hotel | 0 | 7 | 2015 | July | 27 | 1 | 0 | 1 | 1 | 0.0 | 0 | 0 | 0 | 0 | A | C |

> **Note:** The `children` column was originally stored as **float**. After filling 4 missing values (0.0034%), it was converted to **integer** for consistency.

---

### (1) Missing Values Analysis
- Checked for missing values and filled gaps where necessary:  
  - `children`: 4 missing values (0.0034%) → filled with 0.  
- Corrected data types:  
  - Converted `children` from float to integer.  
- No other columns had significant missing values.

**Missing values summary:**  
- `children`: 4 missing values (0.0034%)

---

### (2) Corrupted / Invalid Values Analysis
- Examined column headers to understand relationships and data quality.  
- Created new features:  
  - `total_guests` = `adults + children + babies`  
  - `total_nights` = `stays_in_weekend_nights + stays_in_week_nights`  
- Removed invalid rows:  
  - Bookings with zero adults → **403 rows removed**  
  - Bookings with zero total nights → **645 rows removed**  
- Dropped duplicate rows → **39,814 rows removed**  

**Dataset summary:**  
- Original rows: 119,390  
- Cleaned rows: 78,528  
- Total rows removed: 40,862

---

## 📊 Exploratory Data Analysis

---

### 1. Total Guests (`total_guests.ipynb`)
![total_guests](figures/guests_comparison.png)
> **Note:** The three plots (A, B, C) correspond to these steps.  

**Step A – Initial Inspection**  
- Adults: initially unrestricted.  
- Children: initially unrestricted.  
- Observed bookings with **adults ≥ 5** → unrealistic.  
- **Action:** Removed 16 rows with adults > 4 (0.02% of data).

**Step B – Further Screening**  
- Identified bookings with total guests = 12 and children up to 10.  
- **Action:** Removed 17 rows with adults > 4 and children > 3 (0.02% of data).

**Step C – Final Check**  
- Final distribution:  
  - Adults: 1–5  
  - Children: 0–3  
  - Babies: 0–2  

**Largest non-canceled booking (kept for inspection):**  

```
Total Guests: 12
Breakdown: Adults 2, Children 0, Babies 10
Arrival Date: 2016 - January
Assigned Room Type: D
```

**Performance Comparison:**  
- Compared Pandas vs DuckDB for locating the maximum guest booking.

    - **Pandas:** Using `idxmax()` on the filtered `total_guests` column.
    - **DuckDB:** SQL query on in-memory DataFrame.
    - Each method was run 100 times to compute average execution time.
- Pandas is several times faster for 78,528 rows.

---

### 2. Total Nights (`total_nights.ipynb`)
![total_nights](figures/nights_comparison.png)

#### (A) Distribution & Survival Analysis
- Histogram (linear scale): Most bookings are 1–7 nights.  
- Histogram (log scale): Long-tail behavior for longer stays visible.  

**Survival Function (SF):**
$$
SF(N) = P(X \geq N)
$$
- Monotonic decrease: longer stays less probable.  
- Steep drop for small N; curve flattens for longer stays.  

**Key Observations:**  
- Right-skewed distribution.  
- 99% of guests stay ≤14 nights.  
- Power-law fit: R² = 0.987.

#### (B) Cancellation Rate by Stay Type
- Visualized cancellation probability by weeknights, weekend nights, total stay (≤30 nights).  
- **Insights:**  
  - Longer stays show slightly different cancellation patterns.  
  - Point sizes indicate booking counts; no formal regression applied.

---

## 📈 Lead Time Regression Modeling
- **Goal:** Map Lead Time → Cancellation Probability.  
- **Filter criteria:**  
  - Adults ≤ 4, Children ≤ 3  
  - Total nights ≤ 11 (covers 98% of data)

```python
df_filtered_0 = df_clean[
    (df_clean['adults'] <= 4) & 
    (df_clean['children'] <= 3) & 
    (df_clean['total_nights'] <= 11)
].copy()
```
---

### 1. Lead Time (Days) (`lead_time.ipynb`)

#### (A) Bubble Plot Visualization
![lead_time](figures/leadtime_cancel.png)

- Scatter plot: lead time vs. cancellation rate.
- Bubble size/color ~ booking count (larger → more stable).
- After **350 days**, cancellations often 0 or 1 → long-tail noise.

#### (B) 90% Cumulative Booking Method
![lead_time](figures/leadtime_decay_90p.png)

- Calculated the cumulative sum of bookings by lead time to identify the day covering **90% of total bookings (Day 199)**.
- Ensures statistical reliability: minimum sample size at any day ≤ 199 = 93 bookings.
- Left panel: counts and cancellations vs. lead time, with the 90% cut-off highlighted.
- Right panel: log-scale visualization, showing long-tail behavior beyond the cut-off more clearly.

> Note: For subsequent regression analysis, we only used lead times ≤ 199 days to avoid noise from sparse bookings in the long tail. This ensures that model fitting is based on reliable, well-populated data.

---

### 2. Lead Time Regression: Training & Validation (`regression.ipynb`)
- **Goal**: Map Lead Time → Cancellation Probability.
- **Data Filtering**: Adults ≤ 4, Children ≤ 3, Total nights ≤ 11 (covers 98% of bookings).
- **Cut-off**: Used Day 199 (90% cumulative bookings) to exclude long-tail noise; minimum bookings per day ≤199 = 93.
#### Models Tested (70% Train / 30% Validation):
- Linear, Logarithmic, Power, Quadratic.
- **Training & Validation:**
    - R² and MSE calculated for trusted region (≤199 days) and full range.
    - Scatter plots with color; regression fits overlaid.
    - **Sliding / Rolling Regression:** Calculated moving R² across different lead time horizons to visualize model stability (Training: left panel = scatter + fit, right panel = sliding R²).
- **Results:**

| Model       | R² (Train) | MSE (Train) | R² (Validation) | MSE (Validation) | Best Fit? |
|------------|------------|-------------|-----------------|-----------------|-----------|
| Linear     | 0.354      | 0.0032      | 0.296           | 0.0061          | No        |
| Logarithmic| 0.588      | 0.0020      | 0.423           | 0.0049          | ✅ Yes    |
| Power      | 0.546      | 0.0022      | 0.400           | 0.0052          | No        |
| Quadratic  | 0.480      | 0.0025      | 0.341           | 0.0057          | No        |

#### Visualization:
- Training fit + sliding R²: 
![lead_time](figures/leadtime_regression_fit.png)
- Validation fit: 
![lead_time](figures/leadtime_validation_fit.png)

- **Insight:**

- Logarithmic model captures the rapid increase in cancellation probability in early lead times and stabilizes afterward.
- Using the 199-day cut-off ensures the regression reflects reliable, well-populated data, avoiding long-tail outliers.
- Aggregation + scatter point sizing, combined with sliding R², improves robustness and generalization across the dataset.

> Note: The sliding R² plot provides a clear view of how model performance evolves as more lead time data are included, ensuring the selected model remains stable across different horizons.

---
