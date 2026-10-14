# Food Demand Forecasting

Forecasting weekly meal orders for a food-delivery network, and measuring how much food waste and how many stockouts a better forecast avoids.

A meal-delivery company runs fulfilment centres across many cities. Ingredients are perishable and have to be ordered before the week's orders are known. Ordering too much wastes food; ordering too little loses sales. This project forecasts orders for every (centre, meal) pair two weeks ahead, then turns those forecasts into ordering decisions and compares their cost.

Everything is in one notebook: [`food_demand_forecasting.ipynb`](food_demand_forecasting.ipynb).

## Data

[Food Demand Forecasting (Kaggle)](https://www.kaggle.com/datasets/kannanaikkal/food-demand-forecasting): about 456k weekly rows covering 77 fulfilment centres and 51 meals over 145 weeks, with price, discount and promotion columns.

The data isn't included in this repo. Download these three files from Kaggle into a `data/` folder:

```
data/
├── train.csv
├── fulfilment_center_info.csv
└── meal_info.csv
```

## What the notebook does

| Step | Section | Tooling |
|---|---|---|
| Load with explicit schemas, validate, profile | 1–2 | PySpark |
| Exploratory analysis, promotion and discount lift | 3 | PySpark, matplotlib |
| Leakage-safe feature engineering | 4 | PySpark window functions |
| Rolling-origin backtest: baselines vs a global gradient-boosted model | 5–6 | LightGBM |
| Comparison with per-series exponential smoothing | 7 | statsmodels |
| Error analysis by segment | 8 | pandas |
| Business impact: waste vs stockouts, quantile ordering policy | 9 | LightGBM quantile regression |
| Findings, limitations, recommendations | 10–11 | |

### Design choices worth knowing about

- **No leakage.** Each forecast for week *t* is made at week *t − 2* (the purchasing lead time). Demand features only use weeks up to *t − 2*. The planned price and promotions for week *t* are allowed, because the business sets them in advance. A sanity check recomputes one lag in plain pandas and confirms Spark agrees.
- **Missing weeks stay missing.** About 8% of weeks inside each series' active span have no row. They're treated as unknown, not as zero demand, and lags use range windows over the week number so a gap never shifts the next row into its place.
- **Time-ordered backtest.** Three consecutive 10-week test blocks, each trained on all history ending two weeks before the block. Early stopping uses the end of the training window, never the test block.
- **Log-target bias correction.** The model is trained on `log1p(orders)`, which under-forecasts totals after back-transforming. A smearing factor estimated on validation data corrects it.
- **Ordering at a quantile, not the mean.** If a missed sale costs more than a wasted unit, the cheapest order is a demand quantile (the newsvendor critical ratio). With the default assumption of 3:1 that's the 75th percentile, estimated directly with LightGBM's quantile objective. A sensitivity table repeats the comparison across a range of cost ratios.

### A first look at the data

Measured within each series (every meal compared against its own normal demand), the median week with both an email campaign and homepage feature sells about **3.6×** normal. An email alone sells about 2.2×, a homepage feature alone about 1.6×. A discount of more than 20% with no promotion lifts demand by about 1.3×.

## Results

Backtest over 3 rolling folds of 10 weeks, forecasting 2 weeks ahead. WAPE is total absolute error divided by total demand (lower is better).

| Model | WAPE |
|---|---|
| Naive (demand two weeks ago) | 51.7% |
| 4-week moving average | 45.8% |
| **LightGBM global model** | **28.3%** |

- LightGBM cuts error by 38% compared with the moving average.
- The bias correction matters: before it, LightGBM under-forecast total demand by 9.2%; after it, by 0.5%.
- On the 20 highest-volume series, per-series exponential smoothing does better than LightGBM (13.5% vs 15.8% WAPE). Big, stable series have enough history to model on their own. The global model earns its keep on the thousands of smaller, noisier series.
- Ordering at the 75th percentile instead of following the moving average raises the fill rate from 83.2% to 92.8% and lowers weekly waste-plus-stockout cost by 37.9%, under the assumed 3:1 cost ratio.
- The most useful features are the series' long-run average, its 13-week rolling mean, and price relative to its recent average.

The findings cell (section 10) regenerates these numbers from the results, so they stay correct when the notebook runs on new data.

## Running it

### Google Colab (easiest)

Upload the notebook and the `data/` folder, uncomment the install line in the first code cell, and run all cells.

### Locally

Requirements: Python 3.10+ and **Java 17+** (needed by Spark).

```bash
pip install pyspark lightgbm statsmodels pyarrow pandas matplotlib
```

Then open the notebook and run all cells. Paths can be changed with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `FOOD_DATA_DIR` | `data` | folder with the three CSVs |
| `FOOD_OUTPUT_DIR` | `outputs` | where the feature table is saved |

The forecast horizon, number of backtest folds and the cost assumptions are set in the configuration block of the setup cell.

**Windows notes**
- PySpark is a large download (300+ MB). Install it from a terminal so you can see progress, not from inside the notebook.
- Spark's own file writer needs Hadoop's `winutils.exe` on Windows. The notebook avoids that by saving the feature table with pandas/pyarrow, so no Hadoop setup is needed.

### Databricks

Works on Databricks Free Edition. The notebook detects the runtime and uses the existing Spark session. Set `FOOD_DATA_DIR` to your Volume path.

## Limitations

- **Orders aren't demand.** The data records fulfilled orders. When a centre ran out, true demand was higher, and the model learns those shortfalls. Fixing this needs stockout flags the dataset doesn't have.
- **Missing weeks are ambiguous.** If they actually mean zero demand, forecasts for intermittent series run high.
- **The costs are assumptions.** Money comparisons are only as good as `COST_OVER` and `COST_UNDER`. Replace them with real figures before quoting any number.
- **Spark is more than this dataset needs.** 456k rows fit in pandas. Spark is used so the same feature pipeline scales to daily data or many more products without changes.
