[ 🇺🇸 English ] | [ 🇨🇱 [Leer en Español](README.es.md) ]

# Retail demand forecasting, scored as an inventory decision

![Python](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![Polars](https://img.shields.io/badge/polars-rust--backed-CD792C)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient_boosting-4C7A3E)
![Tests](https://img.shields.io/badge/tests-19_passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

A demand forecasting pipeline on **3,000,888 rows of real grocery sales**, built around
a question the competition metric does not answer: *does this forecast lead to a better
purchase order?*

The short answer, measured here, is that the two questions pick different models. The
model with the best RMSLE leaves a stockout on **41% of store-days**. The model that
wins the purchase decision is **the worse forecast by every accuracy metric except one**.

---

## The dataset

| | |
|---|---|
| Source | [Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) (Corporación Favorita, Ecuador) |
| Downloaded | 2026-10-02, via the Kaggle API |
| Licence | Competition rules apply; the raw data is **not redistributed** in this repository |
| Size | 3,000,888 rows · 54 stores × 33 product families · 2013-01-01 to 2017-08-15 |
| Covariates | daily oil price, national/regional holidays, promotions, store metadata |

**Why Ecuadorian grocery data in a Chile-focused portfolio, stated plainly:** granular
retail demand (SKU × store × day) is never public, in Chile or anywhere else. Chile's
own open-data catalogue lists INE's supermarket index with no file attached, and the
Central Bank's series portal requires an account. The choice was between real demand
from a comparable South American grocery chain and invented demand with a Chilean
label. This repository takes the real data and says where it comes from. The pipeline
is not country-specific; the payday and holiday features transfer directly.

```bash
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw --unzip
```

---

## 1. A quarter of the "zero demand" in this dataset is not demand

The panel is 31.3% zeros, which reads as severe intermittency and sends you looking for
Croston or a zero-inflated model. That number is wrong, and the reason is not in the
schema.

![Where the zeros come from](outputs/figures/zeros_decomposition.png)

**Eight of the 54 stores open after the panel begins** — the last of them on 2017-04-20,
four months before the data ends. The panel is padded, not truncated: until a store
opens, all 33 of its families report zero sales every single day. That is **222,057 rows,
7.4% of the panel, and 23.6% of every zero in the dataset**, recording the absence of a
store rather than the absence of demand.

Three things follow:

- The real zero rate is **25.8%**, not 31.3%.
- The count of severely intermittent series (>90% zeros) drops from **173 to 135** — 22%
  of them were an artifact.
- A model trained on those rows learns "this store does not sell" about a store that had
  not opened, and a metric computed over them collects credit for predicting zero where
  there was nothing to predict.

`src/data.py` flags these rows rather than dropping them silently, so every downstream
step chooses whether to use them and both versions of each number stay visible.

![Intermittency across the 1,782 series](outputs/figures/intermittency_distribution.png)

Even after the correction, **not one of the 1,782 series sells every day**. The
distribution is bimodal — a dense group that sells almost always, a long tail that almost
never does. This is also why MAPE is absent from this repository: it divides by the
actual value, which is zero on a quarter of the observations.

---

## 2. The forecast that wins the metric loses the decision

The competition scores RMSLE on a point forecast. But nobody orders the mean. A buyer
orders a **quantity**, and being short (lost sale) does not cost the same as being long
(excess stock, markdown, tied-up capital). That is a newsvendor problem, and its answer
is not the mean but a **quantile set by the cost ratio**: `Cu / (Cu + Co)`.

The scenario below assumes a stockout costs **4x** an overstock — a stated assumption,
not a fact in the data, which is why §3 varies it.

![The same three forecasts, scored three ways](outputs/figures/accuracy_vs_decision.png)

| Model | RMSLE ↓ | MASE ↓ | Pinball@80% ↓ | Service level ↑ | Fill rate ↑ | Total cost ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal naive | 0.6666 | 1.762 | 64.11 | 60.3% | 87.8% | 9.14M |
| LightGBM (mean) | **0.4148** | **0.811** | 29.78 | 59.0% | 94.3% | 4.25M |
| LightGBM (quantile 80%) | 0.5313 | 1.177 | **27.50** | **84.0%** | **97.3%** | **3.92M** |

Read the first two columns and the mean model wins comfortably. Read the last three and
it is beaten by a model whose RMSLE is 28% worse.

**The mean model's service level is 59.0% — below the seasonal naive's 60.3%**, despite
being dramatically more accurate. That is not a paradox: ordering the conditional mean
leaves you short roughly half the time by construction, however good the mean estimate
is. Accuracy improved; the decision did not.

The quantile model gets there by spending overstock to buy service: its understock cost
falls from 3.06M to 1.45M while its overstock cost rises from 1.19M to 2.47M. At a 4:1
ratio that trade is worth making. At 1:1 it is not — which is the subject of the next
section.

---

## 3. The result does not hold at every cost ratio, and that is worth saying

The critical ratio is an assumption about the business. If the conclusion flips under a
plausible alternative, quoting only the favourable column is how a portfolio project
becomes misleading.

![Cost-ratio sensitivity and quantile calibration](outputs/figures/cost_ratio_sensitivity.png)

| Cost ratio | Critical quantile | Quantile cost | Mean cost | Saving |
|---:|---:|---:|---:|---:|
| 1:1 | 50% | 1.93M | 1.95M | +1.0% |
| 2:1 | 67% | 2.88M | 2.72M | **−5.9%** |
| 2.5:1 | 71% | 3.23M | 3.10M | **−4.3%** |
| 3:1 | 75% | 3.50M | 3.48M | **−0.4%** |
| 4:1 | 80% | 3.92M | 4.25M | +7.7% |
| 6:1 | 86% | 4.71M | 5.77M | +18.5% |
| 9:1 | 90% | 5.68M | 8.07M | +29.5% |

**Below roughly 3:1, ordering the mean is cheaper.** The extra service the quantile buys
costs more in excess stock than it saves in lost sales, and the method only starts paying
above that crossover. Quoting the 4:1 and 9:1 rows alone would make it look universally
better than it is.

The right-hand panel is the check that the models are doing what they were asked: the
delivered service level tracks the requested quantile closely at every level (50%→59.9%,
67%→73.7%, 80%→84.0%, 90%→92.1%), sitting consistently just above the diagonal.

---

## Method

| Step | Choice | Why |
|---|---|---|
| Split | Last **16 days** held out by date | The horizon the competition asks for, so the backtest measures the stated problem. Never a random split: it would leak the future into the past. |
| Features | Lags 7/14/21/28, rolling mean/std, rolling zero rate, all shifted within each series | Every feature is available at decision time. The shift is applied inside `over(["store_nbr","family"])`, so no series borrows another's history. |
| Calendar | Day of week, month, day of year, **payday** (15th and month-end) | Ecuador, like Chile, pays salaries on the 15th and the last working day, and grocery demand follows it. |
| Covariates | Oil price **forward-filled**, holidays excluding transferred days | Oil quotes only on market days; the price that matters for a Sunday is Friday's, not an average that peeks ahead. |
| Target | `log1p(sales)` | Matches RMSLE and keeps three orders of magnitude of family size from dominating the gradient. |
| Baseline | Seasonal naive (same weekday) | A serious rival in retail. A model that cannot beat it is learning the average, not the demand. |

The 2016 Ecuador earthquake (16 April, magnitude 7.8) is flagged rather than removed: a
model that does not know it happened learns it as April seasonality and repeats it every
year.

---

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/activate              # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

kaggle competitions download -c store-sales-time-series-forecasting -p data/raw --unzip

python -m pytest -q                 # 19 tests
python -m src.pipeline              # full run -> outputs/results.json
python -m src.make_figures          # redraws the four figures above
```

Every number in this README comes from `outputs/results.json`, which the pipeline writes
and which is version-controlled so the figures can be checked without re-running
anything.

---

## Tests

19 tests, all on exact arithmetic or textbook identities rather than on the 3M-row
download, so they run in CI in under a second.

| Property | Why it matters |
|---|---|
| Opening date is the first **positive sale**, not the first row | Taking the first row returns 2013-01-01 for every store and erases the whole §1 finding |
| Pre-opening rows are flagged and are all zero | The flag is what separates "no demand" from "no store" |
| Intermittency changes when pre-opening is counted | Encodes the 173 → 135 correction as a test on a toy panel |
| Oil price is forward-filled, never interpolated | An interpolated price is a look-ahead leak that would never show up as a failure |
| Pinball loss is asymmetric in the direction the quantile says | A sign error here would silently invert the project's conclusion |
| Critical ratio equals `Cu/(Cu+Co)` | The one line connecting a cost assumption to a model choice |
| Ordering the mean leaves ~50% service on symmetric demand | The project's central claim, as an assertion |
| Ordering the critical quantile hits its target service | Confirms the mechanism, not just the outcome |

---

## Honest limitations

- **Single-period newsvendor.** No inventory carried between days, so today's excess does
  not cushion tomorrow's shortage. Both cost columns would fall under a carry model; what
  survives is the asymmetry between the two errors, which is the point being made.
- **The cost ratio is assumed, not measured.** Favorita publishes no margin or holding
  cost. §3 is the mitigation, and it shows the conclusion is ratio-dependent.
- **No hierarchical reconciliation.** Store × family forecasts are not constrained to sum
  to store or national totals. For a buyer working at family level this does not bind,
  but a planner reporting up a hierarchy would need it.
- **Ecuador, not Chile.** Stated at the top and repeated here. The seasonality that
  transfers (payday cycles, Christmas, school calendar) is explicit in the features; the
  demand levels do not transfer and are not claimed to.
- **16-day horizon, one split.** A rolling-origin backtest over several windows would put
  an error bar on these comparisons. This one does not, so the differences in the tables
  above carry no confidence interval.

---

## License

MIT — see [LICENSE](LICENSE). The Favorita dataset is governed by the competition's own
rules and is not redistributed here.
