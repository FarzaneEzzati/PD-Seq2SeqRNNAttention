# Quantile Load Forecasting with Seq2Seq GRU + Bahdanau Attention

Day-ahead quantile forecasts (q = 0.1, 0.5, 0.9) of residential electricity demand for socially
vulnerable households in **Houston** (ERCOT COAST zone) and **Austin** (ERCOT SCENT zone), Texas.

The hourly forecasts are intended as inputs to community-microgrid decision models:
- **Unit commitment and economic dispatch:** the median (q0.5) feeds deterministic models, and the quantiles
  provide demand uncertainty for stochastic or robust ones.
- **Emergency operation:** planning how power is shared or traded during outages.
- **Local energy markets:** pricing electricity trades between consumers and prosumers, within and across communities.

The model sees the last 48 hours of weather, calendar features and demand, and predicts the next 24 hours of demand per household (kW) as three quantiles. The q0.1–q0.9 band is an 80 % prediction interval.

## Data

| Source | Content | Raw location |
|---|---|---|
| [ERCOT load history](https://www.ercot.com/gridinfo/load/load_hist) | Hourly native load (MW) per weather zone, 2015 – present | `data/raw/ercot/Native_Load_<year>.xlsx` |
| [Wunderground history](https://www.wunderground.com/history) | Hourly temperature (F), humidity, wind (mph), pressure (in), precipitation (in), condition | `data/raw/weather/<location>.csv` |
| US Census ACS 5-year (B11001) | Number of households per county | `data/raw/census/` |

Demand per vulnerable household is derived as
`1000 * (1 - 0.1005) * zone_load_MW / n_households`, since these households consume about 10 % less than average.
Location settings are at the bottom of [config.yaml](config.yaml).

## Method

A sequence-to-sequence network: a bidirectional multi-layer GRU encoder, and a GRU decoder with
Bahdanau (additive) attention that emits one forecast hour per step.

$$
e_{t,j} = V^\top \tanh\big(W [h^{dec}_{t-1}; h^{enc}_j]\big), \qquad
\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{k=1}^{T}\exp(e_{t,k})}, \qquad
c_t = \sum_{j=1}^{T} \alpha_{t,j} h^{enc}_j
$$

$$
h^{dec}_t = \mathrm{GRU}\big([\hat{y}_{t-1}; x^{f}_t; c_t],\ h^{dec}_{t-1}\big), \qquad
\hat{y}_t = \mathrm{Linear}\big([\hat{y}_{t-1}; x^{f}_t; c_t; h^{dec}_t]\big) \in \mathbb{R}^{3}
$$

At each forecast hour $t$ the decoder gets:

* $\hat{y}_{t-1}$, the previous hour's demand: the true value under teacher forcing (ratio decayed during
  training), otherwise the model's own median forecast.
* $x^{f}_t$, the time features (hour, day, week, month, weekend) and the weather of hour $t$.
  The time features are known exactly; the weather would come from a weather forecast in practice
  (see [Future weather](#future-weather-perfect-vs-noisy)).
* $c_t$, the attention context over the 48 input hours.

The loss is the pinball loss summed over quantiles:

$$L_q = q\,(y-\hat{y})_+ + (1-q)\,(\hat{y} - y)_+$$

## Project structure

```
config.yaml           all settings (city, dates, model size, training)
1_fetch_data.py       step 1: check the raw data, or download what is missing
2_prepare_data.py     step 2: clean the data and build the model table
3_train.py            step 3: train the model
4_evaluate.py         step 4: test the model on 2024
5_predict.py          step 5: forecast the next 24 hours
src/                  helper code used by the steps
    ercot.py          download and read ERCOT load files
    weather.py        scrape and clean weather data
    features.py       build the model table, split it, cut it into windows
    model.py          the network and the quantile loss
    noise.py          forecast-like errors for the future weather
    metrics.py        scores
    plots.py          figures
    utils.py          read config.yaml, fill data gaps
notebooks/            01_eda.ipynb (explore the data), 02_results.ipynb (look at results)
data/raw/             original data (kept in git)
data/interim/         cleaned data             (created by step 2)
data/processed/       model table + scaler     (created by step 2)
models/               trained models           (created by step 3)
reports/              scores, forecasts, figures (created by steps 3-5)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

## How to run

Set what you want in `config.yaml`, then run the steps in order from the project folder:

```bash
python 1_fetch_data.py
python 2_prepare_data.py
python 3_train.py
python 4_evaluate.py
python 5_predict.py
```

Each step saves its results, so you can rerun any step without repeating the earlier ones.
To switch city, change `location` in `config.yaml` and run the steps again.

To keep several settings files, give the file name when running a step, e.g.
`python 3_train.py houston_noisy.yaml`. Without a name, `config.yaml` is used.

### 1. Fetch data

The raw data is already in `data/raw`, so by default nothing is downloaded. The reason is the slow fetch of data, specifically weather data, from its source.
To download new data, set `fetch_from_internet: true` and the dates you want in `config.yaml`:

* **ERCOT load:** downloads the yearly files that are missing.
* **Weather:** scrapes only the days that are missing. This is slow (one web page per day). It saves
  after every month, so if it stops, run it again and it continues. It needs
  Google Chrome and the `selenium` package.

### 2. Prepare data

* Converts ERCOT times (which mark the *end* of each hour) to the start of the hour.
* Removes units and impossible readings (e.g. 0 inHg pressure) from the weather data.
* Fills gaps of up to 24 hours. Longer gaps are left out.
* Joins weather and load by time, and adds weather-condition and calendar columns.
* Splits by date: train 2015-2022, validation 2023, test 2024.
* Scales the data using the training years only.

### 3. Train

Trains for up to `n_epochs` and keeps the model with the lowest validation loss. It stops early if
the validation loss does not improve for `patience` epochs. To try different settings, change them in
`config.yaml` and give the run a new `run_name`, so earlier results are kept.

The model is saved in `models/<location>/<run_name>-<future_weather>/`, so the `perfect` and `noisy`
versions of a run are kept side by side.

### 4. Evaluate

Makes a 24-hour forecast at midnight for every day of 2024 and compares it with the actual demand.
Scores (in kW per household) include:

* **MAE / RMSE / MAPE** of the median forecast
* **Baselines:** the error of simply using yesterday's or last week's demand. The model should beat these.
* **Interval coverage:** how often the actual value falls inside the q0.1–q0.9 band (should be about 80 %)

Every model is tested twice: with the observed weather (`perfect_weather`) and with forecast-like
weather (`noisy_weather`). Both are saved as columns in `test_metrics.csv`.

### 5. Predict

Forecasts one day and compares it with what actually happened. By default this is the last full day
in the data. To pick another day, set `predict_start: "2024-07-15"` in `config.yaml`.

The model needs the weather of the hours it forecasts. Here the observed weather stands in for it.
To forecast a real future day, you need a weather forecast for that day.

## Future weather: perfect vs noisy

The decoder uses the weather of the hours it forecasts. In real use this comes from a weather
forecast, which has errors. The data only has the *observed* weather, so two versions are trained
(setting `future_weather` in `config.yaml`):

* **perfect:** the model gets the observed weather. This is the best case: it shows what the model can
  do with a perfect weather forecast.
* **noisy:** the model gets the observed weather plus random errors that look like forecast errors.
  The errors are new in every training batch, so the model learns how far to trust the weather.
  This is the realistic case.

### How the noise is made

All errors use the normal distribution. Sizes are set under `weather_noise` in `config.yaml`;
the code is in [src/noise.py](src/noise.py).

**Temperature, humidity, wind, pressure.** Forecast errors are not independent from hour to hour: if a
forecast is 3 °F too warm in the morning, it is usually too warm all day. So each error has two parts:

$$
\varepsilon_t = a + b_t, \qquad a \sim \mathcal{N}(0,\ (0.8\,\sigma)^2), \qquad b_t \sim \mathcal{N}(0,\ (0.6\,\sigma)^2)
$$

* $a$ is a daily offset, drawn once and shared by all 24 hours of a forecast.
* $b_t$ is a smaller hour-to-hour wiggle, drawn again every hour.
* Together they have a standard deviation of $\sigma$ (because $0.8^2 + 0.6^2 = 1$).

| Variable | $\sigma$ | Note |
|---|---|---|
| Temperature | 3.0 °F | average error ≈ 2.4 °F, typical for day-ahead forecasts |
| Humidity | 0.08 (8 percentage points) | kept between 0 and 1 |
| Wind | 3.0 mph | kept at 0 or above |
| Pressure | 0.03 inHg | |

**Precipitation.** Adding normal noise would create rain on dry hours and negative rain. Instead, the
amounts of a forecast are multiplied by one random factor $e^{z}$ with $z \sim \mathcal{N}(0,\ 0.5^2)$.
Dry hours stay dry.

**Weather condition (categorical).** Normal noise does not apply to categories. Instead, each hour, with
probability 0.2, the condition is replaced by one drawn at random according to how often each condition
occurs in the training years (e.g. Cloudy is common, Snow/Ice is rare). Because the draw can return the
same condition, about 10 % of hours actually change.

Is noise on the condition necessary? The condition is a separate input from temperature: changing it
does not change the temperature value the model sees. Its effect on demand is small compared with
temperature, so this noise matters little for the results. It is kept so that *all* forecast weather
is treated the same way, and so the model does not rely on a perfectly known condition.

**What is not changed:** the time features (hour, day, week, month, weekend), because they are known exactly.

**Limitations.** Real forecast errors also grow with lead time and depend on season and weather type.
The noise here is a simple, fixed-size approximation. The most realistic option would be to train and
test with archived weather forecasts instead of observed weather.

## Results (test year 2024)

Each model makes one 24-hour forecast at midnight for every day of 2024. Errors are in kW per household.
"Tested with" is the weather the model gets for the forecast hours. The realistic case is **noisy**.

Settings: run `T48-L3-HD128` (48 input hours, 3 GRU layers, 128 hidden units, teacher forcing 0.9),
up to 500 epochs with early stopping (patience 30), seed 42.

**Houston** (training stopped after 201 epochs for perfect, 149 for noisy)

| Model trained with | Tested with | MAE | MAPE | Coverage of q0.1–q0.9 (target 80 %) | Band width |
|---|---|---|---|---|---|
| perfect weather | perfect | 0.149 | 3.0 % | 79.5 % | 0.46 |
| perfect weather | noisy | 0.201 | 3.9 % | **66.9 %** | 0.46 |
| noisy weather | perfect | 0.173 | 3.5 % | 83.4 % | 0.58 |
| noisy weather | noisy | 0.180 | 3.6 % | 81.6 % | 0.58 |
| *baseline: same hour yesterday* | | *0.284* | | | |

**Austin** (training stopped after 144 epochs for perfect, 116 for noisy)

| Model trained with | Tested with | MAE | MAPE | Coverage of q0.1–q0.9 (target 80 %) | Band width |
|---|---|---|---|---|---|
| perfect weather | perfect | 0.161 | 4.3 % | 82.9 % | 0.54 |
| perfect weather | noisy | 0.172 | 4.6 % | 79.2 % | 0.54 |
| noisy weather | perfect | 0.176 | 4.7 % | 83.1 % | 0.61 |
| noisy weather | noisy | 0.181 | 4.8 % | 82.2 % | 0.61 |
| *baseline: same hour yesterday* | | *0.233* | | | |

What this shows:

* All models clearly beat the simple baseline (about 20–37 % lower error).
* With perfect weather the error is lowest, but this is a best case that cannot be reached in practice.
* The model trained with perfect weather becomes **overconfident** when it gets forecast-like weather.
  In Houston its band holds the actual value only 67 % of the time instead of 80 %. The model trained with
  noisy weather keeps the coverage at about 80 % by making the band a little wider. For use with real
  weather forecasts, the noisy-trained model is the safer choice.


Results are in `reports/<location>/T48-L3-HD128-<perfect|noisy>/test_metrics.csv`.
