# Air-Quality-Predictor-

# Air Quality Forecasting with Deep Learning

A deep learning project for predicting particulate matter concentrations (PM1.0, PM2.5, PM10) using time-series data from a PurpleAir sensor, with EPA-standard AQI computation and a systematic comparison of 10 recurrent neural network architectures.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## Project Overview

This project builds and evaluates multiple deep learning models to forecast ambient particulate matter levels from 2-minute resolution sensor data. The workflow covers data cleaning, AQI computation, STL decomposition for trend/seasonality analysis, feature engineering with lag variables, and training/evaluating 10 distinct RNN-family architectures across three pollutant targets.

**Targets:** PM1.0, PM2.5, PM10  
**Evaluation metrics:** MAE, RMSE, MAPE, RAE, R²

---

## Dataset

- **Source:** PurpleAir sensor (ID 57343)
- **Period:** 2024-01-01 to 2024-09-25
- **Resolution:** ~2-minute average readings
- **Records:** 173,921 rows

**Columns:**

| Column | Description |
|---|---|
| `time_stamp` | UTC timestamp |
| `humidity` | Relative humidity (%) |
| `pm1.0_atm` | PM1.0 concentration (µg/m³) |
| `pm2.5_atm` | PM2.5 concentration (µg/m³) |
| `pm10.0_atm` | PM10 concentration (µg/m³) |
| `temperature` | Temperature (°F) |
| `pressure` | Atmospheric pressure (hPa) |

---

## Pipeline

### 1. Data Cleaning
- Converted `time_stamp` from object to `datetime64[ns, UTC]`
- Forward-filled 1,879 missing values in `humidity`, `temperature`, and `pressure`
- Confirmed zero duplicate rows

### 2. AQI Computation
PM AQI is computed per the EPA breakpoint standard for both PM2.5 (truncated to 1 decimal) and PM10 (truncated to integer). The final `pm_aqi` column takes the maximum of the two sub-index values.

### 3. Exploratory Analysis
- **Time-series plot** of PM AQI over the full observation window
- **Distribution histograms** (with KDE) for PM1.0, PM2.5, PM10, and AQI — all right-skewed; AQI shows a bimodal pattern
- **Correlation heatmap** — PM2.5 and PM1.0 are strongly correlated (r = 0.98); weather variables show weak negative correlations with PM concentrations

### 4. STL Decomposition
Seasonal-Trend decomposition (period = 24) applied to all three PM targets reveals:
- High pollution at the start of 2024, dipping mid-year, rising again in late 2024 through early 2025
- Regular diurnal seasonality, likely driven by temperature inversions and human activity cycles
- Residual spikes indicating episodic pollution events, increasing in frequency toward the end of the period

### 5. Preprocessing
- **MinMax scaling** applied to all features
- **Lag features** created at 24h (720 steps), 48h (1440 steps), and 7d (5040 steps) for each PM target, giving models access to historical context across multiple timescales
- Records with NaN lag values dropped, leaving **168,821 usable samples**
- **Sequence windows** of 60 time steps (~2 hours) constructed per target, yielding input shape `(168821, 60, 15)`
- **Train / Val / Test split:** 70% / 15% / 15% (no shuffling, preserving temporal order)
- Data converted to `tf.float32` tensors and loaded into `tf.data.Dataset` pipelines with `batch_size=64` and `prefetch(AUTOTUNE)`
- **Mixed precision** (`mixed_float16`) enabled for faster GPU training
- **Early stopping** on `val_loss` with `patience=3` and best-weight restoration

---

## Models

Ten architectures were benchmarked. All use a 60-step (2-hour) input window and are trained independently on each of the three PM targets.

| # | Architecture | Key Layers |
|---|---|---|
| 1 | LSTM-GRU Stack | LSTM(128) → GRU(64) → GRU(32) → FCN(64,32,1) |
| 2 | Stacked RNN | RNN(128) → RNN(32) → FCN(64,16,1) |
| 3 | Stacked GRU | GRU(128) → GRU(64) → FCN(64,16,1) |
| 4 | BiLSTM-GRU-TDL | BiLSTM(32) → GRU(32) → TimeDistributed(8) → Dense(1) |
| 5 | BiLSTM-TDL | BiLSTM(64) → TimeDistributed(8) → FCN(32,16,1) |
| 6 | LSTM-TDL-LSTM | LSTM(128) → TimeDistributed(8) → LSTM(64) → FCN(32,16,1) |
| 7 | RNN-Flatten-RNN | RNN(128) → TD Flatten → RNN(64) → FCN(16,1) |
| 8 | GRU-Flatten | GRU(128) → TD Flatten → FCN(64,16,1) |
| 9 | BiLSTM + Attention | BiLSTM(64) → BiLSTM(32) → Attention → Dropout → FCN(32,16,1) |
| 10 | CNN + BiLSTM | Conv1D(64) → MaxPool → BiLSTM(64) → Dropout → FCN(32,16,1) |

---

## Results

### PM1.0

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Model 1 (LSTM-GRU) | — | — | — | — |
| Model 2 (RNN) | — | — | — | — |
| Model 3 (GRU) | 0.0163 | 0.0340 | 9.90% | 0.875 |
| Model 4 (BiLSTM-GRU-TDL) | 0.0214 | 0.0375 | 14.37% | 0.847 |
| Model 5 (BiLSTM-TDL) | 0.0162 | 0.0343 | 9.50% | 0.872 |
| **Model 6 (LSTM-TDL-LSTM)** | **0.0154** | **0.0335** | **8.77%** | **0.878** |
| Model 7 (RNN-Flatten-RNN) | 0.0164 | 0.0344 | 9.54% | 0.871 |
| Model 8 (GRU-Flatten) | 0.0270 | 0.0549 | 16.52% | 0.672 |
| Model 9 (BiLSTM+Attention) | 0.0161 | 0.0341 | 9.56% | 0.873 |
| Model 10 (CNN+BiLSTM) | 0.0164 | 0.0345 | 9.30% | 0.871 |

### PM2.5

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Model 3 (GRU) | 0.0171 | 0.0334 | 9.93% | 0.906 |
| Model 4 (BiLSTM-GRU-TDL) | 0.0145 | 0.0323 | 7.92% | 0.912 |
| Model 5 (BiLSTM-TDL) | 0.0159 | 0.0326 | 9.02% | 0.910 |
| **Model 6 (LSTM-TDL-LSTM)** | **0.0141** | **0.0314** | **7.70%** | **0.917** |
| Model 7 (RNN-Flatten-RNN) | 0.0173 | 0.0351 | 8.90% | 0.896 |
| Model 8 (GRU-Flatten) | 0.0295 | 0.0598 | 16.11% | 0.699 |
| Model 9 (BiLSTM+Attention) | 0.0158 | 0.0337 | 8.20% | 0.904 |
| **Model 10 (CNN+BiLSTM)** | 0.0146 | **0.0310** | 8.16% | **0.919** |

### PM10

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| Model 3 (GRU) | 0.0331 | 0.0683 | 15.84% | 0.612 |
| Model 4 (BiLSTM-GRU-TDL) | 0.0334 | 0.0659 | 15.32% | 0.639 |
| **Model 5 (BiLSTM-TDL)** | 0.0335 | **0.0609** | 17.02% | **0.691** |
| Model 6 (LSTM-TDL-LSTM) | 0.0336 | 0.0565 | 17.15% | 0.735 |
| Model 7 (RNN-Flatten-RNN) | 0.0368 | 0.0675 | 17.58% | 0.622 |
| Model 8 (GRU-Flatten) | 0.0411 | 0.0835 | 17.77% | 0.420 |
| Model 9 (BiLSTM+Attention) | 0.0342 | 0.0617 | 16.83% | 0.684 |
| Model 10 (CNN+BiLSTM) | 0.0338 | 0.0585 | 16.56% | 0.716 |

**Key takeaways:**
- **Model 6 (LSTM-TDL-LSTM)** is the strongest overall performer for PM1.0 and PM2.5
- **Model 10 (CNN+BiLSTM)** is competitive on PM2.5 and PM10, with the best RMSE on PM2.5
- **PM10 is the hardest target** across all models (R² peaks around 0.73), likely due to its sensitivity to coarse dust events not well-captured by the 2-hour window
- **Model 8 (GRU-Flatten)** consistently underperforms — the sequence collapsing via TimeDistributed Flatten before a dense head loses temporal structure

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/air-quality-forecasting.git
cd air-quality-forecasting

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow
```

Developed and tested on **Google Colab** with TensorFlow 2.18 and Python 3.11.

---

## Usage

1. Upload your sensor data Excel file to Google Drive
2. Update the file path in the data loading cell:
   ```python
   xls = pd.ExcelFile('/content/drive/MyDrive/.../your_file.xlsx')
   ```
3. Run all cells sequentially. Each model section is self-contained and can be run independently after preprocessing is complete.

To train only a specific model on a specific pollutant:
```python
model6 = build_model6()
history = model6.fit(train_dataset_2_5, validation_data=val_dataset_2_5,
                     epochs=20, callbacks=[early_stopping])
results = evaluate_and_plot(model6, test_dataset_2_5, y_2_5_test, "Model 6", "PM2.5")
```

---

## Project Structure

```
air-quality-forecasting/
│
├── notebook.ipynb          # Main Colab notebook (all steps end-to-end)
├── README.md               # This file
│
└── data/
    └── Air quality 57343 2024-01-01 2024-09-25 0-Minute Average.xlsx
```

---

## Dependencies

| Package | Version |
|---|---|
| Python | 3.11 |
| TensorFlow / Keras | 2.18.0 |
| pandas | ≥ 2.0 |
| numpy | 1.26.4 |
| scikit-learn | ≥ 1.3 |
| statsmodels | ≥ 0.14 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |
