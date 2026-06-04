# 📈 RevenueVision AI: Demand & Revenue Forecasting System

An end-to-end, production-ready time-series forecasting system that predicts future order volumes and projects financial revenues for an e-commerce business. Utilizing transaction-level records from the [Online Retail II Dataset (Kaggle)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci), the system builds, validates, and compares **five predictive models** (Prophet, XGBoost, LightGBM, Random Forest, and LSTM) using rigorous chronological cross-validation and recursive forecasting logic to support strategic business planning.

---

## 🔄 ML Lifecycle & Pipeline Workflow

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Data Scientist / User
    participant CSV as Raw Dataset (UCI Kaggle)
    participant Prep as Preprocessing Pipeline
    participant Feat as Feature Engineer
    participant Trainer as Model Trainer (5 Models)
    participant Forecaster as Recursive Forecaster
    participant Revenue as Revenue & Scenario Engine
    participant BI as Exports & Plots

    Dev->>CSV: Ingest online_retail_II.csv
    CSV-->>Prep: Return raw transaction streams
    note over Prep: Data Cleaning: Filter returns (Invoice starts with 'C') and invalid price/qty
    Prep->>Prep: Reindex daily timeline (Continuous gaps filled)
    Prep-->>Feat: Processed daily time series
    note over Feat: Feature Extraction: Date parts, UK holidays, simulated promo flags
    Feat->>Feat: Calculate rolling features (mean, std over 7, 14, 30 days) and lags (1, 7, 14, 30)
    Feat-->>Trainer: Complete chronological feature matrix
    note over Trainer: 80/20 chronological split (Prevents look-ahead leak)
    Trainer->>Trainer: 5-Fold TimeSeriesSplit Cross-Validation
    note over Trainer: Fits & tunes Prophet, XGBoost, LightGBM, Random Forest, LSTM
    Trainer-->>Trainer: Compare MAE, RMSE, MAPE, R2
    Trainer->>Forecaster: Select & Serialize Best Model (XGBoost)
    
    loop 365 Days Forecast Horizon
        Forecaster->>Forecaster: Predict Day t+1
        Forecaster->>Forecaster: Feedback Prediction as Lag 1 feature for Day t+2
        Forecaster->>Forecaster: Update Rolling Stats for next step
    end
    
    Forecaster-->>Revenue: Return 365-day expected order volume
    note over Revenue: Apply rolling 30-day Average Order Value (AOV = $665.47)
    Revenue->>Revenue: Project Expected Case Scenario
    Revenue->>Revenue: Apply Best Case Scenario (Orders * 1.20)
    Revenue->>Revenue: Apply Worst Case Scenario (Orders * 0.80)
    Revenue-->>BI: Compiled multi-horizon scenario forecasts
    note over BI: Generate SHAP attributions, export CSV files, save 9 dashboard plots
    BI-->>Dev: Exports (forecast_*.csv, plots/*.png, model_metrics.csv)
```

---

## 🧠 Deep-Dive: Core ML Principles Visualized

### 1. Preventing Target Leakage (Chronological Validation)
```mermaid
graph TD
    subgraph A["Chronological Split (Correct - No Leakage)"]
        direction LR
        Train1["Train Set (Past: Day 1 - 300)"] -->|Train Model| Model1["Model"]
        Model1 -->|Predict| Test1["Test Set (Future: Day 301 - 365)"]
        style Train1 fill:#e8f5e9,stroke:#2e7d32
        style Test1 fill:#ffebee,stroke:#c62828
    end
    subgraph B["Shuffled Split (Incorrect - Target Leakage)"]
        direction LR
        Mixed["Mixed History (Past & Future Shuffled)"] -->|Split Randomly| Train2["Train Set (Includes Future Days)"] & Test2["Test Set (Includes Past Days)"]
        Train2 -->|Model Memorizes Future| Model2["Model"]
        Model2 -->|Predicts Test Set| Test2
        style Mixed fill:#fffde7,stroke:#fbc02d
        style Train2 fill:#ffebee,stroke:#c62828
        style Test2 fill:#ffebee,stroke:#c62828
    end
```

### 2. Recursive Out-of-Sample Forecasting
```mermaid
graph LR
    subgraph Step1["Step 1: Predict Day t+1"]
        H1["Historical Lags & Rolling Mean"] -->|Predict| P1["Forecast (t+1)"]
    end
    subgraph Step2["Step 2: Predict Day t+2"]
        P1 -->|Feedback as Lag 1| H2["Updated Lags & Rolling Mean"]
        H2 -->|Predict| P2["Forecast (t+2)"]
    end
    Step1 --> Step2
    style P1 fill:#f3e5f5,stroke:#7b1fa2
    style P2 fill:#f3e5f5,stroke:#7b1fa2
```

### 3. Dynamic Average Order Value (AOV) Revenue Framework
```mermaid
graph TD
    O["Forecasted Orders (Expected / Best / Worst)"] -->|Multiply| R["Projected Revenue ($)"]
    AOV["Rolling 30-Day Average Order Value (AOV)"] -->|Multiply| R
    style R fill:#ede7f6,stroke:#5e35b1,stroke-width:2px
```

### 4. SHAP Feature Attribution (Explainable AI)
```mermaid
graph LR
    Base["Base Value (Average Orders: e.g., 50)"] -->|"+35 (Promo Flag = 1)"| P1["+"]
    P1 -->|"+15 (Weekend = 1)"| P2["+"]
    P2 -->|"-10 (Holiday = 1)"| P3["-"]
    P3 -->|Final Prediction: 90 Orders| Out["Output Prediction"]
    style Base fill:#e0f7fa,stroke:#00acc1
    style Out fill:#e8f5e9,stroke:#43a047,stroke-width:2px
```

---

## 📸 Key Visual Highlights

### 1. Future Demand & Revenue Forecast (365-Day Horizon)
Recursive daily projections of order volumes and risk-adjusted financial revenue alongside a $\pm20\%$ operational confidence interval.

| Order Demand Forecast | Projected Financial Revenue |
| :---: | :---: |
| ![Demand Forecast](RevenueVision_AI_Project/plots/future_forecast.png) | ![Revenue Forecast](RevenueVision_AI_Project/plots/revenue_forecast.png) |

---

### 2. Model Performance & Evaluation
Comparing predicted vs. actual order volumes on the unseen test partition and benchmarking Mean Absolute Percentage Error (MAPE) across models.

| Actual vs. Predicted Orders (XGBoost) | Model Benchmark Comparison (MAPE %) |
| :---: | :---: |
| ![Actual vs Predicted](RevenueVision_AI_Project/plots/actual_vs_predicted.png) | ![Model Comparison](RevenueVision_AI_Project/plots/model_comparison.png) |

---

### 3. Seasonality Patterns & Explainable AI (SHAP)
Decomposing hourly sales into day-of-week and month patterns, combined with SHAP feature attributions showing why forecasts are generated.

| Seasonality Analysis | SHAP Feature Attribution |
| :---: | :---: |
| ![Seasonality](RevenueVision_AI_Project/plots/seasonality.png) | ![SHAP Summary](RevenueVision_AI_Project/plots/shap_summary.png) |

---

## 🛠 What Was Accomplished (System Pipeline)

1. **Robust Ingestion & Cleaning**:
   - Processed transaction records from `online_retail_II.csv`.
   - Excluded return orders (invoices starting with `C` or non-positive quantities/prices) and removed duplicate rows.
2. **continuous Indexing & Gap Filling**:
   - Reindexed raw timestamp sequences to a daily frequency to prevent timeline gaps.
   - Filled inactive days (zero orders/revenue) and forward-filled Average Order Value (AOV).
3. **Feature Engineering**:
   - Extracted calendar variables (Day of week, month, quarter, year, week of year, weekend indicators).
   - Generated UK Bank Holidays via the `holidays` library.
   - Modeled simulated promotional flags corresponding to major retail events (Black Friday, Cyber Monday, Christmas, and quarterly clearances).
   - Computed recursive lags (1, 7, 14, 30 days), rolling averages (mean, standard deviation over 7, 14, 30 days), and growth features.
4. **Chronological Splitting (80-20)**:
   - Partitioned the time series chronologically into training (80%) and testing (20%) datasets without shuffling to prevent target leakage.
5. **Multi-Model Development**:
   - Developed **5 distinct modeling techniques**: Facebook Prophet, XGBoost, LightGBM, Random Forest, and LSTM (TensorFlow Keras).
   - Employed `TimeSeriesSplit` cross-validation with grid search parameter tuning.
6. **Recursive Forecasting & Scenario Analysis**:
   - Implemented a day-by-day recursive forecasting loop to dynamically feed forward predictions as future lags and rolling stats.
   - Mapped order volumes to rolling AOV ($665.47) for revenue projections.
   - Engineered three scenario projections: **Expected**, **Best Case** ($1.20 \times \text{Expected}$), and **Worst Case** ($0.80 \times \text{Expected}$).
7. **Explainable AI**:
   - Generated SHAP tree explanations mapping feature attributions.

---

## 📊 Benchmarking Results

| Model | MAE | RMSE | MAPE (%) | $R^2$ Score | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🏆 **XGBoost** | **3.76** | **5.26** | **6.42%** | **0.980** | **Selected Best Model** |
| ⚡ LightGBM | 4.49 | 7.14 | 7.75% | 0.963 | Backup |
| 🌲 RandomForest | 4.27 | 7.39 | 7.84% | 0.960 | Backup |
| 📈 Prophet | 13.07 | 16.99 | 18.19% | 0.791 | Baseline |
| 🧠 LSTM | 17.55 | 21.74 | 21.03% | 0.671 | Baseline |

---

## 📂 Project Structure

```bash
RevenueVision_AI_Project/
├── RevenueVision_AI.ipynb  # Comprehensive, fully documented Jupyter Notebook (20 sections)
├── pipeline.py             # Executable production Python script running the entire pipeline
├── data/
│   └── online_retail_II.csv # Input dataset (must be placed here before executing)
├── exports/
│   ├── forecast_30_days.csv # 30-day forecast horizons (Orders & Revenue scenarios)
│   ├── forecast_90_days.csv # 90-day forecast horizons
│   ├── forecast_365_days.csv# 365-day complete forecast horizons
│   ├── revenue_forecast.csv # Projected daily financial revenue scenarios
│   └── model_metrics.csv    # Benchmark table for all models
├── models/
│   ├── best_model.joblib    # Serialized production XGBoost model
│   ├── xgboost_model.joblib # Model weights
│   ├── lightgbm_model.joblib# Model weights
│   ├── random_forest.joblib # Model weights
│   ├── lstm_model.h5        # Model weights
│   ├── best_lstm_weights.keras # Checkpointed Keras weights
│   └── prophet_model.pkl    # Serialized Prophet model
└── plots/
    ├── actual_vs_predicted.png
    ├── confidence_intervals.png
    ├── feature_importance.png
    ├── future_forecast.png
    ├── model_comparison.png
    ├── revenue_forecast.png
    ├── seasonality.png
    ├── shap_summary.png
    └── trend_analysis.png
```

---

## 🚀 Setup & Execution

### Prerequisites
Make sure Python 3.10+ is installed on your machine.

### Installation & Run

1. **Clone/Open the workspace** in your preferred editor.
2. **Navigate into the Project Folder**:
   ```bash
   cd RevenueVision_AI_Project
   ```
3. **Ingest the Raw Data**: Place the `online_retail_II.csv` dataset in the `data/` directory.
4. **Execute the Jupyter Notebook**:
   - Open [RevenueVision_AI.ipynb](RevenueVision_AI_Project/RevenueVision_AI.ipynb).
   - Run Section **2.5** (Auto-Install cell). It will dynamically verify and install all missing dependencies directly into your current kernel environment:
     ```python
     # Installs numpy, pandas, xgboost, lightgbm, tensorflow, shap, prophet, etc. if missing.
     ```
   - Run all cells to execute the pipeline.
5. **Execute via Terminal**:
   - Alternatively, you can run the full pipeline script from your command line:
     ```bash
     python pipeline.py
     ```

---
