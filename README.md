# 🚗 Car Price Prediction System

An end-to-end **Machine Learning web application** that predicts the selling price of a used car based on its specifications. Built with **Python**, **Random Forest**, and **Streamlit**.

---

## 📸 App Preview

| Home Page | Predict Price | Data Insights |
|---|---|---|
| KPI cards, charts | Form-based prediction | EDA visualizations |

---

## 🎯 Features

- 🔮 **Predict Price** — Enter car details and get an instant price estimate with a confidence range
- 📊 **Data Insights** — Interactive EDA with box plots, violin charts, correlation heatmap, trendlines
- 🏠 **Home Dashboard** — KPI cards, price distribution, fuel share, price vs age scatter
- 📋 **Raw Data** — Filter, browse and explore the full dataset with descriptive stats

---

## 🧠 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Number of Trees | 300 |
| R² Score | **0.93** |
| Mean Absolute Error | **INR 66,729** |
| Training Samples | 6,502 |
| Test Samples | 1,626 |
| Target Transform | `log1p(price)` |

---

## 📂 Project Structure

```
Car_price_dataset/
├── app.py                     # Streamlit frontend (4 pages)
├── preprocess_train.py        # Data cleaning + model training
├── car_price_prediction.csv   # Dataset (8,128 records)
├── requirements.txt           # Python dependencies
├── run_app.bat                # One-click Windows launcher
└── model/
    ├── car_price_model.pkl    # Trained Random Forest model
    └── feature_meta.pkl       # Encoders + feature metadata
```

---

## 📊 Dataset

| Column | Description |
|---|---|
| `name` | Car brand and model |
| `year` | Year of manufacture |
| `selling_price` | Target — price in INR |
| `km_driven` | Total kilometres driven |
| `fuel` | Diesel / Petrol / CNG / LPG |
| `seller_type` | Individual / Dealer / Trustmark |
| `transmission` | Manual / Automatic |
| `owner` | First / Second / Third / Fourth+ |
| `mileage` | Fuel efficiency (km/ltr/kg) |
| `engine` | Engine displacement (cc) |
| `max_power` | Maximum power (bhp) |
| `seats` | Number of seats |

- **Total Records:** 8,128
- **Year Range:** 1983 – 2020
- **Price Range:** INR 29,999 – INR 1,00,00,000

---

## 🏆 Top Feature Importances

| Feature | Importance |
|---|---|
| Car Age | 47.4% |
| Max Power (bhp) | 38.4% |
| Engine CC | 6.8% |
| KM Driven | 2.7% |
| Mileage | 2.6% |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend UI | Streamlit |
| Charts | Plotly Express + Graph Objects |
| ML Model | scikit-learn (RandomForestRegressor) |
| Data Processing | Pandas, NumPy |
| Model Saving | Joblib |
| Trendlines | statsmodels |
| Language | Python 3.x |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/car-price-prediction.git
cd car-price-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python preprocess_train.py
```

Expected output:
```
Loaded 8,128 rows x 12 columns
After cleaning: 8,128 rows
-- Model Performance ------------------
  R2  (log scale) : 0.9319
  MAE (INR)       : 66,729
---------------------------------------
Saved model  -> model\car_price_model.pkl
Saved meta   -> model\feature_meta.pkl
```

### 4. Launch the App
```bash
# Option A — if streamlit is on PATH
streamlit run app.py

# Option B — using Python module
python -m streamlit run app.py

# Option C — Windows only (double-click)
run_app.bat
```

### 5. Open in Browser
```
http://localhost:8501
```

---

## 💡 Key Insights

- 🏎️ **Car Age** is the #1 price driver (47.4% importance) — newer cars are significantly more expensive
- ⚡ **Max Power** is #2 (38.4%) — higher bhp cars command premium prices
- 🛢️ **Diesel cars** average INR 7.91L vs INR 4.62L for petrol — a **71% premium**
- 🔄 **Automatic cars** are 4x more expensive than manual on average
- 👤 **First-owner cars** are worth 2x more than second-owner cars
- 🏅 **Maruti** dominates with 30% of all listings

---

## 📦 Requirements

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
statsmodels>=0.14.0
```

---

## 🙌 Acknowledgements

- Dataset sourced from [Kaggle — Car Price Prediction](https://www.kaggle.com/)
- Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> Made with ❤️ using Python, Streamlit & Random Forest
