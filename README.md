# 🏠 Boston House Price Predictor

A machine learning project that predicts the price of a house in the Boston area using a **Decision Tree Regressor**, with an interactive **Streamlit** web app to explore how each feature affects the price.

---

## 📌 Overview

The goal of this project is to build a model that estimates the median value of a home from three simple features, and to understand the model's behavior through performance analysis (learning curves, complexity curves, and grid search).

**Dataset:** Boston Housing (cleaned version, 489 homes)

| Feature | Description |
|---|---|
| `RM` | Average number of rooms per home |
| `LSTAT` | Percentage of lower-income residents in the neighborhood |
| `PTRATIO` | Number of students per teacher in nearby schools |
| `MEDV` (target) | Median home price |

---

## 🔍 What was done

1. **Data exploration:** min, max, mean, median, and standard deviation of prices.
   - Min: $105,000 | Max: $1,024,800
   - Mean: $454,343 | Median: $438,900
2. **Feature analysis:** correlation with price
   - `RM`: +0.70 (more rooms, higher price)
   - `LSTAT`: -0.76 (higher poverty, lower price)
   - `PTRATIO`: -0.52 (more students per teacher, lower price)
3. **Performance metric:** R² score (`sklearn.metrics.r2_score`).
4. **Train/test split:** 80% training, 20% testing, shuffled with a fixed `random_state`.
5. **Learning curves:** effect of training set size for `max_depth` = 1, 3, 6, 10.
6. **Complexity curve:** effect of model complexity (`max_depth` from 1 to 10) to study the bias-variance tradeoff.
7. **Model tuning:** Grid Search with `ShuffleSplit` cross-validation over `max_depth`.
   - Optimal `max_depth` = **5**
8. **Predictions:** price estimates for three sample clients.

### Bias vs. Variance

| max_depth | Behavior |
|---|---|
| 1 | High bias (underfitting): low training and validation scores |
| 4-5 | Best balance between bias and variance |
| 10 | High variance (overfitting): training score near 1.0, validation score much lower |

### Sample predictions

| Client | RM | LSTAT | PTRATIO | Predicted price |
|---|---|---|---|---|
| 1 | 5 | 17% | 15 | $324,450 |
| 2 | 4 | 32% | 22 | $287,100 |
| 3 | 8 | 3% | 12 | $927,500 |

---

## 🖥️ Streamlit App

An interactive web app where you can:

- Adjust the house features from the **sidebar**
- See the **estimated price** update instantly
- Compare the estimate with the **average market price**
- View **charts** showing how each feature affects the price (the other two features are held fixed)

---

## 📁 Project Structure

```
.
├── notebook.ipynb          # Analysis, training, and evaluation
├── visuals.py              # Helper functions for learning/complexity curves
├── housing.csv             # Dataset
└── house_app/
    ├── app.py              # Streamlit app
    ├── house_model.pkl     # Trained model
    └── .streamlit/
        └── config.toml     # Light theme settings
```

> Adjust the file names above to match your repository.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/abdogad5100/machine_learning-project.git
cd machine_learning-project
```

### 2. Create an environment and install dependencies

```bash
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS / Linux

pip install numpy pandas scikit-learn matplotlib joblib streamlit plotly
```

### 3. Run the notebook (optional)

Open the notebook in Jupyter or VS Code to reproduce the analysis, then save the model:

```python
import joblib
joblib.dump(reg, "house_app/house_model.pkl")
```

### 4. Launch the app

```bash
cd house_app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🛠️ Tech Stack

- Python
- NumPy, Pandas
- scikit-learn
- Matplotlib, Plotly
- Streamlit

---

## ⚠️ Disclaimer

The data comes from the 1970s Boston housing market. This project is for **learning purposes only** and should not be used to price real homes.

---

## 👤 Author

**Abdelrahman Amr Gad**

