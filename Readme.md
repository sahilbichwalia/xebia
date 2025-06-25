# 🧠 MediSureAI – Medicine Quality & Safety Classifier

**MediSureAI** is a machine learning-powered solution that predicts whether a medicine is **Safe** or **Not Safe** based on its physical and chemical attributes. It handles missing data smartly using **KNN imputation** and performs feature-based analysis to assist in pharmaceutical quality assurance.

## 📂 Project Structure
```
📁 MediSureAI
│
├── readmegenerate.ipynb          # Jupyter notebook with full EDA and model pipeline
├── medicine_quality_dataset.csv  # Input dataset
├── models/                       # Trained models (optional)
└── README.md                     # Project documentation
```

## 📌 Features
- 🔍 Exploratory Data Analysis (EDA)
- 🧼 Missing value handling using **KNN Imputer**
- 📊 Visualizations: boxplots, violin plots, correlation heatmaps
- 🤖 Classification Models: Logistic Regression, Random Forest, XGBoost
- 📈 Performance metrics: Accuracy, Precision, Recall, F1 Score

## 📊 Dataset Overview
The dataset contains 400+ records of various medicines with attributes such as:
- `Dissolution Rate (%)`
- `Assay Purity (%)`
- `Impurity Level (%)`
- `Storage Temperature`
- `Disintegration Time`
- `Active Ingredient` (categorical)
- `Days Until Expiry`
- `Safe/Not Safe` (Target variable)

## 🔧 How It Works
1. **Load & Analyze** the dataset
2. **Handle Missing Values**:
   - Numerical columns → KNN Imputer
   - Categorical columns → Mode
3. **Encode Categorical Features**
4. **Train Classification Models**
5. **Visualize** distributions and correlations
6. **Evaluate** using key performance metrics

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/sahilbichwalia/MediSureAI.git
cd MediSureAI
```

### 2. Install Dependencies
```bash
pip install pandas numpy seaborn scikit-learn matplotlib
```

### 3. Run the Notebook
Open `readmegenerate.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run all cells step-by-step.

## 📈 Example Visualizations
- Violin plots to compare feature distributions by class
- Heatmaps to understand feature correlation
- t-SNE for 2D visual clustering
- Radar charts for profile comparison

## 🧠 Model Performance (Sample)
| Metric     | Score  |
|------------|--------|
| Accuracy   | 91.3%  |
| Precision  | 88.7%  |
| Recall     | 92.1%  |
| F1 Score   | 90.3%  |

## 4 🧠 Trained Models

This repository includes pre-trained components:

- `best_random_forest_model.pkl` – Serialized Random Forest model trained on the medicine quality dataset
- `label_encoder.pkl` – LabelEncoder instance used to transform target labels (`"Safe"` ↔ `1`, `"Not Safe"` ↔ `0`)

To use them:

```python
import pickle

# Load model
with open("best_random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Example prediction
X_sample = df.drop("Safe/Not Safe", axis=1).iloc[[0]]
y_pred = model.predict(X_sample)
print("Predicted label:", le.inverse_transform(y_pred))
```



## 🤝 Contributors
- **[Sushant Shekhar](https://github.com/Jhasushant99)**
- **[Preet Sharma](https://github.com/ZDannn)**
- **[Bhavya Sharma](https://github.com/bhavyacosmo)**
- **[Ashit Vijay](https://github.com/ashitvijay99)**

## 📜 License
This project is licensed under the **MIT License**. Feel free to use and adapt with credit.
