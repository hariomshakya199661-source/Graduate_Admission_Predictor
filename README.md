# 🎓 Graduate Admission Prediction using Linear Regression

A machine learning project that predicts a student's probability of admission to graduate schools based on academic and profile-related factors. 

## 📌 Project Overview

Jamboree Education aims to help students estimate their chances of admission into top graduate colleges. This project analyzes the factors affecting admission probability and builds a Multiple Linear Regression model to predict the **Chance of Admit**.

---

## 🎯 Business Objective

- Identify the factors that influence graduate admission chances.
- Analyze relationships between academic and profile variables.
- Build a Linear Regression model for prediction.
- Validate Linear Regression assumptions.
- Provide business recommendations for Jamboree Education.

---

## 📂 Dataset

The dataset contains **500 student records** with **9 features**, including:

- GRE Score
- TOEFL Score
- University Rating
- SOP Strength
- LOR Strength
- CGPA
- Research Experience
- Chance of Admit (Target Variable)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Streamlit
- Pickle

---

## 📊 Project Workflow

### 1. Data Understanding
- Dataset exploration
- Missing value check
- Duplicate check
- Column profiling

### 2. Exploratory Data Analysis (EDA)
- Univariate Analysis
- Outlier Detection
- Bivariate Analysis
- Correlation Heatmap

### 3. Data Preprocessing
- Removed unnecessary columns
- Train-Test Split (80:20)

### 4. Model Building
- Multiple Linear Regression (Scikit-learn)
- Linear Regression (Statsmodels)

### 5. Assumption Testing
- Mean of Residuals
- Linearity
- Normality of Residuals
- Multicollinearity (VIF)

### 6. Model Evaluation
- MAE
- RMSE
- R² Score
- Adjusted R²

### 7. Deployment
- Saved trained model using Pickle
- Built a simple Streamlit web application

---

## 📈 Key Insights

- CGPA is the strongest predictor of admission chances.
- GRE and TOEFL scores positively influence admission probability.
- Research experience significantly improves admission chances.
- University Rating has a positive impact.
- Strong SOP and LOR also contribute to better admission probability.
- Multicollinearity was detected and handled using VIF analysis.

---

## 💡 Business Recommendations

- Build an admission probability calculator for students.
- Suggest personalized improvements based on predicted admission chances.
- Recommend TOEFL/GRE preparation for students with lower scores.
- Encourage research internships for students without research experience.
- Segment students into Low, Medium, and High probability groups for customized counseling.

---

## 📁 Project Structure

```
Graduate-Admission-Prediction/
│
├── Linear Regression - Jamboree Education Company.ipynb
├── model.pkl
├── app.py                  # Streamlit application
├── README.md
├── requirements.txt
└── dataset/
    └── Jamboree_Admission.csv
```

---



## 📌 Future Improvements

- Compare Linear Regression with Decision Tree, Random Forest, and XGBoost.
- Hyperparameter tuning.
- Deploy the application on Streamlit Cloud.
- Improve UI/UX.
- Add feature importance visualizations.

---

## 👨‍💻 Author

Hariom

If you found this project useful, consider giving it a ⭐ on GitHub!
