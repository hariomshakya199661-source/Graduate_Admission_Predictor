#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# **Jamboree wants to help students estimate their chances of admission into top graduate colleges based on academic and profile related factors.**
# 
# **The Goal of this project is to:**  
#     1) Understand which factors influence graduate admission chances.  
#     2) Study relationships among variables such as GRE, GPA, university rating, research experience.  
#     3) Build linear regression model to predict the chances of admit.  
#     4) Validate whether linear regression assumptions are satisfied.  
#     5) Provide actionable business insights and recommendations for Jamboree Education Company.  
#     
# 

# **Importing the required libraries.**  

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

import warnings
warnings.filterwarnings("ignore")


# **Importing the dataset**  

# In[2]:


df = pd.read_csv('Jamboree_Admission.csv')
df.head(2)


# In[3]:


df.tail(2)


# **Basic Data understanding**  

# In[4]:


df.shape


# The dataset contains 500 rows and 9 columns.  
# Each row represents once student applicant.  

# In[5]:


df.columns


# Here we can see that columns like LOR and Chance of Admit contains space at the end, therefore it will a good choice to remove the extra spaces from the column names.  

# In[6]:


df.columns=df.columns.str.strip()
df.columns


# In[7]:


df.info()


# Most columns are numerical.  
# We do not have any nulls in the entire dataset.  
# Serial No. is a unique identifier for each row.  
# 

# In[8]:


df.describe().T # Transpose


# GRE Score ranges from 290 to 340  
# TOEFL Score ranges from 92 to 120  
# CGPA ranges from 6.80 to 9.92  
# Chance of admit ranges from 0.34 to 0.97  
# SOP stands for Statement of Purpose  
# LOR stands for Letter of Recommendation  
# 
# **Column Profiling:**
# 
# Serial No. (Unique row ID)  
# GRE Scores (out of 340)  
# TOEFL Scores (out of 120)  
# University Rating (out of 5)  
# Statement of Purpose and Letter of Recommendation Strength (out of 5)  
# Undergraduate GPA (out of 10)  
# Research Experience (either 0 or 1)  0 means NO , and 1 means YES  
# **Continuous Target Variable:** Chance of Admit (ranging from 0 to 1)  

# In[9]:


df.drop('Serial No.',axis=1,inplace=True)


# In[10]:


df.shape


# Dropping the column named Serial No. as it is just a unique identifies and does not contains any useful business information that that may affect the chance of admit.

# In[11]:


df.duplicated().sum() # shows that dataset does not have the duplicates
#df.drop_duplicates(inplace=True) # incase we want to drop duplicates


# In[12]:


df.isnull().sum() # shows that dataset does not have any null values either. this was clear form the .info() method as well.


# **Had there been any nulls, I would have chosen to fill the nulls column wise and not as a whole at once because each column is unique and has different requirements.**    

# **UNIVARIATE ANALYSIS**: Means studying one variable at a time.  

# **The Continuous variables are:**  
# GRE Score, TOEFL Score, SOP, LOR, CGPS, Chance of Admit  

# In[13]:


continuous_cols= df.drop(['Research','University Rating'],axis=1)
continuous_cols
#continuous_cols = df.columns[~df.columns.isin(['Research', 'University Rating'])]


# In[14]:


plt.figure(figsize=(15,12))
for i, col in enumerate(continuous_cols,1):
    plt.subplot(3,2,i)
    sns.histplot(df[col],kde=True)
    plt.title(f"Distribution of {col}")
    
plt.tight_layout()
plt.show()


# **Observations:**  
# GRE Score appears approximately normally distributed but with a very slight skew.  
# TOEFL Score shows most students scoring in the higher range, indicating that many applicants have strong english proficiency.  
# CGPA is concentrated around higher values, suggesting academically strong applicants.  
# Chance of Admit is spread between low and high values, which will help the model to learn from the varied admission outcomes.  
# 
# 

# **Outlier Analysis**

# In[15]:


plt.figure(figsize=(15,10))

for i, col in enumerate(continuous_cols,1):
    plt.subplot(3,2,i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()


# **IQR Method for Outlier Detection**

# In[16]:


for col in continuous_cols:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    
lower_limit=Q1-1.5*IQR
upper_limit=Q3+1.5*IQR

outliers=df[(df[col]<lower_limit)|(df[col]>upper_limit)]

print(col,":",outliers.shape[0],"outliers")


# **Observations**  
# Outliers were checked using boxplot and IQR method.  
# Since admission datasets naturally contain students with very high or very low scores, extreme values were not removed.  

# **BIvariate Analysis**

# In[17]:


predictors = ["GRE Score","TOEFL Score","SOP","LOR","CGPA"]

plt.figure(figsize=(15,12))

for i,col in enumerate(predictors,1):
    plt.subplot(3,2,i)
    
    sns.scatterplot(x=df[col],y=df["Chance of Admit"])
    
    plt.title(f"{col} vs Chance of Admit")
    
plt.tight_layout()
plt.show()


# **Observations**
# 
# GRE Score shows a positive relationship with chance of admit.  
# TOEFL Score also appears positively related to admission chances.  
# CGPA has a strong positive relationship with chance of admit.  
# SOP and LOR show positive relationships, but the trend is weaker as compared to CGPA and GRE.  

# In[18]:


# BOXPLOT: Research vs Chance of Admit

sns.boxplot(x=df["Research"],y=df["Chance of Admit"])

plt.title("Research vs Chance of Admit")
plt.show()


# **Observations**
# Students with research experience generally appear to have higher chances of admission compared to students without research experience.  

# In[19]:


# BOXPLOT: University rating vs Chance of Admit

sns.boxplot(x=df["University Rating"],y=df["Chance of Admit"])

plt.title("University rating vs Chance of Admit")
plt.show()


# **Observations**  
# Students from higher rated universities generally have higher chances of admission. This suggests that university reputation may influence admission outcomes.  
# 

# **Correlation Analysis**

# In[20]:


corr_matrix=df.corr()
plt.figure(figsize=(8,4))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")

plt.show()


# **Observations**
# 
# CGPA shows strong positive correlation with chance of admit.  
# GRE Score and TOEFL score are also positively correlated with chance of admit.  
# GRE score, TOEFL score and CGPA may be correlated with each other, indicating possible multicollinearity.  
# Research has a positive relationship with chance of admit, but the strength may be lower than academic score variables.   

# **Data Preparation for Modeling**

# In[21]:


X=df.drop(columns=["Chance of Admit"])
y=df["Chance of Admit"]


#Train-Test Split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# The data has been split into 80% training data and 20% testing data.  
# The model will learn patters from training data and will be evaluated on the unseen test data.  

# **Multiple Linear Regression**

# In[22]:


# Creating and Model training
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[23]:


# Model Evaluation
#Let's evaluate the model by checking out it's coefficients and how we can interpret them.
print(lm.intercept_)


# In[24]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# **This Tells us How much the predicted target changes when one feature increases by 1 unit, while all other features are kept constant.**  
# If GRE goes from 320 to 321, predicted admit chance increases by about 0.0024.  

# **Predictions from our Model**

# In[25]:


predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)


# In[26]:


sns.distplot((y_test-predictions),bins=50)


# In[27]:


from sklearn import metrics


# In[28]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# **MAE = 0.0427 → on average, predictions are off by about 0.043 units**  
# **MSE = 0.0037 → average squared error is very low (penalizes bigger mistakes more)**  
# **RMSE = 0.0609 → typical prediction error is about 0.061 units**  

# **Linear Regression using StatsModels**  

# In[29]:


X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)


# In[30]:


lr_model = sm.OLS(y_train, X_train_sm).fit()


# In[31]:


print(lr_model.summary())


# In[32]:


coefficients = pd.DataFrame({
    "Feature": X_train_sm.columns,
    "Coefficient": lr_model.params
})

coefficients


# **Multicollinearity Check Using VIF**

# In[33]:


def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    return vif_data.sort_values(by="VIF", ascending=False)


# In[34]:


calculate_vif(X_train)


# In[35]:


X_train_vif = X_train.copy()

while True:
    vif_df = calculate_vif(X_train_vif)
    print(vif_df)
    
    max_vif = vif_df["VIF"].max()
    
    if max_vif > 5:
        feature_to_drop = vif_df.iloc[0]["Feature"]
        print("Dropping:", feature_to_drop)
        X_train_vif = X_train_vif.drop(columns=[feature_to_drop])
    else:
        break


# In[36]:


X_test_vif = X_test[X_train_vif.columns]


# In[37]:


X_train_vif_sm = sm.add_constant(X_train_vif)
X_test_vif_sm = sm.add_constant(X_test_vif)

final_model = sm.OLS(y_train, X_train_vif_sm).fit()
print(final_model.summary())


# VIF was used to detect multicollinearity.  
# Variables with VIF greater than 5 were removed one by one.  
# The final model contains only variables with acceptable VIF values.  

# **Linear Regression Assumption Testing**

# **Assumption 1: Mean of Residuals Should Be Nearly Zero**  

# In[38]:


y_train_pred = final_model.predict(X_train_vif_sm)
residuals = y_train - y_train_pred
residuals.mean()


# **Assumption 2: Linearity of Variables**  

# In[39]:


sns.scatterplot(x=y_train_pred, y=residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()


# **Assumption 3: Normality of Residuals**  

# In[40]:


sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.show()


# **There are slight deviations from normality, but the model is still acceptable for interpretation and prediction.**

# **Model Evaluation**

# In[41]:


y_train_pred = final_model.predict(X_train_vif_sm)
y_test_pred = final_model.predict(X_test_vif_sm)


# In[42]:


def adjusted_r2_score(r2, n, p):
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


# In[43]:


# Train metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
train_adj_r2 = adjusted_r2_score(train_r2, X_train_vif.shape[0], X_train_vif.shape[1])

# Test metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
test_adj_r2 = adjusted_r2_score(test_r2, X_test_vif.shape[0], X_test_vif.shape[1])

metrics = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "MAE": [train_mae, test_mae],
    "RMSE": [train_rmse, test_rmse],
    "R2 Score": [train_r2, test_r2],
    "Adjusted R2": [train_adj_r2, test_adj_r2]
})

metrics


# **Train and test performance are close to each other, indicating that the model is not heavily overfitting.**

# In[44]:


final_coefficients = pd.DataFrame({
    "Feature": X_train_vif_sm.columns,
    "Coefficient": final_model.params
}).sort_values(by="Coefficient", ascending=False)

final_coefficients


# # Actionable Insights

# 1. CGPA is one of the strongest predictors of admission chances.  
# Students with higher undergraduate GPA generally have higher chances of admission.  
# 
# 2. GRE and TOEFL scores show positive relationships with admission chances.  
# This indicates that standardized test performance is still important for graduate admissions.  
# 
# 3. Research experience improves admission chances.  
# Students with research exposure may be perceived as better prepared for graduate-level academic work.  
# 
# 4. University Rating has a positive relationship with admission chances.  
# Applicants from higher-rated universities tend to have better admission outcomes.  
# 
# 5. SOP and LOR are also positively related to admission chances.  
# Strong application documents can support the overall profile of a student.  
# 
# 6. Some independent variables are correlated with each other.  
# For example, students with high CGPA may also have high GRE and TOEFL scores. 
# Therefore, multicollinearity was checked using VIF.  

# # Business Recommendations for Jamboree

# Jamboree can use this model to build a student-facing admission probability calculator where students enter GRE, TOEFL, CGPA, SOP/LOR strength, university rating, and research experience to get an estimated admission chance.  
# Instead of only showing admission probability, Jamboree can show improvement suggestions.  
# For example, if a student's TOEFL score is low, the tool can recommend TOEFL preparation support.  
# If research experience is missing, it can recommend research internships or publication guidance.  
# Students can be segmented into low, medium, and high probability groups.  
# This will help Jamboree provide customized counseling plans.  
# Counselors can use the model output to identify which factor is limiting a student's admission chance.  
# This makes counseling more data-driven and personalized.  

# # END

# **Simple User Input Test**

# # Take user input
# gre = float(input("Enter GRE Score (out of 340): "))
# toefl = float(input("Enter TOEFL Score (out of 120): "))
# univ_rating = float(input("Enter University Rating (1-5): "))
# sop = float(input("Enter SOP strength (1-5): "))
# lor = float(input("Enter LOR strength (1-5): "))
# cgpa = float(input("Enter CGPA (out of 10): "))
# research = int(input("Research Experience (0 or 1): "))
# 
# # Create input dataframe
# user_input = pd.DataFrame({
#     "GRE Score": [gre],
#     "TOEFL Score": [toefl],
#     "University Rating": [univ_rating],
#     "SOP": [sop],
#     "LOR": [lor],
#     "CGPA": [cgpa],
#     "Research": [research]
# })
# 
# # Predict
# prediction = lm.predict(user_input)
# 
# print(f"\nEstimated Chance of Admit: {round(prediction[0]*100, 3)} Percent")

# **User Interface using Streamlit**

# In[45]:


X = df.drop(columns=["Chance of Admit"])
y = df["Chance of Admit"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, y_train)


# In[46]:


print(X_train.columns)


# In[47]:


import pickle

pickle.dump(lm, open("model.pkl", "wb"))


# In[48]:


import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🎓 Graduate Admission Predictor")

st.write("Enter your details to predict admission chances")

# Input fields
gre = st.slider("GRE Score", 260, 340, 300)
toefl = st.slider("TOEFL Score", 80, 120, 100)
univ_rating = st.slider("University Rating", 1, 5, 3)

# ✅ Continuous sliders
sop = st.slider("SOP Strength", 1.0, 5.0, 3.0, step=0.1)
lor = st.slider("LOR Strength", 1.0, 5.0, 3.0, step=0.1)

cgpa = st.slider("CGPA", 6.0, 10.0, 8.0)
research = st.selectbox("Research Experience (0 = No, 1 = Yes)", [0, 1])

# Button
if st.button("Predict Admission Chance"):

    user_input = pd.DataFrame({
        "GRE Score": [gre],
        "TOEFL Score": [toefl],
        "University Rating": [univ_rating],
        "SOP": [sop],
        "LOR": [lor],
        "CGPA": [cgpa],
        "Research": [research]
    })

    prediction = model.predict(user_input)

    # ✅ Convert to percentage
    percentage = prediction[0] * 100

    st.success(f"Estimated Chance of Admit: {round(percentage, 2)}%")

    # ✅ Suggestions (keep in 0–1 scale)
    if prediction[0] < 0.5:
        st.warning("Low chance. Improve CGPA and test scores.")
    elif prediction[0] < 0.75:
        st.info("Moderate chance. Improve SOP and LOR.")
    else:
        st.success("Strong profile! Good chances 🎉")


# In[ ]:




