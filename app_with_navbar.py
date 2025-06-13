
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_model
from sklearn.preprocessing import LabelEncoder
from pathlib import Path 


st.set_page_config(page_title="Income Prediction App", layout="wide")

# Navigation bar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

# Load model and accuracy
model, accuracy = train_model()

# Load sample dataset
try:
    df = pd.read_csv("adult_sample.csv")
except FileNotFoundError:
    df = pd.DataFrame()


if page == "Home":
    st.title("Income Classification using Machine Learning")
    st.write("This web application predicts whether an individual's income exceeds $50K/year based on census data.")
    st.write(f"Model Accuracy: **{accuracy*100:.2f}%**")

elif page == "Dataset":
    st.title("Dataset Preview")
    if not df.empty:
        st.write(df.head(20))
        st.write(f"Total Records: {df.shape[0]}")
        st.write(f"Total Features: {df.shape[1]}")
    else:
        st.warning("Dataset not found. Please upload or check the file path.")

elif page == "Summary":
    st.title("Dataset Summary")
    if not df.empty:
        st.write(df.describe(include='all'))
    else:
        st.warning("Dataset not found.")

elif page == "Graphs":
    st.title("Feature Visualizations")
    if not df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["age"], kde=True, bins=30, ax=ax)
        st.pyplot(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.countplot(data=df, x="education", order=df["education"].value_counts().index, ax=ax2)
        plt.xticks(rotation=45)
        st.pyplot(fig2)
    else:
        st.warning("Dataset not available.")

elif page == "Predict":
    st.title("Income Prediction ( >50K or <=50K )")

    st.subheader("Input Features")
    age = st.slider("Age", 17, 90, 30)
    education_num = st.slider("Education Level (numeric)", 1, 16, 10)
    hours = st.slider("Hours per Week", 1, 99, 40)
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)

    workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'State-gov'])
    education = st.selectbox("Education", ['Bachelors', 'HS-grad', 'Some-college'])
    marital_status = st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced'])
    occupation = st.selectbox("Occupation", ['Exec-managerial', 'Craft-repair', 'Adm-clerical'])
    relationship = st.selectbox("Relationship", ['Husband', 'Wife', 'Not-in-family'])
    race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander'])
    sex = st.selectbox("Sex", ['Male', 'Female'])
    native_country = st.selectbox("Native Country", ['United-States', 'India', 'Mexico'])

    input_df = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [100000],  # Placeholder
        'education': [education],
        'education_num': [education_num],
        'marital_status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital_gain': [capital_gain],
        'capital_loss': [capital_loss],
        'hours_per_week': [hours],
        'native_country': [native_country]
    })

    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = LabelEncoder().fit_transform(input_df[col])

    if st.button("Predict Income"):
        prediction = model.predict(input_df)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        st.success(f"Predicted Income: **{result}**")
