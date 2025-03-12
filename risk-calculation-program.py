import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('heart_disease_risk_dataset_earlymed.csv')

# Separate features and target variable
X = data.drop('Heart_Risk', axis=1)
y = data['Heart_Risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Heart Disease Risk Predictor")
st.write("Answer the following questions to assess your risk of heart disease.")

# Collect user input using Streamlit widgets
chest_pain = st.radio("Do you have chest pain?", [1, 0])
shortness_of_breath = st.radio("Do you experience shortness of breath?", [1, 0])
fatigue = st.radio("Do you feel fatigued?", [1, 0])
palpitations = st.radio("Do you have palpitations?", [1, 0])
dizziness = st.radio("Do you feel dizzy?", [1, 0])
swelling = st.radio("Do you have swelling?", [1, 0])
pain_arms_jaw_back = st.radio("Do you have pain in arms, jaw, or back?", [1, 0])
cold_sweats_nausea = st.radio("Do you experience cold sweats or nausea?", [1, 0])
high_bp = st.radio("Do you have high blood pressure?", [1, 0])
high_cholesterol = st.radio("Do you have high cholesterol?", [1, 0])
diabetes = st.radio("Do you have diabetes?", [1, 0])
smoking = st.radio("Do you smoke?", [1, 0])
obesity = st.radio("Are you obese?", [1, 0])
sedentary_lifestyle = st.radio("Do you lead a sedentary lifestyle?", [1, 0])
family_history = st.radio("Do you have a family history of heart disease?", [1, 0])
chronic_stress = st.radio("Do you experience chronic stress?", [1, 0])
gender = st.radio("What is your gender? (1 for Male, 0 for Female)", [1, 0])
age = st.number_input("What is your age?", min_value=1, max_value=120, value=30)

# Predict the risk if the user clicks the button
if st.button("Predict Heart Disease Risk"):
    user_data = pd.DataFrame({
        'Chest_Pain': [chest_pain],
        'Shortness_of_Breath': [shortness_of_breath],
        'Fatigue': [fatigue],
        'Palpitations': [palpitations],
        'Dizziness': [dizziness],
        'Swelling': [swelling],
        'Pain_Arms_Jaw_Back': [pain_arms_jaw_back],
        'Cold_Sweats_Nausea': [cold_sweats_nausea],
        'High_BP': [high_bp],
        'High_Cholesterol': [high_cholesterol],
        'Diabetes': [diabetes],
        'Smoking': [smoking],
        'Obesity': [obesity],
        'Sedentary_Lifestyle': [sedentary_lifestyle],
        'Family_History': [family_history],
        'Chronic_Stress': [chronic_stress],
        'Gender': [gender],
        'Age': [age]
    })
    
    prediction = model.predict(user_data)
    
    if prediction[0] == 1:
        st.error("You are at risk of heart disease. Please consult a doctor.")
    else:
        st.success("You are not at risk of heart disease.")
