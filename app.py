import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# Load the pre-trained model
model = joblib.load('best_model.pkl')

# Title for the app
st.title("Autism Risk Prediction")

# Function to take user inputs
def user_input_features():
    age = st.number_input("Age in Months", min_value=0, max_value=72, value=28)
    gender = st.selectbox("Gender", ["M", "F"])
    ethnicity = st.selectbox("Ethnicity", ["Middle Eastern", "White-European", "Hispanic", "Asian", "Others"])
    jaundice = st.selectbox("Born with Jaundice", ["YES", "NO"])
    family_history = st.selectbox("Family Member with ASD", ["YES", "NO"])
    test_completed_by = st.selectbox("Who completed the test?", ["Parent", "Self", "Family Member", "Health Worker", "Other"])
    qchat_score = st.slider("Qchat-10 Score", min_value=0, max_value=10, value=3)

    # Autism-related questions
    q1 = st.selectbox("A1: Does your child look at you when you call his/her name?", ["YES", "NO"])
    q2 = st.selectbox("A2: How easy is it for you to get eye contact with your child?", ["YES", "NO"])
    q3 = st.selectbox("A3: Does your child point to indicate that s/he wants something?", ["YES", "NO"])
    q4 = st.selectbox("A4: Does your child point to share interest with you?", ["YES", "NO"])
    q5 = st.selectbox("A5: Does your child pretend?", ["YES", "NO"])
    q6 = st.selectbox("A6: Does your child follow where you’re looking?", ["YES", "NO"])
    q7 = st.selectbox("A7: If you or someone else is visibly upset, does your child comfort them?", ["YES", "NO"])
    q8 = st.selectbox("A8: Would you describe your child’s first words as unusual?", ["YES", "NO"])
    q9 = st.selectbox("A9: Does your child use simple gestures?", ["YES", "NO"])
    q10 = st.selectbox("A10: Does your child stare at nothing with no apparent purpose?", ["YES", "NO"])

    # Store inputs in a dictionary
    data = {
        'Age in Months': age,
        'Gender': gender,
        'Ethnicity': ethnicity,
        'Jaundice': jaundice,
        'Family_mem_with_ASD': family_history,
        'Who completed the test': test_completed_by,
        'Qchat-10-Score': qchat_score,
        'A1': q1,
        'A2': q2,
        'A3': q3,
        'A4': q4,
        'A5': q5,
        'A6': q6,
        'A7': q7,
        'A8': q8,
        'A9': q9,
        'A10': q10
    }
    return data

# Function to preprocess the input data
def preprocess_input(data):
    # Convert categorical inputs to match model training
    ordinal_encoder = OrdinalEncoder(categories=[
        ['M', 'F'],  # Gender
        ['Middle Eastern', 'White-European', 'Hispanic', 'Asian', 'Others'],  # Ethnicity
        ['YES', 'NO'],  # Jaundice
        ['YES', 'NO'],  # Family history
        ['Parent', 'Self', 'Family Member', 'Health Worker', 'Other'],  # Who completed the test
        ['YES', 'NO'], ['YES', 'NO'], ['YES', 'NO'], ['YES', 'NO'], ['YES', 'NO'],
        ['YES', 'NO'], ['YES', 'NO'], ['YES', 'NO'], ['YES', 'NO'], ['YES', 'NO']
    ])

    # Order features as expected by the model
    feature_values = [[
        data['Gender'], data['Ethnicity'], data['Jaundice'], data['Family_mem_with_ASD'], data['Who completed the test'],
        data['A1'], data['A2'], data['A3'], data['A4'], data['A5'], data['A6'], data['A7'], data['A8'], data['A9'], data['A10']
    ]]

    # Encode categorical features
    categorical_encoded = ordinal_encoder.fit_transform(feature_values)

    # Scale numerical input (age)
    scaler = MinMaxScaler()
    age_scaled = scaler.fit_transform(np.array([[data['Age in Months']]]))

    # Combine preprocessed features
    final_input = np.hstack((categorical_encoded, age_scaled, np.array([[data['Qchat-10-Score']]])))
    return final_input

# Function to predict the outcome
def predict_autism_risk(processed_data):
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[0][1]
    return prediction, probability

# User inputs
input_data = user_input_features()

# When the button is clicked
if st.button('Predict Autism Risk'):
    # Preprocess the input data
    processed_data = preprocess_input(input_data)
    
    # Make the prediction
    prediction, probability = predict_autism_risk(processed_data)

    # Display the result
    if prediction == 1:
        st.write(f"### Prediction: The child is likely to have Autism Spectrum Disorder (ASD).")
    else:
        st.write(f"### Prediction: The child is not likely to have Autism Spectrum Disorder (ASD).")

    st.write(f"### Probability of ASD: {probability * 100:.2f}%")
