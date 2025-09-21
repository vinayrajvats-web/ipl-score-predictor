import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained pipeline (ensure the path is correct)
pipe = pickle.load(open('pipe1.pkl', 'rb'))

teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Kolkata Knight Riders', 'Royal Challengers Bangalore',
    'Mumbai Indians', 'Sunrisers Hyderabad', 'Punjab Kings', 'Gujarat Titans', 'Rajasthan Royals',
    'Lucknow Super Giants'
]

cities = [
    'Mumbai', 'Kolkata', 'Delhi', 'Chennai', 'Hyderabad', 'Jaipur', 'Bangalore', 'Chandigarh',
    'Pune', 'Dubai', 'Ahmedabad', 'Abu Dhabi', 'Bengaluru', 'Sharjah', 'Lucknow', 'Visakhapatnam',
    'Durban', 'Dharamsala', 'Centurion', 'Rajkot', 'Navi Mumbai', 'Mohali', 'Indore', 'Johannesburg',
    'Port Elizabeth', 'Cuttack', 'Ranchi', 'Cape Town', 'Raipur', 'Guwahati', 'Kochi'
]

st.title('IPL Score Predictor')

# Create 2 columns for team selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

# City input
city = st.selectbox('Select City', sorted(cities))

# Create 3 columns for numeric inputs
col3, col4, col5 = st.columns(3)
with col3:
    current_score = st.number_input('Current Score', min_value=0, value=0)
with col4:
    overs = st.number_input('Overs Done (works for overs > 5)', min_value=0.1, value=5.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets Down', min_value=0, value=0)

# Input for Runs Scored in Last 5 Overs
last_five = st.number_input('Runs Scored in Last Five Overs', min_value=0, value=0)

# Prediction button
if st.button('Predict Score'):
    # Validate overs input
    if overs <= 5:
        st.error('Please enter overs > 5.')
    else:
        # Calculate extra features
        ball_left = 120 - (overs * 6)  # Remaining balls
        wicket_left = 10 - wickets     # Remaining wickets
        crr = current_score / overs    # Current Run Rate (CRR)

        # Prepare the data for prediction
        input_data = {
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'overs': [overs],
            'wickets': [wickets],
            'crr': [crr],
            'wicket_left': [wicket_left],
            'ball_left': [ball_left],
            'last_five': [last_five]
        }

        # Convert input data into a DataFrame
        input_df = pd.DataFrame(input_data)

        # Make the prediction
        try:
            result = pipe.predict(input_df)
            st.success(f"The predicted final score is: {result[0]:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
