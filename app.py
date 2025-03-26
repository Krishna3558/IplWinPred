import streamlit as st
import pickle
import pandas as pd

# Define teams and cities
teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bengaluru',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['M Chinnaswamy Stadium',
       'Punjab Cricket Association Stadium, Mohali', 'Feroz Shah Kotla',
       'Wankhede Stadium', 'Eden Gardens', 'Sawai Mansingh Stadium',
       'Rajiv Gandhi International Stadium, Uppal',
       'MA Chidambaram Stadium, Chepauk', 'Dr DY Patil Sports Academy',
       'Newlands', "St George's Park", 'Kingsmead', 'SuperSport Park',
       'Buffalo Park', 'New Wanderers Stadium', 'De Beers Diamond Oval',
       'OUTsurance Oval', 'Brabourne Stadium',
       'Sardar Patel Stadium, Motera', 'Barabati Stadium',
       'Brabourne Stadium, Mumbai',
       'Vidarbha Cricket Association Stadium, Jamtha',
       'Himachal Pradesh Cricket Association Stadium', 'Nehru Stadium',
       'Holkar Cricket Stadium',
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
       'Subrata Roy Sahara Stadium',
       'Maharashtra Cricket Association Stadium',
       'Shaheed Veer Narayan Singh International Stadium',
       'JSCA International Stadium Complex', 'Sheikh Zayed Stadium',
       'Sharjah Cricket Stadium', 'Dubai International Cricket Stadium',
       'Punjab Cricket Association IS Bindra Stadium, Mohali',
       'Saurashtra Cricket Association Stadium', 'Green Park',
       'M.Chinnaswamy Stadium',
       'Punjab Cricket Association IS Bindra Stadium',
       'Rajiv Gandhi International Stadium', 'MA Chidambaram Stadium',
       'Arun Jaitley Stadium', 'MA Chidambaram Stadium, Chepauk, Chennai',
       'Wankhede Stadium, Mumbai', 'Narendra Modi Stadium, Ahmedabad',
       'Arun Jaitley Stadium, Delhi', 'Zayed Cricket Stadium, Abu Dhabi',
       'Dr DY Patil Sports Academy, Mumbai',
       'Maharashtra Cricket Association Stadium, Pune',
       'Eden Gardens, Kolkata',
       'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh',
       'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
       'Rajiv Gandhi International Stadium, Uppal, Hyderabad',
       'M Chinnaswamy Stadium, Bengaluru',
       'Barsapara Cricket Stadium, Guwahati',
       'Sawai Mansingh Stadium, Jaipur',
       'Himachal Pradesh Cricket Association Stadium, Dharamsala',
       'Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur',
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam']

# Load the prediction pipeline
pipeline = pickle.load(open('pipe.pkl', 'rb'))

st.title('IPL Win Predictor')

# Create columns for team selection
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# Select city
selected_city = st.selectbox('Select host city', sorted(cities))

# Target input
target = st.number_input('Target', min_value=1, step=1)

# Overs input with cricket format (e.g., 10.3 means 10 overs, 3 balls -> 10.5)
overs_input = st.text_input('⏳ Overs Completed (e.g., 10.3 for 10 overs 3 balls)', '0.0')

# Convert overs to decimal format (10.3 -> 10.5)
overs = 0
try:
    if '.' in overs_input:
        overs_parts = overs_input.split('.')
        if int(overs_parts[1]) > 6 :
            st.error("One over can't have more than 6 balls")
        else:
            overs = int(overs_parts[0]) + (int(overs_parts[1]) / 6)
    else:
        overs = float(overs_input)
except:
    st.error("Invalid overs format! Please enter in format like '10.3' for 10 overs 3 balls.")

# Score, overs, and wickets columns
col3 , col4 = st.columns(2)

with col3:
    score = st.number_input('Score', min_value=0, step=1)

with col4:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Calculate probabilities
if st.button('Predict Probability'):
    if overs == 0:  # Avoid division by zero
        st.error("Overs completed can't be 0 or some error occured")
    elif batting_team == bowling_team:
        st.error("Batting team and bowling team can't be same")
    else:
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs
        rrr = (runs_left * 6) / balls_left

        # Prepare input DataFrame
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'venue': [selected_city],
            'crr': [crr],
            'rrr': [rrr],
            'curr_score': [score],
            'total_runs_y': [target],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left]
        })

        # Make prediction
        result = pipeline.predict_proba(input_df)
        loss_prob = result[0][0]
        win_prob = result[0][1]

        # Display results
        st.subheader(f"{batting_team} - {round(win_prob * 100)}% Win Probability")
        st.subheader(f"{bowling_team} - {round(loss_prob * 100)}% Win Probability")
