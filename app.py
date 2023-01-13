import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np
import xgboost
from xgboost import XGBRegressor
import base64

pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = ['Australia',
         'India',
         'Bangladesh',
         'New Zealand',
         'South Africa',
         'England',
         'West Indies',
         'Afghanistan',
         'Pakistan',
         'Sri Lanka']

cities = ['Colombo',
          'Mirpur',
          'Johannesburg',
          'Dubai',
          'Auckland',
          'Cape Town',
          'London',
          'Pallekele',
          'Barbados',
          'Sydney',
          'Melbourne',
          'Durban',
          'St Lucia',
          'Wellington',
          'Lauderhill',
          'Hamilton',
          'Centurion',
          'Manchester',
          'Abu Dhabi',
          'Mumbai',
          'Nottingham',
          'Southampton',
          'Mount Maunganui',
          'Chittagong',
          'Kolkata',
          'Lahore',
          'Delhi',
          'Nagpur',
          'Chandigarh',
          'Bangalore',
          'St Kitts',
          'Cardiff',
          'Christchurch',
          'Trinidad']

@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("img_1.png")

# st.title("T20 Score Predictor")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"]>.main{{
background-image: url("data:image/png;base64,{img}");
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>T20 Score Predictor</h1>", unsafe_allow_html=True)
with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select Bowling Team', sorted(teams))

city = st.selectbox('Select City', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs Done(Over>5)')
with col5:
    wickets = st.number_input('Wickets Out')

last_five = st.number_input('Runs Scored in last 5 Overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [city],
         'current_score': [current_score], 'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr],
         'last_five': [last_five]})
    result = pipe.predict(input_df)
    st.header("Predicted Score : " + str(int(result[0])))
