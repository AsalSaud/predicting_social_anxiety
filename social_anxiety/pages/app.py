import streamlit as st
import requests

st.title("Social Anxiety Predictor")
st.markdown("**Fill in your information below to predict your social anxiety level.**")

st.markdown("### Profile")
profile_option = st.selectbox(
    "Select your profile",
    ('Student', 'Worker')
)
st.markdown("---")

alcohol_consumption = 0
heart_rate = 0
breathing_rate = 0

if profile_option == 'Student':
    age = st.number_input('🧓 &nbsp; Age', min_value=10, max_value=80, value=20)
    gender = st.selectbox('🚻 &nbsp; Gender', ('Male', 'Female', 'Other'))
    occupation = st.selectbox('💼 &nbsp; Occupation', ('Student', 'Worker', 'Unworker', 'Other'),index=0)
    sleep_hours = st.number_input('🛏️ &nbsp; Sleep Hours per Night', min_value=0, max_value=12, value=7)
    physical_activity = st.number_input('🏃‍♂️ &nbsp; Physical Activity (hrs/week)', min_value=0, max_value=20, value=3)
    caffeine_intake = st.number_input('☕️ &nbsp; Caffeine Intake (mg/day)', min_value=0, max_value=500, value=100)
    alcohol_consumption = st.number_input('🍻 &nbsp; Alcohol Consumption (drinks/week) (optional)', min_value=0, max_value=20, value=0)
    smoking = st.selectbox('🚬 &nbsp; Smoking', ('Yes', 'No'),index=0)
    family_history = st.selectbox('🧬 &nbsp; Family History of Anxiety', ('Yes', 'No'),index=0)
    heart_rate = st.number_input('❤️ &nbsp; Heart Rate (bpm) (optional)', min_value=40, max_value=120, value=70)
    breathing_rate = st.number_input('🌬️ &nbsp;  Breathing Rate (breaths/min) (optional)', min_value=10, max_value=30, value=16)
    dizziness = st.selectbox('😵‍💫 &nbsp;  Experience of Dizziness', ('Yes', 'No'),index=1)
    medication = st.selectbox('💊 &nbsp; Currently on Medication', ('Yes', 'No'),index=0)
    therapy_sessions = st.number_input('🧠 &nbsp; Therapy Sessions per Month', min_value=0, max_value=20, value=1)
    recent_life_event = st.selectbox('🔔 &nbsp; Recent Major Life Event', ('Yes', 'No'),index=0)

    stress_level = st.slider('😰 &nbsp; Stress Level (1-10)', 1, 10, 5)
    sweating_level = st.slider('💦 &nbsp; Sweating Level (1-5)', 1, 5, 3)
    diet_quality = st.slider('🥗 &nbsp; Diet Quality (1-10)', 1, 10, 7)

elif profile_option == 'Worker':
    age = st.number_input('🧓 &nbsp; Age', min_value=10, max_value=80, value=30)
    gender = st.selectbox('🚻 &nbsp; Gender', ('Male', 'Female', 'Other'))
    occupation = st.selectbox('💼 &nbsp; Occupation', ('Student', 'Worker', 'Unworker', 'Other'),index=1)
    sleep_hours = st.number_input('🛏️ &nbsp; Sleep Hours per Night', min_value=0, max_value=12, value=7)
    physical_activity = st.number_input('🏃‍♂️ &nbsp; Physical Activity (hrs/week)', min_value=0, max_value=20, value=4)
    caffeine_intake = st.number_input('☕️ &nbsp; Caffeine Intake (mg/day)', min_value=0, max_value=500, value=150)
    alcohol_consumption = st.number_input('🍻 &nbsp; Alcohol Consumption (drinks/week) (optional)', min_value=0, max_value=20, value=3)
    smoking = st.selectbox('🚬 &nbsp; Smoking', ('Yes', 'No'))
    family_history = st.selectbox('🧬 &nbsp; Family History of Anxiety', ('Yes', 'No'))
    heart_rate = st.number_input('❤️ &nbsp; Heart Rate (bpm) (optional)', min_value=40, max_value=120, value=80)
    breathing_rate = st.number_input('🌬️ &nbsp; Breathing Rate (breaths/min) (optional)', min_value=10, max_value=30, value=18)
    dizziness = st.selectbox('😵‍💫 &nbsp; Experience of Dizziness', ('Yes', 'No'))
    medication = st.selectbox('💊 &nbsp; Currently on Medication', ('Yes', 'No'))
    therapy_sessions = st.number_input('🧠 &nbsp; Therapy Sessions per Month', min_value=0, max_value=20, value=1)
    recent_life_event = st.selectbox('🔔 &nbsp; Recent Major Life Event', ('Yes', 'No'))

    stress_level = st.slider('😰 &nbsp; Stress Level (1-10)', 1, 10, 5)
    sweating_level = st.slider('💦 &nbsp; Sweating Level (1-5)', 1, 5, 3)
    diet_quality = st.slider('🥗 &nbsp; Diet Quality (1-10)', 1, 10, 7)

else:
    age = st.number_input('🧓 &nbsp; Age', min_value=10, max_value=80, value=25)
    gender = st.selectbox('🚻 &nbsp; Gender', ('Male', 'Female', 'Other'))
    occupation = st.selectbox('💼 &nbsp; Occupation', ('Student', 'Worker', 'Unworker', 'Other'))
    sleep_hours = st.number_input('🛏️ &nbsp; Sleep Hours per Night', min_value=0, max_value=12, value=7)
    physical_activity = st.number_input('🏃‍♂️ &nbsp; Physical Activity (hrs/week)', min_value=0, max_value=20, value=3)
    caffeine_intake = st.number_input('☕️ &nbsp; Caffeine Intake (mg/day)', min_value=0, max_value=500, value=100)
    alcohol_consumption = st.number_input('🍻 &nbsp; Alcohol Consumption (drinks/week) (optional)', min_value=0, max_value=20, value=2)
    smoking = st.selectbox('🚬 &nbsp; Smoking', ('Yes', 'No'))
    family_history = st.selectbox('🧬 &nbsp; Family History of Anxiety', ('Yes', 'No'))
    heart_rate = st.number_input('❤️ &nbsp; Heart Rate (bpm) (optional)', min_value=40, max_value=120, value=70)
    breathing_rate = st.number_input('🌬️ &nbsp; Breathing Rate (breaths/min) (optional)', min_value=10, max_value=30, value=16)
    dizziness = st.selectbox('😵‍💫 &nbsp; Experience of Dizziness', ('Yes', 'No'))
    medication = st.selectbox('💊 &nbsp; Currently on Medication', ('Yes', 'No'))
    therapy_sessions = st.number_input('🧠 &nbsp; Therapy Sessions per Month', min_value=0, max_value=20, value=2)
    recent_life_event = st.selectbox('🔔 &nbsp; Recent Major Life Event', ('Yes', 'No'))

    stress_level = st.slider('😰 &nbsp; Stress Level (1-10)', 1, 10, 5)
    sweating_level = st.slider('💦 &nbsp; Sweating Level (1-5)', 1, 5, 3)
    diet_quality = st.slider('🥗 &nbsp; Diet Quality (1-10)', 1, 10, 7)

# After all the fields are filled, show the predict button
predict_button = st.button('Get Social Anxiety', use_container_width=True)

if predict_button:
    # Collect all the inputs for prediction
    inputs = {
        'age': age,
        'gender': gender,
        'occupation': occupation,
        'sleep_hours': sleep_hours,
        'physical_activity': physical_activity,
        'caffeine_intake': caffeine_intake,
        'smoking': smoking,
        'family_history': family_history,
        'dizziness': dizziness,
        'medication': medication,
        'therapy_sessions': therapy_sessions,
        'recent_life_event': recent_life_event,
        'stress_level': stress_level,
        'sweating_level': sweating_level,
        'diet_quality': diet_quality
    }

    # Add optional fields if provided by the user
    if alcohol_consumption != 0:
        inputs['alcohol_consumption'] = alcohol_consumption
    if heart_rate != 0:
        inputs['heart_rate'] = heart_rate
    if breathing_rate != 0:
        inputs['breathing_rate'] = breathing_rate

    # Now, you can use the 'inputs' for making the prediction
    try:

        api_url = f'https://socialanxiety-739509359989.europe-west1.run.app/predict?Age={age}&Gender={gender}&Occupation={occupation}&Sleep_Hours={sleep_hours}&Physical_Activity={physical_activity}&Caffeine_Intake={caffeine_intake}&Alcohol_Consumption={alcohol_consumption}&Smoking={smoking}&Family_History={family_history}&Stress_level={stress_level}&Heart_Rate={heart_rate}&Breathing_Rate={breathing_rate}&Sweating_Level={sweating_level}&Dizziness={dizziness}&Medication={medication}&Therapy_sessions={therapy_sessions}&Recent_life_events={recent_life_event}&Diet_Quality={diet_quality}'
        response = requests.get(api_url)
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('anxiety_prediction', 'the result was not found')
            st.success(f'{prediction}')


            if "low anxiety" in prediction.lower():
                st.markdown("🟢 **Great! Your social anxiety seems to be low.** Keep maintaining your well-being, and don’t hesitate to seek support if things ever change. 😊")

            elif "moderate anxiety" in prediction.lower():
                st.markdown("🟠 **Your social anxiety appears to be moderate.** It might help to explore coping strategies like mindfulness, journaling, or speaking with a friend. 💛")

            elif "high anxiety" in prediction.lower():
                st.markdown("🔴 **It seems you're experiencing a high level of social anxiety.** You're not alone — many people go through this. Consider reaching out to someone you trust. 💙")

        else:
            st.error(f'an error occurred while calling: {response.status_code} - {response.text}')
    except Exception as e:
        st.error(f'exception occurred while calling: {str(e)}')
