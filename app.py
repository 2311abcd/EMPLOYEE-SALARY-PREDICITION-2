import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_model.pkl")

# App config
st.set_page_config(page_title="💼 Employee Salary Predictor", layout="centered")

# Custom background and styling
st.markdown("""
    <style>
        body {
            background-color: #87CEFA; /* Sky Blue */
        }
        .stApp {
            background-color: #87CEFA;
        }
        .css-18e3th9 {
            padding: 2rem;
        }
        .css-1d391kg {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .stAlert {
            background-color: #002b5c !important;
            color: #0f5132 !important;
            border: 1.5px solid #0f5132 !important;
            border-radius: 10px;
            padding: 1rem;
            font-weight: 600;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        section[df-testid="stSidebar"] {
            background-color: #002b5c;
            color: white;
            padding: 1rem;
        }
        section[df-testid="stSidebar"] .css-1d391kg {
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #002b5c;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='
        color: #002b5c;
        background-color: #e0f0ff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-family: Arial, sans-serif;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    '>
        🧑‍💼 Employee Salary Prediction
    </h1>
""", unsafe_allow_html=True)

st.markdown("Use this app to predict whether an employee earns **more than $50K** or not based on their profile.")

# Sidebar - Input form
st.sidebar.header("📋 Enter Employee Details")

age = st.sidebar.slider("Age", 18, 90, 30)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov",
    "Without-pay", "Never-worked"
])
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", value=123456)
education = st.sidebar.selectbox("Education", [
    "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc",
    "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"
])
educational_num = st.sidebar.slider("Education Number", 1, 16, 13)
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
    "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
    "Priv-house-serv", "Protective-serv", "Armed-Forces"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
capital_gain = st.sidebar.number_input("Capital Gain", value=0)
capital_loss = st.sidebar.number_input("Capital Loss", value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "Iran", "Other"
])
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Prepare input
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'education': [education],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country],
    'experience': [experience]
})

# Show input df preview
st.markdown("### 🔍 Input Summary")
st.dataframe(input_df.style.set_properties(**{'text-align': 'left'}), use_container_width=True)

# Predict button
if st.button("🔮 Predict Salary"):
    with st.spinner("Predicting..."):
        try:
            prediction = model.predict(input_df)
            if prediction[0] == ">50K":
                st.success("✅ This employee is likely to earn **more than $50K**.")
            else:
                st.info("ℹ️ This employee is likely to earn **$50K or less**.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# Batch Prediction
st.markdown("---")
st.markdown("## 📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV file for batch salary prediction", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("📝 Uploaded df Preview")
        st.dataframe(batch_df.head())

        batch_preds = model.predict(batch_df)
        batch_df['Predicted Salary Class'] = batch_preds

        st.write("✅ Prediction Results")
        st.dataframe(batch_df.head())

        # Download option
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Prediction CSV",
            data=csv,
            file_name="salary_predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

