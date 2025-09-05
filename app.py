# ================== IMPORTS ==================
import streamlit as st                  # Web app framework for building interactive UI
import pandas as pd                     # Data manipulation and handling
import joblib                           # For loading pre-trained models and preprocessing artifacts

# ================== CONFIGURATION ==================
# List of features expected by the model for prediction
FEATURES = [
    'EstimatedSalary', 'Age', 'CreditScore', 'AnnualExpenses',
    'InternetUsagePerDay', 'LoanAmount', 'NumOfDependents', 'EmploymentStatus'
]  

# Columns that are categorical and need encoding before prediction
categorical_cols = ['EmploymentStatus']  

# Set Streamlit page configuration (title and layout)
st.set_page_config(page_title="🚗 EV Car Purchase Prediction 🚗", layout="wide")

# ================== LOAD MODEL ARTIFACTS ==================
@st.cache_resource(show_spinner=True)  
def load_artifacts(path="model_artifacts.pkl"):
    """
    Load model artifacts including trained model, scaler, and encoders from a pickle file.
    Caching ensures the artifacts are loaded only once per session for efficiency.
    """
    artifacts = joblib.load(path)  # Load the pickle file containing model & preprocessing objects
    return artifacts

# Load the artifacts
artifacts = load_artifacts()
model = artifacts.get("model", None)        # The trained XGBoost model
scaler = artifacts.get("scaler", None)      # Scaler for feature normalization
encoders = artifacts.get("encoders", {})    # Dictionary of label encoders for categorical columns

# Check if essential artifacts are missing and stop the app if so
if model is None or scaler is None or not encoders:
    st.error("Model artifacts are missing or incomplete. Please check your file.")
    st.stop()

# ================== SESSION STATE ==================
# Initialize a session state variable to track resets
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0  

def reset_inputs():
    """
    Reset all input fields in the Streamlit form to their default values.
    """
    st.session_state.estimated_salary = 50000
    st.session_state.credit_score = 650
    st.session_state.internet_usage = 2.0
    st.session_state.age = 30
    st.session_state.annual_expenses = 20000
    st.session_state.num_dependents = 0
    st.session_state.loan_amount = 15000

    # Set default employment status if encoder exists
    le = encoders.get('EmploymentStatus')
    if le is not None:
        options = le.classes_
        st.session_state.employment_status = (
            'Unemployed' if 'Unemployed' in options else options[0]
        )

# ================== CUSTOM STYLES ==================
# Apply custom CSS styles to improve app appearance
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8, #c9d6df);
        background-attachment: fixed;
        color: black;
    }
    .main { padding: 2rem 1rem; }
    h1 { color: white !important; }
    label, .stSelectbox label, .stNumberInput label { color: black !important; }
    div.stButton > button:first-child {
        background-color: #003366 !important;
        color: white !important;
        border: 1px solid #003366 !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #005599 !important;
    }
    div.stButton > button:nth-child(2) {
        background-color: #d9534f !important;
        color: white !important;
        border: 1px solid #d9534f !important;
    }
    div.stButton > button:nth-child(2):hover {
        background-color: #c9302c !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== HEADER ==================
# Render the app title and description
st.markdown(
    """
    <h1 style='text-align:center; background-color: #003366; padding: 15px; border-radius: 8px;'>
        🚗 EV Car Purchase Prediction App 🚗
    </h1>
    <p style='text-align:center; font-size:18px; margin-top:10px;'>
        Enter the customer features below to predict if they will purchase an SUV car.
    </p>
    """,
    unsafe_allow_html=True
)

# ================== INPUT FORM ==================
def user_input_features():
    """
    Render input fields for the user to enter customer features.
    Splits the form into three columns for better layout.
    """
    col1, col2, col3 = st.columns(3)  # Split the input form into 3 columns

    with col1:
        st.number_input("Estimated Salary ($)", 0, 1_000_000, 50000, key="estimated_salary")
        st.number_input("Credit Score", 0, 1000, 650, key="credit_score")
        st.number_input("Internet Usage per Day (hours)", 0.0, 24.0, 2.0, format="%.2f", key="internet_usage")

    with col2:
        st.number_input("Age", 0, 120, 30, key="age")
        st.number_input("Annual Expenses ($)", 0, 1_000_000, 20000, key="annual_expenses")
        st.number_input("Number of Dependents", 0, 20, 0, key="num_dependents")

    with col3:
        st.number_input("Loan Amount ($)", 0, 1_000_000, 15000, key="loan_amount")
        le = encoders.get('EmploymentStatus')  # Load the encoder for EmploymentStatus
        if le is not None:
            st.selectbox("Employment Status", le.classes_, key="employment_status")
        else:
            st.error("EmploymentStatus encoder not found.")

# Call function to render the input form
user_input_features()

# ================== BUTTONS ==================
# Create two buttons: Predict and Clear
col_pred, col_clear = st.columns([1, 1])
with col_pred:
    predict_button = st.button("Predict")  # Button to trigger prediction
with col_clear:
    clear_button = st.button("Clear", on_click=reset_inputs)  # Button to reset input fields

# ================== PREDICTION ==================
if predict_button:
    # Create a DataFrame from the user inputs
    input_data = { 
        'EstimatedSalary': st.session_state.estimated_salary,
        'Age': st.session_state.age,
        'CreditScore': st.session_state.credit_score,
        'AnnualExpenses': st.session_state.annual_expenses,
        'InternetUsagePerDay': st.session_state.internet_usage,
        'LoanAmount': st.session_state.loan_amount,
        'NumOfDependents': st.session_state.num_dependents,
        'EmploymentStatus': st.session_state.employment_status
    }
    input_df = pd.DataFrame([input_data])  # Convert to DataFrame

    # Display input data before encoding
    st.markdown("### Input Data (Before Encoding)")
    st.dataframe(input_df)

    # Encode categorical columns
    for col in categorical_cols:
        if col not in input_df.columns:
            st.error(f"Expected column '{col}' not found.")
            st.stop()
        le = encoders[col]
        input_df[col] = le.transform(input_df[col].astype(str))

    # Reorder columns and scale features
    input_df = input_df[FEATURES]
    input_scaled = scaler.transform(input_df)

    # Run prediction
    prediction = model.predict(input_scaled)[0]  # Predicted class (0 or 1)
    prediction_proba = model.predict_proba(input_scaled)[0][prediction]  # Confidence probability

    # ================== RESULTS ==================
    st.markdown("---")
    st.subheader("Prediction Results")

    if prediction == 1:
        # Customer predicted to purchase
        st.markdown(
            f"""
            <div style='background-color:#d4edda; padding:20px; border-radius:10px; text-align:center;'>
            <h2 style='color:#155724;'>✅ Purchased: Yes</h2>
            <p style='font-size:20px; color:green;'>Confidence Score: <strong>{prediction_proba:.2%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Customer predicted not to purchase
        st.markdown(
            f"""
            <div style='background-color:#f8d7da; padding:20px; border-radius:10px; text-align:center;'>
            <h2 style='color:#721c24;'>❌ Purchased: No</h2>
            <p style='font-size:20px; color:red;'>Confidence Score: <strong>{prediction_proba:.2%}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )

# End of app.py thanks to the user for using the app