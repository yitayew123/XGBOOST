import streamlit as st                   # Import Streamlit for creating web apps
import pandas as pd                     # Import pandas for data handling
import numpy as np                      # Import NumPy for numerical operations (not used directly here)
import joblib                          # Import joblib to load saved model files

# List of features expected by the model in correct order for prediction
FEATURES = ['EstimatedSalary', 'Age', 'CreditScore', 'AnnualExpenses',
            'InternetUsagePerDay', 'LoanAmount', 'NumOfDependents', 'EmploymentStatus']

# List of categorical columns that need encoding
categorical_cols = ['EmploymentStatus']

# Use Streamlit cache to load model artifacts only once and reuse, speeding up the app
@st.cache_data(show_spinner=True)
def load_artifacts(path="model_artifacts.pkl"):
    artifacts = joblib.load(path)        # Load the saved artifacts dictionary from file
    return artifacts                     # Return the dictionary (model, scaler, encoders)

# Load model, scaler, and encoders once at app start
artifacts = load_artifacts()
model = artifacts["model"]               # Extract the trained classification model
scaler = artifacts["scaler"]             # Extract the scaler used for numeric features
encoders = artifacts["encoders"]         # Extract label encoders for categorical features

# --- Session State Management ---
# Initialize a counter in session state if not already present (can be used for reset tracking)
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

# Define function to reset all input widgets by setting default values in session state
def reset_inputs():
    """Reset all form input values to their defaults by updating session state."""
    st.session_state.estimated_salary = 50000
    st.session_state.credit_score = 650
    st.session_state.internet_usage = 2.0
    st.session_state.age = 30
    st.session_state.annual_expenses = 20000
    st.session_state.num_dependents = 0
    st.session_state.loan_amount = 15000
    
    # For categorical selectbox, set default based on encoder classes
    le = encoders.get('EmploymentStatus', None)
    if le is not None:
        options = le.classes_
        # Choose 'Unemployed' as default if it exists, else first option
        if 'Unemployed' in options:
            st.session_state.employment_status = 'Unemployed'
        else:
            st.session_state.employment_status = options[0]

# Configure Streamlit app page title and layout
st.set_page_config(page_title="🚗SUV Car Purchase Prediction🚗", layout="wide")

# Add custom CSS styling for background, text, labels, and buttons
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8, #c9d6df);
        background-attachment: fixed;
        color: black;
    }
    .main {
        padding: 2rem 1rem 2rem 1rem;
    }
    h1 {
        color: white !important;
    }
    label, .stSelectbox label, .stNumberInput label {
        color: black !important;
    }
    /* Style for Predict button */
    div.stButton > button:first-child {
        background-color: #003366 !important;
        color: white !important;
        border: 1px solid #003366 !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #005599 !important;
        color: white !important;
    }
    /* Style for Clear button */
    div.stButton > button:nth-child(2) {
        background-color: #d9534f !important;
        color: white !important;
        border: 1px solid #d9534f !important;
    }
    div.stButton > button:nth-child(2):hover {
        background-color: #c9302c !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True               # Allow raw HTML/CSS injection
)

# Display app title and subtitle with styling and center alignment
st.markdown(
    """
    <h1 style='text-align:center; background-color: #003366; padding: 15px; border-radius: 8px;'>
        🚗 SUV Car Purchase Prediction App 🚗
    </h1>
    <p style='text-align:center; font-size:18px; margin-top:10px;'>
        Enter the customer features below to predict if they will purchase an SUV car.
    </p>
    """, unsafe_allow_html=True
)

# Define the function to create input widgets, linked with session state keys for value retention
def user_input_features():
    # Create 3 columns for layout of input widgets
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Estimated Salary input widget with session state key
        st.number_input(
            "Estimated Salary ($)", min_value=0, max_value=1_000_000, value=50000,
            help="Annual estimated salary in dollars.", key="estimated_salary"
        )
        # Credit Score input widget
        st.number_input(
            "Credit Score", min_value=0, max_value=1000, value=650,
            help="Credit score ranging from 0 to 1000.", key="credit_score"
        )
        # Internet Usage per Day input widget
        st.number_input(
            "Internet Usage per Day (hours)", min_value=0.0, max_value=24.0, value=2.0, format="%.2f",
            help="Average daily internet usage in hours.", key="internet_usage"
        )

    with col2:
        # Age input widget
        st.number_input(
            "Age", min_value=0, max_value=120, value=30,
            help="Age of the customer.", key="age"
        )
        # Annual Expenses input widget
        st.number_input(
            "Annual Expenses ($)", min_value=0, max_value=1_000_000, value=20000,
            help="Annual expenses in dollars.", key="annual_expenses"
        )
        # Number of Dependents input widget
        st.number_input(
            "Number of Dependents", min_value=0, max_value=20, value=0,
            help="Number of dependents.", key="num_dependents"
        )

    with col3:
        # Loan Amount input widget
        st.number_input(
            "Loan Amount ($)", min_value=0, max_value=1_000_000, value=15000,
            help="Current loan amount in dollars.", key="loan_amount"
        )
        
        # Employment Status selectbox widget with options from label encoder classes
        le = encoders.get('EmploymentStatus', None)
        if le is not None:
            options = le.classes_
            st.selectbox(
                "Employment Status", options,
                help="Employment status of the customer.", key="employment_status"
            )
        else:
            st.error("EmploymentStatus encoder not found.")

# Call the function to display input widgets on the page
user_input_features()

# Create two columns for placing Predict and Clear buttons side by side
col_pred, col_clear = st.columns([1, 1])

# Place Predict button in the first column and store its clicked state
with col_pred:
    predict_button = st.button("Predict")

# Place Clear button in the second column, which calls reset_inputs function on click
with col_clear:
    clear_button = st.button("Clear", on_click=reset_inputs)

# If the Predict button was clicked, proceed to prediction logic
if predict_button:
    # Build input dictionary from current session state values
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
    # Convert input data to a single-row DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Display input data before encoding
    st.markdown("### Input Data (Before Encoding)")
    st.dataframe(input_df)

    # Encode categorical columns using saved label encoders
    for col in categorical_cols:
        # Safety check if column exists
        if col not in input_df.columns:
            st.error(f"Error: Expected column '{col}' not found in input data.")
            st.stop()  # Stop execution if error
        le = encoders[col]
        # Transform categorical string value to numeric label
        input_df[col] = le.transform(input_df[col].astype(str))

    # Reorder columns to the order expected by the model
    input_df = input_df[FEATURES]

    # Scale numeric features using the saved scaler
    input_scaled = scaler.transform(input_df)

    # Make prediction (0 or 1)
    prediction = model.predict(input_scaled)[0]

    # Get confidence score for predicted class
    prediction_proba = model.predict_proba(input_scaled)[0][prediction]

    # Add a horizontal separator line
    st.markdown("---")
    # Add section header for results
    st.subheader("Prediction Results")

    # Display prediction results with styled messages
    if prediction == 1:
        # Green success box for "Purchased: Yes"
        st.markdown(
            f"""
            <div style='background-color:#d4edda; padding:20px; border-radius:10px; text-align:center; color: black;'>
            <h2 style='color:#155724;'>✅ Purchased: Yes</h2>
            <p style='font-size:20px; color:green;'>Confidence Score: <strong>{prediction_proba:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        # Red error box for "Purchased: No"
        st.markdown(
            f"""
            <div style='background-color:#f8d7da; padding:20px; border-radius:10px; text-align:center; color: black;' >
            <h2 style='color:#721c24;'>❌ Purchased: No</h2>
            <p style='font-size:20px; color:green;'>Confidence Score: <strong>{prediction_proba:.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True
        )
