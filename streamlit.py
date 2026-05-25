import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle as pk
import joblib


# -------------- Page Configuration ----------------------------------------
st.set_page_config(page_title='Used Car Price AI', layout='wide')

# -------------- Indian Formatter -------------------------------------------
def format_indian(n):
    s = str(int(n))

    if len(s) <= 3:
        return s
    
    last3 = s[-3:]
    rest = s[:-3]

    rest = ",".join([rest[max(i-2,0):i] for i in range(len(rest), 0, -2)][::-1])

    return rest + "," + last3
# ==============================================================================
# STYLE
# ==============================================================================
st.markdown("""
<style>
            
/* Remove default padding */
.block-container {
    padding-top: 5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Banner Styling */
.banner {
    border-radius:15px;
    overflow:hidden;
    box-shadow:0px 6px 20px rgba(0,0,0,0.25);
    margin-bottom:20px;
}
/* Number Input Box */
div[data-baseweb="input"] > div {
    background-color: #f8f9fa;
    border-radius: 10px;
    border: 2px solid #ff4b2b;
    padding: 5px;
}

/* Selectbox Styling */
div[data-baseweb="select"] > div {
    background-color: #f8f9fa;
    border-radius: 10px;
    border: 2px solid #ff416c;
}

/* Hover Effect */
div[data-baseweb="input"] > div:hover,
div[data-baseweb="select"] > div:hover {
    border: 2px solid #ff4b2b;
}

/* Label Styling */
.stNumberInput label,
.stSelectbox label {
    font-weight: 700;
    color: #ff4b2b;
    font-size: 16px;
}
           

</style>
""", unsafe_allow_html=True)


# ----------------------------- Load Models ------------------------------------------------
model = joblib.load('RandomForestRegressor_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# ----------------------------- Load Data --------------------------------------------------
df = pd.read_csv("cleaned_dataset.csv")

# ----------------------------- DropDowns --------------------------------------------------
brands = sorted(df['Brand'].unique())
fuel_types = sorted(df['Fuel_Type'].unique())
transmissions = sorted(df['Transmission_Type'].unique())
locations = sorted(df['Location'].unique())

# ----------------------------- Precompute for Charts ---------------------------------------
@st.cache_data
def load_predictions(df, _model, _preprocessor):
    x = df.drop('Price', axis=1)
    x_final = _preprocessor.transform(x)
    df_copy = df.copy()
    df_copy['Pred_Price'] = _model.predict(x_final)
    return df_copy

df_pred = load_predictions(df, model, preprocessor)

# ============================================================================================
# SIDEBAR
# ============================================================================================
# Sidebar Cardekho Banner image
image = Image.open("Images/cardekho_image.jpg")
st.sidebar.image(image, width=340)

# Sidebar Navigation
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 Prediction", "🚗 Suggestion"]
)

st.sidebar.info("""
This AI model predicts used car prices based on:

• Brand  
• Vehicle Age  
• KM Driven  
• Fuel Type  
• Transmission  
• Engine Capacity  

Built using Machine Learning and deployed with Streamlit.
""")

# ==============================================================================================
# HOME PAGE
# ==============================================================================================

# -------------------------------------------- Page Routing ------------------------------------
if menu == "🏠 Home":
    st.markdown("""
        <style>

        /* Main background */
        .stApp {
            background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
            color: white;
        }

        /* Section cards */
        .card {
            background: rgba(255,255,255,0.05);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0px 6px 20px rgba(0,0,0,0.4);
        }

        /* Highlight text */
        .highlight {
            color: #ff416c;
            font-weight: bold;
        }

        </style>
        """, unsafe_allow_html=True)

    # ------------------------------- Cardekho banner -------------------------------------
    banner = Image.open("Images/banner.png")
    st.image(banner, use_container_width=True)

    # ------------------------------- Project Objective ------------------------------------
    st.markdown("""
    <div class='card'>

    ### 🎯 Project Objective

    The objective of this project is to build a <span class='highlight'>machine learning system that predicts the price of used cars</span>
    based on various vehicle attributes such as brand, model, vehicle age, fuel type, transmission, and kilometers driven.

    This system helps users estimate a <span class='highlight'>fair market price for used cars</span> and supports better buying and selling decisions.

    </div>
    """, unsafe_allow_html=True)

    # ------------------------------ Business Problem -----------------------------------------
    st.markdown("""
    <div class='card'>
                
    ### 💼 Business Problem
                
    The used car market is large and complex. Buyers and sellers often struggle to determine the <span class='highlight'> correct price </span>
    for a vehicle because car prices depend on many factors such as:

    - Vehicle age
    - Brand and model
    - Fuel type
    - Transmission type
    - Kilometers driven
    - Number of owners
    - Location

    Without proper data analysis, pricing decisions are often <span class='highlight'> inaccurate or subjective </span>.
                
    </div>

    """, unsafe_allow_html=True)

    # ------------------------------ Solution -----------------------------------------
    st.markdown("""
    <div class='card'>
                
    ### 💡 Solution
                
    This project uses <span class='highlight'> machine learning algorithms </span> to analyze historical used car data
    and predict the expected market price of a vehicle.

    The model learns patterns from past sales and uses them to estimate prices for new car inputs.
                
    </div>

    """, unsafe_allow_html=True)

    # ------------------------------ Machine Learning Model -----------------------------------------
    st.markdown("""
    <div class='card'>
                
    ### 🤖 Machine Learning Model
                
    The following machine learning techniques were used:

    - Data Cleaning and Feature Engineering
    - One-Hot Encoding for categorical variables
    - Feature Scaling
    - Model Training

    The final prediction model used in this system is:

    <span class='highlight'> Random Forest Regressor </span>

    Random Forest works well for regression problems because it can capture
    non-linear relationships and handle complex feature interactions.
                
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------ Key Features -----------------------------------------
    st.markdown("""
    <div class='card'>
                
    ### ✨ Key Features of the Application
                
    - Predict used car price instantly
    - Market price comparison
    - Car recommendation system
    - Interactive user interface built with Streamlit
    - Data-driven pricing insights
                
    </div>

    """, unsafe_allow_html=True)
    
# ==========================================================================================================
# PREDICTION PAGE
# ==========================================================================================================
elif menu == "🔮 Prediction" :
    
    # ------------------------- Main Layout ----------------------------------------------
    left, space, right = st.columns([1.4,0.1,1.4])
    
    # --------------------------------------- LEFT COLUMN ---------------------------------
    with left:
        st.header("🏷️ Price Prediction and Summary")

        tab_pred, tab_summary = st.tabs([
            "🔮 Price Prediction",
            "📊 Market Summary"
        ])

        # Tab 1: Price Prediction
        with tab_pred:
            col1, col2 = st.columns(2)
            
            # col1 -> selectbox
            with col1:
                brand = st.selectbox("🚗Brand",["Select Brand"] + brands)

                if brand != "Select Brand":
                    models = sorted(df[df['Brand']==brand]['Model'].unique())

                else:
                    models = []

                model_name = st.selectbox("🚙Model",
                                           models if models else ['Select Model'],
                                            disabled=(brand == 'Select Brand'))
                
                fuel_type = st.selectbox("⛽Fuel Type", fuel_types)
                transmission = st.selectbox("⚙️Transmission Type", transmissions)
                seats = st.selectbox("💺Seats", [2,3,4,5,6,7,8,9,10])
                location = st.selectbox("📍Location", locations)

            # col2 -> numer_input
            with col2:
                vehicle_age = st.number_input("📆Vehicle Age(Years)", 0, 50)
                km_driven = st.number_input("🛣️KM Driven", 0)
                mileage = st.number_input("⏱️Mileage(kmpl)", 0.0)
                engine = st.number_input("🦾Engine(CC)", 0)
                max_power = st.number_input("⚡Max Power(bhb)", 0.0)
                owner = st.number_input("👤No Of Owners", 0,5)

            # Prdict button
            predict_btn = st.button("Predict Price")

            
            prediction = None
            predicted_price = None

            if predict_btn:
                input_data = pd.DataFrame([{
                    "Brand" : brand,
                    "Model" : model_name,
                    "Vehicle_Age" : vehicle_age,
                    "KM_Driven" : km_driven,
                    "No_Of_Owners" : owner,
                    "Fuel_Type" : fuel_type,
                    "Transmission_Type" : transmission,
                    "Mileage" : mileage,
                    "Engine_CC" : engine,
                    "Max_Power" : max_power,
                    "No_Of_Seats" : seats,
                    "Location" : location
                }])

                input_final = preprocessor.transform(input_data)
                prediction = model.predict(input_final)[0]
                predicted_price = round(np.expm1(prediction))
                
                
                st.markdown(f"""
                                <div style="
                                background: linear-gradient(90deg,#ff4b2b,#ff416c);
                                padding:0px;
                                border-radius:15px;
                                text-align:center;
                                color:white;
                                font-size:25px;
                                font-weight:bold;
                                box-shadow:0px 8px 20px rgba(0,0,0,0.3);
                                margin-top:20px;
                                ">
                                💰 Estimated Car Price: <br> <span style="color:#1cff57; font-weight:bold;">₹ {format_indian(predicted_price)}</span>
                                </div>
                                """, unsafe_allow_html=True)
                
            # Tab 2: Market Summary        
            with tab_summary:
                # ---------------------- Market Comparison -----------------------------------------
                market_data = df[(df['Brand'] == brand) & (df['Model'] == model_name)]

                if not market_data.empty and market_data['Price'].notna().any():
                    avg_price = round(np.expm1(market_data['Price'].mean()))
                    median_price = round(np.expm1(market_data['Price'].median()))
                else:
                    avg_price = 0
                    median_price = 0

                st.metric("Market Average Price", f"₹{format_indian(avg_price)}")
                st.metric("Market Median Price", f"₹{format_indian(median_price)}")

                # --------------------- Expected Price Range ---------------------------------------
                price_std = market_data['Price'].std() if not market_data.empty else 0
                
                if (prediction is not None) and (not pd.isna(price_std)):
                    lower_price = round(np.expm1(prediction - price_std))
                    upper_price = round(np.expm1(prediction + price_std))

                    st.metric("Expected Price Range", f"₹{format_indian(lower_price)} - ₹{format_indian(upper_price)}")

                else:
                    st.info("Click 'Predict Price' to see the expected price range")

    with right:
        # ---------------------- Similar Cars Dataset ------------------------------------
        similar_cars = df_pred[
            (df_pred['Brand'] == brand) & (df_pred['Model'] == model_name)
        ]

        # ---------------------- Similar Cars with low Km_Driven -------------------------
        low_km_cars = similar_cars.sort_values("KM_Driven").head(5)
        low_km_cars = low_km_cars.copy()
        low_km_cars['Price'] = np.expm1(low_km_cars['Price']).astype(int)

        st.subheader("🚗  Low KM_Driven Cars")
        st.dataframe(low_km_cars[['Brand','Model','KM_Driven','Mileage','Engine_CC','Price']])
        st.markdown("---")

        # ---------------------- Similar Cars with Highest Mileage -------------------------
        high_mileage_cars = similar_cars.sort_values("Mileage", ascending=False).head(5)
        high_mileage_cars = high_mileage_cars.copy()
        high_mileage_cars['Price'] = np.expm1(high_mileage_cars['Price']).astype(int)

        st.subheader("⛽ Best Mileage Cars")

        st.dataframe(
            high_mileage_cars[['Brand','Model','Mileage','KM_Driven','Engine_CC','Price']]
        )
        st.markdown("---")

        if predicted_price is not None:
            similar_cars['price_diff'] = abs(
                np.expm1(similar_cars['Pred_Price']) - predicted_price
            )

            best_value_cars = similar_cars.sort_values("price_diff").head(5)
            best_value_cars = best_value_cars.copy()
            best_value_cars['Predicted_Price'] = np.expm1(best_value_cars['Pred_Price']).astype(int)

            st.subheader("⭐ Best Matching Cars Near Your Predicted Price")

            st.dataframe(
                best_value_cars[['Brand','Model','KM_Driven','Mileage','Engine_CC','Predicted_Price']]
            )

elif menu == "🚗 Suggestion":

    st.markdown("""
    <style>
    .stApp {
            background: linear-gradient(120deg,#0f2027,#203a43,#2c5364);
            color: white;
        }
                
    .card{
    background:#111827;
    padding:20px;
    border-radius:15px;
    box-shadow:0 6px 20px rgba(0,0,0,0.4);
    color:white;
    margin-bottom:20px;
    transition:0.3s;
    }

    .card:hover{
    transform:scale(1.03);
    box-shadow:0 10px 30px rgba(0,0,0,0.6);
    }

    .car-title{
    font-size:20px;
    font-weight:bold;
    color:#FF7000;
    margin-bottom:10px;
    }

    .car-price{
    font-size:18px;
    font-weight:bold;
    color:#00ffcc;
    margin-bottom:10px;
    }

    .car-details{
    font-size:14px;
    line-height:1.6;
    }
                
    /* Tab text color */
    .stTabs [data-baseweb="tab"] {
        color: white;
        font-weight: 600;
    }

    </style>
    """, unsafe_allow_html=True)
    # -------- Card Function --------
    def car_card(row):

        st.markdown(f"""
        <div class="card">

        <div class="car-title">
        🚘 {row['Brand']} {row['Model']}
        </div>

        <div class="car-price">
        💰 ₹ {int(np.expm1(row['Price'])):,}
        </div>

        <div class="car-details">

        ⛽ Fuel : {row['Fuel_Type']} <br>

        ⚙ Transmission : {row['Transmission_Type']} <br>

        🛣 KM Driven : {int(row['KM_Driven']):,} km <br>

        📅 Vehicle Age : {row['Vehicle_Age']} years <br>

        🏎 Engine : {row['Engine_CC']} CC

        </div>

        </div>
        """, unsafe_allow_html=True)

    premium_cars,budget_cars,high_power_cars = st.tabs([
            "🔮 Premium Cars",
            "📊 Budget Cars",
            "⚡High Power Cars"
        ])
    

    with premium_cars:

        # -------- Filter Premium Cars --------
        df['Actual_Price'] = np.expm1(df['Price'])
        premium_df = df.sort_values(by='Actual_Price', ascending=False).head(50)

        # -------- Create 3 Column Layout --------
        cols = st.columns(3)

        # -------- Display Cards --------
        for i, (_, row) in enumerate(premium_df.iterrows()):
            with cols[i % 3]:
                car_card(row)


    with budget_cars:
        budget_df = df.sort_values(by='Actual_Price', ascending=True).head(50)

        # -------- Create 3 Column Layout --------
        cols = st.columns(3)

        # -------- Display Cards --------
        for i, (_, row) in enumerate(budget_df.iterrows()):
            with cols[i % 3]:
                car_card(row)
                

    with high_power_cars:
        power_df = df.sort_values(by='Engine_CC', ascending=False).head(50)

        # -------- Create 3 Column Layout --------
        cols = st.columns(3)

        # -------- Display Cards --------
        for i, (_, row) in enumerate(power_df.iterrows()):
            with cols[i % 3]:
                car_card(row)

    

    








         
