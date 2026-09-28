
import os
import requests
import pandas as pd
import streamlit as st

# Backend API URL
API_URL = os.getenv("API_URL", "http://backend:7860")

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("SuperKart Sales Prediction")
st.write("Predict product-store sales using the trained machine learning model.")

# ============================================================
# Single Prediction
# ============================================================

st.header("Single Prediction")

col1, col2 = st.columns(2)

with col1:
    product_weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=12.0,
        step=0.01
    )

    product_sugar_content = st.selectbox(
        "Product Sugar Content",
        ["Low Sugar", "Regular", "No Sugar"]
    )

    product_allocated_area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        value=0.05,
        step=0.001
    )

    product_mrp = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=150.0,
        step=0.01
    )

    store_size = st.selectbox(
        "Store Size",
        ["Small", "Medium", "High"]
    )

with col2:
    store_location_city_type = st.selectbox(
        "Store Location City Type",
        ["Tier 1", "Tier 2", "Tier 3"]
    )

    store_type = st.selectbox(
        "Store Type",
        [
            "Departmental Store",
            "Supermarket Type1",
            "Supermarket Type2",
            "Supermarket Type3",
            "Food Mart"
        ]
    )

    product_id_char = st.selectbox(
        "Product ID Character",
        ["FD", "DR", "NC"]
    )

    store_age_years = st.number_input(
        "Store Age (Years)",
        min_value=0,
        value=16,
        step=1
    )

    product_type_category = st.selectbox(
        "Product Type Category",
        ["Perishables", "Non Perishables"]
    )


if st.button("Predict Sales"):

    input_data = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar_content,
        "Product_Allocated_Area": product_allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_location_city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Store_Age_Years": store_age_years,
        "Product_Type_Category": product_type_category
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=input_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            prediction = result["predictions"][0]

            st.success(
                f"Predicted Product Store Sales: {prediction:.2f}"
            )

        else:
            st.error(
                f"Prediction failed. API returned status "
                f"{response.status_code}: {response.text}"
            )

    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to the backend API: {e}")


# ============================================================
# Batch Prediction
# ============================================================

st.header("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload SuperKart batch CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    st.write("Uploaded data:")
    st.dataframe(batch_df)

    if st.button("Run Batch Prediction"):

        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv"
                )
            }

            response = requests.post(
                f"{API_URL}/predict_batch",
                files=files,
                timeout=60
            )

            if response.status_code == 200:

                result_df = pd.DataFrame(
                    response.json()
                )

                st.success(
                    f"Batch prediction completed for "
                    f"{len(result_df)} records."
                )

                st.dataframe(result_df)

                csv_data = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_data,
                    file_name="SuperKart_Batch_Predictions.csv",
                    mime="text/csv"
                )

            else:
                st.error(
                    f"Batch prediction failed. API returned "
                    f"status {response.status_code}: "
                    f"{response.text}"
                )

        except requests.exceptions.RequestException as e:
            st.error(
                f"Unable to connect to the backend API: {e}"
            )
