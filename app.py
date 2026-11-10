
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AMP Prediction System",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Antimicrobial Peptide Prediction")
st.write(
    "A machine learning application for classifying peptide "
    "sequences using an optimised Random Forest model."
)

st.divider()

st.subheader("Enter Peptide Sequence")

sequence = st.text_area(
    "Peptide sequence",
    placeholder="Example: GWLIRGAIHAGKAIHGLIHRRRH",
    height=120
)

if st.button("Predict AMP", type="primary"):

    if not sequence.strip():
        st.warning("Please enter a peptide sequence.")

    else:
        with st.spinner("Analysing peptide sequence..."):

            try:
                response = requests.post(
                    API_URL,
                    json={"sequence": sequence},
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("Prediction completed!")

                    st.subheader("Prediction Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            "Classification",
                            result["prediction"]
                        )

                    with col2:
                        st.metric(
                            "AMP-class score",
                            f'{result["amp_class_score"] * 100:.2f}%'
                        )

                    st.progress(result["amp_class_score"])

                    st.write("**Input sequence:**", result["sequence"])
                    st.write("**Model:**", result["model"])
                    st.write("**Features used:**", result["feature_count"])

                    st.info(result["interpretation"])

                else:
                    try:
                        detail = response.json().get("detail", "Unknown error")
                    except ValueError:
                        detail = response.text

                    st.error(f"API error: {detail}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the API. "
                    "Please make sure FastAPI is running."
                )

            except requests.exceptions.RequestException as error:
                st.error(f"Request failed: {error}")

st.divider()

st.caption(
    "Research prototype | Predictions are computational estimates "
    "and require experimental validation."
)
