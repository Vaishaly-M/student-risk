import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Risk Predictor", layout="wide")


@st.cache_resource
def load():
    return joblib.load("models/model.joblib")


b = load()
st.title("🎓 Student Performance Early-Warning System")
st.write("Enter a student's profile to estimate the risk of failing. "
         "This is a decision-support demo, not a replacement for teacher judgement.")

cols = st.columns(3)
row = {}
i = 0
for c in b["num_cols"]:
    lo, hi, med = b["num_ranges"][c]
    row[c] = cols[i % 3].slider(c, lo, hi, med)
    i += 1
for c in b["cat_cols"]:
    row[c] = cols[i % 3].selectbox(c, b["cat_options"][c])
    i += 1

if st.button("Predict risk", type="primary"):
    p_pass = b["pipeline"].predict_proba(pd.DataFrame([row]))[0, 1]
    risk = 1 - p_pass
    st.metric("Estimated risk of failing", f"{risk:.0%}")
    st.progress(float(risk))
    if risk >= 0.5:
        st.error("High risk: consider early support or intervention.")
    else:
        st.success("Lower risk based on the entered profile.")

st.divider()
st.subheader("What drives the predictions?")
st.image("reports/feature_importance.png")