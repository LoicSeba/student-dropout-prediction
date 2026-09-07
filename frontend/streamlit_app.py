import json
import os
from pathlib import Path

import labels
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")

DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "models" / "feature_defaults.json"
 
 
def load_defaults() -> dict:
    if DEFAULTS_PATH.exists():
        with open(DEFAULTS_PATH) as f:
            return json.load(f)
    return {}
 
 
def num_default(defaults: dict, key: str, fallback):
    """Get the value from the default dictionary casting it to the correct type using a fallback."""
    value = defaults.get(key)
    if value is None:
        return fallback
    return type(fallback)(value)

dict_defaults = load_defaults()

st.set_page_config(page_title="Student Dropout Prediction", page_icon="🎓", layout="centered")

st.title("🎓 Student Dropout Prediction")
st.caption("Fill in a student's profile to predict Graduate / Dropout / Enrolled.")
if not dict_defaults:
    st.caption("⚠️ No feature_defaults.json found — using placeholder defaults. Run `python src/train.py` to generate it.")

with st.form("student_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        marital_options = labels.format_options(labels.MARITAL_STATUS)
        marital_status_option = st.selectbox(
            "Marital status", marital_options,
            index=labels.default_index(labels.MARITAL_STATUS, marital_options, dict_defaults.get("Marital status"))
        )
        marital_status = labels.code_from_option(marital_status_option)
 
        gender_default = num_default(dict_defaults, "Gender", 1)
        gender = st.selectbox("Gender", [0, 1], index=gender_default, format_func=lambda x: "Female" if x == 0 else "Male")
 
        age_at_enrollment = st.number_input("Age at enrollment", min_value=17, max_value=70, value=num_default(dict_defaults, "Age at enrollment", 20))
 
        nacionality_options = labels.format_options(labels.NACIONALITY)
        nacionality_option = st.selectbox(
            "Nationality", nacionality_options,
            index=labels.default_index(labels.NACIONALITY, nacionality_options, dict_defaults.get("Nacionality"))
        )
        nacionality = labels.code_from_option(nacionality_option)
 
        international_default = num_default(dict_defaults, "International", 0)
        international = st.selectbox("International student", [0, 1], index=international_default, format_func=lambda x: "No" if x == 0 else "Yes")
    with col2:
        displaced = st.selectbox("Displaced", [0, 1], index=num_default(dict_defaults, "Displaced", 0), format_func=lambda x: "No" if x == 0 else "Yes")
        educational_special_needs = st.selectbox("Educational special needs", [0, 1], index=num_default(dict_defaults, "Educational special needs", 0), format_func=lambda x: "No" if x == 0 else "Yes")
        debtor = st.selectbox("Debtor", [0, 1], index=num_default(dict_defaults, "Debtor", 0), format_func=lambda x: "No" if x == 0 else "Yes")
        tuition_fees_up_to_date = st.selectbox("Tuition fees up to date", [0, 1], index=num_default(dict_defaults, "Tuition fees up to date", 1), format_func=lambda x: "No" if x == 0 else "Yes")
        scholarship_holder = st.selectbox("Scholarship holder", [0, 1], index=num_default(dict_defaults, "Scholarship holder", 0), format_func=lambda x: "No" if x == 0 else "Yes")
 
    st.subheader("Academic background")
    col3, col4 = st.columns(2)
    with col3:
        course_options = labels.format_options(labels.COURSE)
        course_option = st.selectbox(
            "Course", course_options,
            index=labels.default_index(labels.COURSE, course_options, dict_defaults.get("Course"), key_is_str=True)
        )
        course = str(labels.code_from_option(course_option))
 
        app_mode_options = labels.format_options(labels.APPLICATION_MODE)
        application_mode_option = st.selectbox(
            "Application mode", app_mode_options,
            index=labels.default_index(labels.APPLICATION_MODE, app_mode_options, dict_defaults.get("Application mode"))
        )
        application_mode = labels.code_from_option(application_mode_option)
 
        application_order = st.number_input("Application order", min_value=0, max_value=9, value=num_default(dict_defaults, "Application order", 1))
        daytime_evening_attendance = st.selectbox("Attendance", [0, 1], index=num_default(dict_defaults, "Daytime/evening attendance", 1), format_func=lambda x: "Evening" if x == 0 else "Daytime")
    with col4:
        prev_qual_options = labels.format_options(labels.PREVIOUS_QUALIFICATION)
        previous_qualification_option = st.selectbox(
            "Previous qualification", prev_qual_options,
            index=labels.default_index(labels.PREVIOUS_QUALIFICATION, prev_qual_options, dict_defaults.get("Previous qualification"))
        )
        previous_qualification = labels.code_from_option(previous_qualification_option)
 
        previous_qualification_grade = st.number_input("Previous qualification grade", min_value=0.0, max_value=200.0, value=num_default(dict_defaults, "Previous qualification (grade)", 150.0))
        admission_grade = st.number_input("Admission grade", min_value=0.0, max_value=200.0, value=num_default(dict_defaults, "Admission grade", 140.0))
 
    st.subheader("Parents")
    col5, col6 = st.columns(2)
    with col5:
        mq_options = labels.format_options(labels.PARENT_QUALIFICATION)
        mothers_qualification_option = st.selectbox(
            "Mother's qualification", mq_options,
            index=labels.default_index(labels.PARENT_QUALIFICATION, mq_options, dict_defaults.get("Mother's qualification"))
        )
        mothers_qualification = labels.code_from_option(mothers_qualification_option)
 
        mo_options = labels.format_options(labels.PARENT_OCCUPATION)
        mothers_occupation_option = st.selectbox(
            "Mother's occupation", mo_options,
            index=labels.default_index(labels.PARENT_OCCUPATION, mo_options, dict_defaults.get("Mother's occupation"))
        )
        mothers_occupation = labels.code_from_option(mothers_occupation_option)
    with col6:
        fq_options = labels.format_options(labels.PARENT_QUALIFICATION)
        fathers_qualification_option = st.selectbox(
            "Father's qualification", fq_options,
            index=labels.default_index(labels.PARENT_QUALIFICATION, fq_options, dict_defaults.get("Father's qualification"))
        )
        fathers_qualification = labels.code_from_option(fathers_qualification_option)
 
        fo_options = labels.format_options(labels.PARENT_OCCUPATION)
        fathers_occupation_option = st.selectbox(
            "Father's occupation", fo_options,
            index=labels.default_index(labels.PARENT_OCCUPATION, fo_options, dict_defaults.get("Father's occupation"))
        )
        fathers_occupation = labels.code_from_option(fathers_occupation_option)
 
    st.subheader("1st semester")
    c1, c2, c3 = st.columns(3)
    with c1:
        cu1_credited = st.number_input("Units credited (S1)", min_value=0, value=num_default(dict_defaults, "Curricular units 1st sem (credited)", 0))
        cu1_enrolled = st.number_input("Units enrolled (S1)", min_value=0, value=num_default(dict_defaults, "Curricular units 1st sem (enrolled)", 6))
    with c2:
        cu1_evaluations = st.number_input("Units evaluations (S1)", min_value=0, value=num_default(dict_defaults, "Curricular units 1st sem (evaluations)", 8))
        cu1_approved = st.number_input("Units approved (S1)", min_value=0, value=num_default(dict_defaults, "Curricular units 1st sem (approved)", 5))
    with c3:
        cu1_grade = st.number_input("Grade (S1)", min_value=0.0, max_value=20.0, value=num_default(dict_defaults, "Curricular units 1st sem (grade)", 13.5))
        cu1_without_eval = st.number_input("Units without evaluation (S1)", min_value=0, value=num_default(dict_defaults, "Curricular units 1st sem (without evaluations)", 0))
 
    st.subheader("2nd semester")
    c4, c5, c6 = st.columns(3)
    with c4:
        cu2_credited = st.number_input("Units credited (S2)", min_value=0, value=num_default(dict_defaults, "Curricular units 2nd sem (credited)", 0))
        cu2_enrolled = st.number_input("Units enrolled (S2)", min_value=0, value=num_default(dict_defaults, "Curricular units 2nd sem (enrolled)", 6))
    with c5:
        cu2_evaluations = st.number_input("Units evaluations (S2)", min_value=0, value=num_default(dict_defaults, "Curricular units 2nd sem (evaluations)", 7))
        cu2_approved = st.number_input("Units approved (S2)", min_value=0, value=num_default(dict_defaults, "Curricular units 2nd sem (approved)", 4))
    with c6:
        cu2_grade = st.number_input("Grade (S2)", min_value=0.0, max_value=20.0, value=num_default(dict_defaults, "Curricular units 2nd sem (grade)", 12.8))
        cu2_without_eval = st.number_input("Units without evaluation (S2)", min_value=0, value=num_default(dict_defaults, "Curricular units 2nd sem (without evaluations)", 0))
 
    st.subheader("Economic context")
    e1, e2, e3 = st.columns(3)
    with e1:
        unemployment_rate = st.number_input("Unemployment rate (%)", value=num_default(dict_defaults, "Unemployment rate", 10.8))
    with e2:
        inflation_rate = st.number_input("Inflation rate (%)", value=num_default(dict_defaults, "Inflation rate", 1.4))
    with e3:
        gdp = st.number_input("GDP", value=num_default(dict_defaults, "GDP", 1.74))
 
    submitted = st.form_submit_button("Predict")


if submitted:
    payload = {
        "Marital status": marital_status,
        "Application mode": application_mode,
        "Application order": application_order,
        "Course": course,
        "Daytime/evening attendance\t": daytime_evening_attendance,
        "Previous qualification": previous_qualification,
        "Previous qualification (grade)": previous_qualification_grade,
        "Nacionality": nacionality,
        "Mother's qualification": mothers_qualification,
        "Father's qualification": fathers_qualification,
        "Mother's occupation": mothers_occupation,
        "Father's occupation": fathers_occupation,
        "Admission grade": admission_grade,
        "Displaced": displaced,
        "Educational special needs": educational_special_needs,
        "Debtor": debtor,
        "Tuition fees up to date": tuition_fees_up_to_date,
        "Gender": gender,
        "Scholarship holder": scholarship_holder,
        "Age at enrollment": age_at_enrollment,
        "International": international,
        "Curricular units 1st sem (credited)": cu1_credited,
        "Curricular units 1st sem (enrolled)": cu1_enrolled,
        "Curricular units 1st sem (evaluations)": cu1_evaluations,
        "Curricular units 1st sem (approved)": cu1_approved,
        "Curricular units 1st sem (grade)": cu1_grade,
        "Curricular units 1st sem (without evaluations)": cu1_without_eval,
        "Curricular units 2nd sem (credited)": cu2_credited,
        "Curricular units 2nd sem (enrolled)": cu2_enrolled,
        "Curricular units 2nd sem (evaluations)": cu2_evaluations,
        "Curricular units 2nd sem (approved)": cu2_approved,
        "Curricular units 2nd sem (grade)": cu2_grade,
        "Curricular units 2nd sem (without evaluations)": cu2_without_eval,
        "Unemployment rate": unemployment_rate,
        "Inflation rate": inflation_rate,
        "GDP": gdp,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(
            "Can't reach the API at localhost:8000. Start it first with:\n\n"
            "`uvicorn api.main:app --reload`"
        )
        st.stop()

    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        probabilities = result["probabilities"]

        icon = {"Graduate": "🎓", "Dropout": "⚠️", "Enrolled": "📘"}.get(prediction, "")
        st.success(f"{icon} Predicted outcome: **{prediction}**")

        st.subheader("Class probabilities")
        st.bar_chart(probabilities)

    elif response.status_code == 422:
        st.error("Invalid input — check the values entered.")
        st.json(response.json())
    else:
        st.error(f"Prediction failed ({response.status_code})")
        st.json(response.json())