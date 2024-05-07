import pandas as pd
import streamlit as st
import pickle
import time

df = pd.read_csv('House_Rent_Dataset.csv')
df = df.drop(['Area Locality', 'Floor', 'Posted On', 'Rent'], axis=1)
mlp_model = pickle.load(open("mlp_model.pkl", "rb"))

def show_output ():
    l = [int(bhk), int(size), area_type, city, furnishing_status, tenant_preferred, int(bathroom), point_of_contact]
    result = process_input(l)
    return result[0]

def process_input (list_input, df=df, mlp_model=mlp_model):
    df_temp = df.copy()
    df_temp.loc[len(df_temp)] = list_input
    df_temp = pd.get_dummies(df_temp)
    return mlp_model.predict(df_temp.tail(1))


html_temp = """
    <div>
    <h1 style="color:#c85103;text-align:center;">Machine Learning Project</h2>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

html_temp2 = """
    <div style="background-color:#013120;padding:10px;border-radius: 10px">
    <h2 style="color:white; text-align:center"> Predicting Rental Rates</h2>
    </div>
    """

st.markdown(html_temp2, unsafe_allow_html=True)

with st.form(key="form"):
    bhk = st.number_input("Please insert the BHK", value=None, placeholder="Type a number...", min_value=1, max_value=6)
    size = st.number_input("Please insert the Size", value=None, placeholder="Type a number...", min_value=10, max_value=8000)
    area_type = st.selectbox("Please select Area Type", ['Super Area', 'Carpet Area', 'Built Area'])
    city = st.selectbox("Please select City", ['Kolkata', 'Mumbai', 'Bangalore', 'Delhi', 'Chennai', 'Hyderabad'])
    furnishing_status = st.selectbox("Please select Furnishing Status", ['Unfurnished', 'Semi-Furnished', 'Furnished'])
    tenant_preferred = st.selectbox("Please select Tenant Preferred", ['Bachelors/Family', 'Bachelors', 'Family'])
    bathroom = st.number_input("Please insert the Number of Bathrooms", value=None, placeholder="Type a number...", min_value=1, max_value=10)
    point_of_contact= st.selectbox("Please select Point of Contact", ['Contact Owner', 'Contact Agent', 'Contact Builder'])
    submit = st.form_submit_button()

if submit:
    start_time = time.time()
    st.write(f"Prediction: {show_output()}")
    print("--- %s seconds ---" % (time.time() - start_time))