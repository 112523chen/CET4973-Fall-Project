import streamlit as st 
import pandas as pd 
import numpy as np
import pickle
from src.helpers.functions import *

data = pd.read_csv("src/data/listings_Sep_22.csv", low_memory=False)
model = pickle.load(open('./src/model/reducedLinearRegressionModel.pkl','rb'))
not_important_features = open("./src/helpers/notImportantFeatures.txt", "r").read().split("\n")[:-1]
important_features = open("./src/helpers/importantFeatures.txt", "r").read().split("\n")[:-1]
important_columns = ['host_location','host_response_time','host_response_rate','neighbourhood_cleansed','neighbourhood_group_cleansed','latitude','longitude','property_type','room_type','accommodates','bedrooms','beds','instant_bookable',]
df = create_df(data, important_columns, True)

prediction = None

st.set_page_config(
    page_title="Airbnb Listing Price Predictor",
    layout="wide"
)

c1, c2, c3 = st.columns(3, gap='medium')

with c1:
    st.header("Listing Features Info")
    propertyType = st.selectbox("Property Type",data.property_type.value_counts().index)
    roomType = st.selectbox("Room Type", data.room_type.value_counts().index)
    maxCapacity = st.number_input("Max Capacity of Listing", int(df['accommodates'].min()), int(df['accommodates'].max()), 1, 1)
    numberOfBedrooms = st.number_input("Number of Bedroom(s)",int(df['bedrooms'].min()), int(df['bedrooms'].max()), step=1)
    numberOfBeds = st.number_input("Number of Bed(s)", int(df['beds'].min()), int(df['beds'].max()), step=1)
    isInstantBookable = st.checkbox("Will be Instant Bookable")

with c2:
    st.header("Listing Location Info")
    borough = st.selectbox("Borough", data.neighbourhood_group_cleansed.value_counts().index)
    neighborhood = st.selectbox("Neighborhood", data[data.neighbourhood_group_cleansed == borough].neighbourhood_cleansed.value_counts().index)
    lat = (data.groupby('neighbourhood_cleansed')['latitude'].get_group(neighborhood).mean())
    log = (data.groupby('neighbourhood_cleansed')['longitude'].get_group(neighborhood).mean())
    st.header("Host Info")
    hostLocation = st.text_input("Host Location")
    hostResponseTime = st.selectbox("Host Response Time", data.host_response_time.value_counts().index)
    hostResponseRate = st.number_input('Host Response Rate Percentage', 0, 100, 90, 1)

with c3:
    st.header("Prediction")
    if st.button("Make Prediction"):
        input_array = np.array([hostLocation, hostResponseTime, hostResponseRate, borough, neighborhood, lat, log, propertyType, roomType, maxCapacity, numberOfBedrooms, numberOfBeds, isInstantBookable], dtype=object)
        prediction = get_prediction(model, data, important_columns, important_features, input_array)[0]
        st.header(f"The predicted price for this listing for one day is ${round(prediction, 2)}")