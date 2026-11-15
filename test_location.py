
import streamlit as st
from streamlit_geolocation import streamlit_geolocation

st.title("Campus Location Test")
st.write("Click the location button and allow browser permission.")

location = streamlit_geolocation()

if location and location.get("latitude") is not None:
    st.success("Location received")
    st.write("Latitude:", location["latitude"])
    st.write("Longitude:", location["longitude"])
    st.write("Accuracy:", location.get("accuracy"), "metres")
else:
    st.warning("Location not received yet.")
