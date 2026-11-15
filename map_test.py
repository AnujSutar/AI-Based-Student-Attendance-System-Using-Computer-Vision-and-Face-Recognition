import streamlit as st
import pandas as pd
import pydeck as pdk

lat = 19.13721980926153
lon = 74.14914710929455

st.title("Campus Marker Test")

data = pd.DataFrame([{"latitude": lat, "longitude": lon}])

layer = pdk.Layer(
    "ScatterplotLayer",
    data=data,
    get_position="[longitude, latitude]",
    get_radius=30,
    radius_units="pixels",
    get_fill_color=[255, 0, 0, 255],
    filled=True,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=16,
    ),
)

st.pydeck_chart(deck, height=500)
