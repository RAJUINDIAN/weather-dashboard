import streamlit as st
import requests

st.set_page_config(page_title="Weather Predictor", page_icon="⛅")

st.title("🌦️ Weather Predictor App")
st.write("Enter a city name below to get the current weather details.")

API_KEY = "YOUR_REAL_API_KEY"   # Replace with your key

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    return response.json()

city = st.text_input("Enter City Name")

if st.button("Get Weather"):
    if city.strip() == "":
        st.error("Please enter a valid city name.")
    else:
        data = get_weather(city)
        st.write(data)  # DEBUG: shows actual API response

        if data.get("cod") != 200:
            st.error(f"Error: {data.get('message', 'City not found')}")
        else:
            st.success(f"Weather in {city.title()}")

            temperature = data["main"]["temp"]
            description = data["weather"][0]["description"].title()
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]

            st.metric("🌡️ Temperature (°C)", temperature)
            st.metric("🌥️ Description", description)
            st.metric("💧 Humidity (%)", humidity)
            st.metric("🌬️ Wind Speed (m/s)", wind_speed)
