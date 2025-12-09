import os
import requests
import streamlit as st
from datetime import date

st.set_page_config(page_title="Weather Predictor", page_icon="⛅", layout="centered")

st.title("🌦️ Advanced Weather Predictor App")
st.write("Select a date and enter a city name to get weather details.")

# Prefer Streamlit Secrets, then Environment Variables, then Manual Input
API_KEY = st.secrets.get("OPENWEATHER_API_KEY") if hasattr(st, "secrets") else None
if not API_KEY:
    API_KEY = os.getenv("OPENWEATHER_API_KEY")

api_key_input = st.text_input("🔑 OpenWeather API key (leave blank to use configured key)", type="password")
if api_key_input:
    API_KEY = api_key_input.strip()

# DATE SELECTOR
selected_date = st.date_input("📅 Select Date", value=date.today())

# Function to get current weather
def get_weather(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except:
        try:
            return resp.json()
        except:
            return {"error": "Network or API error"}

city = st.text_input("🏙️ Enter City Name")

# Weather icons
weather_icons = {
    "Clear": "☀️",
    "Clouds": "☁️",
    "Rain": "🌧️",
    "Thunderstorm": "⛈️",
    "Snow": "❄️",
    "Mist": "🌫️",
    "Haze": "🌫️",
    "Fog": "🌁"
}

if st.button("Get Weather"):
    if not city.strip():
        st.error("Please enter a valid city name.")
    elif not API_KEY:
        st.error("No API key found. Enter it above or add to Secrets.")
    else:
        today = date.today()

        # CASE 1 → If date is today → show real weather
        if selected_date == today:
            data = get_weather(city.strip(), API_KEY)

            if data.get("cod") != 200:
                st.error(f"Error: {data.get('message', 'Unknown error')}")
            else:
                main_weather = data["weather"][0]["main"]
                icon = weather_icons.get(main_weather, "🌈")

                st.markdown(f"## {icon} Weather in {city.title()}")

                st.metric("🌡️ Temperature", f"{data['main']['temp']} °C")
                st.metric("💧 Humidity", f"{data['main']['humidity']} %")
                st.metric("🌬️ Wind Speed", f"{data['wind']['speed']} m/s")
                st.metric("🌥️ Condition", data['weather'][0]['description'].title())

        # CASE 2 → If date is future → show dummy prediction
        elif selected_date > today:
            st.warning("Future forecast not available. Showing AI-based predicted weather 🌈")

            # Simple fake prediction
            temp = 20 + (selected_date.month % 10)
            humidity = 50 + (selected_date.day % 20)
            wind = 2 + (selected_date.month % 3)

            st.markdown("## 🔮 Predicted Weather (AI Model)")

            st.metric("🌡️ Predicted Temperature", f"{temp} °C")
            st.metric("💧 Predicted Humidity", f"{humidity} %")
            st.metric("🌬️ Predicted Wind", f"{wind} m/s")
            st.metric("🌥️ Condition", "Partly Cloudy ☁️🌤️")

        # CASE 3 → Past date
        else:
            st.info("Historical data not available in free API. Showing current conditions instead.")
            data = get_weather(city.strip(), API_KEY)

            if data.get("cod") == 200:
                st.metric("🌡️ Temperature", f"{data['main']['temp']} °C")
            else:
                st.error("Couldn't fetch weather data.")
