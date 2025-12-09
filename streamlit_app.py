import os
import requests
import streamlit as st

st.set_page_config(page_title="Weather Predictor", page_icon="⛅")

st.title("🌦️ Weather Predictor App")
st.write("Enter a city name below to get the current weather details.")

# Prefer Streamlit secrets, then environment variable, then allow runtime entry
API_KEY = st.secrets.get("OPENWEATHER_API_KEY") if hasattr(st, "secrets") else None
if not API_KEY:
    API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Allow user to type a key at runtime (hidden input)
api_key_input = st.text_input("OpenWeather API key (leave blank to use configured key)", type="password")
if api_key_input:
    API_KEY = api_key_input.strip()

def get_weather(city, api_key):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError:
        # Return JSON if available (OpenWeather returns JSON error bodies)
        try:
            return resp.json()
        except Exception:
            return {"error": f"HTTP error {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {e}"}

city = st.text_input("Enter City Name")

if st.button("Get Weather"):
    if not city.strip():
        st.error("Please enter a valid city name.")
    elif not API_KEY:
        st.error(
            "No OpenWeather API key provided. "
            "Set OPENWEATHER_API_KEY in Streamlit secrets or as an environment variable, "
            "or enter it in the field above."
        )
    else:
        data = get_weather(city.strip(), API_KEY)

        if data.get("error"):
            st.error(data["error"])
        else:
            # cod can be int or string; convert safely
            try:
                cod = int(data.get("cod", 0))
            except (TypeError, ValueError):
                cod = 0

            if cod != 200:
                message = data.get("message", "Unknown error")
                st.error(f"Error ({cod}): {message}")
                if cod == 401:
                    st.info("401 Unauthorized — check that your API key is correct and active.")
            else:
                st.success(f"Weather in {city.title()}")

                temperature = data["main"]["temp"]
                description = data["weather"][0]["description"].title()
                humidity = data["main"]["humidity"]
                wind_speed = data["wind"]["speed"]

                st.metric("🌡️ Temperature (°C)", f"{temperature:.1f}")
                st.metric("🌥️ Description", description)
                st.metric("💧 Humidity (%)", f"{humidity}%")
                st.metric("🌬️ Wind Speed (m/s)", f"{wind_speed} m/s")
