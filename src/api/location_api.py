# backend/src/api/location_api.py

from flask import Blueprint, request, jsonify
from geopy.geocoders import Nominatim   # Will not be used now except fallback
import pandas as pd
import os
import requests

# Import safety + weather logic from recommend_api
from api.recommend_api import (
    get_weather,
    get_disaster_status,
    evaluate_site_safety
)

location_api = Blueprint("location_api", __name__)

# Load datasets
base_dir = os.path.dirname(__file__)
sites_path = os.path.join(base_dir, "../../data/heritage_sites.csv")
events_path = os.path.join(base_dir, "../../data/heritage_events.csv")

df_sites = pd.read_csv(sites_path)
df_events = pd.read_csv(events_path)

# Google API Key
GOOGLE_API_KEY = "AIzaSyCmDTmBIMU9QquyjZiYpsgnnQ0mg3QkrwA"

# GOOGLE GEOCODING API
# Convert GPS → Human-readable address
def get_google_address(lat, lon):
    url = (
        f"https://maps.googleapis.com/maps/api/geocode/json?"
        f"latlng={lat},{lon}&key={GOOGLE_API_KEY}"
    )

    res = requests.get(url).json()

    if res.get("status") == "OK" and len(res.get("results")) > 0:
        return res["results"][0]["formatted_address"]

    return "Unknown location"

# GOOGLE DIRECTIONS API
# Calculate ROAD DISTANCE between user and heritage site
def get_road_distance(lat1, lon1, lat2, lon2):
    url = (
        f"https://maps.googleapis.com/maps/api/directions/json?"
        f"origin={lat1},{lon1}&destination={lat2},{lon2}"
        f"&key={GOOGLE_API_KEY}"
    )

    res = requests.get(url).json()

    if res.get("routes"):
        leg = res["routes"][0]["legs"][0]
        distance_km = leg["distance"]["value"] / 1000       # meters → km
        duration_text = leg["duration"]["text"]             # "2 hours 30 min"
        return distance_km, duration_text

    return None, None

# MAIN /location ENDPOINT
@location_api.route("/location", methods=["POST"])
def receive_location():
    data = request.get_json()
    lat = data.get("latitude")
    lon = data.get("longitude")

    if lat is None or lon is None:
        return jsonify({"status": "error", "message": "Missing lat/lon"}), 400

    # Get Accurate Address (Google Geocoding)
    address = get_google_address(lat, lon)

    # Compute ROAD Distance to each heritage site
    results = []

    for _, row in df_sites.iterrows():
        site_id = int(row["id"])
        site_name = row["site"]
        site_lat = float(row["lat"])
        site_lon = float(row["lon"])

        road_km, duration = get_road_distance(lat, lon, site_lat, site_lon)

        if road_km is not None:
            results.append({
                "site_id": site_id,
                "site": site_name,
                "site_lat": site_lat,
                "site_lon": site_lon,
                "road_distance_km": road_km,
                "duration": duration
            })

    # Pick nearest site by ROAD DISTANCE
    nearest = min(results, key=lambda x: x["road_distance_km"])

    # Get events for that site
    related_events = df_events[df_events["site_id"] == nearest["site_id"]].to_dict(orient="records")

    # WEATHER + DISASTER STATUS
    weather = get_weather(nearest["site_lat"], nearest["site_lon"])
    disaster = get_disaster_status(nearest["site_lat"], nearest["site_lon"])
    safety_status = evaluate_site_safety(weather, disaster)

    # Return JSON Response
    return jsonify({
        "status": "success",
        "place_name": address,
        "latitude": lat,
        "longitude": lon,

        "nearest_site": nearest["site"],
        "site_id": nearest["site_id"],
        "site_lat": nearest["site_lat"],
        "site_lon": nearest["site_lon"],

        "road_distance_km": round(nearest["road_distance_km"], 1),
        "duration": nearest["duration"],

        "weather": weather,
        "disaster": disaster,
        "safety_status": safety_status,

        "events": related_events
    })
