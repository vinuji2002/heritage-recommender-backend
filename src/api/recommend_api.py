# backend/src/api/recommend_api.py

from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
import requests
from math import radians, sin, cos, sqrt, atan2

recommend_api = Blueprint("recommend_api", __name__)

# Load model + scaler + dataset only once (FAST - cached)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))    # backend/src/api
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))  # backend/
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
df = pd.read_csv(os.path.join(DATA_DIR, "heritage_clustered.csv"))

# API Keys
GOOGLE_API_KEY = "AIzaSyCmDTmBIMU9QquyjZiYpsgnnQ0mg3QkrwA"
WEATHER_API_KEY = "b611f2e4c820427aa1642014261902"

# Helper: Haversine Distance (Straight Distance)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters

    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)

    a = (sin(dLat / 2) ** 2 +
         cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon / 2) ** 2)

    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# GOOGLE ROAD DISTANCE
def get_road_distance(lat1, lon1, lat2, lon2):
    url = (
        f"https://maps.googleapis.com/maps/api/directions/json?"
        f"origin={lat1},{lon1}&destination={lat2},{lon2}"
        f"&key={GOOGLE_API_KEY}"
    )

    res = requests.get(url).json()

    if res.get("routes"):
        leg = res["routes"][0]["legs"][0]
        distance_km = leg["distance"]["value"] / 1000
        return distance_km

    return None

# WEATHER FUNCTION (WeatherAPI)
def get_weather(lat, lon):
    try:
        url = (
            f"http://api.weatherapi.com/v1/current.json?"
            f"key={WEATHER_API_KEY}&q={lat},{lon}&aqi=no"
        )

        response = requests.get(url)
        data = response.json()

        current = data.get("current", {})

        return {
            "temperature_c": current.get("temp_c"),
            "condition": current.get("condition", {}).get("text"),
            "icon": current.get("condition", {}).get("icon"),
            "wind_kph": current.get("wind_kph"),
            "humidity": current.get("humidity")
        }

    except Exception as e:
        return {
            "temperature_c": None,
            "condition": "Unavailable",
            "icon": None,
            "wind_kph": None,
            "humidity": None
        }

# DISASTER FUNCTION (WeatherAPI alert endpoint)
def get_disaster_status(lat, lon):
    try:
        url = (
            f"http://api.weatherapi.com/v1/forecast.json?"
            f"key={WEATHER_API_KEY}"
            f"&q={lat},{lon}"
            f"&days=1"
            f"&alerts=yes"
        )

        response = requests.get(url)
        data = response.json()

        alerts = data.get("alerts", {}).get("alert", [])

        if not alerts:
            return {
                "has_alert": False,
                "risk_level": "Low",
                "message": None
            }

        # Take first alert
        alert = alerts[0]

        severity = alert.get("severity", "Moderate")
        headline = alert.get("headline")

        if severity.lower() in ["severe", "extreme"]:
            risk = "High"
        elif severity.lower() in ["moderate"]:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "has_alert": True,
            "risk_level": risk,
            "message": headline
        }

    except Exception:
        return {
            "has_alert": False,
            "risk_level": "Unknown",
            "message": None
        }

# SAFETY EVALUATION FUNCTION
def evaluate_site_safety(weather, disaster):

    if disaster and disaster.get("has_alert"):
        risk = disaster.get("risk_level")

        if risk == "High":
            return "UNSAFE"

        if risk == "Medium":
            return "CAUTION"

    if weather:
        condition = (weather.get("condition") or "").lower()
        wind = weather.get("wind_kph") or 0

        # ---- Immediate UNSAFE conditions ----
        if wind > 60:
            return "UNSAFE"

        if any(keyword in condition for keyword in [
            "storm", "thunder", "torrential", "cyclone"
        ]):
            return "UNSAFE"

        # ---- Medium Risk ----
        if wind > 40:
            return "CAUTION"

        if any(keyword in condition for keyword in [
            "heavy rain", "moderate rain"
        ]):
            return "CAUTION"

    return "SAFE"

# Convert Area Name to longitude/latitude
def get_coordinates_from_area(area_name):
    try:
        url = (
            f"https://maps.googleapis.com/maps/api/geocode/json?"
            f"address={area_name}&key={GOOGLE_API_KEY}"
        )

        res = requests.get(url).json()

        if res["results"]:
            location = res["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]

        return None, None

    except Exception:
        return None, None

# Get the place name
def get_place_name(lat, lon):
    try:
        url = (
            f"https://maps.googleapis.com/maps/api/geocode/json?"
            f"latlng={lat},{lon}&key={GOOGLE_API_KEY}"
        )

        res = requests.get(url).json()

        if res["results"]:
            return res["results"][0]["formatted_address"]

        return None

    except Exception:
        return None

# API 1: Predict cluster for user's GPS location
@recommend_api.route("/predict-cluster", methods=["POST"])
def predict_cluster():
    data = request.get_json()
    lat, lon = data.get("lat"), data.get("lon")

    if lat is None or lon is None:
        return jsonify({"error": "lat & lon required"}), 400

    # scale input
    scaled = scaler.transform(np.array([[lat, lon]]))

    # predict cluster
    cluster_id = int(model.predict(scaled)[0])

    return jsonify({"cluster": cluster_id})

# API 2: Get all sites inside a cluster
@recommend_api.route("/cluster-sites/<int:cluster_id>", methods=["GET"])
def cluster_sites(cluster_id):
    sites = df[df["cluster"] == cluster_id][["site_id", "site_name", "lat", "lon"]]

    return jsonify({
        "cluster_id": cluster_id,
        "count": len(sites),
        "sites": sites.to_dict(orient="records")
    })

# API 3: Hybrid Recommendation (ML + Geofence + Events)
@recommend_api.route("/recommend-nearby", methods=["POST"])
def recommend_nearby():
    data = request.get_json()
    lat, lon = data.get("lat"), data.get("lon")

    if lat is None or lon is None:
        return jsonify({"error": "lat & lon required"}), 400

    # 1. Predict ML cluster
    cluster_scaled = scaler.transform(np.array([[lat, lon]]))
    cluster_id = int(model.predict(cluster_scaled)[0])

    # 2. Filter to sites in that cluster
    cluster_subset = df[df["cluster"] == cluster_id]

    # 3. Check geofence & return events for that site
    unique_sites = cluster_subset.drop_duplicates(subset=["site_id"])
    candidates = []

    # 4. Compute haversine for all sites in cluster
    for _, row in unique_sites.iterrows():
        straight_dist = haversine(lat, lon, row["lat"], row["lon"])

        candidates.append({
            "site_id": row["site_id"],
            "site_name": row["site_name"],
            "lat": row["lat"],
            "lon": row["lon"],
            "straight_distance_m": straight_dist
        })

    # 5. Take TOP 5 by straight-line distance
    candidates.sort(key=lambda x: x["straight_distance_m"])
    top_5 = candidates[:5]

    final_results = []

    # 6. Compute ROAD distance + weather only for top 5
    for site in top_5:

        road_km = get_road_distance(lat, lon, site["lat"], site["lon"])

        if road_km is None:
            continue

        weather = get_weather(site["lat"], site["lon"])
        disaster = get_disaster_status(site["lat"], site["lon"])

        site_events = df[df["site_id"] == site["site_id"]][
            ["event_name", "year", "description"]
        ].to_dict(orient="records")

        final_results.append({
            "site_id": site["site_id"],
            "site_name": site["site_name"],
            "road_distance_km": round(road_km, 2),
            "lat": site["lat"],
            "lon": site["lon"],
            "events": site_events,
            "weather": weather,
            "disaster": disaster
        })

    # 7. Re-rank by ROAD distance
    final_results.sort(key=lambda x: x["road_distance_km"])

    # 8. Return TOP 3
    top_3 = final_results[:3]

    # SAFETY EVALUATION FOR TOP 3
    safe_top3 = []
    for site in top_3:
        status = evaluate_site_safety(site["weather"], site["disaster"])
        site["safety_status"] = status

        if status == "SAFE":
            safe_top3.append(site)

    highlight_site_id = None
    alternative_site = None
    recommendation_message = ""

    # STEP 1: IF ANY SAFE IN TOP 3 → STOP
    if safe_top3:
        best_site = min(safe_top3, key=lambda x: x["road_distance_km"])
        highlight_site_id = best_site["site_id"]

        recommendation_message = (
            f"{best_site['site_name']} is currently safe to visit and is the closest recommended heritage site."
        )

    # ELSE → ESCALATION BEGINS
    else:
        # NEW DYNAMIC STATUS DETECTION
        top3_statuses = [site["safety_status"] for site in top_3]

        if all(status == "UNSAFE" for status in top3_statuses):
            overall_status = "unsafe"
        elif all(status == "CAUTION" for status in top3_statuses):
            overall_status = "under caution"
        else:
            overall_status = "at elevated risk"

        # ESCALATION STEP 1: CHECK REMAINING INSIDE TOP 5
        remaining_top5 = final_results[3:]  # site 4 & 5

        safe_candidates = []

        for site in remaining_top5:
            status = evaluate_site_safety(site["weather"], site["disaster"])
            site["safety_status"] = status

            if status == "SAFE":
                safe_candidates.append(site)

        if safe_candidates:
            best_site = min(safe_candidates, key=lambda x: x["road_distance_km"])
            highlight_site_id = best_site["site_id"]
            alternative_site = best_site

            recommendation_message = (
                f"The nearest heritage sites are currently {overall_status}. "
                "A safer nearby alternative is suggested."
            )

        # ESCALATION STEP 2: CHECK REMAINING INSIDE CLUSTER
        else:

            cluster_remaining = df[
                (df["cluster"] == cluster_id) &
                (~df["site_id"].isin([s["site_id"] for s in top_5]))
            ].drop_duplicates(subset=["site_id"])

            cluster_safe = []

            for _, row in cluster_remaining.iterrows():

                straight_dist = haversine(lat, lon, row["lat"], row["lon"])
                if straight_dist > 50000:
                    continue

                road_km = get_road_distance(lat, lon, row["lat"], row["lon"])
                if road_km is None:
                    continue

                weather = get_weather(row["lat"], row["lon"])
                disaster = get_disaster_status(row["lat"], row["lon"])
                status = evaluate_site_safety(weather, disaster)

                if status == "SAFE":
                    cluster_safe.append({
                        "site_id": row["site_id"],
                        "site_name": row["site_name"],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "road_distance_km": road_km,
                        "weather": weather,
                        "disaster": disaster,
                        "safety_status": status,
                        "events": df[df["site_id"] == row["site_id"]][
                            ["event_name", "year", "description"]
                        ].to_dict(orient="records")
                    })

            if cluster_safe:
                best_site = min(cluster_safe, key=lambda x: x["road_distance_km"])
                highlight_site_id = best_site["site_id"]
                alternative_site = best_site

                recommendation_message = (
                    f"The nearest heritage sites are currently {overall_status}. "
                    "A safer alternative within your region is suggested."
                )

            # ESCALATION STEP 3: SEARCH WITHIN 50 KM (GLOBAL)
            else:

                global_safe = []

                for _, row in df.drop_duplicates(subset=["site_id"]).iterrows():

                    straight_dist = haversine(lat, lon, row["lat"], row["lon"])
                    if straight_dist > 50000:  # 50km radius
                        continue

                    road_km = get_road_distance(lat, lon, row["lat"], row["lon"])
                    if road_km is None:
                        continue

                    weather = get_weather(row["lat"], row["lon"])
                    disaster = get_disaster_status(row["lat"], row["lon"])
                    status = evaluate_site_safety(weather, disaster)

                    if status == "SAFE":
                        global_safe.append({
                            "site_id": row["site_id"],
                            "site_name": row["site_name"],
                            "lat": row["lat"],
                            "lon": row["lon"],
                            "road_distance_km": road_km,
                            "weather": weather,
                            "disaster": disaster,
                            "safety_status": status,
                            "events": df[df["site_id"] == row["site_id"]][
                                ["event_name", "year", "description"]
                            ].to_dict(orient="records")
                        })

                if global_safe:
                    best_site = min(global_safe, key=lambda x: x["road_distance_km"])
                    highlight_site_id = best_site["site_id"]
                    alternative_site = best_site

                    recommendation_message = (
                        "All sites in your immediate region are unsafe. "
                        "A safe alternative within 50 km is recommended."
                    )

                else:
                    recommendation_message = (
                        f"All heritage sites within 50 km are currently {overall_status} "
                        "due to severe weather."
                    )

    return jsonify({
        "user_cluster": cluster_id,
        "recommended": top_3,
        "highlight_site_id": highlight_site_id,
        "recommendation_message": recommendation_message,
        "alternative_site": alternative_site
    })

# API 4: Manual Area Search
@recommend_api.route("/search-area", methods=["POST"])
def search_area():
    data = request.get_json()
    area = data.get("area")
    user_lat = data.get("user_lat")
    user_lon = data.get("user_lon")

    if not area or user_lat is None or user_lon is None:
        return jsonify({"error": "area and user location required"}), 400

    # 1. Convert area to coordinates
    area_lat, area_lon = get_coordinates_from_area(area)

    if area_lat is None:
        return jsonify({"error": "Area not found"}), 404

    # 2. Predict cluster for that area
    cluster_scaled = scaler.transform(np.array([[area_lat, area_lon]]))
    cluster_id = int(model.predict(cluster_scaled)[0])

    # 3. Get cluster sites
    cluster_sites = df[df["cluster"] == cluster_id] \
        .drop_duplicates(subset=["site_id"])

    results = []

    for _, row in cluster_sites.iterrows():

        # Distance from USER location
        road_user_km = get_road_distance(
            user_lat, user_lon,
            row["lat"], row["lon"]
        )

        # Distance from SEARCHED AREA
        road_area_km = get_road_distance(
            area_lat, area_lon,
            row["lat"], row["lon"]
        )

        # If either distance fails, skip this site
        if road_user_km is None or road_area_km is None:
            continue

        # Get Place Name
        place_name = get_place_name(row["lat"], row["lon"])

        weather = get_weather(row["lat"], row["lon"])
        disaster = get_disaster_status(row["lat"], row["lon"])
        status = evaluate_site_safety(weather, disaster)

        events = df[df["site_id"] == row["site_id"]][
            ["event_name", "year", "description"]
        ].to_dict(orient="records")

        results.append({
            "site_id": row["site_id"],
            "site_name": row["site_name"],
            "place_name": place_name,
            "lat": row["lat"],
            "lon": row["lon"],
            "distance_from_user_km": round(road_user_km, 2),
            "distance_from_area_km": round(road_area_km, 2),
            "weather": weather,
            "disaster": disaster,
            "safety_status": status,
            "events": events
        })

    # Sort by distance from user
    results.sort(key=lambda x: x["distance_from_area_km"])

    return jsonify({
        "searched_area": area,
        "cluster": cluster_id,
        "results": results[:3]   # limit top 3
    })

# API 5: Search Heritage Site by Name
@recommend_api.route("/search-site", methods=["POST"])
def search_site():

    data = request.get_json()
    site_name = data.get("site_name")
    user_lat = data.get("user_lat")
    user_lon = data.get("user_lon")

    if not site_name:
        return jsonify({"error": "site_name required"}), 400

    # Find matching sites
    matches = df[
        df["site_name"].str.contains(site_name, case=False, na=False)
    ].drop_duplicates(subset=["site_id"])

    if matches.empty:
        return jsonify({"error": "Site not found"}), 404

    results = []

    for _, row in matches.iterrows():

        road_km = None

        if user_lat and user_lon:
            road_km = get_road_distance(
                user_lat, user_lon,
                row["lat"], row["lon"]
            )

        place_name = get_place_name(row["lat"], row["lon"])

        weather = get_weather(row["lat"], row["lon"])
        disaster = get_disaster_status(row["lat"], row["lon"])
        status = evaluate_site_safety(weather, disaster)

        events = df[df["site_id"] == row["site_id"]][
            ["event_name", "year", "description"]
        ].to_dict(orient="records")

        results.append({
            "site_id": row["site_id"],
            "site_name": row["site_name"],
            "lat": row["lat"],
            "lon": row["lon"],
            "place_name": place_name,
            "distance_km": road_km,
            "weather": weather,
            "disaster": disaster,
            "safety_status": status,
            "events": events
        })

    return jsonify({
        "results": results
    })
