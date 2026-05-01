# backend/src/api/events_api.py

from flask import Blueprint, request, jsonify
import pandas as pd
import os
import re

# Create blueprint for modular API structure
events_api = Blueprint('events_api', __name__)

# --- Load events dataset ---
base_dir = os.path.dirname(__file__)
events_path = os.path.join(base_dir, "../../data/heritage_events.csv")
df_events = pd.read_csv(events_path)

# --- Convert year (BC/AD/Century) to numeric year (BC → negative, AD → positive) ---
def parse_year(year_str):
    try:
        s = str(year_str).strip().upper()
        if "BC" in s:
            return -int(re.sub(r"[^0-9]", "", s))
        if "AD" in s:
            return int(re.sub(r"[^0-9]", "", s))
        if "CENTURY" in s:
            n = int(re.search(r"(\d+)", s).group(1))
            return (n - 1) * 100 + 50
        return int(s)
    except Exception:
        return None

df_events["year_num"] = df_events["year"].apply(parse_year)

# --- POST endpoint for filtering by site & year range ---
@events_api.route("/events/filter", methods=["POST"])
def filter_events():
    data = request.get_json()
    site_id = data.get("site_id")
    year_min = data.get("year_min", -9999)
    year_max = data.get("year_max", 9999)

    if site_id is None:
        return jsonify({"status": "error", "message": "Missing site_id"}), 400

    filtered = df_events[
        (df_events["site_id"] == site_id) &
        (df_events["year_num"].between(year_min, year_max))
    ]

    return jsonify({
        "status": "success",
        "filtered_events": filtered.to_dict(orient="records")
    })
