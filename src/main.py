# backend/src/main.py

from flask import Flask
from flask_cors import CORS

from api.location_api import location_api
from api.events_api import events_api
from api.recommend_api import recommend_api

app = Flask(__name__)
CORS(app)

# Register additional API routes
app.register_blueprint(location_api)
app.register_blueprint(events_api)
app.register_blueprint(recommend_api) # NEW route

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
