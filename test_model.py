from infer import IrradianceReconstructor

# Instantiate once at service startup — loads model weights into GPU
rec = IrradianceReconstructor()

# Every hour, call predict() with your live sentinel readings
readings = {
    "PANEL_042": 523.1,   # W/m2 — whatever your monitoring system gives you
    "PANEL_107": 489.7,
    # ... all 15 sentinel panel IDs
}
result = rec.predict(readings)

# result["irradiance"] is a (1149,) numpy array of W/m2 for every panel
# result["panel_ids"]  is the matching array of panel ID strings