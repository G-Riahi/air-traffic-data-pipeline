import requests
import json
import time
import math
import sys
import folium
import pandas as pd
import numpy as np
from folium import Element
from folium.plugins import HeatMap
from itertools import product
from pyspark.sql import SparkSession, functions as F, types as T


# Getting flight dataset, depending on a predefined zone
def getFlightData(credFile, params):
    with open(credFile) as f:
        creds = json.load(f)

    client_id = creds.get("clientId")
    client_secret = creds.get("clientSecret")

    if not client_id or not client_secret:
        raise ValueError("Set CLIENT_ID and CLIENT_SECRET environment variables before running.")

    token_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"

    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    response = requests.post(token_url, data=payload, headers=headers)
    response.raise_for_status()

    access_token = response.json().get("access_token")

    #print(f"Access token: {access_token}")

    url = "https://opensky-network.org/api/states/all"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()  # raise error if request failed

    data = response.json()

    if response.status_code == 200:

        data_json = response.text

        spark = SparkSession.builder.appName("flight-data-pipeline").getOrCreate()

        # Create RDD from JSON string (split by lines if multiline JSON)
        rdd = spark.sparkContext.parallelize([data_json])

        # Read JSON from RDD
        df = spark.read.json(rdd)

        if data.get("states") is None:          # None → empty list
            data["states"] = []

        if not data["states"]:                  # still empty? nothing to process
            print("No aircraft in the selected area.")
            sys.exit(0)                         # or `return` inside a function

        df_states = df.withColumn("state", F.explode("states"))

        schema_def = [
        (0,  "icao24",          T.StringType()),
        (1,  "callsign",        T.StringType()),
        (2,  "origin_country",  T.StringType()),
        (3,  "time_position",   T.LongType()),
        (4,  "last_contact",    T.LongType()),
        (5,  "longitude",       T.DoubleType()),
        (6,  "latitude",        T.DoubleType()),
        (7,  "baro_altitude",   T.DoubleType()),
        (8,  "on_ground",       T.BooleanType()),
        (9,  "velocity",        T.DoubleType()),
        (10, "true_track",      T.DoubleType()),
        (11, "vertical_rate",   T.DoubleType()),
        (12, "sensors",         T.ArrayType(T.IntegerType())),
        (13, "geo_altitude",    T.DoubleType()),
        (14, "squawk",          T.StringType()),
        (15, "spi",             T.BooleanType()),
        (16, "position_source", T.IntegerType())
        ]

        cols = []

        for idx, name, dtype in schema_def:
            c = F.col("state")[idx]

            # sensors comes in as a JSON-style string like "[1,2,3]"
            if name == "sensors":
                c = F.when(
                        c.isNull(), None
                    ).otherwise(
                        F.split(                    
                            F.regexp_replace(c, r'[\[\]\s]', ''),
                            ','
                        ).cast(dtype)              
                    )
            else:
                c = c.cast(dtype)

            cols.append(c.alias(name))

        
        df_typed = df_states.select(*cols)
        print(df_typed.count())

        return spark, df_typed
    else:
        print(f"Error: {response.status_code} - {response.text}")

# credFile = '/home/gesser/air-traffic-data-pipeline/credentials.json'

#params = {
#        "lamin": 47.001917,
#        "lomin": -1.919083, 
#        "lamax": 47.340556, 
#        "lomax": -1.181750  
#    }   ---> example

EARTH_RAD = 6371000.0
R_MAX     = 20000.0        # m radius cut-off
REF_ALT   = 27.0            # m, ground reference altitude

def stepslat(lat_deg, step_m, n_steps):
    """
    Latitude rings around `lat_deg` (decimal°) up to ±n_steps·step_m metres.
    Returns a 1-D NumPy array sorted south→north, length 2·n_steps+1.
    """
    lat0 = np.radians(lat_deg)

    # i = −n … 0 … +n  →   δ = i·step_m / R   (signed angular distance, radians)
    i = np.arange(-n_steps, n_steps + 1)
    delta = i * step_m / EARTH_RAD

    # due north/south: bearing = 0° or 180°, so cos(bearing)=±1 → sign handled by i
    phi = lat0 + delta                          # exact for a meridian track
    return np.degrees(phi)


def stepslong(lat_deg, lon_deg, step_m, n_steps):
    """
    Longitude rings around (`lat_deg`, `lon_deg`) up to ±n_steps·step_m metres.
    Returns a 1-D NumPy array sorted west→east, length 2·n_steps+1.
    """
    lat0 = np.radians(lat_deg)
    lon0 = np.radians(lon_deg)

    # metres→radians in longitude shrinks by cos(latitude)
    metres_per_rad_lon = EARTH_RAD * np.cos(lat0)
    i = np.arange(-n_steps, n_steps + 1)
    lmbda = lon0 + i * (step_m / metres_per_rad_lon)
    return np.degrees(lmbda)


def genCoords(airport_coords, step_dist, step_number):
    """
    Returns
    -------
    coords      : (M, 2) ndarray of [lat, lon] pairs (M = (2n+1)²)
    params      : dict  – {'lamin', 'lomin', 'lamax', 'lomax'}
    """
    lats = stepslat(airport_coords['Latitude'], step_dist, step_number)
    lons = stepslong(airport_coords['Latitude'], airport_coords['Longitude'], step_dist, step_number)

    # full latitude/longitude grid in one call
    lon_grid, lat_grid = np.meshgrid(lons, lats)        # shapes (2n+1, 2n+1)
    coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    params = {
        "lamin": float(lats[0]),        # already sorted S→N, W→E
        "lomin": float(lons[0]),
        "lamax": float(lats[-1]),
        "lomax": float(lons[-1]),
    }
    return coords, params


#Nantes Airport coordinates in degrees:
Nantes = {
    'Latitude':     47.1542,
    'Longitude':    -1.6044
}

def distance(gr_lat, gr_lon, pl_lat, pl_lon, alt):
    """
    All inputs are NumPy arrays **or** scalars that broadcast.
    Returns an array of 3-D distances in metres.
    """
    # Convert to radians in one shot
    gr_lat_rad = np.radians(gr_lat)
    pl_lat_rad = np.radians(pl_lat)
    
    d_lat = gr_lat_rad - pl_lat_rad
    d_lon = np.radians(gr_lon - pl_lon)
    
    a = np.sin(d_lat / 2.0)**2
    b = np.cos(gr_lat_rad) * np.cos(pl_lat_rad) * np.sin(d_lon / 2.0)**2
    hor_dist = 2.0 * EARTH_RAD * np.arcsin(np.sqrt(a + b))
    
    ver_dist = np.abs(alt - 27)           # works element-wise
    return np.sqrt(hor_dist**2 + ver_dist**2)

def decibelEstimationSource(spark, df):
    a = df.select('longitude', 'latitude', 'on_ground', 'vertical_rate', 'geo_altitude')
    dataset = a.collect()
    currentDecibels = {}
    for i in dataset:
        if i['on_ground']==True:
            currentDecibels[(i['latitude'], i['longitude'])] = [80, 27]
        else:
            if i['vertical_rate']<-1.5:
                currentDecibels[(i['latitude'], i['longitude'])] = [110, i['geo_altitude']]
            elif i['vertical_rate']>1.5:
                currentDecibels[(i['latitude'], i['longitude'])] = [130, i['geo_altitude']]
            else:
                currentDecibels[(i['latitude'], i['longitude'])] = [90, i['geo_altitude']]
    return currentDecibels

def haversine_matrix(lat_a, lon_a, lat_b, lon_b):
    """
    All inputs are 1-D arrays in *radians*.
    Returns an (len(a), len(b)) matrix of great-circle distances in metres.
    """
    dlat = lat_a[:, None] - lat_b[None, :]
    dlon = lon_a[:, None] - lon_b[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lat_a[:,None])*np.cos(lat_b[None,:])*np.sin(dlon/2)**2
    return 2*EARTH_RAD*np.arcsin(np.sqrt(a))

def decibel_estimation_ground(df_source, coordinates):
    # -------- source columns → NumPy --------------------------------------
    src_lat, src_lon, src_db, src_alt = map(
        np.asarray,
        zip(*[(la, lo, db_alt[0], db_alt[1]) for (la, lo), db_alt in df_source.items()])
    )
    # -------- ground pixels ------------------------------------------------
    g_lat, g_lon = map(np.asarray, zip(*coordinates))

    # radians once
    src_lat_rad = np.radians(src_lat)
    src_lon_rad = np.radians(src_lon)
    g_lat_rad   = np.radians(g_lat)
    g_lon_rad   = np.radians(g_lon)

    # -------- distance matrix (|G|×|S|) -----------------------------------
    dist = haversine_matrix(g_lat_rad, g_lon_rad, src_lat_rad, src_lon_rad)

    # discard far sources
    mask = dist <= R_MAX
    if not mask.any():
        return {}

    # inverse-square law in dB
    db_loss   = 20*np.log10(dist, where=mask, out=np.zeros_like(dist))  # avoid log10(0) warnings
    contrib   = np.where(mask, src_db - db_loss, 0)                     # dB SPL at ground

    # -------- combine in power domain -------------------------------------
    #  P_total = Σ 10^(dB/10)   →   dB_total = 10 log10(P_total)
    power     = np.where(mask, 10**(contrib/10), 0)
    sum_power = power.sum(axis=1)

    out_db    = np.round(10*np.log10(sum_power, where=sum_power>0), 2)
    valid     = sum_power > 0

    # -------- build the (lat,lon) → dB dict -------------------------------
    return { (float(g_lat[i]), float(g_lon[i])): float(out_db[i])
             for i in np.where(valid)[0] }


#------------------------ test -------------------------------------------

coords, bounds = genCoords(Nantes, 200, 500)

spark, df_typed = getFlightData('/home/gesser/air-traffic-data-pipeline/credentials.json', bounds)

test = decibel_estimation_ground(decibelEstimationSource(spark, df_typed), coords)

center_lat = (bounds['lamin'] + bounds['lamax']) / 2
center_lon = (bounds['lomin'] + bounds['lomax']) / 2

m = folium.Map(location=[center_lat, center_lon], zoom_start=11)

corners = [
    [bounds['lamin'], bounds['lomin']],
    [bounds['lamin'], bounds['lomax']],
    [bounds['lamax'], bounds['lomax']],
    [bounds['lamax'], bounds['lomin']],
    [bounds['lamin'], bounds['lomin']]
]

folium.PolyLine(corners, color='red', weight=3).add_to(m)

heat_data = [[lat, lon, db / 130] for (lat, lon), db in test.items()]  # Normalize to 0–1 range

HeatMap(
    heat_data,
    radius=15,
    blur=25,
    max_zoom=13,
    max_opacity=0.3,
).add_to(m)


legend_html = """
<div style="
    position: fixed; 
    bottom: 50px; left: 50px; width: 180px; height: 120px; 
    background-color: rgba(255, 255, 255, 0.8);
    border:2px solid grey; z-index:9999; font-size:14px;
    padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
    <b>Heatmap Intensity</b><br>
    <i style="background:blue; width: 18px; height: 10px; display:inline-block;"></i> Low<br>
    <i style="background:orange; width: 18px; height: 10px; display:inline-block;"></i> Medium<br>
    <i style="background:red; width: 18px; height: 10px; display:inline-block;"></i> High<br>
</div>
"""
m.get_root().html.add_child(Element(legend_html))

m.save("/home/gesser/air-traffic-data-pipeline/test_maps/dec_flight_heat_map.html")

spark.stop()