import requests
import json
import time
import math
import pandas as pd
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

def stepslat(lat, lon, d, R, num):
    coLat_rad = math.radians(lat)
    coLon_rad = math.radians(lon)

    bearing_north = math.radians(0)
    bearing_south = math.radians(180)

    coords = [coLat_rad]

    temp1, temp2 = coLat_rad, coLat_rad
    for i in range(0,num):
        temp1 = math.asin(math.sin(temp1) * math.cos(d / R) +
                            math.cos(temp1) * math.sin(d / R) * math.cos(bearing_north))
        temp2 = math.asin(math.sin(temp2) * math.cos(d / R) +
                            math.cos(temp2) * math.sin(d / R) * math.cos(bearing_south))
        
        coords.append(temp1)
        coords.append(temp2)

    coords.sort()
    return [math.degrees(i) for i in coords]


def stepslong(lat, lon, d, R, num_steps):
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    bearing_east = math.radians(90)
    bearing_west = math.radians(270)

    coords = [lon_rad]
    
    temp_east = lon_rad
    temp_west = lon_rad
    
    for i in range(num_steps):
        delta = math.atan2(
            math.sin(bearing_east) * math.sin(d / R) * math.cos(lat_rad),
            math.cos(d / R) - math.sin(lat_rad) * math.sin(lat_rad)
        )
        temp_east += delta
        coords.append(temp_east)
        
        delta = math.atan2(
            math.sin(bearing_west) * math.sin(d / R) * math.cos(lat_rad),
            math.cos(d / R) - math.sin(lat_rad) * math.sin(lat_rad)
        )
        temp_west += delta
        coords.append(temp_west)

    coords.sort()
    return [math.degrees(i) for i in coords]


def genCoords(airportLat, airportLon, stepDist, stepNumber): 
    earthRad = 6371000
    lats = stepslat(airportLat, airportLon, stepDist, earthRad, stepNumber)
    lons = stepslong(airportLat, airportLon, stepDist, earthRad, stepNumber)
    params = {
        "lamin": min(lats),
        "lomin": min(lons), 
        "lamax": max(lats), 
        "lomax": max(lons)  
    }
    coord_sets = [set(coord) for coord in product(lats, lons)]

    return coord_sets, params


#Nantes Airport coordinates in degrees:
Nantes = {
    'Latitude':     47.1542,
    'Longitude':    -1.6044
}

def distance(grLat, grLon, plLat, plLon):
    R = 6371000

    # Calculating horizental distance using the haversine_distance
    radCoLat, radPlLat= math.radians(grLat), math.radians(plLat)
    deltaLat = math.radians(grLat - plLat)
    deltaLon = math.radians(grLon - plLon)

    a = math.sin(deltaLat/2) ** 2
    b = math.cos(radCoLat) * math.cos(radPlLat) * (math.sin(deltaLon/2) ** 2)
    horDist = 2 * R * math.asin(math.sqrt(a+b))

    # Calculating vertical distance, simple because variables in meters not degrees
    verDist = abs(2461.26 - 27)

    # Pythagores
    return math.sqrt((horDist**2)+(verDist**2))

def decibelEstimationSource(spark, df):
    a = df.select('longitude', 'latitude', 'on_ground', 'vertical_rate')
    dataset = a.collect()
    currentDecibels = {}
    for i in dataset:
        if i['on_ground']==True:
            currentDecibels[(i['latitude'], i['longitude'])] = 80
        else:
            if i['vertical_rate']<-1.5:
                currentDecibels[(i['latitude'], i['longitude'])] = 110
            elif i['vertical_rate']>1.5:
                currentDecibels[(i['latitude'], i['longitude'])] = 130
            else:
                currentDecibels[(i['latitude'], i['longitude'])] = 90
    return currentDecibels

def combine_decibels(decibel_list):
    if not decibel_list:
        return 0
    total = sum(10 ** (d / 10) for d in decibel_list)
    return round(10 * math.log10(total), 2)

def decibelEstimationGround(dfSource, coordinates):
    decibelsList = {}

    for (lat, lon), i0 in dfSource.items():
        for (i, j) in coordinates:
            if abs(i - lat) > 0.18 or abs(j - lon) > 0.246:
                continue

            dist = max(distance(i, j, lat, lon), 1e-3)
            sourceDecibel = i0 - 20 * math.log10(dist)

            if sourceDecibel >= 30:
                decibelsList.setdefault((i, j), []).append(sourceDecibel)

    return {coord: combine_decibels(vals) for coord, vals in decibelsList.items()}