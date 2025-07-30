# Aviation noise monitoring pipeline

## Table of contents

- [Goal and motivation](#goal-and-motivation)
- [Requirements](#requirements)
- [Data source]()

## Goal and motivation

This project aims to build a data engineering pipeline that collects data from the OpenSky REST API, maps aircraft coordinates within a defined geographic area, performs data transformation, cleaning, and feature engineering, and outputs the data in a format suitable for visualizing ground-level noise pollution caused by aircraft.

Initially, this project aimed to create a heatmap showing zones with frequent aircraft activity within a specific region. However, due to API limitations and storage constraints, the project focus shifted to the current data pipeline approach.

## Requirements

As of now, this project has been built using only Python and some of its libraries:

- **NumPy**: for efficient handling of numerical values and optimized calculations  
- **PySpark**: used for dataset management (even though the dataset is small, this is for training purposes)  
- **Folium**: for generating a heatmap of calculated decibel levels  
- **Standard libraries**: such as `sys`, `json`, and `requests` for general-purpose scripting and API calls

> It is recommended to install all dependencies within a **virtual environment** to avoid conflicts and keep the project isolated.

In the future, the following tools and technologies may be integrated:
- **Apache Airflow**: for workflow orchestration  
- **Docker**: for containerization and reproducibility  
- **Web development tools**: for building a user-facing interface or dashboard

## Data source

The data used in this project is collected from the **[OpenSky Network API](https://opensky-network.org/)**, an open platform providing real-time and historical flight tracking data.

The relevant information provided by the API for this project includes:

- **longitude**: The longitudinal position of the aircraft  
- **latitude**: The latitudinal position of the aircraft  
- **on_ground**: A boolean indicating whether the aircraft is on the ground (e.g., taxiing, parked)  
- **vertical_rate**: The rate of climb or descent (in meters/second), used to infer the flight phase (e.g., takeoff, cruising, landing)  
- **geo_altitude**: The geometric altitude of the aircraft (in meters above sea level)

> This data is collected from OpenSky's REST API via HTTP requests using the `requests` library in Python and stored for decibel estimation.