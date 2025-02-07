import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import searoute as sr
from shapely.geometry import LineString
from itertools import product
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

w2e_coords = pd.read_csv("data/w2e_coordinates.csv") 
w2e_heat = pd.read_csv("data/w2e_data_all.csv", delimiter=";") 
w2e_plants = pd.merge(w2e_coords, w2e_heat, on="Name", how="inner")

bio_coords = pd.read_csv("data/bio_coordinates.csv") 
bio_heat = pd.read_csv("data/bio_data_all.csv", delimiter=";") 
bio_plants = pd.merge(bio_coords, bio_heat, on="Name", how="inner")

min_plant = min([w2e_plants["Heat output (MWheat)"].min(), bio_plants["Heat output (MWheat)"].min()])
max_plant = max([w2e_plants["Heat output (MWheat)"].max(), bio_plants["Heat output (MWheat)"].max()])

w2e_plants["Size"] = (w2e_plants["Heat output (MWheat)"] - min_plant) / \
                 (max_plant - min_plant) * 300
bio_plants["Size"] = (bio_plants["Heat output (MWheat)"] - min_plant) / \
                 (max_plant - min_plant) * 300

# Define origins and destinations
origins = [
    [22.2, 65.6],   # Luleå
    [17.3, 62.4],   # Sundsvall
    [18.0, 59.3],   # Stockholm
    [17.1, 58.6],   # Oxelösund
    [13.0, 55.6],   # Malmö
    [11.9, 57.6],   # Göteborg
    [11.4, 58.2],   # Lysekil
]
destinations = [
    [8.3, 55.5],    # Greensand
    [4.2, 60.4],    # Northern Lights
    [-1.7, 57.5],   # Acorn Project
    [4.3, 56.2],    # Project Bifrost
]

routes = list(product(origins, destinations))
europe = gpd.read_file("shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
europe.plot(ax=ax, edgecolor="black", facecolor="whitesmoke")  # Plot landmass

# Compute route lengths
route_data = []
for origin, destination in routes:
    try:
        route = sr.searoute(origin, destination)
        route_length = route.properties["length"]  
        route_geom = LineString(route.geometry["coordinates"])
        route_data.append((route_geom, route_length))
        print("{:.1f} {}".format(route.properties['length'], route.properties['units']))
    except Exception as e:
        print(f"Could not process route from {origin} to {destination}: {e}")

# Normalize route lengths for colormap
if route_data:
    lengths = [length for _, length in route_data]
    norm = mcolors.Normalize(vmin=min(lengths), vmax=max(lengths))
    cmap = cmap = cm.coolwarm  # Choose colormap

    for route_geom, length in route_data:
        color = cmap(norm(length))
        gpd.GeoSeries(route_geom).plot(ax=ax, linewidth=2, color=color, alpha=0.2)
        
sc = ax.scatter(
    bio_plants["Longitude"], bio_plants["Latitude"], linewidths=1,
    s=bio_plants["Size"], c="mediumseagreen", alpha=0.8, label="Biomass-fired plants"
)
sc = ax.scatter(
    w2e_plants["Longitude"], w2e_plants["Latitude"], linewidths=1,
    s=w2e_plants["Size"], c="grey", alpha=0.8, label="Waste-fired plants"
)

for origin in origins:
    ax.scatter(*origin, color="crimson", marker="D", s=25, alpha=1, label="Origin" if origin == origins[0] else "")

for destination in destinations:
    ax.scatter(*destination, color="deepskyblue", marker="D", s=25, alpha=1, label="Destination" if destination == destinations[0] else "")



ax.set_xlim(-5, 25)
ax.set_ylim(54, 74)
ax.set_aspect(1.90) # Hard code the aspect ratio to compensate for colormap
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Shipping Routes from Sweden - Colored by Distance")
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array for colorbar
cbar = fig.colorbar(sm, ax=ax, location="right", fraction=0.03, pad=0.02)
cbar.set_label("Route Length (km)")

plt.legend()
plt.show()