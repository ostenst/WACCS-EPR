import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import searoute as sr
from shapely.geometry import LineString
from itertools import product
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Preparing data
w2e_heat = pd.read_csv("data/w2e_data_all.csv", delimiter=";") 
bio_heat = pd.read_csv("data/bio_data_all.csv", delimiter=";")  
w2e_coords = pd.read_csv("data/w2e_coordinates.csv", delimiter=",") 
bio_coords = pd.read_csv("data/bio_coordinates.csv", delimiter=",")  

combined_df = pd.concat([w2e_heat, bio_heat], ignore_index=True)
all_coords = pd.concat([w2e_coords, bio_coords], ignore_index=True)
combined_df = combined_df.merge(all_coords, on=["Name", "City"], how="left")
city_coords = all_coords.drop(columns=["Name"]).drop_duplicates(subset=["City"])
combined_df = combined_df.merge(city_coords, on="City", how="left", suffixes=("", "_city"))

for col in ["Latitude", "Longitude"]: 
    combined_df[col] = combined_df[col].fillna(combined_df[f"{col}_city"])
combined_df.drop(columns=["Latitude_city", "Longitude_city"], inplace=True)
plants = combined_df
print(plants.head(len(plants)))

# Determine the emissions from each plant (86-94 % rates)
bio_experiments = pd.read_csv("data/swe_data/all_bio_experiments.csv", delimiter=",")
w2e_experiments = pd.read_csv("data/swe_data/all_w2e_experiments.csv", delimiter=",")
bio_outcomes = pd.read_csv("data/swe_data/all_bio_outcomes.csv", delimiter=",")
w2e_outcomes = pd.read_csv("data/swe_data/all_w2e_outcomes.csv", delimiter=",")

bio_outcomes["rate"] = bio_experiments["rate"]
w2e_outcomes["rate"] = w2e_experiments["rate"]
bio_outcomes = bio_outcomes[bio_outcomes["rate"] >= 0.86]
w2e_outcomes = w2e_outcomes[w2e_outcomes["rate"] >= 0.86]

# Calculate means and rename the column to make it distinct
bio_mean_captured = bio_outcomes.groupby("Name")["captured"].mean().reset_index()
w2e_mean_captured = w2e_outcomes.groupby("Name")["captured"].mean().reset_index()

bio_mean_captured.rename(columns={"captured": "mean_captured"}, inplace=True)
w2e_mean_captured.rename(columns={"captured": "mean_captured"}, inplace=True)
bio_outcomes = bio_outcomes.merge(bio_mean_captured, on="Name", how="left")
w2e_outcomes = w2e_outcomes.merge(w2e_mean_captured, on="Name", how="left")

all_mean_captured = pd.concat([bio_mean_captured, w2e_mean_captured], ignore_index=True)
plants = plants.merge(all_mean_captured, on="Name", how="left")
print(plants.columns)

# Calculate sizes
plants["Size"] = (plants["mean_captured"] - plants["mean_captured"].min()) / \
                 (plants["mean_captured"].max() - plants["mean_captured"].min()) * 300

# Map this!
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

        
fuel_colors = {"B": "mediumseagreen", "W": "grey"}
for fuel, color in fuel_colors.items():
    subset = plants[plants["Fuel (W=waste, B=biomass)"] == fuel]
    ax.scatter(
        subset["Longitude"], subset["Latitude"], linewidths=1,
        s=subset["Size"], c=color, alpha=0.8, label=f"{fuel}-fired plants"
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