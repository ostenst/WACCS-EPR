import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import searoute as sr
from shapely.geometry import LineString
from itertools import product
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from geopy.distance import geodesic

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

# Define origins and destinations
origins = [
    ("Lulea", 22.2, 65.6),
    ("Sundsvall", 17.3, 62.4),
    ("Stockholm/Norvik", 17.9, 58.9),
    # ("Oxelosund", 17.1, 58.6),
    ("Malmo", 13.0, 55.6),
    ("Goteborg", 11.8, 57.6),
    # ("Lysekil", 11.4, 58.2)
]

# List of destinations (lat, lon)
destinations = [
    ("Greensand", 8.3, 55.5),
    ("Northern Lights", 4.2, 60.4),
    ("Acorn Project", -1.7, 57.5),
    ("Project Bifrost", 4.3, 56.2)
]

# Define color mapping for each Origin
origin_colors = {
    "Lulea": "crimson", 
    "Sundsvall": "dodgerblue", 
    "Stockholm/Norvik": "forestgreen", 
    # "Oxelosund": "darkorange", 
    "Malmo": "purple", 
    "Goteborg": "goldenrod", 
    # "Lysekil": "pink"
}

# Function to calculate the nearest origin for each plant
def assign_origin(row, origins):
    plant_location = (row["Latitude"], row["Longitude"])
    distances = []
    
    for origin_name, lon, lat in origins:
        origin_location = (lat, lon)
        distance = geodesic(plant_location, origin_location).km  # Calculate distance in kilometers
        distances.append((origin_name, distance))
    
    closest_origin = min(distances, key=lambda x: x[1])[0]
    return closest_origin
plants["Origin"] = plants.apply(assign_origin, origins=origins, axis=1)
sum_captured_by_origin = plants.groupby("Origin")["mean_captured"].sum().reset_index()
print(sum_captured_by_origin)

# MAPPING
routes = list(product(origins, destinations))
europe = gpd.read_file("shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
europe.plot(ax=ax, edgecolor="black", facecolor="whitesmoke")  # Plot landmass

# Compute route lengths
route_data = []
for origin, destination in routes:
    try:
        route = sr.searoute(origin[1:3], destination[1:3])
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
    cmap = cm.coolwarm  # Choose colormap

    for route_geom, length in route_data:
        color = cmap(norm(length))
        gpd.GeoSeries(route_geom).plot(ax=ax, linewidth=2, color=color, alpha=0.2)

# Plot plants, coloring by Origin
for origin_name, color in origin_colors.items():
    subset = plants[plants["Origin"] == origin_name]
    ax.scatter(
        subset["Longitude"], subset["Latitude"], linewidths=1,
        s=subset["Size"], c=color, alpha=0.8, label=f"{origin_name} plants"
    )

# Plot Origins and Destinations
for origin in origins:
    ax.scatter(*origin[1:3], color=origin_colors[origin[0]], marker="D", s=25, alpha=1, label="Origin" if origin == origins[0] else "")

for destination in destinations:
    ax.scatter(*destination[1:3], color="deepskyblue", marker="D", s=25, alpha=1, label="Destination" if destination == destinations[0] else "")

# Formatting the plot
ax.set_xlim(-5, 25)
ax.set_ylim(54, 74)
ax.set_aspect(1.90)  # Hard code the aspect ratio to compensate for colormap
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Shipping Routes from Sweden - Colored by Origin")

# Colorbar for route length
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array for colorbar
cbar = fig.colorbar(sm, ax=ax, location="right", fraction=0.03, pad=0.02)
cbar.set_label("Route Length (km)")

# Display the legend
plt.legend()
plt.show()