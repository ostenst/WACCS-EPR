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
from shipcost import uncertain_cost

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
# print(plants.head(len(plants)))

# Determine the emissions from each plant (86-94 % rates)
bio_experiments = pd.read_csv("data/swe_data/all_bio_experiments.csv", delimiter=",")
bio_experiments = bio_experiments[bio_experiments["duration_increase"] == 0]            #NOTE: Filtering 
w2e_experiments = pd.read_csv("data/swe_data/all_w2e_experiments.csv", delimiter=",")
bio_outcomes = pd.read_csv("data/swe_data/all_bio_outcomes.csv", delimiter=",")
bio_outcomes = bio_outcomes[bio_outcomes["penalty_biomass"] == 0]
w2e_outcomes = pd.read_csv("data/swe_data/all_w2e_outcomes.csv", delimiter=",")

bio_outcomes["rate"] = bio_experiments["rate"]
w2e_outcomes["rate"] = w2e_experiments["rate"]
bio_outcomes = bio_outcomes[bio_outcomes["rate"] >= 0.86]
w2e_outcomes = w2e_outcomes[w2e_outcomes["rate"] >= 0.86]

# Calculate total number of CHP and W2E plants and how many times these are costly!
total_chp = bio_outcomes.shape[0]
total_w2e = w2e_outcomes.shape[0]
chp_high_cost = bio_outcomes[bio_outcomes["capture_cost"] > 200].shape[0]
w2e_high_cost = w2e_outcomes[w2e_outcomes["capture_cost"] > 200].shape[0]
chp_high_cost_pct = (chp_high_cost / total_chp) * 100 if total_chp > 0 else 0
w2e_high_cost_pct = (w2e_high_cost / total_w2e) * 100 if total_w2e > 0 else 0
print(f"Number of CHP plants with capture_cost > 200: {chp_high_cost} ({chp_high_cost_pct:.2f}%) out of N total scenarios: {total_chp}")
print(f"Number of W2E plants with capture_cost > 200: {w2e_high_cost} ({w2e_high_cost_pct:.2f}%) out of N total scenarios: {total_w2e}")

# Calculate means and rename the column to make it distinct - needed for plotting bubbles later!
bio_mean_captured = bio_outcomes.groupby("Name")["captured"].mean().reset_index()
w2e_mean_captured = w2e_outcomes.groupby("Name")["captured"].mean().reset_index()

bio_mean_captured.rename(columns={"captured": "mean_captured"}, inplace=True)
w2e_mean_captured.rename(columns={"captured": "mean_captured"}, inplace=True)
bio_outcomes = bio_outcomes.merge(bio_mean_captured, on="Name", how="left")
w2e_outcomes = w2e_outcomes.merge(w2e_mean_captured, on="Name", how="left")

all_mean_captured = pd.concat([bio_mean_captured, w2e_mean_captured], ignore_index=True)
plants = plants.merge(all_mean_captured, on="Name", how="left")
plants["Size"] = (plants["mean_captured"] - plants["mean_captured"].min()) / \
                 (plants["mean_captured"].max() - plants["mean_captured"].min()) * 300
print(plants.columns)

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
    # ("Greensand", 8.3, 55.5),
    ("Northern Lights", 4.2, 60.4),
    # ("Acorn Project", -1.7, 57.5),
    # ("Project Bifrost", 4.3, 56.2)
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

# MAPPING
routes = list(product(origins, destinations))
europe = gpd.read_file("shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
europe.plot(ax=ax, edgecolor="black", facecolor="whitesmoke")  # Plot landmass

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
origin_distances = {
    "Lulea": 2321.3,
    "Sundsvall": 1874.9,
    "Stockholm/Norvik": 1521.1,
    "Malmo": 936.2,
    "Goteborg": 616.6
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
plants['Route_Length'] = plants['Origin'].map(origin_distances)
print("Note: no truck transport is considered yet!")

# # Now I must assign a transport cost to each SCENARIO, not to each PLANT
def transport_scenario(row):
    distance = row['Route_Length']
    flow, cost = uncertain_cost(distance)
    return pd.Series([flow, cost], index=['Flow', 'Cost'])

bio_outcomes['Origin'] = bio_outcomes['Name'].map(plants.set_index('Name')['Origin'])
bio_outcomes['Route_Length'] = bio_outcomes['Name'].map(plants.set_index('Name')['Route_Length'])

bio_outcomes[['Flow', 'Cost']] = bio_outcomes.apply(transport_scenario, axis=1)
bio_outcomes['capture_cost'] = bio_outcomes['capture_cost'] * 11.03 #Convert to SEK
bio_outcomes['total_cost'] = bio_outcomes['capture_cost']*1.10 + bio_outcomes['Cost'] + 15*11.03 #Assumed 15 EUR storage cost and that truck transport is 10% of capture cost
print(bio_outcomes[['Name', 'Origin', 'Route_Length', 'capture_cost', 'mean_captured', 'Flow', 'Cost', 'total_cost']])

w2e_outcomes['Origin'] = w2e_outcomes['Name'].map(plants.set_index('Name')['Origin'])
w2e_outcomes['Route_Length'] = w2e_outcomes['Name'].map(plants.set_index('Name')['Route_Length'])

w2e_outcomes[['Flow', 'Cost']] = w2e_outcomes.apply(transport_scenario, axis=1)
w2e_outcomes['capture_cost'] = w2e_outcomes['capture_cost'] * 11.03 #Convert to SEK
w2e_outcomes['total_cost'] = w2e_outcomes['capture_cost']*1.10 + w2e_outcomes['Cost'] + 15*11.03 #Assumed 15 EUR storage cost and that truck transport is 10% of capture cost
print(w2e_outcomes[['Name', 'Origin', 'Route_Length', 'capture_cost', 'mean_captured', 'Flow', 'Cost', 'total_cost']])

# NOTE: If our aim is PURELY BECCS, then the W2E plants incur additional ETS costs for each biogenic CO2 realized.
# I therefore add ETS costs of [1000, 1500, 2000, 2500] SEK/tCO2fossil, based on EON feedback.
ETS = 2500
ffraction = 0.40
bfraction = 1-ffraction
difference = w2e_outcomes['total_cost'] - ETS # The extra cost incurred per fossil tCO2 when trying to realize BECCS.
difference_total = difference * w2e_outcomes['captured']*ffraction # [SEK/t]*[kt/yr]=[kSEK/yr]

# This difference can be allocated EITHER to all tons of CO2, or to just the biogenic fraction - depends on purpose!
extra_cost = difference_total / w2e_outcomes['captured'] # [kSEK/yr]/[kt/yr]=[SEK/t], allocated to all
# extra_cost = difference_total / (w2e_outcomes['captured']*bfraction) # [kSEK/yr]/[kt/yr]=[SEK/t], allocated to biogenic
w2e_outcomes['total_cost'] = w2e_outcomes['total_cost'] + extra_cost 

# Good. Now it is time to calculate FEATURES relevant for PLOTTING! For example, %density of CRC scenarios
total_counts_bio = bio_outcomes.groupby('Name').size()
above_150_counts_bio = bio_outcomes[bio_outcomes['total_cost'] < 3000].groupby('Name').size()
above_150_counts_bio = above_150_counts_bio.reindex(total_counts_bio.index, fill_value=0)

density_bio = above_150_counts_bio / total_counts_bio

total_counts_w2e = w2e_outcomes.groupby('Name').size()
above_150_counts_w2e = w2e_outcomes[w2e_outcomes['total_cost'] < 3000].groupby('Name').size()
above_150_counts_w2e = above_150_counts_w2e.reindex(total_counts_w2e.index, fill_value=0)

density_w2e = above_150_counts_w2e / total_counts_w2e

# --- ADD TO PLANTS DATAFRAME ---
plants['density_bio'] = plants['Name'].map(density_bio)
plants['density_w2e'] = plants['Name'].map(density_w2e)
plants.to_csv("plants_mapping.csv", index=False)
print("Updated plants dataframe:\n", plants)

# Set figure size
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Bar chart for density_bio
axes[0].bar(plants['Name'], plants['density_bio'], color='blue', alpha=0.7)
axes[0].set_title("Density of Bio Outcomes")
axes[0].set_ylabel("Density")
axes[0].set_xlabel("Plant Name")
axes[0].tick_params(axis='x', rotation=45)

# Bar chart for density_w2e
axes[1].bar(plants['Name'], plants['density_w2e'], color='green', alpha=0.7)
axes[1].set_title("Density of W2E Outcomes")
axes[1].set_xlabel("Plant Name")
axes[1].tick_params(axis='x', rotation=45)

# Adjust layout
plt.tight_layout()
# plt.show()
