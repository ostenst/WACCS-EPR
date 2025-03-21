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
from scipy.interpolate import interp1d

print("NOTE: The density depends on the selected price level in the density.py script. Select what plants to map also!")

# Load data
sector = "W"
plants = pd.read_csv("plants_mapping.csv", delimiter=",") 
plants = plants[plants["Fuel (W=waste, B=biomass)"] == sector]  # Filter for biomass plants

# Define origins and destinations
origins = [
    ("Lulea", 22.2, 65.6),
    ("Sundsvall", 17.3, 62.4),
    ("Stockholm/Norvik", 17.9, 58.9),
    ("Malmo", 12., 55.6),
    ("Goteborg", 11.8, 57.6),
]

destinations = [
    ("Northern Lights", 4.2, 60.4),
]

routes = list(product(origins, destinations))
europe = gpd.read_file("shapefiles/Europe/Europe_merged.shp").to_crs("EPSG:4326")

# Initialize figure
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

# Plot routes
if route_data:
    lengths = [length for _, length in route_data]
    norm_routes = mcolors.Normalize(vmin=min(lengths), vmax=max(lengths))
    cmap_routes = cm.coolwarm  # Colormap for routes

    for route_geom, length in route_data:
        color = cmap_routes(norm_routes(length))
        gpd.GeoSeries(route_geom).plot(ax=ax, linewidth=2, color=color, alpha=0.2)

# --- COLOR PLANTS BASED ON DENSITY ---
if sector == "B":
    density_values = plants["density_bio"].fillna(0)  # Ensure no NaNs
else:
    density_values = plants["density_w2e"].fillna(0)  # Ensure no NaNs
norm_density = mcolors.Normalize(vmin=density_values.min(), vmax=density_values.max())  # Normalize density
norm_density = mcolors.Normalize(vmin=0, vmax=1)
cmap_density = cm.RdYlGn  # Colormap for density

# Scatter plot for plants, colored by density
scatter = ax.scatter(
    plants["Longitude"], plants["Latitude"], linewidths=0,
    s=plants["Size"]*2.4, c=density_values, cmap=cmap_density, norm=norm_density, alpha=0.8
)

# CHANGE BELOW FOR WASTE vs BIO
largest_plant = plants.loc[plants["Size"].idxmax()]
print(largest_plant["mean_captured"], " ktCO2 emitted by largest plant")
ax.scatter(
    4, 67, 
    s=largest_plant["Size"] * 2.4, 
    color="black", linewidths=0, alpha=1
)
ax.annotate(
    "500 ktCO2/yr", 
    (5, 66.6), 
    textcoords="offset points", xytext=(5, 5), ha='left', fontsize=10, color='black'
)
# sysav = plants[plants["Name"] == "Sjolunda 1 "]
# ax.scatter(
#     sysav["Longitude"], sysav["Latitude"], linewidths=0,
#     s=sysav["Size"]*4, c=0.65, cmap=cmap_density, norm=norm_density, alpha=0.8
# )

# Plot Origins and Destinations
for origin in origins:
    ax.scatter(*origin[1:3], color="grey", marker="D", s=45, alpha=1, label="Hubs/storage" if origin == origins[0] else "")
for destination in destinations:
    ax.scatter(*destination[1:3], color="grey", marker="D", s=45, alpha=1)

# Formatting the plot
ax.set_xlim(2, 24)
ax.set_ylim(53.5, 70)
ax.set_aspect(1.90)  # Adjust aspect ratio
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Shipping Routes from Sweden - Plants Colored by Density")

# Colorbar for plant density
sm_density = cm.ScalarMappable(cmap=cmap_density, norm=norm_density)
sm_density.set_array([])  # Dummy array for colorbar
cbar = fig.colorbar(sm_density, ax=ax, location="right", fraction=0.03, pad=0.02)
cbar.set_label("Fraction of 'cheap' scenarios (cost < 1700 SEK/tCO2)")

# Display the legend
plt.legend()
ax.text(2.5, 69.78, "Waste-to-energy CCS costs\n2000 scenarios per plant, e.g.:\n7800-8200 h/yr\n20-160 EUR/MWh elec.", fontsize=10, color="grey", ha="left", va="top")
# ax.text(2.5, 69.78, "Biomass-fired CCS costs\n2000 scenarios per plant, e.g.:\n4000-6000 h/yr\n20-160 EUR/MWh elec.", fontsize=10, color="grey", ha="left", va="top")

plt.savefig("UNNAMED.png", dpi=600, bbox_inches="tight")
plt.show()
