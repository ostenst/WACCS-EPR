import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import random

# Load data
data = {
    "Distance": [1250, 1145, 1030, 935, 870, 765, 650, 555],
    "1000000": [813.3348493, 787.8440136, 757.898899, 744.3118925, 730.5392357, 707.9449625, 673.0982995, 650.5940268],
    "2000000": [750.5540143, 729.6108548, 702.7277159, 691.8816558, 679.9969846, 660.3998211, 628.8064311, 557.0792473],
    "3000000": [723.9813168, 704.1813286, 678.4338466, 668.5014354, 657.2345549, 591.337993, 571.4562927, 554.4960328],
}

df = pd.DataFrame(data)
df['Distance'] = df['Distance'] / 0.54 #NOTE Converting from nautical miles to km

# Define function for linear regression
def predict_with_regression(distances, costs, new_distances):
    model = LinearRegression()
    distances = np.array(distances).reshape(-1, 1)
    costs = np.array(costs)
    model.fit(distances, costs)
    return model.predict(np.array(new_distances).reshape(-1, 1))

# New distances to estimate
new_distances = [350, 700]

# Perform linear regression for each Flow
estimates = {flow: predict_with_regression(df["Distance"], df[flow], new_distances) for flow in ["1000000", "2000000", "3000000"]}

# Create a DataFrame with new estimates
new_df = pd.DataFrame({"Distance": new_distances, "1000000": estimates["1000000"], "2000000": estimates["2000000"], "3000000": estimates["3000000"]})

# Append to original DataFrame
df_extended = pd.concat([df, new_df]).sort_values(by="Distance", ascending=False).reset_index(drop=True)

# Print the extended DataFrame
print(df_extended)

# Plot the extended data
plt.figure(figsize=(8, 6))
for flow in ["1000000", "2000000", "3000000"]:
    plt.plot(df_extended["Distance"], df_extended[flow], marker="o", label=f"Flow {flow}")
df_extended.to_csv('shipcost.csv', index=False)

plt.xlabel("Distance")
plt.ylabel("Cost")
plt.title("Cost vs Distance (Linear Regression)")
plt.legend()
plt.grid(True)
# plt.show()


### A NEW SECTION: HERE, WE DEFINE A FUNCTION THAT ESTIMATES SHIPPING COST BASED ON THE DISTANCE
df = pd.read_csv("shipcost.csv")

# Create interpolation functions for each flow
def create_interpolation_function(flow):
    # Extract the relevant columns for the specified flow
    distances = df['Distance']
    costs = df[flow]
    
    # Create an interpolation function for the given flow
    interp_func = interp1d(distances, costs, kind='linear', fill_value="extrapolate")
    
    return interp_func

# Create separate interpolation functions for each flow
interp_1000000 = create_interpolation_function('1000000')
interp_2000000 = create_interpolation_function('2000000')
interp_3000000 = create_interpolation_function('3000000')

# Function to calculate cost based on distance and a randomly selected flow
def uncertain_cost(distance):
    # Randomly choose a flow (1000000, 2000000, or 3000000)
    flow = random.choice([1000000, 2000000, 3000000])
    
    # Calculate the cost using the corresponding interpolation function
    if flow == 1000000:
        cost = interp_1000000(distance)
    elif flow == 2000000:
        cost = interp_2000000(distance)
    elif flow == 3000000:
        cost = interp_3000000(distance)
    else:
        raise ValueError("Invalid flow specified")
    
    return flow, cost

# Example usage
distance = 400  # Example distance

# Get the cost with a random flow and variance
flow, cost = uncertain_cost(distance)
print(f"Estimated cost for distance {distance} and flow {flow}: {cost:.2f}")