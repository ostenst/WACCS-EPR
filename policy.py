import numpy as np
import matplotlib.pyplot as plt

def calculate_tax_revenue(quantity, tax_rate):
    return quantity * tax_rate

def calculate_annualized_cost(total_cost, years):
    return total_cost / years

# Given data
mapped_products = 884393  # t per year
all_products = 1245000  # t per year
low_tax = 2900  # SEK/t
high_tax = 5000  # SEK/t
sorting_facility_cost = 134000000  # SEK total investment
investment_years = 20  # Years to annualize over

years = list(range(2025, 2041))  # List of years for even distribution
num_facilities = np.arange(len(years))  # Number of facilities increasing each year

# Calculate tax revenues
mapped_low = calculate_tax_revenue(mapped_products, low_tax)
mapped_high = calculate_tax_revenue(mapped_products, high_tax)
all_low = calculate_tax_revenue(all_products, low_tax)
all_high = calculate_tax_revenue(all_products, high_tax)

# Store revenues in lists for proper plotting
tax_revenues = [
    [mapped_low] * len(years),
    [mapped_high] * len(years),
    [all_low] * len(years),
    [all_high] * len(years)
]

labels = [
    f"Mapped - Low Tax: {mapped_low:,} SEK",
    f"Mapped - High Tax: {mapped_high:,} SEK",
    f"All - Low Tax: {all_low:,} SEK",
    f"All - High Tax: {all_high:,} SEK"
]

colors = ['b', 'r', 'g', 'purple']
linestyles = ['--', '--', '-', '-']

# Calculate subsidies (negative values) for sorting facilities
annualized_facility_cost = calculate_annualized_cost(sorting_facility_cost, investment_years)
subsidies = -num_facilities * annualized_facility_cost  # More negative as more facilities are added

# Plot horizontal lines distributed over time
plt.figure(figsize=(10, 5))
for revenue, label, color, linestyle in zip(tax_revenues, labels, colors, linestyles):
    plt.plot(years, revenue, linestyle=linestyle, color=color, label=label)

# Add subsidy line
plt.plot(years, subsidies, linestyle='-.', color='black', label=f"Sorting Facility Subsidy (Increasing)")

plt.xlabel("Year")
plt.ylabel("Annual Tax Revenue (SEK)")
plt.title("Projected Annual Tax Revenue and Subsidies Over Time")
plt.xticks(years, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
