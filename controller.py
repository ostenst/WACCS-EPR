"""Stuff here controller"""

import numpy as np
from model import *
import seaborn as sns
import pandas as pd
from ema_workbench import (
    Model,
    RealParameter,
    IntegerParameter,
    CategoricalParameter,
    ScalarOutcome,
    ArrayOutcome,
    Constant,
    Samplers,
    ema_logging,
    perform_experiments
)

# -------------------------------------- Read data and initiate a plant ----------------------------------
plants_df = pd.read_csv("plant_data_all.csv",delimiter=";")
plants_df = plants_df.iloc[0].to_frame().T # This row makes us only iterate over the 1st plant
# plants_df = plants_df.iloc[:3] # This row makes us only iterate over the 4 first plant
all_experiments = pd.DataFrame()
all_outcomes = pd.DataFrame()

# Load CHP Aspen data
aspen_df = pd.read_csv("amine.csv", sep=";", decimal=',')
aspen_interpolators = create_interpolators(aspen_df)

for index, plant_data in plants_df.iterrows():

    print(f"||| MODELLING {plant_data['Plant Name']} WASTE CHP |||")
    CHP = WASTE_PLANT(
        name=plant_data["Plant Name"],
        fuel=plant_data["Fuel (W=waste, B=biomass)"],
        Qdh=plant_data["Heat output (MWheat)"],
        P=plant_data["Electric output (MWe)"],
        Qfgc=plant_data["Existing FGC heat output (MWheat)"],
        Tsteam=plant_data["Live steam temperature (degC)"],
        psteam=plant_data["Live steam pressure (bar)"],
    )
    CHP.estimate_nominal_cycle() 

    # ----------------------------------------- Begin RDM analysis  ---------------------------------------------
    model = Model("WACCSEPR", function=WACCS_EPR)
    model.uncertainties = [
        RealParameter("dTreb", 7, 14),
        RealParameter("Tsupp", 78, 100),
        RealParameter("Tlow", 43, 55),     
        RealParameter("dTmin", 5, 12),
        RealParameter("U", 1300, 1700),
        RealParameter("COP", 2.3, 3.8),

        RealParameter("alpha", 6, 7),
        RealParameter("beta", 0.6, 0.7),
        RealParameter("CEPCI", 1.386, 1.57),
        RealParameter("fixed", 0.04, 0.08),
        RealParameter("ownercost", 0.1, 0.3),
        RealParameter("WACC", 0.03, 0.09),
        IntegerParameter("yexpenses", 3, 6),
        RealParameter("rescalation", 0.00, 0.06),
        RealParameter("i", 0.05, 0.12),
        IntegerParameter("t", 20, 30),

        RealParameter("celc", 20, 160),
        RealParameter("cheat", 0.25, 1.00),
        RealParameter("cMEA", 1.5, 2.5),
        RealParameter("cHP", 0.76, 0.96),
        RealParameter("cHEX", 0.470, 0.670), 
        RealParameter("cETS", 50, 180), 
        RealParameter("ctrans", 20, 40), 
        RealParameter("cstore", 30, 60), 

        RealParameter("time", 7800, 8200),
    ]
    model.levers = [
        RealParameter("rate", 0.78, 0.94),
        RealParameter("tax", 0.5, 1.5),
    ]
    model.outcomes = [
        ScalarOutcome("q", ScalarOutcome.MAXIMIZE),
        ScalarOutcome("eta", ScalarOutcome.MAXIMIZE),
        ScalarOutcome("NPV", ScalarOutcome.MAXIMIZE),
        ArrayOutcome("cash"),
    ]
    model.constants = [
        Constant("interpolators", aspen_interpolators),
        Constant("CHP", CHP),
    ]

    ema_logging.log_to_stderr(ema_logging.INFO)
    n_scenarios = 30
    n_policies = 10

    results = perform_experiments(model, n_scenarios, n_policies, uncertainty_sampling = Samplers.LHS, lever_sampling = Samplers.LHS)
    experiments, outcomes = results

    # ---------------------------------------- Process results  ---------------------------------------------
    plant_experiments = pd.DataFrame(experiments)
    plant_experiments["Name"] = CHP.name
    all_experiments = pd.concat([all_experiments, plant_experiments], ignore_index=True)
    all_experiments.to_csv("all_experiments.csv", index=False) 

    processed_outcomes = {} # Multi-dimensional outcomes need to be put into neat columns
    for k, v in outcomes.items():
        # print(f"Key: {k}, Shape of v: {v.shape}")  # Add this debug line
        if isinstance(v, np.ndarray) and v.ndim > 1: 
            for i in range(v.shape[1]):
                # processed_outcomes[v[0,i,0]] = v[:,i,1]
                processed_outcomes[f"{k}_{i}"] = v[:, i]  # Corrected indexing for 2D array
        else:
            processed_outcomes[k] = v

    plant_outcomes = pd.DataFrame(processed_outcomes)
    plant_outcomes["Name"] = CHP.name
    all_outcomes = pd.concat([all_outcomes, plant_outcomes], ignore_index=True)
    all_outcomes.to_csv("all_outcomes.csv", index=False)

    if plant_experiments.shape[0] == plant_outcomes.shape[0]:
        if all(plant_experiments.index == plant_outcomes.index):
            print(" ")
    else:
        print("Mismatch in the number of rows between plant_experiments and plant_outcomes.")

    plant_outcomes["tax"] = experiments["tax"]
    # sns.pairplot(plant_outcomes, hue="time", vars=list(outcomes.keys())) # This plots ALL outcomes
    sns.pairplot(plant_outcomes, hue="tax", vars=["eta","NPV"])

plt.show()