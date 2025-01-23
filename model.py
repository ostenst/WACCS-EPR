"""Model file. Here are useful links:
Plastic mapping in Sweden:
https://www.naturvardsverket.se/4ac864/globalassets/media/publikationer-pdf/7000/978-91-620-7038-0.pdf

A key missing component is this:
(A) I need to determine ALTERNATIVES for how the tax can be applied to upstream products (e.g. based on mC and translated to PE, PVC, PET etc.)
(B) I need to determine the quantity of these materials that is treated at each W2E plant.
"""
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
from pyXSteam.XSteam import XSteam
from scipy.interpolate import LinearNDInterpolator

# Here I insert various helper functions:
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
class State:
    def __init__(self, Name, p=None, T=None, s=None, satL=False, satV=False, mix=False):
        self.Name = Name
        if satL==False and satV==False and mix==False:
            self.p = p
            self.T = T
            self.s = steamTable.s_pt(p,T)
            self.h = steamTable.h_pt(p,T)
        if satL==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = steamTable.sL_p(p)
            self.h = steamTable.hL_p(p) 
        if satV==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = steamTable.sV_p(p)
            self.h = steamTable.hV_p(p)
        if mix==True:
            self.p = p
            self.T = steamTable.tsat_p(p)
            self.s = s
            self.h = steamTable.h_ps(p,s)
        if self.p is None or self.T is None or self.s is None or self.h is None:
            raise ValueError("Steam properties cannot be determined")
        
def create_interpolators(aspen_df):
    # extract 'Flow' and 'Rcapture' columns as x values, the rest are y values
    x1 = aspen_df['Flow']
    x2 = aspen_df['Rcapture']
    x_values = np.column_stack((x1, x2))

    y_values = aspen_df.drop(columns=['Flow', 'Rcapture']).values  
    aspen_interpolators = {}

    for idx, column_name in enumerate(aspen_df.drop(columns=['Flow', 'Rcapture']).columns):
        y = y_values[:, idx]
        interp_func = LinearNDInterpolator(x_values, y)
        aspen_interpolators[column_name] = interp_func

    return aspen_interpolators

class WASTE_PLANT:
    def __init__(self, name, Qdh, P, Qfgc, Tsteam, psteam):
        self.name = name
        self.Qfuel = None
        self.Qdh = Qdh
        self.P = P
        self.Qfgc = Qfgc
        self.Tsteam = Tsteam
        self.psteam = psteam

        self.assumptions = None
        self.interpolators = None
        self.results = {}
        self.nominal_state = {}

    def estimate_nominal_cycle(self):
        print("... rankine cycle estimated")

    def future_scenarios(self, assumptions):
        h = assumptions["time"]
        mplastic = 5678 # this should include plastics scenarios! Of what fractions are treated in W2E! Increasing vs decreasing plastic volumes/shares compared to biogenic
        CAPEX = 100000
        NPV = -CAPEX
        for t in range(0,5):

            electricity_revenue = 6 * h #NOTE: we could remove the CCS electricity tax-deduction for W2E if they have EPR policy, granting some revenues to government?
            heat_revenue = 4 * h
            carbon_revenue = 3 * mplastic # no reversed auction, only EPR policy incentive assumed!
            revenues = electricity_revenue + heat_revenue + carbon_revenue

            transport_cost = 1000
            storage_cost = 2000
            auxiliary_costs = 150 # maybe two contingency scenario? of energy crises (elc prices)
            costs = transport_cost + storage_cost + auxiliary_costs

            NPV += (revenues - costs)*0.78
            print("NPV =", NPV)

        return NPV

    def reset(self):
        print("... resetting plant")


# here I create a main model function, which relies on the helper functions:
def WACCS_EPR( 
    time= 8000,
    rate = 0.90,
    tax = 1,
    CHP = None,
    interpolators = None
):
    assumptions = {
        "time": time,
        "rate": rate,
        "molar_mass": 29.55
    }
    # The below functions should only return objects that are relevant to our results
    CHP.assumptions = assumptions
    CHP.interpolators = interpolators

    # Size a capture plant and power it
    emissions_nominal = CHP.burn_fuel() # NOTE: burn_fuel() massflows of carbon must be linked to the tax in future_scenarios, i.e. mplastic
    CHP.size_amine()
    emissions_captured = CHP.power_amine()

    # Calculate recoverable heat and integrate
    CHP.merge_heat() # move the CHP.select_streams() function into the merge_heat() function AND save composite curve to self.composite_curve
    energy_balance = CHP.recover_heat() # NOTE: add prints of how much (in%) heat is recovered, relative to reboiler duty
    
    # Calculate CAPEX and NPV
    CAPEX = CHP.CAPEX(escalate=True)
    NPV = CHP.future_scenarios()

    # Calculate product/waste cost increases for "this plant" (will be the same for all plants)
    buildings += tax    # cost increase of insulation, carried by building companies
    packaging += tax    # cost increase of plastic bag, carried by consumers 
    tires += tax        # cost increase of tires, carried by consumers
    imported += tax     # cost increase of imported waste, carried by exporters
    mixed += tax        # cost increase of mixed waste (what product?), carried py public authorities (?)

    CHP.reset()
    return NPV


if __name__ == "__main__":

    # the main function resembles the controller
    plants_df = pd.read_csv("plant_data.csv",delimiter=";")
    plant_data = plants_df.iloc[0]
    print(plant_data)

    aspen_df = pd.read_csv("amine.csv", sep=";", decimal=',')
    aspen_interpolators = create_interpolators(aspen_df)

    # initate a CHP and calculate its nominal energy balance
    CHP = WASTE_PLANT(
        name=plant_data["Plant Name"],
        Qdh=plant_data["Heat output (MWheat)"],
        P=plant_data["Electric output (MWe)"],
        Qfgc=plant_data["Existing FGC heat output (MWheat)"],
        Tsteam=plant_data["Live steam temperature (degC)"],
        psteam=plant_data["Live steam pressure (bar)"],
    )

    CHP.estimate_nominal_cycle() 

    # the RDM evaluation starts below
    NPV = WACCS_EPR(CHP=CHP, interpolators=aspen_interpolators)
    print("NPV =", NPV)