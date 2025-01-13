"""Model file"""
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

    def do_things(self, assumptions):
        NPV = 3.1415
        return NPV

    def reset(self):
        print("resetting...")


# here I create a main model function, which relies on the helper functions:
def WACCS_EPR( 
    time=8000,
    rate=0.90,
    CHP=None,
    interpolators=None
):
    assumptions = {
        "time": time,
        "rate": rate,
        "molar_mass": 29.55
    }
    
    NPV = CHP.do_things(assumptions)

    CHP.reset()
    return NPV


if __name__ == "__main__":

    # load data
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

    # CHP.estimate_nominal_cycle() 
    # CHP.print_energybalance()

    # The RDM evaluation starts below:
    NPV = WACCS_EPR(CHP=CHP, interpolators=aspen_interpolators)
    print("Outcomes: ", NPV)