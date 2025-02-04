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
import searoute as sr

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
    def __init__(self, name, fuel, Qdh, P, Qfgc, Tsteam, psteam):
        self.name = name
        self.fuel = fuel
        self.Qfuel = None
        self.Qdh = Qdh
        self.P = P
        self.Qfgc = Qfgc
        self.Tsteam = Tsteam
        self.psteam = psteam
        self.msteam = None
        self.states = {}

        self.technology_assumptions = None
        self.economic_assumptions = None
        self.interpolators = None
        self.results = {}
        self.nominal_state = {}

    def get(self, parameter):
        return np.round( self.aspen_data[parameter].values[0] )

    def estimate_nominal_cycle(self):
        live = State("live", self.psteam, self.Tsteam)

        Ptarget = self.P
        max_iterations = 10000
        pcond_guess = self.psteam
        Pestimated = 0
        i = 0
        tol = 0.05
        while abs(Pestimated - Ptarget) > Ptarget*tol and i < max_iterations:
            pcond_guess = pcond_guess - 0.1
            mix = State("mix", p=pcond_guess, s=live.s, mix=True)
            boiler = State("boiler", pcond_guess, satL=True)
            msteam = self.Qdh/(mix.h-boiler.h)
            Pestimated = msteam*(live.h-mix.h)
            i += 1
        if i == max_iterations:
            raise ValueError("Couldn't estimate Rankine cycle!")

        self.Qfuel = msteam*(live.h-boiler.h)
        self.P = Pestimated
        self.msteam = msteam
        self.states = {"boiler": boiler, "mix": mix, "live": live}

        # This is a dict with the attribute values of the nominal state. But we have to exclude the initial 'nominal_state' from being copied!
        self.nominal_state = copy.deepcopy({k: v for k, v in self.__dict__.items() if k != 'nominal_state'})

        if msteam is not None and Pestimated is not None and self.Qfuel > 0 and pcond_guess > 0:
            return
        else:
            raise ValueError("One or more of the variables (msteam, Pestimated, Qfuel, pcond_guess) is not positive.")
        
    def burn_fuel(self):
        mfuel = self.Qfuel / 11.1 # Assumed LHV [MJ/kg] from Hammar's thesis
        C_fuel = 0.298 * mfuel   # [kg/s]
        m_fluegas = 5.99 * mfuel # [kg/s]
        V_fluegas = 4.70 * mfuel # [Nm3/s]
        m_CO2 = 1.192 * mfuel    # [kg/s]

        duration = self.technology_assumptions["time"]
        duration_increase = self.technology_assumptions["duration_increase"]

        self.results["Qextra"] = self.Qfuel*duration_increase #[MWh/yr]
        self.results["C_fuel"] = C_fuel*3600/1000 *(duration+duration_increase) /1000                                   #[ktC/yr]
        self.gases = {
            "nominal_emissions": m_CO2*3600/1000 *(duration) /1000,
            "boiler_emissions": m_CO2*3600/1000 *(duration+duration_increase) /1000,                                    #[ktCO2/yr]
            "captured_emissions": m_CO2*3600/1000 *(duration+duration_increase) /1000*self.technology_assumptions["rate"],   
            "m_fluegas": m_fluegas,                                                                                     #[kg/s]
            "V_fluegas": V_fluegas,                                                                                     #[Nm3/s]
        }
        return self.gases["nominal_emissions"]
    
    def size_amine(self):   
        new_Flow = self.gases["m_fluegas"]  # [kg/s]
        if new_Flow < 3:                    # the CHP interpolators only work between 3 kg/s / above 140kgs
            new_Flow = 3
        if new_Flow > 140:
            new_Flow = 140

        new_Rcapture = self.technology_assumptions["rate"]*100
        new_y_values = {}

        for column_name, interp_func in self.interpolators.items():
            new_y = interp_func(([new_Flow], [new_Rcapture]))
            new_y_values[column_name] = new_y

        new_data = pd.DataFrame({
            'Flow': [new_Flow],
            'Rcapture': [new_Rcapture],
            **new_y_values  # Unpack new y values dictionary
        })

        self.aspen_data = new_data
        self.results["Q_reboiler"] = self.get("Qreb")/1000 # [MW]
        return
    
    def power_amine(self):
        boiler = self.states["boiler"]
        mix = self.states["mix"]
        live = self.states["live"]
        dTreb = self.technology_assumptions["dTreb"]

        # Find the reboiler states [a,d] and calculate required mass m,CCS 
        mtot = self.Qfuel*1000 / (live.h-boiler.h) 
        TCCS = self.get("Treb") + dTreb
        pCCS = steamTable.psat_t(TCCS)

        a = State("a",pCCS,s=live.s,mix=True) 
        d = State("d",pCCS,satL=True)
        mCCS = self.get("Qreb") / (a.h-d.h)
        mB = mtot-mCCS

        W = 0
        for Wi in ["Wpumps","Wcfg","Wc1","Wc2","Wc3","Wrefr1","Wrefr2","Wrecomp"]:
            W += self.get(Wi)

        # The new power output depends on the pressures of p,mix and p,CCS
        if a.p > mix.p: 
            Pnew = mtot*(live.h-a.h) + mB*(a.h-mix.h) - W
        else: 
            Pnew = mtot*(live.h-mix.h) + mCCS*(mix.h-a.h) - W

        Plost = (mtot*(live.h-mix.h) - Pnew)
        Qnew = mB*(mix.h-boiler.h)
        Qlost = (mtot*(mix.h-boiler.h) - Qnew)

        self.P = Pnew/1000
        self.Qdh = Qnew/1000
        self.reboiler_steam = [a,d]
        self.results["W_captureplant"] = W/1000 
        self.results["Plost"] = Plost/1000
        self.results["Qlost"] = Qlost/1000
        return self.gases["captured_emissions"]

    def merge_heat(self):
        # Extract stream data from Aspen data using self.get()
        considered_streams = ['wash', 'strip', 'lean', 'int2', 'int1', 'dhx', 'dry', 'rcond', 'rint', 'preliq'] # For CHPs
        consider_dcc = False
        if consider_dcc: # For industrial (pulp) cases
            considered_streams.append('dcc') 

        stream_data = {}
        for component in considered_streams:
            stream_data[component] = {
                'Q': -self.get(f"Q{component}"),
                'Tin': self.get(f"Tin{component}")-273.15,
                'Tout': self.get(f"Tout{component}")-273.15
            }

        # Identify temperature ranges of the streams
        temperatures = []
        for component, data in stream_data.items():
            temperatures.extend([data['Tin'], data['Tout']])

        unique_temperatures = list(dict.fromkeys(temperatures)) 
        unique_temperatures.sort(reverse=True)

        temperature_ranges = []
        for i in range(len(unique_temperatures) - 1):
            temperature_range = (unique_temperatures[i + 1], unique_temperatures[i])
            temperature_ranges.append(temperature_range)
        
        # Construct and save the composite curves using the temperature ranges
        composite_curve = [[0, temperature_ranges[0][1]]] # First data point has 0 heat and the highest temperature
        Qranges = []
        for temperature_range in temperature_ranges:
            Ctot = 0
            for component, data in stream_data.items():
                TIN = data['Tin']
                TOUT = data['Tout']
                if TIN == TOUT:
                    TIN += 0.001 # To avoid division by zero
                Q = data['Q']
                C = Q/(TIN-TOUT)
                
                if TIN >= temperature_range[1] and TOUT <= temperature_range[0]:
                    Ctot += C

            Qrange = Ctot*(temperature_range[1]-temperature_range[0])
            Qranges.append(Qrange)
            composite_curve.append([sum(Qranges), temperature_range[0]])
        self.composite_curve = composite_curve
        return

    def recover_heat(self):
        composite_curve = self.composite_curve
        Tsupp = self.technology_assumptions["Tsupp"]
        Tlow = self.technology_assumptions["Tlow"]
        dTmin = self.technology_assumptions["dTmin"]
        
        shifted_curve = [[point[0], point[1] - dTmin] for point in composite_curve]
        curve = shifted_curve

        # Find the elbow point (point of maximum curvature) from the distance of each point
        def distance(p1, p2, p):
            x1, y1 = p1
            x2, y2 = p2
            x0, y0 = p
            return np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances = [distance(curve[0], curve[-1], point) for point in curve]
        differences = np.diff(distances)
        max_curvature_index = np.argmax(differences) + 1

        # Finding low and high points.
        def linear_interpolation(curve, ynew):
            # Find the nearest points, then perform inverse linear interpolation
            y_values = [point[1] for point in curve]    
            nearest_index = min(range(len(y_values)), key=lambda i: abs(y_values[i] - ynew))
            x1, y1 = curve[nearest_index]
            if nearest_index == 0:
                x2, y2 = curve[1]
            elif nearest_index == len(curve) - 1:
                x2, y2 = curve[-2]
            else:
                x2, y2 = curve[nearest_index + 1]

            if y2 == y1:
                return x1 + (x2 - x1) * (ynew - y1) / (y2 - y1)
            else:
                return x1 + (x2 - x1) * (ynew - y1) / (y2 - y1)
        
        Qsupp = linear_interpolation(curve, Tsupp) # This sets a threshold: no heat above Tsupp=86C is recoverable
        Qlow = linear_interpolation(curve, Tlow)
        Qpinch, Tpinch = curve[max_curvature_index][0], curve[max_curvature_index][1]
        if Qlow < Qpinch:
            Qlow = Qpinch # Sometimes Qlow is poorly estimated, then just set the low grade heat to zero
        Qhex = (Qpinch-Qsupp) + (Qlow-Qpinch)

        # Qhp = composite_curve[-1][0] - Qlow # This overestimates Qhp for smaller plants where the composite curve is less accurate
        Qmax_beiron = self.get("Qreb")*1.18 # Assumptions from (Beiron, 2022)
        Qhp = Qmax_beiron - Qlow
        if not self.technology_assumptions["heat_pump"]:
            Qhp = 0
        Php = Qhp/self.technology_assumptions["COP"]

        self.P -= Php/1000
        self.results["Plost"] += Php/1000
        self.Qdh += (Qhex + Qhp)/1000
        self.results["Qlost"] -= (Qhex + Qhp)/1000
        self.results["Qrecovered"] = (Qhex + Qhp)/1000
        self.results["Qhp"] = Qhp/1000
        self.results["qrecovered"] = (Qhex + Qhp)/self.get("Qreb")
        self.results["efficiency"] = (self.P + self.Qdh + self.Qfgc) / self.Qfuel
        self.QTdict = {
            "supp": [Qsupp, Tsupp], # Also hard-coded
            "low": [Qlow, Tlow],
            "pinch": [Qpinch, Tpinch]
        }

        if Qhex<0 or Qhp<0:
            self.plot_hexchange(show=True)
            raise ValueError("Infeasible heat exchange")
        return (self.Qdh+self.Qfgc), self.P, self.Qfuel

    def estimate_CAPEX(self, escalate=True):
        X = self.economic_assumptions

        # Estimating base cost of capture plant
        CAPEX = X['alpha'] * (self.gases["V_fluegas"])**X['beta']   #[MEUR](Eliasson, 2022)
        CAPEX *= X['CEPCI'] *1000                                   #[kEUR]

        # Adding cost of HEXs (+~1% cost) and HP (+~21% cost)
        Qhex = self.results["Qrecovered"] - self.results["Qhp"]     #[MW]
        U = self.technology_assumptions["U"] /1000                  #[kW/m2K]
        A = Qhex*1000/(U * self.technology_assumptions["dTmin"])    # This overestimates the area as dTmin<dTln so it is pessimistic costwise
        CAPEX_hex = X["cHEX"]*A**0.9145                             #[kEUR] Original val: 0.571 (Eliasson, 2022)
        CAPEX += CAPEX_hex

        if self.technology_assumptions["heat_pump"]:
            Qhp = self.results["Qhp"]
            CAPEX_hp = X["cHP"]*1000 * Qhp                          #[kEUR], probably represents 2-4 pumps
            CAPEX += CAPEX_hp  

        if escalate:
            CAPEX *= 1 + X['ownercost']
            escalation = sum((1 + X['rescalation']) ** (n - 1) * (1 / X['yexpenses']) for n in range(1, X['yexpenses'] + 1)) # equals ~1.03
            cfunding = sum(X['WACC'] * (X['yexpenses'] - n + 1) * (1 + X['rescalation']) ** (n - 1) * (1 / X['yexpenses']) for n in range(1, X['yexpenses'] + 1)) # equals ~0.10
            CAPEX *= escalation + cfunding     
        fixed_OPEX = X['fixed'] * CAPEX 

        annualization = (X['i'] * (1 + X['i']) ** X['t']) / ((1 + X['i']) ** X['t'] - 1)
        aCAPEX = annualization * CAPEX   

        self.results["CAPEX"] = CAPEX           #[kEUR]    
        self.results["aCAPEX"] = aCAPEX         #[kEUR/yr]
        self.results["fixed_OPEX"] = fixed_OPEX #[kEUR/yr]
        return                       

    def future_scenarios(self):
        # I should save the cash flows to be able to print these! Call the array "cash"
        h = self.economic_assumptions["time"]
        mC = self.results["C_fuel"] * self.technology_assumptions["fossil"]  #[ktC/yr] NOTE: should include scenarios of plastic fractions evolving!

        cashflow = []
        cashflow.append(-self.results["CAPEX"]*1000)   #[EUR]

        for n in range(0, self.economic_assumptions["t"]):
            electricity_revenue = - self.results["Plost"]* h * self.economic_assumptions["celc"] #[EUR/yr]
            heat_revenue =        - self.results["Qlost"]* h * self.economic_assumptions["cheat"]*self.economic_assumptions["celc"]
            carbon_revenue = mC*3.67*1000 * self.economic_assumptions["tax"]        
            carbon_revenue += self.economic_assumptions["cETS"] * self.gases["captured_emissions"]*1000 * self.technology_assumptions["fossil"] #[EUR/yr] NOTE: half is avoided fossil CO2
            revenues = electricity_revenue + heat_revenue + carbon_revenue

            transport_cost = self.economic_assumptions["ctrans"] * self.gases["captured_emissions"]*1000 #[EUR/yr]
            storage_cost   = self.economic_assumptions["cstore"] * self.gases["captured_emissions"]*1000 #[EUR/yr]
            auxiliary_costs = self.results["fixed_OPEX"]*1000
            costs = transport_cost + storage_cost + auxiliary_costs

            cashflow.append( (revenues - costs)/(1 + self.economic_assumptions["i"])**n )
            
        NPV = sum(cashflow)

        # Ensures cash array is always of the expected length
        length = 31
        if len(cashflow) < length:
            cashflow = np.pad(cashflow, (0, length - len(cashflow)), mode='constant', constant_values=np.nan)
        elif len(cashflow) > length:
            cashflow = cashflow[:length]
            
        return NPV, cashflow

    def print_energybalance(self):
        print(f"\n{'Heat output (Qdh)':<20}: {self.Qdh} MWheat")
        print(f"{'Electric output (P)':<20}: {self.P} MWe")
        print(f"{'Existing FGC (Qfgc)':<20}: {self.Qfgc} MWheat")
        print(f"{'Fuel input (Qfuel)':<20}: {self.Qfuel} MWheat")

        for key,value in self.results.items():
            print(f"{' ':<5} {key:<20} {value}")
        for key,value in self.gases.items():
            print(f"{key:<20} {value}")

    def plot_hexchange(self, show=False): 
        Qsupp, Tsupp = self.QTdict["supp"]
        Qlow, Tlow = self.QTdict["low"]
        Qpinch, Tpinch = self.QTdict["pinch"]
        dTmin = self.technology_assumptions["dTmin"]

        plt.figure(figsize=(10, 8))
        composite_curve = self.composite_curve
        shifted_curve = [[point[0], point[1] - dTmin] for point in composite_curve]
        (Qpinch-Qsupp) + (Qlow-Qpinch)

        plt.plot([0, self.get("Qreb")], [self.get("Treb"), self.get("Treb")], marker='*', color='#a100ba', label='Qreboiler')
        plt.plot([point[0] for point in composite_curve], [point[1] for point in composite_curve], marker='o', color='red', label='T of CCS streams')
        plt.plot([point[0] for point in shifted_curve], [point[1] for point in shifted_curve], marker='o', color='pink', label='T shifted')
        plt.plot([Qpinch, Qlow], [Tpinch, Tlow], marker='x', color='#069AF3', label='Qlowgrade')
        plt.plot([Qpinch, Qsupp], [Tpinch, Tsupp], marker='x', color='blue', label='Qhighgrade')
        plt.plot([Qlow, composite_curve[-1][0]], [20, 15], marker='o', color='#0000FF', label='Cooling water') # NOTE: hard-coded CW temps.

        plt.text(26000, 55, f'dTmin={round(dTmin,2)} C', color='black', fontsize=12, ha='center', va='center')
        plt.text(26000, 115, f'Qreb={round(self.get("Qreb")/1000)} MW', color='#a100ba', fontsize=12, ha='center', va='center')       
        plt.text(5000, 60, f'Qhighgrade={round((Qpinch-Qsupp)/1000)} MW', color='#0000FF', fontsize=12, ha='center', va='center')
        plt.text(5000, 40, f'Qlowgrade={round((Qlow-Qpinch)/1000)} MW', color='#069AF3', fontsize=12, ha='center', va='center')
        plt.text(10000, 15, f'Qcoolingwater={round((composite_curve[-1][0]-Qlow)/1000)} MW', color='#0000FF', fontsize=12, ha='center', va='center')

        plt.xlabel('Q [kW]')
        plt.ylabel('T [C]')
        plt.title(f'[{self.name}] Heat exchange between composite curve and district heating')
        plt.legend()
        if show:
            plt.show()
        return
    
    def reset(self):
        self.__dict__.update(copy.deepcopy(self.nominal_state)) # Resets to the nominal state values

# Here I create a main model function, which relies on the helper functions:
def WACCS_EPR( 
    dTreb=10,
    Tsupp=86,
    Tlow=38,
    dTmin=7,
    COP = 3,
    U = 1500,
    fossil = 0.5,

    alpha=6.12,
    beta=0.6336,
    CEPCI=600/550,
    fixed=0.06,
    ownercost=0.2,
    WACC=0.05,
    yexpenses=3,
    rescalation=0.03,
    i=0.075,
    t=25,

    celc=40,
    cheat=0.80,
    cbio=99999999,
    cMEA=2,
    cHP=0.86,
    cHEX=0.571,
    cETS=70,
    ctrans=30,
    cstore=60,

    time= 8000,
    rate = 0.90,
    tax = 300, #EUR/tCO2
    CHP = None,
    interpolators = None
):
    CHP.interpolators = interpolators
    CHP.technology_assumptions = {
        'U': U,
        "time": time,
        "duration_increase": 0,
        "rate": rate,
        "heat_pump": True,
        "dTreb": dTreb,
        "Tsupp": Tsupp,
        "Tlow": Tlow,
        "dTmin": dTmin,
        "COP": COP,
        "fossil": fossil,
    }

    CHP.economic_assumptions = {
        'time': time,
        'alpha': alpha,
        'beta': beta,
        'CEPCI': CEPCI,
        'fixed': fixed,
        'ownercost': ownercost,
        'WACC': WACC,
        'yexpenses': yexpenses,
        'rescalation': rescalation,
        'i': i,
        't': t,
        'celc': celc,
        'cheat': cheat,
        'cbio': cbio,
        'cMEA': cMEA,
        'cHP': cHP,
        'cHEX' : cHEX,
        'cETS' : cETS,
        'ctrans' : ctrans,
        'cstore' : cstore,
        'tax' : tax
    }

    # Size a capture plant and power it
    emissions_nominal = CHP.burn_fuel() # NOTE: burn_fuel() massflows of carbon must be linked to the tax in future_scenarios, i.e. mplastic
    CHP.size_amine()
    emissions_captured = CHP.power_amine()
    # CHP.print_energybalance()

    # Calculate recoverable heat and integrate
    CHP.merge_heat()
    energy_balance = CHP.recover_heat()

    # Calculate CAPEX and NPV
    CHP.estimate_CAPEX(escalate=True)
    NPV, cash = CHP.future_scenarios()

    # Calculate product/waste cost increases for "this plant" (will be the same for all plants)
    bag = (3.1415*10**-5 * tax) /0.3
    floor = (0.00217998 * tax)  /50  
    tires = (0.02889024 * tax)  /120     
    imported = (0.734 * tax)    /55
    # mixed = tax   # cost increase of mixed waste (what product?), carried py public authorities (?)

    q, eta = CHP.results["qrecovered"], CHP.results["efficiency"]
    CHP.reset()
    return eta, NPV, bag, floor, tires, imported, cash


if __name__ == "__main__":

    # the main function resembles the controller
    plants_df = pd.read_csv("data/w2e_data.csv",delimiter=";")
    plant_data = plants_df.iloc[0]
    print(plant_data)

    aspen_df = pd.read_csv("data/amine.csv", sep=";", decimal=',')
    aspen_interpolators = create_interpolators(aspen_df)

    # initate a CHP and calculate its nominal energy balance
    CHP = WASTE_PLANT(
        name=plant_data["Name"],
        fuel=plant_data["Fuel (W=waste, B=biomass)"],
        Qdh=plant_data["Heat output (MWheat)"],
        P=plant_data["Electric output (MWe)"],
        Qfgc=plant_data["Existing FGC heat output (MWheat)"],
        Tsteam=plant_data["Live steam temperature (degC)"],
        psteam=plant_data["Live steam pressure (bar)"],
    )
    CHP.estimate_nominal_cycle() 

    # the RDM evaluation starts below
    eta, NPV, bag, floor, tires, imported, cash = WACCS_EPR(CHP=CHP, interpolators=aspen_interpolators)
    outcomes = {
        "eta": eta,
        "NPV": NPV,
        "bag": bag,
        "floor": floor,
        "tires": tires,
        "imported": imported,
        "cash": cash
    }
    for name, value in outcomes.items():
        print(f"{name} = {value}")

    # plotting here
    # categories = ["bag", "floor", "tires", "imported"]
    # values1 = [bag * 100, floor * 100, tires * 100]  # First y-axis (scaled)
    # values2 = [imported * 100]  # Second y-axis (scaled)
    # fig, ax1 = plt.subplots(figsize=(8, 5))

    # # X locations for bars
    # x = np.arange(len(categories))
    # bar_width = 0.6  
    # # First y-axis bars (bag, floor, tires)
    # ax1.bar(x[:-1], values1, width=bar_width, color=['blue', 'green', 'red'], label="Bag, Floor, Tires")
    # ax1.set_ylabel("Bag, Floor, Tires [%]", color='black')
    # # Create second y-axis for "imported"
    # ax2 = ax1.twinx()
    # ax2.bar(x[-1], values2, width=bar_width, color='purple', label="Imported")
    # ax2.set_ylabel("Imported [%]", color='black')
    # # Set x-ticks and labels
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(categories)
    # plt.title("Bar Plot with Dual Y-Axis (Scaled to %)")

    # # plotting cash flows
    # time_steps = range(len(cash))  
    # plt.figure(figsize=(8, 5))
    # plt.plot(time_steps, cash, marker='o', linestyle='-', color='b', label="Cash Flow (€)")
    # plt.xlabel("Time Steps")
    # plt.ylabel("Cash (€)")
    # plt.title("Cash Flow Over Time")
    # plt.legend()
    # plt.grid(True)
    plt.show()