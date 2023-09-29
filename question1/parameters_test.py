import numpy as np

simulationTiming = {
    # start simulating from (seconds)
    'from': 0,
    # end simulation at - (seconds)
    'until': 1,
    # increase time step by this amount - (seconds)
    'incrementBy': 1e-3,
}
populationSize = 100  # number of parallel, uncoupled units
V = {
    'threshold': 10e-3,  # V_thr -  threshold potential in V
    'reset': 0,  # V_reset - reset potential in V
    'rest': 0  # V_resting - resting potential in V
}
R = {
    'm': 1e6 # R_m - membrane resistance in Ohms (Ω)
}
T = {
    'm': 10e-3   # tau_m - membrane time-constant - seconds
}
C = T['m']/R['m']  # capacitance - s/Ω

inputCurrents = np.linspace(0.9e-8, 2e-8, 1000) 
#inputCurrents = [1.8e-8, 1.9e-8]
showSingularPlot = False


# Do not edit below this line
timeSteps = np.arange(
    simulationTiming['from'], simulationTiming['until'], simulationTiming['incrementBy'])

V['m'] = V['rest']*np.ones(len(timeSteps))

singleNeuron = {
    'V': V,
    'R': R,
    'T': T,
    'C': C,
    'isSpikingAtTime': np.zeros(timeSteps.size),
    'numberOfSpikes': 0
}
