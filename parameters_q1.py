import numpy as np

simulationTiming = {
    # start simulating from (seconds)
    'from': 0,
    # end simulation at - (seconds)
    'until': 1,
    # increase time step by this amount - (seconds)
    'incrementBy': 0.001,
}
populationSize = 100  # number of parallel, uncoupled units
V = {
    'threshold': 10 * (10**-3),  # V_thr -  threshold potential in V
    'reset': 0 * (10**-3),  # V_reset - reset potential in V
    'rest': 0 * (10**-3)  # V_resting - resting potential in V
}
R = {
    'm': 1 * (10**6)  # R_m - membrane resistance in Ohms (Ω)
}
T = {
    'm': 10 * (10**-3)   # tau_m - membrane time-constant - seconds
}
C = T['m']/R['m']  # capacitance - s/Ω

showSingularPlot = True



# Do not edit below this line
timeSteps = np.arange(simulationTiming['from'], simulationTiming['until'], simulationTiming['incrementBy'])

V['m'] = V['rest']*np.ones(len(timeSteps))

singleNeuron = {
    'V': V,
    'R': R,
    'T': T,
    'C': C,
    'isSpikingAtTime': np.zeros(timeSteps.size),
    'numberOfSpikes': 0
}
