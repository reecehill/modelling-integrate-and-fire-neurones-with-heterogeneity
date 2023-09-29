import numpy as np
# The following units are expected:
# Volts: v
# Ohms: Ω
# Time: s
# Amps: A


simulationTiming = {
    # start simulating from (seconds)
    'from': 0,
    # end simulation at - (seconds)
    'until': 200e-3,
    # increase time step by this amount - (seconds)
    'incrementBy': 1e-3,
}
populationSize = 100 # number of parallel, uncoupled neurones (discrete units)

V = {
    'threshold': 10e-3,  # V_thr -  threshold potential in V
    'reset': 0,  # V_reset - reset potential in V
    'rest': 0  # V_resting - resting potential in V
}
R = {
    'm': 1e6  # R_m - membrane resistance in Ohms (Ω)
}
T = {
    'm': 10e-3   # tau_m - membrane time-constant - seconds
}
C = T['m']/R['m']  # capacitance - s/Ω
I_0 = 20e-9  # external current injected at time=0, (A)
f = 20  # Hz (seconds)


desiredBinWidth = 2e-3  # When simulation is binned, collect into timeseries this large (seconds).

showSingularPlot = False

# DO NOT EDIT BELOW THIS LINE.
timeSteps = np.arange(simulationTiming['from'],simulationTiming['until'],simulationTiming['incrementBy'])
V['m'] = V['rest']*np.ones(len(timeSteps))
singleNeuron = {
    'V': V,
    'R': R,
    'T': T,
    'C': C,
    'isSpikingAtTime': np.zeros(V['m'].size),
    'numberOfSpikes': 0
}
