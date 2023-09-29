import numpy as np
import sys

seeds = list(range(1))
coEfficientOfVariations = np.linspace(0.00001, 200, 8)
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
    'reset': 0e-3,  # V_reset - reset potential in V
    'rest': 0e-3  # V_resting - resting potential in V
}
R = {
    'm': 1e6  # R_m - membrane resistance in Ohms (Ω)
}
T = {
    'm': 10e-3   # tau_m - membrane time-constant - seconds
}
C = T['m']/R['m']  # capacitance - ms/mΩ
I_0 = 20e-9  # external current injected at time=0, (A)
f = 20  # Hz (seconds)


desiredBinWidth = 2e-3  # When simulation is binned, collect into timeseries this large (seconds).

showSingularPlot = False
plotGraphsForQuestion = 1
defaultGaussianMean = 1 * (10**9)  # from MΩ (mega-) to mΩ (milli-)



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
binWidth = int(
    simulationTiming['until'] / desiredBinWidth)
binnedTime = timeSteps.reshape(binWidth, -1).mean(axis=1)


def setSeed(inputSeed):
  global seed
  global randomGenerator
  seed = inputSeed
  randomGenerator = np.random.default_rng(seed)


def getSd(cov):
  return (cov * defaultGaussianMean) / 100


def getFromGaussian(scale, loc):
    x = randomGenerator.normal(scale=scale, loc=loc)
    # If x is sampled as negative, then return the smallest possible number allowed by system (avoids divide by zero)
    return (x) if x > 0 else sys.float_info.min

def generateNeurone(scale, loc=defaultGaussianMean):
    V = {
        'threshold': 10,  # V_thr -  threshold potential in mV
        'reset': 0,  # V_reset - reset potential in mV
        'rest': 0  # V_resting - resting potential in mV
    }
    V['m'] = V['rest']*np.ones(len(timeSteps))

    T = {
        'm': 10  # tau_m - millisecond
    }
    R = {
        'm': getFromGaussian(scale, loc)  # R_m - milliOhms (mΩ)

    }
    C = T['m']/R['m']  # capacitance - ms/mΩ

    return {
        'V': V,
        'R': R,
        'T': T,
        'C': C,
        'isSpikingAtTime': np.zeros(V['m'].size),
        'numberOfSpikes': 0
    }
