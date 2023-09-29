from random import random
import numpy as np
import sys

# The following units are expected:
# Volts: v
# Ohms: Ω
# Time: s
# Amps: A

seeds = list(range(2))
simulationTiming = {
    'from': 0,  # start simulating from - seconds
    'until': 200e-3,  # end simulation at - seconds
    'incrementBy': 1e-3,  # increase time step by this amount - seconds
}
populationSize = 100  # number of parallel, uncoupled units


I_0 = 20e-9  # nanoAmperes (nA) -> amperes (A)

f = 20  # Hz (in s)

desiredBinWidth = 2e-3  # Each bin represents is X amount of time (in seconds).

showSingularPlot = False

defaultGaussianMean = 1e6  # from MΩ (mega-) to ohms (Ω)

#coEfficientOfVariations = np.linspace(1e-20, 0.5e3, 100)
#coEfficientOfVariations = np.linspace(1, 100, 10)


#coEfficientOfVariations = np.linspace(0, 5e27, 200)
coEfficientOfVariations = np.linspace(0, 9e-12, 2)
#coEfficientOfVariations = [100]

varyVThreshold = 1
varyRm = 1
varyTm = 0
varyC = 1

# DO NOT EDIT BELOW THIS LINE.
def setSeed(inputSeed):
  global seed
  global randomGenerator
  seed = inputSeed
  randomGenerator = np.random.default_rng(seed)

timeSteps = np.arange(
    simulationTiming['from'], simulationTiming['until'], simulationTiming['incrementBy'])

def getSd(cov):
  return (cov * defaultGaussianMean) / 100


def getFromGaussian(scale, loc):
    x = randomGenerator.normal(scale=scale, loc=loc)
    # If x is sampled as negative, then return the smallest possible number allowed by system (avoids divide by zero)
    return (x) if x > 0 else sys.float_info.min


def generateNeurone(scale, loc=defaultGaussianMean):
    V = {
        # V_thr -  threshold potential in V
        'threshold': getFromGaussian(scale, 10e-3) if varyVThreshold else 10e-3,
        'reset': 0,  # V_reset - reset potential in V
        'rest': 0  # V_resting - resting potential in V
    }
    V['m'] = V['rest']*np.ones(len(timeSteps))

    R = {
        'm': getFromGaussian(scale, 1e6) if varyRm else 1e6  # R_m - Ohms (Ω)

    }
    if varyC:
        # capacitance - s/Ω
        C = getFromGaussian(scale, 10e-3/R['m'])
        T = {
            # tau_m - seconds
            'm': R['m']*C
        }


    else:
        # capacitance - s/Ω
        T = {
            # tau_m - seconds
            'm': getFromGaussian(scale, 10e-3) if varyTm else 10e-3
        }
        C = T['m']/R['m']
    return {
        'V': V,
        'R': R,
        'T': T,
        'C': C,
        'isSpikingAtTime': np.zeros(V['m'].size),
        'numberOfSpikes': 0
    }


binWidth = int(simulationTiming['until'] / desiredBinWidth)
binnedTime = timeSteps.reshape(binWidth, -1).mean(axis=1)
