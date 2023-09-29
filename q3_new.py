import parameters_q3_new as p
from injectIntoSingleNeurone import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from itertools import product
from plotting import *

plt.close("all")


def generateNeuronalPopulation():
    return [deepcopy(p.singleNeuron) for i in range(p.populationSize)]

def generateCurrentToInject():
  if(p.plotGraphsForQuestion == 1):
    print("Not configured question 1 plotting.")
    pass
  else: 
    return np.ones(p.timeSteps.size) * (0.5*p.I_0*(1 + np.cos(2*np.pi*p.f*p.timeSteps)))


def splitActivityIntoBins(neuronalPopulation):
    binnedSpikes = np.mean([n['isSpikingAtTime'].reshape(
        p.binWidth, -1).sum(axis=1) for n in neuronalPopulation], axis=0)
    return binnedSpikes

def plotQuestion1(currentToInject):
    # Question 1
  plt.figure()
  plt.plot(p.timeSteps, currentToInject)
  plt.xlabel('time (s)')
  plt.ylabel('I_ext (A)')
  plt.title('Current injected (I_ext) into each neurone over time')
  
  
currentToInject = generateCurrentToInject()

# Loop through each seed and coEfficientVariation
for coEfficientOfVariation in p.coEfficientOfVariations:
    binnedSpikesBySeed = np.zeros((len(p.seeds), p.binnedTime.size))
    
    for seedIndex, seed in enumerate(p.seeds):
        p.setSeed(seed)
        neuronalPopulation = [p.generateNeurone(scale=p.getSd(
            coEfficientOfVariation)) for i in range(p.populationSize)]

        for neuroneIndex, neurone in enumerate(neuronalPopulation):
            print('Injecting neurone: '+str(neuroneIndex) +
                  '/'+str(len(neuronalPopulation)))
            neurone = injectIntoSingleNeuron(neurone=neurone, current=currentToInject)

        binnedSpikesBySeed[seedIndex] = splitActivityIntoBins(neuronalPopulation)
    plotQuestion2(binnedSpikesBySeed, neuronalPopulation)
    


# Question 3
plotQuestion3()

plt.show()
