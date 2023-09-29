import parameters_q3_new as p
import matplotlib.pyplot as plt
import numpy as np

def getG_t(A, B, phi):
  return A + B*np.cos(2*np.pi*p.f*p.timeSteps + phi)

def plotQuestion2(binnedSpikesBySeed, neuronalPopulation):
    binnedSpikesBySeedsMean = binnedSpikesBySeed.mean(axis=0)
    binnedSpikesBySeedsSd = binnedSpikesBySeed.std(axis=0)
    meanHz = binnedSpikesBySeedsMean.sum() / (p.simulationTiming['until'])

    fig2 = plt.figure(2)
    plt.suptitle('Averaged spike times of a neuronal population')
    plt.title('('+str(len(neuronalPopulation))+' neurones)\n' +
              "Hz (mean): "+str(meanHz))
    plt.xlabel('Time (s)')
    plt.ylabel('Average number of activity spikes')
    plt.bar(p.binnedTime, binnedSpikesBySeedsMean,  width=p.desiredBinWidth,
            yerr=binnedSpikesBySeedsSd, capsize=2, ecolor='red')


def plotQuestion3():
    ydata = getG_t(1, 10, 1)
    plt.figure()
    plt.title('Guessed parameters')
    plt.plot(p.timeSteps, ydata)
