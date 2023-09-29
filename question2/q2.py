import parameters_q2 as p
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def getDV(neurone, timeIndex, I_ext):
  return (-(1/neurone['R']['m'])*(neurone['V']['m'][timeIndex]-neurone['V']['rest'])+I_ext)/neurone['C']


def injectIntoSingleNeuron(neurone, current):  
  for timeIndex, timeStep in enumerate(p.timeSteps): 
    # Calculate the potential for this timestep.
    # newVm = V_m(t) + 𐤃V_m(t)
    VmAtCurrentTimeStep = neurone['V']['m'][timeIndex] + (p.simulationTiming['incrementBy'] * getDV(
        neurone, timeIndex, current[timeIndex]))
    VmAtNextTimeStep = VmAtCurrentTimeStep
    # If neurone hits threshold...
    if(VmAtCurrentTimeStep >= neurone['V']['threshold']):
      # Override potential, and set to 40mV.
      VmAtCurrentTimeStep = 40e-3
      # Set next potential to V_rest.
      VmAtNextTimeStep = neurone['V']['rest']
      
      # Mark this timestep as a spike.
      neurone['isSpikingAtTime'][timeIndex] = 1
      
    neurone['V']['m'][timeIndex] = VmAtCurrentTimeStep
    if(timeIndex+1 <= p.timeSteps.size-1):
      neurone['V']['m'][timeIndex+1] = VmAtNextTimeStep
      
  if(p.showSingularPlot):    
    plt.figure()
    plt.plot(p.timeSteps, neurone['V']['m'], label='Spiking profile of single neurone')
    plt.plot(p.timeSteps, current, label="I_ext")
    plt.legend()
    plt.xlabel('time (t)')
    plt.ylabel('V_m')
    plt.show()
    input('Paused... hit enter to continue')
    
  neurone['numberOfSpikes'] = np.sum(neurone['isSpikingAtTime'])
  return neurone
  
plt.close("all")


neuronalPopulation = [deepcopy(p.singleNeuron) for i in range(p.populationSize)]
currentToInject = np.ones(p.V['m'].size) * (0.5*p.I_0*(1 + np.cos(2*np.pi*p.f*p.timeSteps)))

fig1 = plt.figure(1)
plt.plot(p.timeSteps*1e3, currentToInject*1e9)
plt.xlabel('Time (ms)')
plt.ylabel('$I_{ext}$ (nA)')
plt.title('Current injected ($I_{ext}$) into each neurone over time')

binWidth = int(p.simulationTiming['until'] / p.desiredBinWidth)
for neuroneIndex, neurone in enumerate(neuronalPopulation):
  print('Injecting neurone: '+str(neuroneIndex)+'/'+str(len(neuronalPopulation)))
  neurone = injectIntoSingleNeuron(neurone=neurone, current=currentToInject)

binnedSpikes = np.mean([n['isSpikingAtTime'].reshape(binWidth, -1).sum(axis=1) for n in neuronalPopulation], axis=0)
binnedTime = p.timeSteps.reshape(binWidth, -1).mean(axis=1)
meanHz = np.sum(binnedSpikes) / (p.simulationTiming['until'])

plt.figure()
plt.suptitle('Averaged spike times of a neuronal population')
plt.title('('+str(len(neuronalPopulation))+' neurones)\n' +
          "Population firing rate (Hz): "+str(meanHz))
plt.xlabel('Time (ms)')
plt.ylabel('Mean number of spikes per neurone')
plt.bar(np.multiply(binnedTime,1e3), binnedSpikes,  width=p.desiredBinWidth*1e3)

plt.figure()
plt.suptitle('Firing rate of neuronal population over time')
plt.title('('+str(len(neuronalPopulation))+' neurones)\n' +
          "Mean population firing rate (Hz): "+str(meanHz))
plt.xlabel('Time (ms)')
plt.ylabel('Firing rate of population (Hz)')
meanHzPerBin = binnedSpikes / p.simulationTiming['until']
meanHz = meanHzPerBin.mean()*p.populationSize
populationFiringRate = meanHzPerBin*p.populationSize
plt.bar(np.multiply(binnedTime, 1e3),
        populationFiringRate,  width=p.desiredBinWidth*1e3)

plt.figure()
plt.suptitle('Averaged spike times of a neuronal population')
plt.title('('+str(len(neuronalPopulation))+' neurones)')
plt.plot(binnedTime, binnedSpikes)
plt.xlabel('Time (s)')
plt.ylabel('Average number of activity spikes')

plt.figure()
color='tab:red'
plt.suptitle('Time course of $I_{ext}$ and a neurone\'s $V_m$')
plt.plot(p.timeSteps*1e3, neuronalPopulation[-1]['V']['m']*1e3, color=color)
plt.tick_params(axis='y', labelcolor=color)
plt.xlabel('Time (ms)')
plt.ylabel('$V_m$ (mV)', color=color)

color = 'tab:blue'
plt.twinx()
plt.tick_params(axis='y', labelcolor=color)
plt.plot(p.timeSteps*1e3, currentToInject*1e9, color=color)
plt.ylabel('$I_{ext}$ (nA)', color=color)
plt.tight_layout()


plt.figure()
plt.title('Spike raster for entire population\n'+str(p.populationSize)+' neurones')
spikeEventPositionsForPopulation = [np.array(n['isSpikingAtTime']).nonzero()[0] for n in neuronalPopulation]
colors1 = ['C{}'.format(i) for i in range(p.populationSize)]
plt.ylabel('Neurone')
plt.xlabel('Time (ms)')
plt.eventplot(spikeEventPositionsForPopulation, colors=colors1, linewidths=5)

plt.show()