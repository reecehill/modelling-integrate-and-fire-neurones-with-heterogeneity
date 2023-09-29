import parameters_test as p
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from matplotlib.ticker import FuncFormatter


def getDV(neurone, timeIndex, I_ext):
  return (-(1/neurone['R']['m'])*(neurone['V']['m'][timeIndex]-neurone['V']['rest'])+I_ext)/neurone['C']


def injectIntoSingleNeuron(neurone, current):
  neurone['numberOfSpikes'] = 0
  for timeIndex, timeStep in enumerate(p.timeSteps):
    # newVm = V_m(t) + 𐤃V_m(t)
    newVm = neurone['V']['m'][timeIndex] + (p.simulationTiming['incrementBy'] * getDV(
        neurone, timeIndex, current[timeIndex]))

    # If neurone hits threshold, override value and set to 40.
    if(newVm >= neurone['V']['threshold']):
      newVm = 40e-3
      neurone['numberOfSpikes'] += 1
      nextVm = neurone['V']['rest']
    else:
      nextVm = newVm

    neurone['V']['m'][timeIndex] = newVm
    if(timeIndex+1 < p.timeSteps.size):
      neurone['V']['m'][timeIndex+1] = nextVm

  if(p.showSingularPlot):
    plt.plot(p.timeSteps, neurone['V']['m'])
    plt.xlabel('time (s)')
    plt.ylabel('V_m')
    plt.show()
  return neurone


plt.close("all")

## PRODUCES I_EXT vs. FIRING RATE PLOT
numberOfSpikesPerSecond = np.zeros(len(p.inputCurrents))
for inputCurrentIndex, inputCurrent in enumerate(p.inputCurrents):
  print("Simulating input current: "+str(inputCurrentIndex)+'/'+str(len(p.inputCurrents)))
  
  # Generate new vector of inputcurrents over time.
  currentToInject = np.ones(p.timeSteps.size) * inputCurrent
  # Generate new neurone, using parameters from file.
  neurone = deepcopy(p.singleNeuron)
  # Simulate neurone. Returns the entire neurone, but we only take the numberOfSpikes
  numberOfSpikesPerSecond[inputCurrentIndex] = injectIntoSingleNeuron(
      neurone, current=currentToInject)['numberOfSpikes']
plt.plot(np.multiply(p.inputCurrents, 1e9), numberOfSpikesPerSecond)
plt.xlabel('$I_{ext} (nA)$')
plt.ylabel('Firing rate (Hz)')


if(len(p.inputCurrents) <=2):
  print("If you specify just two input currents, you can view and compare the timecourses of V_m")
  plt.figure()
  plt.suptitle('$V_m$ timecourses')
  hz = neurone['numberOfSpikes'] / (1 / p.simulationTiming['until'])
  plt.title('Simulated firing rate for both $I_{ext}$ (Hz): '+str(hz))
  plt.xlabel('Time (s)')
  plt.ylabel('$V_m$ (mV)')
  ## PRODUCES SINGLE NEURONE POTENTIAL timeseries
  for inputCurrentIndex, inputCurrent in enumerate(p.inputCurrents):
    print("Simulating input current for V_m of single current")
    currentToInject = np.ones(p.timeSteps.size) * inputCurrent
    neurone = injectIntoSingleNeuron(
        neurone, current=currentToInject)

    
    plt.plot(p.timeSteps.reshape(25, -1)[-1], neurone['V']['m'].reshape(25, -1)[-1] * 1e3, label=str(inputCurrent))
  plt.legend(title='$I_{ext}$ (nA):')


plt.show()
