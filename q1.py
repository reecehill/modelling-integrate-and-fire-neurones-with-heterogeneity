import parameters_q1 as p
import matplotlib.pyplot as plt
import numpy as np


def getDV(V_m, I_ext):
  return (-(1/p.R['m'])*(V_m-p.V['rest'])+I_ext)/p.C


def injectIntoSingleNeuron(current):
  numberOfSpikes = 0;
  for timeIndex, timeStep in enumerate(p.timeSteps):
    # newVm = V_m(t) + 𐤃V_m(t)
    newVm = p.V['m'][timeIndex] + (p.simulationTiming['incrementBy'] * getDV(
        p.V['m'][timeIndex], current[timeIndex]))
    
    if(newVm >= p.V['threshold']): # If neurone hits threshold, override value and set to 40.
      newVm = 40 * (10**-3)
      numberOfSpikes += 1
      nextVm = p.V['rest']
    else:
      nextVm = newVm
      
    p.V['m'][timeIndex] = newVm
    if(timeIndex+1 < p.V['m'].size):
      p.V['m'][timeIndex+1] = nextVm
      
  if(p.showSingularPlot):    
    fig = plt.plot(p.timeSteps, p.V['m'])
    plt.xlabel('time (t)')
    plt.ylabel('V_m')
    plt.show()
  return numberOfSpikes
  
plt.close("all")

#currentToInject = np.zeros(p.V['m'].size)
#currentToInject[int(p.V['m'].size*0.25):int(p.V['m'].size*0.75)] = 800
#inputCurrents = np.linspace(1e-8,4e-7,500)
inputCurrents = [5e-8]
numberOfSpikesPerSecond = np.zeros(len(inputCurrents))
for inputCurrentIndex, inputCurrent in enumerate(inputCurrents):
  currentToInject = np.ones(p.V['m'].size) * inputCurrent
  numberOfSpikesPerSecond[inputCurrentIndex] = injectIntoSingleNeuron(current=currentToInject)

plt.plot(inputCurrents, numberOfSpikesPerSecond)
plt.xlabel('I_ext (V)')
plt.yscale('log')
plt.xscale('log')
plt.ylabel('Firing rate (Hz)')
plt.show()