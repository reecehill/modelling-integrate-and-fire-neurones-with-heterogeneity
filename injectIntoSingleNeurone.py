import parameters_q3_new as p
import numpy as np


def getDV(neurone, timeIndex, I_ext):
    return (-(1/neurone['R']['m'])*(neurone['V']['m'][timeIndex]-neurone['V']['rest'])+I_ext)/neurone['C']


def injectIntoSingleNeuron(neurone, current):
    for timeIndex, timeStep in enumerate(p.timeSteps):
        # Calculate the potential for this timestep.
        # newVm = V_m(t) + 𐤃V_m(t)
        VmAtCurrentTimeStep = neurone['V']['m'][timeIndex] + (
            p.simulationTiming['incrementBy'] * getDV(neurone, timeIndex, current[timeIndex]))
        VmAtNextTimeStep = VmAtCurrentTimeStep
        # If neurone hits threshold...
        if(VmAtCurrentTimeStep >= neurone['V']['threshold']):
            # Override potential, and set to 40.
            VmAtCurrentTimeStep = 40 * (10**-3)
            # Set next potential to V_rest.
            VmAtNextTimeStep = neurone['V']['rest']

            # Mark this timestep as a spike.
            neurone['isSpikingAtTime'][timeIndex] = 1

        neurone['V']['m'][timeIndex] = VmAtCurrentTimeStep
        if(timeIndex+1 <= neurone['V']['m'].size-1):
            neurone['V']['m'][timeIndex+1] = VmAtNextTimeStep

    if(p.showSingularPlot):
        plt.figure()
        plt.plot(p.timeSteps, neurone['V']['m'],
                 label='Spiking profile of single neurone')
        plt.plot(p.timeSteps, current, label="I_ext")
        plt.legend()
        plt.xlabel('time (t)')
        plt.ylabel('V_m')
        plt.show()
        input('Paused... hit enter to continue')

    neurone['numberOfSpikes'] = np.sum(neurone['isSpikingAtTime'])
    return neurone
