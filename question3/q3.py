import parameters_q3 as p
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def normaliseData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def getDV(neurone, timeIndex, I_ext):
    return (-(1/neurone['R']['m'])*(neurone['V']['m'][timeIndex]-neurone['V']['rest'])+I_ext)/neurone['C']


def getG_t(t, A, B, phi):
    return A + B*np.cos(2*np.pi*p.f*t + phi)


def injectIntoSingleNeuron(neurone, current):
    for timeIndex, timeStep in enumerate(p.timeSteps):
        # Calculate the potential for this timestep.
        # newVm = V_m(t) + 𐤃V_m(t)
        VmAtCurrentTimeStep = neurone['V']['m'][timeIndex] + (p.simulationTiming['incrementBy'] * getDV(neurone,
                                                                                                        timeIndex, current[timeIndex]))
        VmAtNextTimeStep = VmAtCurrentTimeStep
        # If neurone hits threshold...
        if(VmAtCurrentTimeStep >= neurone['V']['threshold']):
            # Override potential, and set to 40mV (40e-3 V).
            VmAtCurrentTimeStep = 40e-3
            # Set next potential to V_rest.
            VmAtNextTimeStep = neurone['V']['rest']

            # Mark this timestep as a spike.
            neurone['isSpikingAtTime'][timeIndex] = 1

        neurone['V']['m'][timeIndex] = VmAtCurrentTimeStep
        if(timeIndex+1 <= p.timeSteps.size-1):
            neurone['V']['m'][timeIndex+1] = VmAtNextTimeStep

    if(p.showSingularPlot):
        fig0 = plt.figure(0)
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


plt.close("all")


currentToInject = np.ones(p.timeSteps.size) * \
    (0.5*p.I_0*(1 + np.cos(2*np.pi*p.f*p.timeSteps)))

fig1 = plt.figure(1)
plt.plot(p.timeSteps, currentToInject)
plt.xlabel('time (ms)')
plt.ylabel('I_ext (nanoA)')
plt.title('Current injected (I_ext) into each neurone over time')


simulationCounter = 1
meanSquaredErrorsByCOVFixedInitials = np.zeros(len(p.coEfficientOfVariations))
meanSquaredErrorsByCOVDependantInitials = np.zeros(len(p.coEfficientOfVariations))
for coEfficientOfVariationIndex, coEfficientOfVariation in enumerate(p.coEfficientOfVariations):
    binnedSpikesBySeed = np.zeros((len(p.seeds), p.binnedTime.size))
    binnedSpikesBySeedNew = np.zeros((len(p.seeds), p.binnedTime.size))
    for seedIndex, seed in enumerate(p.seeds):
        p.setSeed(seed)
        neuronalPopulation = [p.generateNeurone(scale=p.getSd(
            coEfficientOfVariation)) for i in range(p.populationSize)]

        for neuroneIndex, neurone in enumerate(neuronalPopulation):
            print('Injecting neurone: '+str(simulationCounter)+'/' +
                  str(len(neuronalPopulation) * len(p.seeds) * len(p.coEfficientOfVariations)))
            neurone = injectIntoSingleNeuron(
                neurone=neurone, current=currentToInject)
            simulationCounter += 1

        binnedSpikesBySeed[seedIndex] = np.mean([
            # Take a 2ms segment in time for one neurone. In here, there may have been multiple spikes.
            n['isSpikingAtTime'].reshape(p.binWidth, -1)
            # Sum them. Now, we have vectors of the number of spikes every 2ms (where each vector is a neurone).
            .sum(axis=1)
            for n in neuronalPopulation],
            # Take the vectors and find their mean, to get the mean activity of all neurones, per 2ms (or "bin").
            # This value is indexed by "seedIndex", and as is later averaged again to get the average activity per population, per 2ms ("bin") for all seeds.
            axis=0)

        binnedSpikesBySeedNew[seedIndex] = np.sum([
            # Take a 2ms segment in time for one neurone. In here, there may have been multiple spikes.
            n['isSpikingAtTime'].reshape(p.binWidth, -1)
            # Sum them. Now, we have vectors of the number of spikes every 2ms (where each vector is a neurone).
            .sum(axis=1)
            for n in neuronalPopulation],
            # Take the vectors and find their mean, to get the mean activity of all neurones, per 2ms (or "bin").
            # This value is indexed by "seedIndex", and as is later averaged again to get the average activity per population, per 2ms ("bin") for all seeds.
            axis=0)

    binnedSpikesBySeedsMean = binnedSpikesBySeed.mean(axis=0)
    binnedSpikesBySeedsNewMean = binnedSpikesBySeedNew.mean(axis=0)
    binnedSpikesBySeedsSd = binnedSpikesBySeed.std(axis=0)
    meanHzPerBin = binnedSpikesBySeedsMean / p.simulationTiming['until']
    meanHz = meanHzPerBin.mean()*p.populationSize
    populationFiringRate = (meanHzPerBin*p.populationSize) / \
        (1/p.simulationTiming['until'])
    
    if(len(p.coEfficientOfVariations) <= 3):
        plt.figure()
        plt.suptitle('Mean number of spikes per neurone over time')
        plt.title(''+str(len(neuronalPopulation))+' neurones, ' +
                  "Hz (mean): "+str(meanHz)+', CV: '+str(np.round(coEfficientOfVariation, 2))+'%')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean number of spikes (/neurone)')
        plt.bar(p.binnedTime, binnedSpikesBySeedsMean,  width=p.desiredBinWidth,
                yerr=np.abs(binnedSpikesBySeedsSd), capsize=2, ecolor='red')

        plt.figure()
        plt.suptitle('Mean frequency per neurone over time')

        plt.title(''+str(len(neuronalPopulation))+' neurones, ' +
                  "Hz (mean): "+str(meanHz)+', CV: '+str(np.round(coEfficientOfVariation, 2))+'%')
        plt.xlabel('Time (s)')
        plt.ylabel('Mean firing rate in Hz (/neurone)')
        plt.bar(p.binnedTime, meanHzPerBin,  width=p.desiredBinWidth,
                yerr=binnedSpikesBySeedsSd, capsize=2, ecolor='red')

        plt.figure()
        plt.suptitle('Neuronal population firing rate over time')
        plt.title('Population size = '+str(len(neuronalPopulation))+' neurones, ' +
                  "Hz (mean): "+str(meanHz)+', CV: '+str(np.round(coEfficientOfVariation, 2))+'%')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean firing rate (Hz)')
        plt.bar(p.binnedTime*1e3, populationFiringRate,  width=p.desiredBinWidth*1e3, label='Population spiking activity'
                #yerr=binnedSpikesBySeedsSd*p.populationSize, capsize=2, ecolor='red'
                )
        plt.axhline(y=populationFiringRate.mean(), label='Mean', color='red', linestyle='--')
        plt.axhline(y=populationFiringRate.max(), label='Maximum', color='green', linestyle='--')
        plt.legend()
    else:
        print("You can produce a figure that shows the neuronal population's firing activity over time if you set p.coEfficientOfVariations to one value.")

    # Guesses are hardcoded, achieved by numerous plotting.
    # translate Y, stretch in Y direction, translate in X
    # initialParametersFixed = {'A': 10/64+250, 'B': 80/512+300, 'Phi': 75/400}
    initialParametersDependant = {
        'A': populationFiringRate.mean(),
        'B': (populationFiringRate.max() - populationFiringRate.mean()),
        #'Phi': 75/400}
        'Phi': p.simulationTiming['incrementBy']}
    
    # approximatedLine = getG_t(p.binnedTime, *(initialParametersFixed.values()))
    # optimalParameters, pcov = curve_fit(f=getG_t, xdata=p.binnedTime,
    #                                     ydata=populationFiringRate, p0=list(initialParametersFixed.values()))
    # optimalLine = getG_t(p.binnedTime, *(optimalParameters))
    # meanSquaredErrorsByCOVFixedInitials[coEfficientOfVariationIndex] = np.mean(
    #     (populationFiringRate-optimalLine)**2)
    
    approximatedLine = getG_t(p.binnedTime, *(initialParametersDependant.values()))
    optimalParameters, pcov = curve_fit(f=getG_t, xdata=p.binnedTime,
                                        ydata=populationFiringRate, p0=list(initialParametersDependant.values()))
    optimalLine = getG_t(p.binnedTime, *(optimalParameters))
    meanSquaredErrorsByCOVDependantInitials[coEfficientOfVariationIndex] = np.mean(
        (populationFiringRate-optimalLine)**2)
    

    if(len(p.coEfficientOfVariations) <= 2):
        #ydata = getG_t(15/80, 12/32, 8/100)
        plt.figure()
        #plt.suptitle('Question 3: curve fitting')
        plt.title(
            #'Initial params (fixed)- A='+str(initialParametersFixed['A'])+' B='+str(initialParametersFixed['B'])+' φ='+str(initialParametersFixed['Phi']) +
            #'\n'+
            '$A_{initial}$: '+str(initialParametersDependant['A'])+', $B_{initial}$: '+str(initialParametersDependant['B'])+', $φ_{initial}$= '+str(initialParametersDependant['Phi']) +
                  '\n'+
                  '$A_{optimal}$: '+str(optimalParameters[0])+', $B_{optimal}$: '+str(optimalParameters[1])+', $φ_{optimal}$= '+str(optimalParameters[2]))
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean firing rate of population\'s neurones (Hz)')
        plt.plot(p.binnedTime*1e3, approximatedLine,
                 color='purple', label="Fitting curve: initial parameters")
        plt.plot(p.binnedTime*1e3, optimalLine, color='green',
                 linestyle='--', label="Fitting curve: optimal parameters")
        plt.bar(p.binnedTime*1e3, populationFiringRate,  width=p.desiredBinWidth*1e3, label="Neuronal activity (data)"
                #yerr=binnedSpikesBySeedsSd, capsize=2, ecolor='red'
                )
        plt.legend()
    else:
        print("You can produce a figure that shows the neuronal population's firing activity over time with fitted curves superimposed if you set p.coEfficientOfVariations to one value.")

if(len(p.coEfficientOfVariations) > 1):
    plt.figure()
    plt.title('The effects of varying CV of $R_m$ on mean-squared error (MSE)')
    plt.xlabel('CV')
    plt.ylabel('MSE')
    #minX = np.min(p.coEfficientOfVariations)
    #minY = np.min(meanSquaredErrorsByCOVFixedInitials)
    #lastX = p.coEfficientOfVariations[-1]
    #lastY = meanSquaredErrorsByCOVFixedInitials[-1]
    #markersOn = [meanSquaredErrorsByCOVFixedInitials.argmin(),
    #             meanSquaredErrorsByCOVFixedInitials.size-1]
    #plt.text(minX+5e3,
    #         minY, '('+str(round(minX, 5))+', '+str(round(
    #             minY, 3))+')', fontsize=12)
    #plt.text(lastX,
    #         lastY+1e-4, '('+str(round(lastX, 5))+', '+str(round(
    #             lastY, 3))+')', fontsize=12, horizontalalignment='right')
    #plt.plot(p.coEfficientOfVariations, meanSquaredErrorsByCOVFixedInitials, label='Fixed', linestyle='-', marker='o', color='red', linewidth=2, markevery=markersOn)
    
    minX = p.coEfficientOfVariations[meanSquaredErrorsByCOVDependantInitials.argmin()]
    minY = np.min(meanSquaredErrorsByCOVDependantInitials)
    lastX = p.coEfficientOfVariations[-1]
    lastY = meanSquaredErrorsByCOVDependantInitials[-1]
    firstX = p.coEfficientOfVariations[0]
    firstY = meanSquaredErrorsByCOVDependantInitials[0]
    plt.text(minX,
             minY, '('+str(minX)+', '+str(round(
                 minY, 3))+')', fontsize=12)
    plt.text(lastX,
             lastY, '('+str(round(lastX, 5))+', '+str(round(
                 lastY, 3))+')', fontsize=12, horizontalalignment='right')
    plt.text(firstX,
             firstY, '('+str(round(firstX, 5))+', '+str(round(
                 firstY, 3))+')', fontsize=12, horizontalalignment='left')
    plt.plot(minX, minY, marker='o', color='blue')
    plt.plot(lastX, lastY, marker='o', color='blue')
    plt.plot(firstX, firstY, marker='o', color='blue')
    plt.plot(p.coEfficientOfVariations, meanSquaredErrorsByCOVDependantInitials,
             #label='Dependant',
             #color='blue',
             linewidth=2)
    #plt.legend()
    
    
    #plt.figure()
    #plt.ylabel('$MSE_{dependant}$ - $MSE_{fixed}$')
    #plt.xlabel('CV')
    #diff = meanSquaredErrorsByCOVDependantInitials - meanSquaredErrorsByCOVFixedInitials
    #plt.plot(p.coEfficientOfVariations[list(range(len(diff)))], diff, label='Data')
    #plt.axhline(diff.mean(), label='Mean (' +
    #            str("{:e}".format(diff.mean()))+')', color='red')
    #plt.legend()
else:
    print("You can print a figure of COV against MSE if you set more COV.")

plt.figure()
neuronalPopulationIndicesSortedByRm = np.argsort(
    np.array([n['R']['m'] for n in neuronalPopulation]), axis=0)
neuronalPopulationOrderedByIncreasingRm = np.take_along_axis(np.array(
    neuronalPopulation), indices=neuronalPopulationIndicesSortedByRm, axis=0)
vectorOfRm = np.multiply([n['R']['m'] for n in neuronalPopulationOrderedByIncreasingRm], 1e-6)

colorMap = mpl.colors.LinearSegmentedColormap.from_list('colorMap', [
                                                        'blue', 'red'])
# Create gradient from lowest Rm to highest Rm
levels = np.linspace(np.min(vectorOfRm), np.max(vectorOfRm), 100)
# Produce a contour map to be thrown away (we just want the colorMap)
CS3 = plt.contourf([[0, 0], [0, 0]], levels, cmap=colorMap)
# Clear the figure
plt.clf()

spikeEventPositionsForPopulation = [np.array(n['isSpikingAtTime']).nonzero()[
    0] for n in neuronalPopulationOrderedByIncreasingRm]
spikeEventColorsForPopulation = [
    colorMap(rm) for rm in normaliseData(vectorOfRm)]

plt.suptitle('The effect of hetereogeneity on population spiking activity')
plt.title('Population size: '+str(p.populationSize)+' neurones, CV = '+str(coEfficientOfVariation)+'%')
plt.ylabel('Neurone')
plt.xlabel('Time (ms)')
#lineoffsets = [n['R']['m'] for n in neuronalPopulation]
plt.eventplot(spikeEventPositionsForPopulation,
              colors=spikeEventColorsForPopulation, linewidths=5)
colorbar = plt.colorbar(
    CS3, ticks=[np.min(vectorOfRm), (np.max(vectorOfRm)/2),  np.max(vectorOfRm)], label='$R_m$ (MΩ)')

plt.show()
