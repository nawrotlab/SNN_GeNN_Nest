import numpy as np
import time
import pickle
import copy
import sys
sys.path.append("..")
from Helper import GeneralHelper
from Defaults import defaultGridsearch as default
from Helper import Gridsearch
import psutil

# ********************************************************************************
#                                    Function
# ********************************************************************************
def SimulateIX(params, IX_all, measurementVar, ID_all, PathOutput, lock, PathSpikes=None, timeout=None, Queue=None):
    """
    Simulation function which is used in Gridsearch and excuted by x workers in parallel.
    The function creates the EI-clustered model and dumps the Results as well as (optional)
    the spiketimes to a pickle file.
    This is the only model dependent function as it adds the changed parameters in EI_Network.set_I_x
    to the mmodel and uses the EI-clustered model simulation function which returns a dict which
    contains the measured firing rates. The used model has to provide these functions.
    """
    from Helper import ClusterModelNEST
    # Import at this point starts one NEST interpreter per process instead of one for all
    params = copy.deepcopy(params)
    EI_Network = ClusterModelNEST.ClusteredNetworkNEST_Timing(default, params)
    EI_Network.PrepareTimingMultipleSimulations()
    EI_Network.setup_network_woPrepare()
    # Creates object which creates the EI clustered network in NEST and adds function to build procedure to modify
    # background stimulation of neuron populations. This could be also done by changing the params in that way.
    for IX, ID in IX_all:
        EI_Network.reset_I_x()
        EI_Network.reinitalizeStateVariables()
        EI_Network.set_I_x(I_XE=IX[0][0], I_XI=IX[1][0])
        EI_Network.reset_RecordingDevice()
        TimePreSim=EI_Network.simulate_nest()
        spiketimes=EI_Network.get_recordings(TimePreSim)
        e_rate, i_rate=EI_Network.get_firing_rates(spiketimes)
        Result={'e_rate': e_rate, 'i_rate': i_rate, 'Timing': EI_Network.get_timing(), 'params': EI_Network.get_parameter(),
                    'spiketimes': spiketimes}
        startDump = time.time()
        if Queue is not None:
            Queue.put(True)
        
        with lock[0]:
            with open(PathOutput, 'ab') as outfile:
                pickle.dump([
                    tuple(Result[var] for var in measurementVar), ID]
                    , outfile)

        if PathSpikes is not None:
            with lock[1]:
                with open(PathSpikes, 'ab') as outfile:
                    pickle.dump([Result['spiketimes'], ID], outfile)
        endDump=time.time()
        Timing = Result['Timing']
    Timing['Load']=np.nan
    Timing['Dump']=np.nan
        #Timing['Dump'].append(endDump - startDump)

    return Timing

#######################################################################################################################################
# main#
#######################################################################################################################################
if __name__ == '__main__':
    startTime = time.time()
    CPUcount=psutil.cpu_count(logical = False)
    #if CPUcount>8:
    #    CPUcount-=2

    CPUsperJob=CPUcount
    params = {'n_jobs': CPUsperJob, 'N_E': 20000, 'N_I': 5000, 'dt': 0.1, 'neuron_type': 'iaf_psc_exp', 'simtime': 9000,
              'delta_I_xE': 0., 'delta_I_xI': 0., 'record_voltage': False, 'record_from': 1, 'warmup': 1000, 'Q': 20}

    jip_ratio = 0.75  # 0.75 default value  #works with 0.95 and gif wo adaptation
    jep = 2.75  # clustering strength
    jip = 1. + (jep - 1) * jip_ratio
    params['jplus'] = np.array([[jep, jip], [jip, jip]])

    I_ths = [0.0, 0.0]  # set background stimulation baseline to 0
    params['I_th_E'] = I_ths[0]
    params['I_th_I'] = I_ths[1]

    measurementVar = ("e_rate", "i_rate")
    #####################################################################################################
    #               Set tune parameter and create function to calculate it                              #
    #####################################################################################################
    params = GeneralHelper.mergeParams(params, default)

    def GlobE(x):   # conversion factor of Rheobase current
        return x * (params['V_th_E'] - params['E_L']) / params['tau_E'] * params['C_m']

    def GlobI(x):
        return x * (params['V_th_I'] - params['E_L']) / params['tau_I'] * params['C_m']
    ###################################################################################################


    Constructor = (
        ('Ix', [GlobE(0.95000000000001 + 0.05 * ii) for ii in range(40)], ('I_X_E',)),
        ('Ix', [GlobI(0.70000000000001 + 0.025 * ii) for ii in range(40)], ('I_X_I',)),
    )

    Grid = Gridsearch.Gridsearch_NEST(SimulateIX, params, Constructor, measurementVar, default,
                                      PathOutput='Test.pkl', Nworkers=CPUcount//CPUsperJob)
    Grid.search(ProgressBar=True, ReuseSimulation=True)
    times = Grid.getTiming()
    endTime = time.time()
    TotalTime = endTime - startTime
    print("Finished!")
    print("TotalTime  : %.4f s" % TotalTime)
    resultsInfo = {"params": params, "TotalTime": TotalTime}
    with open("TimesNest.pkl", 'wb') as outfile:
        pickle.dump(resultsInfo, outfile)
        pickle.dump(Grid.getParamField(), outfile)
        pickle.dump(times, outfile)
