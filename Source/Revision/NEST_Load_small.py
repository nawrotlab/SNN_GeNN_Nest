import numpy as np
import pickle
import sys
import time
import copy
import pandas as pd
sys.path.append("..")
from Defaults import defaultGridsearch as default
from multiprocessing import Process, Queue
from Helper import GeneralHelper

# ********************************************************************************
#                                    Function
# ********************************************************************************
def Simulate(params, queue, PathOutput=None, PathSpikes=None):
    """
    Simulates network with NEST from scratch and saves connectivity if Path
    is given.
    """
    from Helper import ClusterModelNEST
    # Import at this point starts one NEST interpreter per process instead of one for all
    params = copy.deepcopy(params)
    EI_Network = ClusterModelNEST.ClusteredNetworkNEST_Timing(default, params)
        # Creates object which creates the EI clustered network in NEST and adds function to build procedure to modify
    # background stimulation of neuron populations. This could be also done by changing the params in that way.
    Result = EI_Network.get_simulation(timeout=timeout)

    startDump = time.time()
    if PathOutput is not None:
        EI_Network.save_conn_to_file(PathOutput)

    if PathSpikes is not None:
        with open(PathSpikes, 'ab') as outfile:
            pickle.dump(Result['spiketimes'], outfile)

    endDump = time.time()
    Timing = Result['Timing']
    Timing['Dump'] = endDump - startDump
    queue.put(Timing)

def Simulate_load_connectivity(params, queue, PathInput, PathSpikes=None):
    """
    Simulates network with NEST by utilizing provided connectivity.
    """
    from Helper import ClusterModelNEST
    # Import at this point starts one NEST interpreter per process instead of one for all
    params = copy.deepcopy(params)
    EI_Network = ClusterModelNEST.ClusteredNetworkNEST_Timing(default, params)
    EI_Network.set_model_build_pipeline(
        [EI_Network.setup_nest, EI_Network.create_populations, EI_Network.create_stimulation,
         EI_Network.create_recording_devices, lambda: EI_Network.connect_from_file(PathInput)])
    Result = EI_Network.get_simulation(timeout=timeout)

    startDump = time.time()

    if PathSpikes is not None:
        with open(PathSpikes, 'ab') as outfile:
            pickle.dump(Result['spiketimes'], outfile)

    endDump = time.time()
    Timing = copy.deepcopy(Result['Timing'])
    Timing['Dump'] = endDump - startDump
    queue.put(Timing)



if __name__ == '__main__':
    Savepath = "LoadConnectivity_small2.pkl"

    params = {'n_jobs': 4, 'N_E': 4000, 'N_I': 1000, 'dt': 0.1, 'neuron_type': 'iaf_psc_exp', 'simtime': 9000,
              'delta_I_xE': 0., 'delta_I_xI': 0., 'record_voltage': False, 'record_from': 1, 'warmup': 1000, 'Q': 20}

    jip_ratio = 0.75  # 0.75 default value  #works with 0.95 and gif wo adaptation
    jep = 2.75  # clustering strength
    jip = 1. + (jep - 1) * jip_ratio
    params['jplus'] = np.array([[jep, jip], [jip, jip]])

    I_ths = [0.0, 0.0]  # set background stimulation baseline to 0
    params['I_th_E'] = I_ths[0]
    params['I_th_I'] = I_ths[1]

    timeout = 18000  # 5h

    # ********************************************************************************
    #                                    Run
    # ********************************************************************************
    # Loop x times: Simulate from scratch and save connectivity and then simulate again with loaded connectivity
    Timing=[]
    queue = Queue()
    for i in range(10):
        p = Process(target=Simulate, args=(params, queue, "SmallCluster_" + str(i) + ".pkl"))
        p.start()
        p.join()  # this blocks until the process terminates
        p = Process(target=Simulate_load_connectivity, args=(params, queue, "SmallCluster_" + str(i) + ".pkl"))
        p.start()
        p.join()  # this blocks until the process terminates
        TimingDict= queue.get()
        TimingDict['Run'] = i
        TimingDict['Connectivity'] = "Store"
        Timing.append(TimingDict)
        TimingDict= queue.get()
        TimingDict['Run'] = i
        TimingDict['Connectivity'] = "Load"
        Timing.append(TimingDict)
        print("Simulation " + str(i) + " done")

    Timing = pd.DataFrame(Timing)
    print(Timing)
    with open(Savepath, 'ab') as outfile:
        pickle.dump(GeneralHelper.mergeParams(params, default), outfile)
        pickle.dump(Timing, outfile)
