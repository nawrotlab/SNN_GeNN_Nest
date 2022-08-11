import numpy as np
from multiprocessing.pool import ThreadPool as Pool


class SuperPopulation:
    """
    Class which contains names and a list of populations and a function to set global parameters associated with this
    population. Populations can be all objects from GeNN e.g. neuron populations or synapse populations or
    current sources. The global parameters of connectivity initialization need a different command to set the global
    parameter. ()
    """
    def __init__(self, Populations: [], name: str):
        self.Populations = Populations  
        self.name=name

    def get_Populations(self):
        return self.Populations
    
    def get_name(self):
      return self.name

    def set_global_Param(self, GlobalParam, Value):
        """
        Sets the global parameter of all members of super population to value.
        Arguments:
            GlobalParam: Name of global parameter. (Name defined while defining e.g. custom neuron model)
            Value: List of Values with length batch_size. Each entry corresponds to a value set for one network instance
        """
        for pop in self.Populations:
            pop.extra_global_params[GlobalParam].view[:] = Value
            pop.push_extra_global_param_to_device(GlobalParam)


class SuperPopulation_Connectivity(SuperPopulation):
    """
    Class which contains names and a list of populations and a function to set
    global parameters of connectivity initialization. The model needs to be reinitalized to use the new parameter.
    (model.reinitialise())
    """
    def initialize_global_Param(self, GlobalParam, Value):
        for pop in self.Populations:
            pop.connectivity_extra_global_params[GlobalParam].set_values(Value)

    def set_global_Param(self, GlobalParam, Value):
        for pop in self.Populations:
            pop.connectivity_extra_global_params[GlobalParam].view[:] = Value
            pop._push_extra_global_param_to_device(GlobalParam, egp_dict=pop.connectivity_extra_global_params)


def prepareModelForLoad_Neurons(model):
    """
    Prepares a GeNN model with global parameters in neuron models to be loaded by setting all global parameters to 0.
    """
    GlobalParams=False
    for popN in model.neuron_populations:         
        pop = model.neuron_populations[popN]
        try:
            popGlobVar = list(pop.extra_global_params.keys())
            for globPar in popGlobVar:         
                pop.set_extra_global_param(globPar, [0.0 for ii in range(model.batch_size)])
            GlobalParams=True
        except:
            print(pop.name + "has no global parameter")        
    return GlobalParams


def extractPopSpikes(num, NQ, Pop, batch_size, NOffset=0, warmup=0, timeZero=0):
    """
    Extracts the spikes of one neuron population and changes the ID. Specific to the EI-clustered model
    """
    DownloadedData = Pop.spike_recording_data
    if batch_size>1:
        spiketimes=[np.vstack(
            (np.array(DownloadedData[jj][0][DownloadedData[jj][0]>= (warmup+timeZero)])-(warmup+timeZero),
             np.array(DownloadedData[jj][1][DownloadedData[jj][0]>= (warmup+timeZero)]) + num * NQ + NOffset)) for jj in range(batch_size)]
    else:
        spiketimes = [np.vstack(
            (np.array(DownloadedData[0][DownloadedData[0] >= (warmup + timeZero)]) - (warmup + timeZero),
             np.array(DownloadedData[1][DownloadedData[0] >= (warmup + timeZero)]) + num * NQ + NOffset))]
    return spiketimes


def extractSpiketimes(model, params, Populations ,timeZero=0):
    """
    Extracts the spiketimes of two super populations. Specific to the EI-clustered model and the Exc and Inh. population
    """
    batch_size=model.batch_size
    # Download recording data
    model.pull_recording_buffers_from_device()

    with Pool(params['n_jobs']) as p:
        resE=p.map(lambda x: extractPopSpikes(x[0], int(params['N_E']/params['Q']), x[1], batch_size, 0
                                              , params['warmup'],timeZero), enumerate(Populations[0].get_Populations()))
        resI=p.map(lambda x: extractPopSpikes(x[0], int(params['N_I']/params['Q']), x[1], batch_size, params['N_E']
                                              , params['warmup'],timeZero), enumerate(Populations[1].get_Populations()))
   
    TuplesConcat=[tuple(res[jj] for res in resE+resI) for jj in range(batch_size)] 
    spiketimesLoc=[np.hstack(TuplesConcat[jj]) for jj in range(batch_size)]
    spiketimesLoc=[spiketimesLoc[jj][:, spiketimesLoc[jj][0,:].argsort()] for jj in range(batch_size)]
    return spiketimesLoc