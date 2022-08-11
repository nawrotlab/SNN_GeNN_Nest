import numpy as np
import time
import pickle
import sys
sys.path.append("..")
from Defaults import defaultGridsearch as default
from Helper import Gridsearch
from Helper import ClusterModelGeNN
from Helper import GeNN_Models

class ClusteredNetworkGeNN_Grid(ClusterModelGeNN.ClusteredNetworkGeNN_Timing):

    def Measurement(self, Spikes, Ids):
        FiringRates = np.array(self.get_firing_rates(Spikes))
        return [{'e_rate': FiringRates[jj,0].item(), 'i_rate': FiringRates[jj,1].item(), 'ID': Ids[jj]} for jj in range(len(Ids))]


    def simulateIxBatch(self, TuneParameters, Ids, measurementVar, OutputPath,  PathSpikes=None, reinitialise = True):
        startInitialize = time.time()
        if reinitialise:
            self.reinitalizeModel()
        for TuneID in range(len(TuneParameters[0])):
            Values = TuneParameters[1][TuneID]
            if Values.shape[0] < self.params['batch_size']:
                print("Batch is not fully filled!")
                Values = np.pad(Values, (0, batch_size - Values.shape[0]), 'constant', constant_values=0)
            TuneParameters[0][TuneID][1].set_global_Param(TuneParameters[0][TuneID][0], Values)
        endInitalize = time.time()
        Spikes=self.simulate_and_get_recordings(timeZero=self.getModelTime())
        StartDump = time.time()
        Measurement=self.Measurement(Spikes, Ids)


        with open(OutputPath, 'ab') as outfile:
            for Entry in Measurement:
                pickle.dump([tuple(Entry[var] for var in measurementVar), Entry['ID']]
                    , outfile)

        if PathSpikes is not None:
            with open(PathSpikes, 'ab') as outfile:
                for ii, ID in enumerate(Ids):
                    pickle.dump([Spikes[ii], ID]
                                , outfile)

        endDumpData = time.time()
        Timing=self.get_timing()
        self.clearTiming()
        return {"ReInit": endInitalize - startInitialize, "Sim": Timing['Sim'],
                "Download": Timing['Download'], "Dump": endDumpData - StartDump}


#######################################################################################################################################
# main#
#######################################################################################################################################

if __name__ == '__main__':

    SpikeOutput = None
    startTime = time.time()

    params = {'n_jobs': 24, 'N_E': 20000, 'N_I': 5000, 'dt': 0.1, 'neuron_type': 'iaf_psc_exp', 'simtime': 9000,
              'delta_I_xE': 0., 'delta_I_xI': 0., 'record_voltage': False, 'record_from': 1, 'warmup': 1000, 'Q': 20}

    jip_ratio = 0.75  # 0.75 default value  #works with 0.95 and gif wo adaptation
    jep = 2.75  # clustering strength
    jip = 1. + (jep - 1) * jip_ratio
    params['jplus'] = np.array([[jep, jip], [jip, jip]])

    I_ths = [0.0, 0.0]  # set background stimulation baseline to 0
    params['I_th_E'] = I_ths[0]
    params['I_th_I'] = I_ths[1]
    params['matrixType'] = "SPARSE_GLOBALG"
    batch_size = 40

    NetworkModel = ClusteredNetworkGeNN_Grid(default, params, batch_size=batch_size,
                                         NModel=GeNN_Models.define_iaf_psc_exp_Ie_multibatch())
    NetworkModel.set_model_build_pipeline([NetworkModel.setup_GeNN, NetworkModel.create_populations,
                                           NetworkModel.create_stimulation, NetworkModel.create_recording_devices,
                                           NetworkModel.connect, NetworkModel.prepare_global_parameters])

    NetworkModel.setup_network()
    NetworkModel.build_model()
    NetworkModel.load_model()


    measurementVar = ("e_rate", "i_rate")
    #####################################################################################################
    #               Set tune parameter and create function to calculate it                              #
    #####################################################################################################
    params = NetworkModel.get_parameter()

    def GlobE(x):   # conversion factor of Rheobase current
        return x * (params['V_th_E'] - params['E_L']) / params['tau_E'] * params['C_m']

    def GlobI(x):
        return x * (params['V_th_I'] - params['E_L']) / params['tau_I'] * params['C_m']
    ###################################################################################################

    Constructor=(
                ('Ix',[GlobE(0.95+0.05*ii) for ii in range(40)], (NetworkModel.get_populations()[0],)),
                ('Ix',[GlobI(0.7+0.025*ii) for ii in range(40)], (NetworkModel.get_populations()[1],)),
                )

    Grid = Gridsearch.Gridsearch_GeNN(NetworkModel, NetworkModel.simulateIxBatch, Constructor, measurementVar, PathOutput='Test.pkl',
                                      PathSpikes=SpikeOutput)
    Grid.search(ProgressBar=True)
    times = Grid.getTiming()
    endTime = time.time()
    TotalTime = endTime - startTime
    print("Finished!")
    print("TotalTime  : %.4f s" % TotalTime)

    resultsInfo = {"params": params, "TotalTime": TotalTime}
    with open("TimesGeNN.pkl", 'wb') as outfile:
        pickle.dump(resultsInfo, outfile)
        pickle.dump(Grid.getParamField(), outfile)
        pickle.dump(times, outfile)
