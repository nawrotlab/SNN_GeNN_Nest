import nest
import numpy as np
import sys
sys.path.append("..")
from Helper import ClusterHelper
from Helper import GeneralHelper
from Helper import ClusterModelNEST
import shutil

def UniformParameters(mean, CV):
    """ Uniform distribution parameters """
    params = {"min": mean - (mean * CV * np.sqrt(12)) / 2, "max": mean + (mean * CV * np.sqrt(12)) / 2}
    return params


class ClusteredNetworkGeNN_Distributed(ClusterModelNEST.ClusteredNetworkNEST_Timing):
    # connect clusters with distributed synapses
    def create_populations_dist(self, CV=0.01):
        """
        Creates Q excitatory and inhibitory neuron populations with the parameters of the network and uses
        synapses with distributed time constants. Time constants are distributed for the postsynaptic neurons.
        """
        # make sure number of clusters and units are compatible
        assert self.params['N_E'] % self.params['Q'] == 0, 'N_E needs to be evenly divisible by Q'
        assert self.params['N_I'] % self.params['Q'] == 0, 'N_I needs to be evenly divisible by Q'

        N = self.params['N_E'] + self.params['N_I']  # total units
        if self.params['I_th_E'] is None:
            I_xE = self.params['I_xE']
        else:
            I_xE = self.params['I_th_E'] * (self.params['V_th_E'] - self.params['E_L']) / self.params['tau_E'] * \
                   self.params['C_m']

        if self.params['I_th_I'] is None:
            I_xI = self.params['I_xI']
        else:
            I_xI = self.params['I_th_I'] * (self.params['V_th_I'] - self.params['E_L']) / self.params['tau_I'] * \
                   self.params['C_m']

        E_neuron_params = {'E_L': self.params['E_L'], 'C_m': self.params['C_m'], 'tau_m': self.params['tau_E'],
                           't_ref': self.params['t_ref'], 'V_th': self.params['V_th_E'], 'V_reset': self.params['V_r'],
                           'I_e': I_xE}
        I_neuron_params = {'E_L': self.params['E_L'], 'C_m': self.params['C_m'], 'tau_m': self.params['tau_I'],
                           't_ref': self.params['t_ref'], 'V_th': self.params['V_th_I'], 'V_reset': self.params['V_r'],
                           'I_e': I_xI}
        if 'iaf_psc_exp' in self.params['neuron_type']:

            Esyn_dist=UniformParameters(self.params['tau_syn_ex'], CV)
            Isyn_dist=UniformParameters(self.params['tau_syn_in'], CV)

            n = nest.Create('iaf_psc_alpha', 10000, {'V_m': nest.random.normal(mean=-60.0, std=10.0)})
            E_neuron_params['tau_syn_ex'] =  nest.random.uniform(min=Esyn_dist['min'], max=Esyn_dist['max'])
            E_neuron_params['tau_syn_in'] =  nest.random.uniform(min=Isyn_dist['min'], max=Isyn_dist['max'])
            I_neuron_params['tau_syn_in'] =  nest.random.uniform(min=Isyn_dist['min'], max=Isyn_dist['max'])
            I_neuron_params['tau_syn_ex'] =  nest.random.uniform(min=Esyn_dist['min'], max=Esyn_dist['max'])

            # iaf_psc_exp allows stochasticity, if not used - ignore
            try:
                if self.params['delta_'] is not None:
                    E_neuron_params['delta'] = self.params['delta_']
                    I_neuron_params['delta'] = self.params['delta_']
                if self.params['rho'] is not None:
                    E_neuron_params['rho'] = self.params['rho']
                    I_neuron_params['rho'] = self.params['rho']
            except KeyError:
                pass
        else:
            assert 'iaf_psc_exp' in self.params['neuron_type'], "iaf_psc_exp neuron model is the only implemented model"

            # create the neuron populations
        E_pops = []
        I_pops = []
        for q in range(self.params['Q']):
            E_pops.append(nest.Create(self.params['neuron_type'], int(self.params['N_E'] / self.params['Q'])))
            nest.SetStatus(E_pops[-1], E_neuron_params)
        for q in range(self.params['Q']):
            I_pops.append(nest.Create(self.params['neuron_type'], int(self.params['N_I'] / self.params['Q'])))
            nest.SetStatus(I_pops[-1], I_neuron_params)

        if self.params['delta_I_xE'] > 0:
            for E_pop in E_pops:
                I_xEs = nest.GetStatus(E_pop, 'I_e')
                nest.SetStatus(E_pop, [
                    {'I_e': (1 - 0.5 * self.params['delta_I_xE'] + np.random.rand() * self.params['delta_I_xE']) * ixe}
                    for ixe in I_xEs])

        if self.params['delta_I_xI'] > 0:
            for I_pop in I_pops:
                I_xIs = nest.GetStatus(I_pop, 'I_e')
                nest.SetStatus(I_pop, [
                    {'I_e': (1 - 0.5 * self.params['delta_I_xI'] + np.random.rand() * self.params['delta_I_xI']) * ixi}
                    for ixi in I_xIs])
        if self.params['V_m'] == 'rand':
            T_0_E = self.params['t_ref'] + ClusterHelper.FPT(self.params['tau_E'], self.params['E_L'], I_xE,
                                                             self.params['C_m'], self.params['V_th_E'],
                                                             self.params['V_r'])
            if np.isnan(T_0_E):
                T_0_E = 10.
            for E_pop in E_pops:
                nest.SetStatus(E_pop, [{'V_m': ClusterHelper.V_FPT(self.params['tau_E'], self.params['E_L'], I_xE,
                                                                   self.params['C_m'], T_0_E * np.random.rand(),
                                                                   self.params['V_th_E'], self.params['t_ref'])} for i
                                       in range(len(E_pop))])

            T_0_I = self.params['t_ref'] + ClusterHelper.FPT(self.params['tau_I'], self.params['E_L'], I_xI,
                                                             self.params['C_m'], self.params['V_th_I'],
                                                             self.params['V_r'])
            if np.isnan(T_0_I):
                T_0_I = 10.
            for I_pop in I_pops:
                nest.SetStatus(I_pop, [{'V_m': ClusterHelper.V_FPT(self.params['tau_I'], self.params['E_L'], I_xI,
                                                                   self.params['C_m'], T_0_I * np.random.rand(),
                                                                   self.params['V_th_E'], self.params['t_ref'])} for i
                                       in range(len(I_pop))])
        else:
            nest.SetStatus(nest.NodeCollection([x for x in range(1, N + 1)]),
                           [{'V_m': self.params['V_m']} for i in range(N)])
        self.Populations = [E_pops, I_pops]

    def create_populations(self):
        self.create_populations_dist(0.05)


if __name__ == "__main__":
    sys.path.append("..")
    from Defaults import defaultSimulate as default
    import matplotlib.pyplot as plt
    CV=0.05

    params = {'n_jobs': 24, 'N_E': 72000, 'N_I': 16000, 'dt': 0.1, 'neuron_type': 'iaf_psc_exp', 'simtime': 9000,
              'delta_I_xE': 0., 'delta_I_xI': 0., 'record_voltage': False, 'record_from': 1, 'warmup': 1000, 'Q': 20}

    jip_ratio = 0.75  # 0.75 default value  #works with 0.95 and gif wo adaptation
    jep = 2.75  # clustering strength
    jip = 1. + (jep - 1) * jip_ratio
    params['jplus'] = np.array([[jep, jip], [jip, jip]])

    I_ths = [2.13, 1.24]  # set background stimulation baseline to 0
    params['I_th_E'] = I_ths[0]
    params['I_th_I'] = I_ths[1]


    EI_Network = ClusterModelNEST.ClusteredNetworkNEST_Timing(default, params)
    # Creates object which creates the EI clustered network in NEST
    Result = EI_Network.get_simulation(timeout=10000)
    del Result['spiketimes']
    print(Result)
    del EI_Network

    EI_Network = ClusteredNetworkGeNN_Distributed(default, params)
    # Creates object which creates the EI clustered network in NEST
    Result = EI_Network.get_simulation(timeout=10000)
    del Result['spiketimes']
    print(Result)
    del EI_Network

    Cluster = ClusteredNetworkGeNN_Distributed(default, params)

    # Name has to be changed because PyGeNN will be confused if two objects with the same reference are present
    Cluster.set_model_build_pipeline([Cluster.setup_nest, lambda: Cluster.create_populations_dist(CV),
                                      Cluster.create_stimulation, Cluster.create_recording_devices, Cluster.conneect])


    Cluster.setup_network()

    # create histogram of synaptic times
    Tau_EE=Cluster.Populations[0][0].get("tau_syn_ex")
    Tau_EI=Cluster.Populations[0][0].get("tau_syn_in")
    Tau_IE=Cluster.Populations[1][0].get("tau_syn_ex")
    Tau_II=Cluster.Populations[1][0].get("tau_syn_in")
    plt.figure()
    plt.subplot(2,2,1)
    plt.hist(Tau_EE, bins=20, label='E->E')
    plt.legend()
    plt.subplot(2,2,2)
    plt.hist(Tau_EI, bins=20, label='I->E')
    plt.legend()
    plt.subplot(2,2,3)
    plt.hist(Tau_IE, bins=20, label='E->I')
    plt.legend()
    plt.subplot(2,2,4)
    plt.hist(Tau_II, bins=20, label='I->I')
    plt.legend()
    plt.savefig('Tau_hist_NEST.png')
    spiketimes = Cluster.simulate_and_get_recordings()
    rates = np.array(Cluster.get_firing_rates(spiketimes))
    print(rates)
    plt.figure()
    spiketimesplot=(spiketimes[0][1, :]%50)==0
    plt.plot(spiketimes[0][0, spiketimesplot], spiketimes[0][1, spiketimesplot], '.')
    plt.savefig('NEST_1.png')
    print(Cluster.get_timing())