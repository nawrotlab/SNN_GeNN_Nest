from pygenn import genn_model, genn_wrapper
import numpy as np
import sys
sys.path.append("..")
from Helper import ClusterHelper
from Helper import GeNN_Models
from Helper import GeneralHelper
from Helper import GeNNHelper
from Helper import ClusterModelGeNN


def UniformParameters(mean, CV):
    """ Uniform distribution parameters """
    params = {"min": mean - (mean * CV * np.sqrt(12)) / 2, "max": mean + (mean * CV * np.sqrt(12)) / 2}
    return params


def define_psc_exp_distributed():
    """
    Defines a postsynaptic current with exponential shaped which can have different time constants in a synapse group.
    """
    psc_exp_dist = genn_model.create_custom_postsynaptic_class(
        "psc_exp_dist",
        param_names=[],
        var_name_types=[("expDecay", "scalar")],
        decay_code=
        """
        $(inSyn) *= $(expDecay);
        """,
        apply_input_code=
        """
        $(Isyn) += $(inSyn);
        """
        )
    return psc_exp_dist
def define_synaptic_timeconstant_initialization():
    expDecay_init = genn_model.create_custom_init_var_snippet_class(
    "expDecay_init",
    param_names=["min", "max"],
    var_init_code =
    """
    $(value) = exp(-DT/($(min) + ($(gennrand_uniform) * ($(max)-$(min)))));
    """)
    #
    # """
    # const scalar scale = $(max) - $(min);
    # const scalar expDecay = 0.0;
    # do
    # {
    #    expDecay = $(min) + ($(gennrand_uniform) * scale);
    # } while (expDecay < 0.0);
    # $(value) = exp(-DT/expDecay);
    # """)
    return expDecay_init


class ClusteredNetworkGeNN_Distributed(ClusterModelGeNN.ClusteredNetworkGeNN_Timing):
    # connect clusters with distributed synapses
    def connect_uniform_distributed(self, CV=0.01):
        """ Connects the excitatory and inhibitory populations with each other in the EI-clustered scheme
        using synapses with distributed time constants.
        """
        #  self.Populations[0] -> Excitatory super-population
        #  self.Populations[1] -> Inhibitory super-population
        # connectivity parameters

        #define models
        psc_exp_distributed = define_psc_exp_distributed()
        init_psc_exp = define_synaptic_timeconstant_initialization()

        js = self.params['js']  # connection weights
        N = self.params['N_E'] + self.params['N_I']  # total units


        delaySteps = int((self.params['delay'] + 0.5 * self.model.dT) // self.model.dT)
        psc_E = {"expDecay": genn_model.init_var(init_psc_exp, UniformParameters(self.params['tau_syn_ex'], CV))}
        psc_I = {"expDecay": genn_model.init_var(init_psc_exp, UniformParameters(self.params['tau_syn_in'], CV))}

        print(UniformParameters(self.params['tau_syn_ex'], CV), UniformParameters(self.params['tau_syn_in'], CV))
        # if js are not given compute them so that sqrt(K) spikes equal v_thr-E_L and rows are balanced
        if np.isnan(js).any():
            js = ClusterHelper.calc_js(self.params)
        js *= self.params['s']

        # jminus is calculated so that row sums remain constant
        if self.params['Q'] > 1:
            jminus = (self.params['Q'] - self.params['jplus']) / float(self.params['Q'] - 1)
        else:
            self.params['jplus'] = np.ones((2, 2))
            jminus = np.ones((2, 2))

        # define the synapses and connect the populations
        # EE
        j_ee = js[0, 0] / np.sqrt(N)
        if self.params['fixed_indegree']:
            K_EE = int(self.params['ps'][0, 0] * self.params['N_E'] / self.params['Q'])
            print('K_EE: ', K_EE)
            conn_params_EE = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                               {"colLength": K_EE})

        else:
            conn_params_EE = genn_model.init_connectivity("FixedProbabilityNoAutapse",
                                                               {"prob": self.params['ps'][0, 0]})
        for i, pre in enumerate(self.Populations[0].get_Populations()):
            for j, post in enumerate(self.Populations[0].get_Populations()):
                if i == j:
                    # same cluster
                    self.model.add_synapse_population(str(i) + "EE" + str(j), self.params['matrixType'], delaySteps,
                                                 pre, post,
                                                 "StaticPulse", {}, {"g": self.params['jplus'][0, 0] * j_ee}, {}, {},
                                                 psc_exp_distributed, {}, psc_E, conn_params_EE
                                                 )
                else:
                    self.model.add_synapse_population(str(i) + "EE" + str(j), self.params['matrixType'], delaySteps,
                                                      pre, post,
                                                      "StaticPulse", {}, {"g": jminus[0, 0] * j_ee}, {},
                                                      {},
                                                      psc_exp_distributed, {}, psc_E, conn_params_EE
                                                      )

        # EI
        j_ei = js[0, 1] / np.sqrt(N)
        if self.params['fixed_indegree']:
            K_EI = int(self.params['ps'][0, 1] * self.params['N_I'] / self.params['Q'])
            print('K_EI: ', K_EI)
            conn_params_EI = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                          {"colLength": K_EI})
        else:
            conn_params_EI = genn_model.init_connectivity("FixedProbability",
                                                          {"prob": self.params['ps'][0, 1]})
        for i, pre in enumerate(self.Populations[1].get_Populations()):
            for j, post in enumerate(self.Populations[0].get_Populations()):
                if i == j:
                    # same cluster
                    self.model.add_synapse_population(str(i) + "EI" + str(j), self.params['matrixType'], delaySteps,
                                                 pre, post,
                                                 "StaticPulse", {}, {"g": j_ei * self.params['jplus'][0, 1]}, {}, {},
                                                 psc_exp_distributed, {}, psc_I, conn_params_EI
                                                 )
                else:
                    self.model.add_synapse_population(str(i) + "EI" + str(j), self.params['matrixType'], delaySteps,
                                                      pre, post,
                                                      "StaticPulse", {}, {"g": j_ei * jminus[0, 1]}, {},
                                                      {},
                                                      psc_exp_distributed, {}, psc_I, conn_params_EI
                                                      )
        # IE
        j_ie = js[1, 0] / np.sqrt(N)

        if self.params['fixed_indegree']:
            K_IE = int(self.params['ps'][1, 0] * self.params['N_E'] / self.params['Q'])
            print('K_IE: ', K_IE)
            conn_params_IE = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                          {"colLength": K_IE})
        else:
            conn_params_IE = genn_model.init_connectivity("FixedProbability",
                                                          {"prob": self.params['ps'][1, 0]})
        for i, pre in enumerate(self.Populations[0].get_Populations()):
            for j, post in enumerate(self.Populations[1].get_Populations()):
                if i == j:
                    # same cluster
                    self.model.add_synapse_population(str(i) + "IE" + str(j), self.params['matrixType'], delaySteps,
                                                 pre, post,
                                                 "StaticPulse", {}, {"g": j_ie * self.params['jplus'][1, 0]}, {}, {},
                                                 psc_exp_distributed, {}, psc_E, conn_params_IE
                                                 )
                else:
                    self.model.add_synapse_population(str(i) + "IE" + str(j), self.params['matrixType'], delaySteps,
                                                 pre, post,
                                                 "StaticPulse", {}, {"g": j_ie * jminus[1, 0]}, {}, {},
                                                 psc_exp_distributed, {}, psc_E, conn_params_IE
                                                 )

        # II
        j_ii = js[1, 1] / np.sqrt(N)
        if self.params['fixed_indegree']:
            K_II = int(self.params['ps'][1, 1] * self.params['N_I'] / self.params['Q'])
            print('K_II: ', K_II)
            conn_params_II = genn_model.init_connectivity("FixedNumberPreWithReplacement",
                                                            {"colLength": K_II})
        else:
            conn_params_II = genn_model.init_connectivity("FixedProbability",
                                                          {"prob": self.params['ps'][1, 1]})
        for i, pre in enumerate(self.Populations[1].get_Populations()):
            for j, post in enumerate(self.Populations[1].get_Populations()):
                if i == j:
                    # same cluster
                    self.model.add_synapse_population(str(i) + "II" + str(j), self.params['matrixType'], delaySteps,
                                                 pre, post,
                                                 "StaticPulse", {}, {"g": j_ii * self.params['jplus'][1, 1]}, {}, {},
                                                 psc_exp_distributed, {}, psc_I, conn_params_II
                                                 )
                else:
                    self.model.add_synapse_population(str(i) + "II" + str(j), self.params['matrixType'], delaySteps,
                                                 pre, post,
                                                 "StaticPulse", {}, {"g": j_ii * jminus[1, 1]}, {}, {},
                                                 psc_exp_distributed, {}, psc_I, conn_params_II
                                                 )
        print('Js: ', js / np.sqrt(N))


if __name__ == "__main__":
    sys.path.append("..")
    from Defaults import defaultSimulate as default
    import matplotlib.pyplot as plt
    CV=0.05

    params = {'n_jobs': 24, 'N_E': 20000, 'N_I': 5000, 'dt': 0.1, 'neuron_type': 'iaf_psc_exp', 'simtime': 3000,
              'delta_I_xE': 0., 'delta_I_xI': 0., 'record_voltage': False, 'record_from': 1, 'warmup': 1000, 'Q': 20}

    jip_ratio = 0.75  # 0.75 default value  #works with 0.95 and gif wo adaptation
    jep = 2.75  # clustering strength
    jip = 1. + (jep - 1) * jip_ratio
    params['jplus'] = np.array([[jep, jip], [jip, jip]])

    I_ths = [2.13, 1.24]  # set background stimulation baseline to 0
    params['I_th_E'] = I_ths[0]
    params['I_th_I'] = I_ths[1]
    params['matrixType'] = "SPARSE_GLOBALG_INDIVIDUAL_PSM"

    # PROCEDURAL_GLOBALG_INDIVIDUAL_PSM
    # SPARSE_GLOBALG_INDIVIDUAL_PSM



    Cluster = ClusteredNetworkGeNN_Distributed(default, params, batch_size=1)

    # Name has to be changed because PyGeNN will be confused if two objects with the same reference are present
    Cluster.set_model_build_pipeline([lambda: Cluster.setup_GeNN(Name="EICluster2"), Cluster.create_populations,
                                      Cluster.create_stimulation, Cluster.create_recording_devices, lambda: Cluster.connect_uniform_distributed(CV),
                                      Cluster.prepare_global_parameters])


    Cluster.setup_network()
    Cluster.build_model()
    Cluster.load_model()  # set GPUspecificConstraint to a value like 5000000 to test if the splitting works


    spiketimes = Cluster.simulate_and_get_recordings()
    rates = np.array(Cluster.get_firing_rates(spiketimes))
    print(rates)
    plt.figure()
    spiketimesplot=(spiketimes[0][1, :]%50)==0
    plt.plot(spiketimes[0][0, spiketimesplot], spiketimes[0][1, spiketimesplot], '.')
    plt.savefig('GeNN_1.png')
    print(Cluster.get_timing())