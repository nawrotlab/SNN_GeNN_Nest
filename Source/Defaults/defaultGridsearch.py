# GIF-Parameters as similar as possible to lif model

import numpy as np
# default values for parameters
eps = np.finfo(float).eps
# number of units
N_E = 1200
N_I = 300

# connection probabilities
ps = np.array([[0.2,0.5],[0.5,0.5]])

# connections strengths
# weights are js/sqrt(N)
# nan means they are calculated
js = np.ones((2,2))*np.nan
# factors for inhibitory weights
ge = 1.2
gi = 1.
gie = 1.


# cluster number
Q = 6
# cluster weight ratios
jplus = np.ones((2,2))

# synaptic delay
delay =0.1

# factor multiplied with weights
s = 1.
fixed_indegree = False
# neuron parameters
#neuron_type = 'gif_psc_exp'
neuron_type = 'iaf_psc_exp'

E_L = 0.
C_m = 1.
tau_E = 20.
tau_I = 10.
t_ref = 5.
V_th_E = 20.
V_th_I = 20.
V_r = 0.
I_xE = 1.
I_xI = 2.
delta_I_xE = 0.
delta_I_xI = 0.
I_th_E = 1.25#2.13#2.13
I_th_I = 0.78#1.24#1.24
V_m =  'rand' #10 #

tau_syn_ex = 3. #2.  #3.
tau_syn_in = 2. #1. # 2.

#Extra parameters
lambda_0 = 1000. # intensity of point process at firing threshold V_T in 1/s
Delta_V=0.00000000000000000000000000000000000000001  # sharpness of stochasiticity with lim -> 0 deterministic
g_L_E = C_m / tau_E
g_L_I = C_m / tau_I
# adaptation
#adaptive TH
q_sfa = [0.0] #[3.]# #[eps] #
tau_sfa = [2.]#[eps]
#spike triggered current

Q_AdapI=0.0*230
tau_stc = [179.0]#[eps]
q_stc = [Q_AdapI/tau_stc[0]] #[3.]# #[eps] #


n_jobs = 12

#Distribution of synaptic weights
#available distributions= https://nest-simulator.readthedocs.io/en/stable/guides/connection_management.html#dist-params


DistParams={'distribution':'normal', 'sigma': 0.0, 'fraction': False}

syn_params={"U": 0.2, "u": 0.0, "tau_rec": 120.0,
                  "tau_fac": 0.0}


# Defaults otherwise set in the simulation

dt=0.1
simtime=1000.
warmup=0.
record_voltage = False
record_from = 'all'
recording_interval = dt
return_weights = False


# stimulation
stim_clusters = None # clusters to be stimulated
stim_amp = 0.  # amplitude of the stimulation current in pA
stim_starts = []      # list of stimulation start times
stim_ends = []          # list of stimulation end times
