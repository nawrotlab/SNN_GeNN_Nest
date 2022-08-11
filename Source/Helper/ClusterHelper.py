import numpy as np
small = 1e-10

def max_PSP_exp(tau_m, tau_syn, C_m=1., E_l=0.):
    """
    Calculates the maximum psp amplitude for exponential synapses and unit J.
    Parameters:
        tau_m (float):      Membrane time constant [ms]
        tau_syn (float):    Synapse time constant  [ms]
        C_m (float):        Membrane capacity [pF]
        E_l (float):        Resting potential (mV)

    """
    tmax = np.log(tau_syn / tau_m) / (1 / tau_m - 1 / tau_syn)
    B = tau_m * tau_syn / C_m / (tau_syn - tau_m)
    return (E_l - B) * np.exp(-tmax / tau_m) + B * np.exp(-tmax / tau_syn)

def calc_js(params):
    """calculates the synaptic weights for exponential synapses before clustering"""
    N_E = params.get('N_E')  # excitatory units
    N_I = params.get('N_I')  # inhibitory units
    N = N_E + N_I  # total units
    ps = params.get('ps')  # connection probs
    ge = params.get('ge')
    gi = params.get('gi')
    gie = params.get('gie')
    V_th_E = params.get('V_th_E')  # threshold voltage
    V_th_I = params.get('V_th_I')
    tau_E = params.get('tau_E')
    tau_I = params.get('tau_I')
    E_L = params.get('E_L')
    neuron_type = params.get('neuron_type')
    if ('iaf_psc_exp' in neuron_type) or ('gif_psc_exp' in neuron_type):
        tau_syn_ex = params.get('tau_syn_ex')
        tau_syn_in = params.get('tau_syn_in')
        amp_EE = max_PSP_exp(tau_E, tau_syn_ex)
        amp_EI = max_PSP_exp(tau_E, tau_syn_in)
        amp_IE = max_PSP_exp(tau_I, tau_syn_ex)
        amp_II = max_PSP_exp(tau_I, tau_syn_in)
    else:
        amp_EE = 1.
        amp_EI = 1.
        amp_IE = 1.
        amp_II = 1.

    js = np.zeros((2, 2))
    K_EE = N_E * ps[0, 0]
    js[0, 0] = (V_th_E - E_L) * (K_EE ** -0.5) * N ** 0.5 / amp_EE
    js[0, 1] = -ge * js[0, 0] * ps[0, 0] * N_E * amp_EE / (ps[0, 1] * N_I * amp_EI)
    K_IE = N_E * ps[1, 0]
    js[1, 0] = gie * (V_th_I - E_L) * (K_IE ** -0.5) * N ** 0.5 / amp_IE
    js[1, 1] = -gi * js[1, 0] * ps[1, 0] * N_E * amp_IE / (ps[1, 1] * N_I * amp_II)
    return js


def FPT(tau_m, E_L, I_e, C_m, Vtarget, Vstart):
    """ calculate first pasage time between Vstart and Vtarget."""
    return -tau_m * np.log((Vtarget - E_L - tau_m * I_e / C_m) / (Vstart - E_L - tau_m * I_e / C_m + small))


def V_FPT(tau_m, E_L, I_e, C_m, Ttarget, Vtarget, t_ref):
    """ calculate the initial voltage required to obtain a certain first passage time. """
    return (Vtarget - E_L - tau_m * I_e / C_m) * np.exp((Ttarget) / tau_m) + E_L + tau_m * I_e / C_m