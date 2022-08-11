import sys
sys.path.append("..")
from Helper import GeneralHelper


class ClusteredNetworkBase:
    """ 
    Baseobject with basic initilaization and method for firing rate estimation
    """

    def __init__(self, defaultValues, parameters):
        """
        Creates an object with functions to create neuron populations,
        stimulation devices and recording devices for an EI-clustered network.
        Initializes the object. Creates the attributes Populations, RecordingDevices and 
        Currentsources to be filled during network construction.
        Attribute params contains all parameters used to construct network.

        Parameters:
            defaultValues (module): A Module which contains the default configuration
            parameters (dict):      Dictionary with parameters which should be modified from their default values
        """
        self.params = GeneralHelper.mergeParams(parameters, defaultValues)

    def get_firing_rates(self, spiketimes=None):
        """
        Calculates the firing rates of all excitatory neurons and the firing rates of all inhibitory neurons
        created by self.create_populations. If spiketimes are not supplied, they get extracted.
        Parameters:
            spiketimes: (optional, np.array 2xT)   spiketimes of simulation
        Returns:
            (e_rate, i_rate) average firing rate of excitatory/inhibitory neurons (spikes/s)
        """
        if spiketimes is None:
            spiketimes = self.get_recordings()
        e_count = spiketimes[:, spiketimes[1] < self.params['N_E']].shape[1]
        i_count = spiketimes[:, spiketimes[1] >= self.params['N_E']].shape[1]
        e_rate = e_count / float(self.params['N_E']) / float(self.params['simtime']) * 1000.
        i_rate = i_count / float(self.params['N_I']) / float(self.params['simtime']) * 1000.
        return e_rate, i_rate

    def get_recordings(self):
        """
        Placeholder to be filled in children
        """
        return []