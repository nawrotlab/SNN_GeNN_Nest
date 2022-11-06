import os
import pickle
import time
import numpy as np
from Helper import GeneralHelper

from itertools import islice

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


class ParamField:
    """
    Object which contains a meshgrid of N parameters and keeps track of the parameter values which were already sampled
    from it.
    """
    def __init__(self, Constructor):
        """
        Initalizes Parameter field
        Parameter:
           Constructor: Tuple of N-Tuples which describe the parameter to construct the field from.
           Example: NEST
           Constructor = (
            ('Ix', [x1, x2, x3, ... xn], ('I_X_E',)),
            ('Ix', [y1, y2, y3, ... ym], ('I_X_I',)),
            )
            The Ix at the beginning and the I_X_.. are just decriptive to match the format used by GeNN-Gridsearches.
            E.g. Ix could be changed to the unit used for this parameter. I_X_E and I_X_I.
            The order of them is also the order in the outputted parameters if samples are taken from the field.
            Example: GeNN
            Constructor=(
            ('Ix',[x1, x2, x3, ... xn], (Pops[0],)),
            ('Ix',[y1, y2, y3, ... ym], (Pops[1],)),
            )
            The 'Ix' gives the name of the global parameter to be changed in the simulation and the Pops[0] or Pops[1]
            the populations targeted by these global parameters. The list in the middle are the values to construct the
            field from.
        """
        self.shape = tuple([len(dim[1]) for dim in Constructor])
        self.paramField = np.empty(self.shape, dtype=object)
        self.bitfield = np.zeros(self.shape, dtype=bool)
        # Create Indexing array for parameter field and fill it with the parameter tuples.
        for idx in DynIndex(self.shape):
            tempTup = []
            for dimId, elementId in enumerate(idx):
                for Pop in Constructor[dimId][2]:
                    tempTup.append(Constructor[dimId][1][elementId])
            self.paramField[idx] = tempTup
            self.Vectorbase = [(dim[0], Pop) for dim in Constructor for Pop in dim[2]]

    def get_data(self, idx):
        """
        Gets the parameter combination of a specific coordinate, but does not change the status of the bitfield
        -> Coordinate is not marked as sampled afterwards if it hasn't been before.
        """
        return (self.paramField[idx])

    def get_paramField(self):
        """
        Returns the whole parameter field (Array of tuples containing the parameter for that coordinate).
        """
        return self.paramField

    def print(self):
        """
        Prints first the bitfield. (False -> Coordinate not sampled)
        Than the parameter field.
        """
        print(self.bitfield)
        print(self.paramField)

    def get_UnusedIdx(self):
        """
        Returns the coordinates which haven't been sampled from.
        """
        return np.argwhere(self.bitfield == False)

    def get_Params(self, Idxs, Ignore=False):
        """
        Return by Idxs requested parameters. If Ignore is false set coordinates as sampled afterwards and check if they
        were sampled before
        """
        Params = []
        for id in Idxs:
            if Ignore == False:
                assert self.bitfield[id] == False, "Param already pulled"
                self.bitfield[id] = True
            Params.append(self.paramField[id])
        return np.transpose(np.array(Params))

    def get_Vectorbase(self):
        """
        Return vectorbase which is the combination of the name of a global parameter and the population
        for each dimension.
        (NEST returns the combination of the first and third part of the constructor -> Useful for the name and the unit)
        """
        return self.Vectorbase

    def randomSample(self, NumberSamples: int):
        """
        Samples randomly NumberSamples Samples in the parameter field.
        Returns:
            -1: no samples left
             1: Not enough samples left to return the  requested number. THe lower number of samples is returned and
                marked as sampled.
        """
        Ids = self.get_UnusedIdx()
        if Ids.size == 0:
            return -1, ([], []), Ids
        elif len(Ids) < NumberSamples:
            Ids = Ids[np.random.choice(Ids.shape[0], len(Ids), replace=False)]
            Parms = self.get_Params([tuple(i) for i in Ids])
            return 1, (self.get_Vectorbase(), Parms), Ids
        else:
            Ids = Ids[np.random.choice(Ids.shape[0], NumberSamples, replace=False)]
            Parms = self.get_Params([tuple(i) for i in Ids])
            return 0, (self.get_Vectorbase(), Parms), Ids

    def axisSample(self, NumberSamples: int, axis=0):
        """
        Samples NumberSamples Samples along a given axis in the parameter field.
        Returns:
            -1: no samples left
             1: Not enough samples left to return the  requested number. THe lower number of samples is returned and
                marked as sampled.
        """
        if (self.shape[axis] % NumberSamples):
            print("Number of Samples does not fit to the dimensions of the parameter field - Batches not alligned")
        Ids = self.get_UnusedIdx()

        if Ids.size == 0:
            return -1, ([], []), Ids
        else:
            SortTuple = tuple([Ids[:, (jj + axis) % len(Ids[0])] for jj in range(len(Ids[0]))])
            Ids = Ids[np.lexsort(SortTuple)]

        if len(Ids) < NumberSamples:
            Parms = self.get_Params([tuple(i) for i in Ids])
            return 1, (self.get_Vectorbase(), Parms), Ids
        else:
            Ids = Ids[0:NumberSamples]
            Parms = self.get_Params([tuple(i) for i in Ids])
            return 0, (self.get_Vectorbase(), Parms), Ids

    def AllSample(self):
        """
        Returns all samples which are not yet sampled and marks them as sampled.
        """
        Ids = self.get_UnusedIdx()
        if Ids.size == 0:
            return -1, [], Ids
        else:
            Parms = self.get_Params([tuple(Ids[0])])
            return 0, Parms, Ids[0]


def DynIndex(stop):
    """
    Generator function which yields indexing tuples dynamically of a N dimensional array.
    Parameter:
        stop: tuple of N numbers which represent the shape of the N dimensional array
    """
    dims = len(stop)
    if not dims:
        yield ()
        return
    for outer in DynIndex(stop[1:]):
        for inner in range(0, stop[0]):
            yield (inner,) + outer



class Gridsearch:
    """
    General class to setup and run a gridsearch.
    """
    def __init__(self, simFun, params, Constructor, measurementVar, PathOutput="Measurement.pkl", PathSpikes=None):
        """
        Init Gridsearch object, which contains the parameter field used for the gridsearch, the names of the measurements
        which are reported back. The useds parameters are merged with the default values and saved as well.
        """
        startInitGrid = time.time()
        if os.path.exists(PathOutput):
            os.remove(PathOutput)
        if PathSpikes is not None:
            if os.path.exists(PathSpikes):
                os.remove(PathSpikes)
        self.ParamField = ParamField(Constructor)
        self.Parameter = params
        self.measurementVar = measurementVar
        self.OutputPath = PathOutput
        self.SpikesPath = PathSpikes
        self.SimulationFunction = simFun
        with open(self.OutputPath, 'ab') as outfile:
            pickle.dump(self.measurementVar
                        , outfile)
            pickle.dump(self.getParamField()
                        , outfile)
            pickle.dump(self.Parameter, outfile)
        endInitGrid = time.time()
        self.Timing = {'InitGrid': endInitGrid - startInitGrid}

    def getTiming(self):
        """
        Returns the timing information of the gridsearch.
        """
        return self.Timing

    def getParamField(self):
        """
        Returns the parameter field used in the gridsearch.
        """
        return self.ParamField.get_paramField()


class Gridsearch_NEST(Gridsearch):
    """
        Gridsearch with NEST. Adds Nworkers argument controlling the number of parallel simulations running and the
        simulation function itself.
    """
    def __init__(self, simFun, params, Constructor, measurementVar, default, PathOutput="Measurement.pkl",
                 PathSpikes=None, Nworkers=6):
        startInitGrid = time.time()
        Parameter=GeneralHelper.mergeParams(params, default)
        Parameter['Nworker'] = Nworkers
        super().__init__(simFun, Parameter, Constructor, measurementVar, PathOutput, PathSpikes)
        endInitGrid = time.time()
        self.Timing = {'InitGrid': endInitGrid - startInitGrid}

    def search(self, ProgressBar=False, ReuseSimulation=False):
        """
        Run the Gridsearch and set the timing information.
        Arguments:
            ProgressBar: True: Progress bar is shown
        """
        from pathos.multiprocessing import ProcessPool
        import multiprocessing
        import numpy as np
        Queue=None
        Parameterlist = []
        rv, Parm, Ids = self.ParamField.AllSample()
        while rv != -1:
            Parameterlist.append((Parm, Ids))
            rv, Parm, Ids = self.ParamField.AllSample()
        m = multiprocessing.Manager()
        lock = [m.Lock(), m.Lock()]
        if ProgressBar:
            from tqdm import tqdm
            import numpy as np
            numberSamples = np.prod(self.ParamField.shape)
            Queue= m.Queue(numberSamples)
            LQeueTM1=0
            pbar = tqdm(total=numberSamples)

        if ReuseSimulation:
            with ProcessPool(nodes=self.Parameter['Nworker']) as p:
                TimesL = p.amap(lambda x: self.SimulationFunction(self.Parameter, x, self.measurementVar, x,
                                                    self.OutputPath, lock, PathSpikes=self.SpikesPath, timeout=7200, Queue=Queue), [Parameterlist])
        else:
            with ProcessPool(nodes=self.Parameter['Nworker']) as p:
                TimesL = p.amap(lambda x: self.SimulationFunction(self.Parameter, x[0], self.measurementVar, x[1],
                                                    self.OutputPath, lock, PathSpikes=self.SpikesPath, timeout=7200, Queue=Queue), Parameterlist)

        if ProgressBar:
            while TimesL.ready() is False:
                LQeueT = Queue.qsize()
                pbar.update(LQeueT-LQeueTM1)
                LQeueTM1=LQeueT
                time.sleep(0.25)

        TimesL = TimesL.get()

        BuildTimes = []
        CompileTimes = []
        LoadTimes = []
        SimTimes = []
        DownloadTimes = []
        DumpTimes = []

        for Times in TimesL:
            BuildTimes.append(Times["Build"])
            CompileTimes.append(Times["Compile"])
            LoadTimes.append(Times["Load"])
            SimTimes.append(Times["Sim"])
            DownloadTimes.append(Times["Download"])
            DumpTimes.append(Times["Dump"])
        self.Timing['Build'] = BuildTimes
        self.Timing['Compile'] = CompileTimes
        self.Timing['Load'] = LoadTimes
        self.Timing['Simulation'] = SimTimes
        self.Timing['Download'] = DownloadTimes
        self.Timing['Dump'] = DumpTimes
        #Ids are not neccassary as all simulations are independent and the results are reported in an ordered list.


class Gridsearch_GeNN(Gridsearch):
    """
        Gridsearch with GeNN.
    """

    def __init__(self, model, SimulationFunction, Constructor, measurementVar, PathOutput="Measurement.pkl",
                 PathSpikes=None):

        startInitGrid = time.time()
        super().__init__(SimulationFunction, model.get_parameter(), Constructor, measurementVar, PathOutput, PathSpikes)
        endInitGrid = time.time()
        self.Timing = {'InitGrid': endInitGrid - startInitGrid, 'Build': model.get_timing()['Build'],
                       'Compile': model.get_timing()['Compile'], 'Load': model.get_timing()['Load']}

    def search(self, axis=None, ProgressBar=False):
        """
        Run the Gridsearch and set the timing information.
        Arguments:
            axis : controls in which direction the parameter field is sampled and
            thus which ids are contained in a batch. If a batch_size greater 1 is defined the networks in a batch
                will be correlated by the same connectivity matrix
                None: random sample
                int: axis number to sample along
            ProgressBar: True: Progress bar is shown
        """

        ReinitializationTimes = []
        SimTimes = []
        DownloadTimes = []
        DumpTimes = []
        IdTimes =[]

        if axis is None:
            method = lambda x: self.ParamField.randomSample(NumberSamples=x)
        else:
            method = lambda x: self.ParamField.axisSample(NumberSamples=x, axis=axis)

        rv, Parm, Ids = method(self.Parameter['batch_size'])


        if ProgressBar:
            from tqdm import tqdm
            import numpy as np
            numberSamples=np.prod(self.ParamField.shape)
            pbar = tqdm(total=numberSamples)
        while rv != -1:
            if (rv != -1):
                Times = self.SimulationFunction(Parm, Ids, self.measurementVar, self.OutputPath,
                                    PathSpikes=self.SpikesPath)
            else:
                Times = {"TimeInit": -1, "Sim": -1, "Download": -1}
            ReinitializationTimes.append(Times["ReInit"])
            SimTimes.append(Times["Sim"])
            DownloadTimes.append(Times["Download"])
            DumpTimes.append(Times["Dump"])
            IdTimes.append(Ids)
            if ProgressBar:
                pbar.update(len(Ids))
            rv, Parm, Ids = method(self.Parameter['batch_size'])  # prepare for next Run

        self.Timing['ReInit'] = ReinitializationTimes
        self.Timing['Simulation'] = SimTimes
        self.Timing['Download'] = DownloadTimes
        self.Timing['Dump'] = DumpTimes
        self.Timing['Ids'] = IdTimes
