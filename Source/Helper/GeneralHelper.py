from types import ModuleType
import copy
# ********************************************************************************
#                                    Classes
# ********************************************************************************


class TimeoutException(Exception):   # Custom exception class
    pass
# ********************************************************************************
#                                    Functions
# ********************************************************************************


def mergeParams(params, defaultValues):
    """
    Updates default Values defined in a module with params-dictionary.
    The returned dictionary contains the parameters used for simulation.
    """
    return nested_update(moduleVarToDict(defaultValues), params)


def timeout_handler(signum, frame):
    """
    # Custom signal handler for timeout. 
    Should be raised if a set of instructions takes too long
    """   
    raise TimeoutException


def moduleVarToDict(module):
    """
    Creates dict of explicit variables in module which are not imported modules.
    Key names equal the variable name.
    """
    ModuleDict = {}
    if module:
        ModuleDict = {key: value for key, value in module.__dict__.items() if not (key.startswith('__') or key.startswith('_') or isinstance(value, ModuleType))}
    return ModuleDict

def nested_update(d, d2):
    """
    Updates values in d with values of d2. Adds new keys if they are not present.
    Returns Changed dict.
    """
    d_local=copy.deepcopy(d)
    for key in d2:
        if isinstance(d2[key], dict) and key in d_local:
            d_local[key] = nested_update(d_local[key], d2[key])
        else:
            d_local[key] = d2[key]
    return d_local


if __name__ == "__main__":
    import defaultTest as default
    Test = {'Message': "Update was successfull.",
              'Nested_Dict': {'Test':2}}

    defaultValues=moduleVarToDict(default)
    print("Default values:")
    print(defaultValues)
    print("\nUpdated values")
    print(mergeParams(Test, default))
    print(mergeParams({}, default))

"""
defaultTest file content:
Message = "Not updated"
Nested_Dict= {'Test':1, 'NoChange':0}
UnchangedVar=0

defaultTest.py has to be created to test the merger.
"""


