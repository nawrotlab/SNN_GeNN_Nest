import os
import htcondor
import shutil
schedd = htcondor.Schedd()


# Example Two: submit by adding attributes to the Submit() object
sub = htcondor.Submit()
sub['executable']=              '/Benchmark/Gridsearch_Genn/CondorSubmission/RunSimulation.sh'
sub['log']=                     'Logs/Log.log'
sub['output']=                  'Logs/Simulation.out'
sub['error']=                   'Logs/Error.err'
sub['request_cpus']=            '20'
sub['request_gpus']=                    '1'
sub['request_memory']=                  '110GB'
sub['should_transfer_files']=   'No'

ii=1
with schedd.transaction() as txn:
     sub.queue(txn)

print("Submitted " + str(ii) + " Jobs to queue.")


