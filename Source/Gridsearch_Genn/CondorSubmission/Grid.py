import os
import htcondor
import shutil
schedd = htcondor.Schedd()

# Check whether the specified path exists or not
isExist = os.path.exists("Logs")
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs("Logs")

coll = htcondor.Collector()
GPU=0
CPU=0
Slots=coll.query(htcondor.htcondor.AdTypes.Startd, projection=['CPUs', 'GPUs'])
for ii in Slots:
	GPU+=ii.get('GPUs')
	CPU+=ii.get('CPUs')
print("Total GPUs: ", GPU, ", Total CPUs: ", CPU)

# Example Two: submit by adding attributes to the Submit() object
sub = htcondor.Submit()
sub['executable']=              '/Benchmark/Gridsearch_Genn/CondorSubmission/RunSimulation.sh'
sub['log']=                     'Logs/Log.log'
sub['output']=                  'Logs/Simulation.out'
sub['error']=                   'Logs/Error.err'
sub['request_cpus']=            str(CPU)
sub['request_gpus']=                    str(GPU)
sub['request_memory']=			'1GB'           #Memory requirement is not meaningful -> It will need more
sub['should_transfer_files']=   'No'

ii=1
with schedd.transaction() as txn:
     sub.queue(txn)

print("Submitted " + str(ii) + " Jobs to queue.")


