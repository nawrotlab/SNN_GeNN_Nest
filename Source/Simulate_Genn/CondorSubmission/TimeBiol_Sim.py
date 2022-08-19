

import os
import htcondor
import shutil
schedd = htcondor.Schedd()

Outputpath="GeNN_TimeBiol_sim.pkl"

try:
        os.remove("/Benchmark/Simulate_Genn/"+Outputpath)
except:
        print("No Data was written before!")

try:
        shutil.rmtree("/Benchmark/Simulate_Genn/EICluster_CODE", ignore_errors=False, onerror=None)
except:
        print("Code was not generated before!")

# Check whether the specified path exists or not
isExist = os.path.exists("Logs")
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs("Logs")

LogBase="Logs/Log-$(ProcId).log"
OutputBase="Logs/Simulation-$(ProcId).out"
Errorbase="Logs/Error-$(ProcId).err"

Trials=5

MatrixType=[0, 1]
sizes=[50, 500]  #measured in 100 neurons
times=[0, 1, 5, 10, 25, 50,100, 175, 350, 625, 1250, 2500]

coll = htcondor.Collector()
GPU=0
CPU=0
Slots=coll.query(htcondor.htcondor.AdTypes.Startd, projection=['CPUs', 'GPUs'])
for ii in Slots:
	GPU+=ii.get('GPUs')
	CPU+=ii.get('CPUs')
print("Total GPUs: ", GPU, ", Total CPUs: ", CPU)

sub = htcondor.Submit()
sub['executable']=              '/Benchmark/Simulate_Genn/CondorSubmission/RunSimulation.sh'
sub['request_cpus']=            str(CPU)
sub['request_gpus']=                    str(GPU)
sub['request_memory']=			'1GB'           #Memory requirement is not meaningful -> It will need more
sub['should_transfer_files']=   'No'

ii=0


for size in sizes:
        for jj in range(Trials):
                for matType in MatrixType:      
                        for time in times:
                                sub['arguments']=str(size) + " " + str(time) + " " + str(matType) + " " + Outputpath
                                sub['log']=LogBase.replace("$(ProcId)", str(ii))                  
                                sub['output']=OutputBase.replace("$(ProcId)", str(ii))                 
                                sub['error']=Errorbase.replace("$(ProcId)", str(ii))  
                                ii+=1
                                with schedd.transaction() as txn:
                                  sub.queue(txn)

print("Submitted " + str(ii) + " Jobs to queue.")


