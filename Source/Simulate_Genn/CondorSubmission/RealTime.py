import os
import htcondor
import shutil
schedd = htcondor.Schedd()

Outputpath="GeNN_TimeSim_Size_RT.pkl"

try:
        os.remove("/Benchmark/Simulate_Genn/"+Outputpath)
except:
        print("No Data was written before!")

try:
        shutil.rmtree("/Benchmark/Simulate_Genn/EICluster_CODE", ignore_errors=False, onerror=None)
except:
        print("Code was not generated before!")



LogBase="Logs/Log-$(ProcId).log"
OutputBase="Logs/Simulation-$(ProcId).out"
Errorbase="Logs/Error-$(ProcId).err"

Trials=10
MatrixType=[0, 1, 1, 1]
sizes=[944, 191, 192, 193]  #measured in 100 neurons
times=[10]

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
sub['request_memory']=                  '1GB'      #Memory requirement is not meaningful -> It will need more
sub['should_transfer_files']=   'No'

ii=0
for size, matType in zip(sizes, MatrixType):
        for jj in range(Trials):
                for time in times:
                        sub['arguments']=str(size) + " " + str(time) + " " + str(matType) + " " + Outputpath
                        sub['log']=LogBase.replace("$(ProcId)", str(ii))                  
                        sub['output']=OutputBase.replace("$(ProcId)", str(ii))                 
                        sub['error']=Errorbase.replace("$(ProcId)", str(ii))   
                        ii+=1
                        with schedd.transaction() as txn:
                                sub.queue(txn)

print("Submitted " + str(ii) + " Jobs to queue.")