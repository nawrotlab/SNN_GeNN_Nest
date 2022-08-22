import os
import htcondor
import shutil
schedd = htcondor.Schedd()

Outputpath="NEST_TimeBiol_sim.pkl"

try:
    os.remove("/Benchmark/Simulate_NEST/"+Outputpath)
except:
    print("No Data was written before!")

# Check whether the specified path exists or not
isExist = os.path.exists("Logs")
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs("Logs")

LogBase="Logs/Log-$(ProcId).log"
OutputBase="Logs/Simulation-$(ProcId).out"
Errorbase="Logs/Error-$(ProcId).err"


Trials=5

MatrixType=[0]
sizes=[50, 500]  #measured in 100 neurons
times=[0, 1, 5, 10, 25, 50,100, 175, 350, 625, 1250, 2500]

coll = htcondor.Collector()
GPU=0
CPU=0
Slots=coll.query(htcondor.htcondor.AdTypes.Startd, projection=['CPUs'])
for ii in Slots:
    CPU+=ii.get('CPUs')
print("Total CPUs: ", CPU)

sub = htcondor.Submit()
sub['executable']=				'/Benchmark/Simulate_NEST/CondorSubmission/RunSimulation3.sh'
sub['request_cpus']=            str(CPU)
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
