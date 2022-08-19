import os
import htcondor
import shutil
schedd = htcondor.Schedd()

Outputpath="GeNN_TimeSim_Size.pkl"

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

Trials=10
MatrixType=[0, 1]
sizes=[    5,    50,   115,   160,   200,   230,   255,   280,   300,
         320,   340,   360,   565,   805,   985,  1140,  1275,  1395,
        1510,  1610,  1710,  1800,  2205,  2550,  2850,  3120,  3605,
        4030,  6370,  9010, 12740, 15605, 18020, 22070, 25485, 31210,
       36040] #measured in 100 neurons
       #Extended Datapoints are added
times=[10]


sub = htcondor.Submit()
sub['executable']=              '/Benchmark/Simulate_Genn/CondorSubmission/RunSimulation.sh'
sub['request_cpus']=            '20'
sub['request_gpus']=            '1'
sub['request_memory']=          '30GB'
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


