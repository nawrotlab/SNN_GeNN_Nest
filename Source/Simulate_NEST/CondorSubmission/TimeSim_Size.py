import os
import htcondor
import shutil
schedd = htcondor.Schedd()

Outputpath="NEST_TimeSim_Size.pkl"  

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



Trials=10
MatrixType=[0]
sizes=[   5,   50,  115,  160,  200,  230,  255,  280,  300
		320,  340,	360,  565, 
		805,  985, 1140, 1275, 1395, 
	   1510, 1610, 1710, 2205, 2550]  #measured in 100 neurons
       #Extended Datapoints are added
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
