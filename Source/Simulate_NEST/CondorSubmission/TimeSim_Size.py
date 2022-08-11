import os
import htcondor
import shutil
schedd = htcondor.Schedd()

Outputpath="NEST_TimeSim_Size.pkl"  

try:
	os.remove("/Benchmark/Simulate_NEST/"+Outputpath)
except:
	print("No Data was written before!")

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

sub = htcondor.Submit()
sub['executable']=				'/Benchmark/Simulate_NEST/CondorSubmission/RunSimulation3.sh'
sub['request_cpus']=            '24'
sub['request_gpus']= 			'0'
sub['request_memory']=			'100GB'
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