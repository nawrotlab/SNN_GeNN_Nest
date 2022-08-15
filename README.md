This repository contains all accompanying code to reproduce the figures of the following paper:

Felix Schmitt, Vahid Rostami, Martin Nawrot, **Efficient parameter calibration and real-time
simulation of large scale spiking neural networks with GeNN and NEST**

If you use any parts of this code for your own work, please acknowledge us by citing the above paper.

If you have questions or encounter any problems while using this code base, feel free to file a Github issue here and we'll be in touch !

# Project layout
- Data/ contains all simulated data used for the figures of the paper except the spiking data of the gridsearch due to the limitation of filesize. It's split into data of the gridsearches, the simulations of small networks for long periods of time and the simulation of large networks.
- Docker/ contains a build script for our supplied docker images to run the supplied source code without setting up the enviroment. We did not use these docker files to run our simulations
- Figures/ containes the scripts togenerate the figures of the paper from the data and the generated figures
- Source/ containes the Source code of the simulations.
  - Helper/ contains classes for the clustered model in GeNN and NEST, the user speific GeNN models (Neuron models, current sources, ...) and a class to perform gridsearches. The classes of the clustered model can be run directly to test the software enviroment.
  - Defaults/ contains the default parameters of the simulation. It consists of two induvidual files, one for the gridsearch and one for the independent simulatins. All values can be overwritten in the specific simulation scripts.
  - Gridsearch_$Simulator$ contains the gridsearch with the specific $Simulator$ and a folder which contains neccessary scripts to submit the gridsearch as a jo to htcondor.
  - Simulate_$Simulator$ contains the simulation script with the specific $Simulator$ and a folder which contains neccessary scripts to submit the jobs to reproduce the data of figure 3 and 4.

# Docker Images
We supply docker images to run our code without installing NEST and GeNN.
Images can be found under: https://hub.docker.com/repository/docker/fschmitt/snn_genn_nest. 
All images contain htcondor as a scheduler and an anconda basic installation. The images are not optimized like containers used in production environments. 

The relevant tags to test our code are:
- genn
- nest
- genn_nest

- gpu_genn
- gpu_genn_nest
- gpu_genn_nest 

All tags with a "gpu" prefix contain the CUDA drivers needed to use GeNN on the GPU. GPU drivers can be only used in Docker if the  nvidia-container-runtime
is installed. The listed simulators in the tag are available in the image. We recommend using either the images with GPU support or the ones without as we had to change the base image for the GPU support and thus these two families do not share layers. 
We decided to not include the source code in the image as it would increase their size and make it harder to edit parameters and collect results. We use a bind-mount to mount the directory with the source code into the image. Unfortunately this creates some problems with the file permission. 

Please follow the following steps to run the images:
1) `chmod -R 777 Source`
2) `docker pull fschmitt/snn_genn_nest:Tag`
3) `docker run --rm --gpus all --detach --mount type=bind,source=PATH to sourcecode,target=/Benchmark --name=NAME fschmitt/snn_genn_nest:Tag`
4) `docker exec -ti NAME /bin/bash`

You can now enter the Benchmark folder and the subfolder of the different experiments. Now you can run the simulation by python SourceCode.py.
If you want to reproduce our results, please make sure all previously written data \*.pkl is writable by all users (`chmod 777 *.pkl`) and for GeNN simulation no folder EICluster_CODE exists (`rm -RI EICluster_CODE`). Now you can enter the appropriate CondorSubmission folder and run `su submituser`. Experiments can be scheduled by `python CondorSubmission.py`. By running `condor_q` you can see the currently scheduled jobs and by `condor_history` you get the information about finished jobs.
