#!/bin/bash

GPUSupport=true
PushToDocker=false
Prefix="testuser/snn_genn_nest"
Workingdirectory=$PWD

 
while getopts ":dgp:" opt; do
  case $opt in
    d)
      echo "Images will be uploaded to DockerHub" >&2
      PushToDocker=true
      ;;
    g)
      echo "Raw CentOS image used - no GPU support" >&2
      GPUSupport=false
      ;;
    p)
      echo "Prefix of Images: $OPTARG">&2
      Prefix=$OPTARG
      ;;

    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
Prefix="${Prefix}:"


git clone https://github.com/htcondor/htcondor.git

if $GPUSupport
then
  patch -i build-images.patch htcondor/build/docker/services/build-images
  Prefix="${Prefix}gpu_"
fi


cd htcondor/build/docker/services
./build-images --distro=el7 --prefix=local/htcondor
cd $Workingdirectory
rm -R -f htcondor

docker build -f Dockerfile_Conda --build-arg IMAGE=local/htcondormini:latest -t "${Prefix}conda" .
if $GPUSupport
then
  docker build -f Dockerfile_GeNN_GPU --build-arg IMAGE="${Prefix}conda" --build-arg GPU=$GPUSupport -t "${Prefix}genn" .
else
  docker build -f Dockerfile_GeNN --build-arg IMAGE="${Prefix}conda" --build-arg GPU=$GPUSupport -t "${Prefix}genn" .
fi
docker build -f Dockerfile_NEST --build-arg IMAGE="${Prefix}conda" -t "${Prefix}nest" .

docker build -f Dockerfile_NEST --build-arg IMAGE="${Prefix}genn" -t "${Prefix}genn_nest" .

if $PushToDocker
then
  docker push -a "${Prefix%:*}" 
fi
