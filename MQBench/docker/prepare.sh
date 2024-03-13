#! /bin/bash 


# install docker 
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

# add nvidia docker repo 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list


sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 nvidia-smi


###### ==============================================>  New add
# Build image named quan_mqbench
docker build . -t quan_mqbench

# Build docker and mapping some folders
docker run --rm -it \
-v /home/users/kzhao27/imagenet_data:/home/vitis-ai-user/imagenet_data \
-v /home/users/kzhao27/imagenet_data_10categories:/home/vitis-ai-user/imagenet_data_10categories \
-v /home/users/kzhao27/tiny-imagenet-200:/home/vitis-ai-user/tiny-imagenet-200 \
-v /home/users/kzhao27/quantization/important/my_new_mqbench:/home/vitis-ai-user/my_new_mqbench \
--gpus all \
quan_mqbench:latest

# Other commands
docker run --rm -it --gpus all quan_mqbench:latest

docker ps
docker attach CONTAINER_ID