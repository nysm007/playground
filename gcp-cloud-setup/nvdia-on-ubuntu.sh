#!/bin/bash

#Add the ppa repo for NVIDIA graphics driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

#Install the recommended driver (currently nvidia-384)
sudo apt-get install nvidia-384

#check if drivers were installed, but sometime it doesn't work. Don't worry here.
nvidia-smi

# Register at https://developer.nvidia.com/cudnn and download cuDNN. 
# For which cudnn-* to use, visit https://www.tensorflow.org/install/install_sources to get more information.

# Use scp the file to your new instance, or any other method
scp -i ~/.ssh/google_compute_engine cudnn-8.0-linux-x64-v5.1.tgz <external-IP-of-GPU-instance>:

# Back on the remote instance, unzip and copy these files:
tar xzvf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/

rm -rf ~/cuda
rm cudnn-8.0-linux-x64-v5.1.tgz
