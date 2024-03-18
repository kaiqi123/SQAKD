# ======================================================== Set up the environment
1. Creating conda environment
echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc
source .bashrc

# if need remove: conda remove --name mqbench --all

# Note: this is not needed if it has been created
conda create -n mqbench python=3

# Activate the virtual environment
conda activate mqbench


From now on, assume we are in mqbench environment (conda activate mqbench)
2. Install pytorch, support cuda 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

3. Install MQbench
Go to folder "MQBench", run:
python3.9 -m pip install -e .


4. Install Apex
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH

python3.9 -m pip install ninja # makes installation way faster

python3.9 -m pip install  packaging # install apex needs it

git clone https://github.com/NVIDIA/apex
cd apex
python3.9 -m pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Comment this line in setup.py inside folder "apex":
# check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)

5. Install Dali 11.X
python3.9 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110


6. Install tensorboardX and timm
sudo pip3 install tensorboardX
sudo pip3 install timm

7. Activiate conda
source ~/.bashrc
conda activate mqbench

# mkdir results

