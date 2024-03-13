
# ############################################ We run the code on the following envirenment ########################################

# # Ubuntu Version
# lsb_release -a
# Distributor ID: Ubuntu
# Description:    Ubuntu 20.04.1 LTS
# Release:        20.04
# Codename:       focal


# # CUDA Version
# nvidia-smi
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 460.80       Driver Version: 460.80       CUDA Version: 11.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |                               |                      |               MIG M. |
# |===============================+======================+======================|
# |   0  GeForce RTX 208...  Off  | 00000000:3B:00.0 Off |                  N/A |
# | 22%   28C    P8     1W / 250W |      5MiB / 11019MiB |      0%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
# |   1  GeForce RTX 208...  Off  | 00000000:5E:00.0 Off |                  N/A |
# | 22%   29C    P8     4W / 250W |      5MiB / 11019MiB |      0%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
# |   2  GeForce RTX 208...  Off  | 00000000:AF:00.0 Off |                  N/A |
# | 22%   30C    P8    12W / 250W |      5MiB / 11019MiB |      0%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+
# |   3  GeForce RTX 208...  Off  | 00000000:D8:00.0 Off |                  N/A |
# | 22%   30C    P8    13W / 250W |      5MiB / 11019MiB |      0%      Default |
# |                               |                      |                  N/A |
# +-------------------------------+----------------------+----------------------+



# Python 3.8.5


# Pip Version
# pip3 --version
# pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.8)
# ##################################################################################################################################

# Install pytorch, tensorboard, setuptools
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install google-auth
pip3 install tensorboard or python3 -m pip install tensorboard
pip3 install setuptools==59.5.0 or pip3 install setuptools