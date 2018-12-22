This repo holds the codes for the paper "Learning Evasion Strategy in Pursuit-Evasion by Deep Q-Network, ICPR 2018".

To replicate the experiment results, a number of dependencies need to be
installed, namely:
* LuaJIT and Torch 7.0
* nngraph
* Xitari (fork of the Arcade Learning Environment (Bellemare et al., 2013))
* AleWrap (a lua interface to Xitari)
An install script for these dependencies is provided.

Two run scripts are provided: run_cpu and run_gpu. As the names imply,
the former trains the DQN network using regular CPUs, while the latter uses
GPUs (CUDA), which typically results in a significant speed-up.

Installation instructions
-------------------------

The installation requires Linux with apt-get.

Note: In order to run the GPU version of DQN, you should additionally have the
NVIDIA® CUDA® (version 5.5 or later) toolkit installed prior to the Torch
installation below.
This can be downloaded from https://developer.nvidia.com/cuda-toolkit
and installation instructions can be found in
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux


To train the DQN, the following components must be installed:
* LuaJIT and Torch 7.0
* nngraph
* Xitari
* AleWrap

To install all of the above in a subdirectory called 'torch', it should be enough to run

    ./install_dependencies.sh

from the base directory of the package.


Note: The above install script will install the following packages via apt-get:
build-essential, gcc, g++, cmake, curl, libreadline-dev, git-core, libjpeg-dev,
libpng-dev, ncurses-dev, imagemagick, unzip

Training the DQN
---------------------------

    ./run_gpu
    
    
    
    
## Citation
Please cite the following paper if you feel this repository useful.
```
@inproceedings{PURSUITEVASION2018ICPR,
  author    = {Jiagang Zhu and
               Wei Zou and
               Zheng Zhu},
  title     = {Learning Evasion Strategy in Pursuit-Evasion by Deep Q-Network},
  booktitle   = {ICPR},
  year      = {2018},
}

```


## Contact
For any question, please contact
```
Jiagang Zhu: zhujiagang2015@ia.ac.cn
```
