Our Setup
=========

The following machines in the WSI network are CUDA capable and can 
be used to solve the exercises:
- cgpool120[0-7]: 1x Geforce GTX 1080

To access the WSI network, from outside, use
$ ssh [yourwsilogin]@cgcontact.informatik.uni-tuebingen.de
From there, connect to one of the machines via
$ ssh [hostname]
Do not execute any computations on cgcontact directly!

The toolkit 9.0 is installed in /graphics/opt/opt_Ubuntu16.04/cuda/

To setup your environment and be able to use CUDA you need to execute

# CUDA
export OPT_PATH="/graphics/opt/opt_Ubuntu16.04"
export CUDA_INSTALL_PATH=${OPT_PATH}/cuda/toolkit_9.0/cuda
export CUDA_PATH=$CUDA_INSTALL_PATH
export CUDA_LIB_PATH=${CUDA_INSTALL_PATH}/lib64
export CUDA_INC_PATH=${CUDA_INSTALL_PATH}/include
export LD_LIBRARY_PATH=${CUDA_LIB_PATH}:$LD_LIBRARY_PATH
export PATH=${CUDA_INSTALL_PATH}/bin:$PATH
export PATH=${CUDA_INSTALL_PATH}/nvvm/bin/:$PATH

# CUPTI
export CUPIT_LIB_PATH=${OPT_PATH}/cuda/toolkit_9.0/cuda/extras/CUPTI/lib64
export LD_LIBRARY_PATH=${CUPIT_LIB_PATH}:$LD_LIBRARY_PATH

each time you login.
You can also write these lines into a file 
called ~/.bashrc directly in your homedirectory. This executes them
automatically each time you log in. 

To make login shells (e.g. shells which you opened via ssh) to behave 
like local terminals, add the following lines to your ~/.bash_profile file:

if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi



A common linux CUDA setup
=========================

For running CUDA code on your own machines, some paths will most likely 
be different because the toolkit will be installed in /usr/cuda.

Therefore, to be able to use cuda, you need to change the lines to match your
installation.
