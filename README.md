# CUDASTF / NCCL example

This code is based on NCCL documentation https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html# and illustrates how to combine NCCL and CUDASTF.

To compile, please adjust the location of the NCCL directory, and do the following :

```bash
mkdir build;
cd build;

cmake .. -DNCCL_INCLUDE_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/comm_libs/12.6/nccl/include/ -DNCCL_LIBRARY=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/REDIST/comm_libs/12.6/nccl/lib/libnccl.so

make

./example1
```

