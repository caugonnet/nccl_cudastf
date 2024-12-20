#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[])
{

  int size = 32*1024*1024;

  int nDev;
  CUDACHECK(cudaGetDeviceCount(&nDev));

  ::std::vector<int> devs(nDev);
  for (int i = 0; i < nDev; i++) {
      devs.push_back(i);
  }

  context ctx;

  ::std::vector<logical_data<slice<float>>> sendbuff(nDev);
  ::std::vector<logical_data<slice<float>>> recvbuff(nDev);

  // Shape of every buffers
  auto shape = shape_of<slice<float>>(size);

  for (int i = 0; i < nDev; ++i) {
    sendbuff[i] = ctx.logical_data(shape);
    recvbuff[i] = ctx.logical_data(shape);

    ctx.parallel_for(exec_place::device(devs[i]), shape, sendbuff[i].write())->*[]__device__(size_t i, auto buf) {
        buf[i] = 1.0f;
    };

    ctx.parallel_for(exec_place::device(devs[i]), shape, recvbuff[i].write())->*[]__device__(size_t i, auto buf) {
        buf[i] = 0.0f;
    };
  }

  //initializing NCCL
  ::std::vector<ncclComm_t> comms(nDev);
  NCCLCHECK(ncclCommInitAll(comms.data(), nDev, devs.data()));

   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    ctx.task(exec_place::device(devs[i]), sendbuff[i].read(), recvbuff[i].write())->*[&,i](cudaStream_t stream, auto sbuf, auto rbuf) {
        NCCLCHECK(ncclAllReduce((const void*)sbuf.data_handle(), (void*)rbuf.data_handle(), size, ncclFloat, ncclSum, comms[i], stream));
    };
  NCCLCHECK(ncclGroupEnd());

  ctx.finalize();

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
