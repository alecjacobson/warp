// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(call) do { cudaError_t e = (call); if(e != cudaSuccess) { std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::exit(1); } } while(0)
__global__ void write_values(int* p) { int t=threadIdx.x; if(t<129) p[t]=t; }
int main(int argc, char**) {
 const bool defer = argc > 1;
 int* observed; CHECK(cudaMallocHost(&observed,2*129*sizeof(int)));
 cudaStream_t s; CHECK(cudaStreamCreate(&s));
 cudaGraph_t graph; CHECK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
 void* allocations[2][15]{};
 const size_t sizes[]={4112,4112,1028,1028,4,516,12,12,12,1032,1032,1028,1028,1028};
 for(int b=0;b<2;++b){
  for(int i=0;i<14;++i) CHECK(cudaMallocAsync(&allocations[b][i],sizes[i],s));
  write_values<<<1,256,0,s>>>((int*)allocations[b][11]); CHECK(cudaGetLastError());
  CHECK(cudaMemcpyAsync(observed+b*129,allocations[b][11],129*sizeof(int),cudaMemcpyDeviceToHost,s));
  if(!defer){
   for(int i=9;i<14;++i) CHECK(cudaFreeAsync(allocations[b][i],s));
   for(int i=6;i<9;++i) CHECK(cudaFreeAsync(allocations[b][i],s));
  }
  CHECK(cudaMallocAsync(&allocations[b][14],112,s));
 }
 if(defer) for(int b=0;b<2;++b) for(int i=6;i<14;++i) CHECK(cudaFreeAsync(allocations[b][i],s));
 CHECK(cudaStreamEndCapture(s,&graph));
 cudaGraphExec_t exec; CHECK(cudaGraphInstantiate(&exec,graph,0));
 CHECK(cudaGraphLaunch(exec,s)); CHECK(cudaStreamSynchronize(s));
 for(int b=0;b<2;++b) for(int i=0;i<129;++i) if(observed[b*129+i]!=i) {std::fprintf(stderr,"wrong result\n");return 2;}
 CHECK(cudaFreeHost(observed));
 CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(graph));
 for(int b=0;b<2;++b){for(int i=0;i<6;++i) CHECK(cudaFree(allocations[b][i]));CHECK(cudaFree(allocations[b][14]));}
 CHECK(cudaStreamDestroy(s));std::puts("completed");
}
