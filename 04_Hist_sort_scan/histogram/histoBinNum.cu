#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include "histogram_common.h"


__global__ void histogramBinNumKernel(uint *d_PartialHistograms, uint *d_Data,
                                   uint dataCount, uint binNum, uint Wc) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Per-warp subhistogram storage
  uint warps_per_block = cta.size()/32;
  uint s_mem_elems = (warps_per_block/Wc) * binNum;
  extern __shared__ uint s_Hist[];

  cg::thread_group tile = cg::tiled_partition(cg::this_thread_block(), 32*Wc);
  uint tile_id = threadIdx.x / tile.size();
  uint *s_WarpHist = s_Hist + tile_id * binNum;

// Clear shared memory storage for current threadblock before processing
#pragma unroll

  for (uint i = 0; i < (s_mem_elems / cta.size()); i++) {
    s_Hist[threadIdx.x + i * cta.size()] = 0;
  }

  cg::sync(cta);

  uint mask = binNum-1;
  // grid stride kernel
  for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x)) {
    uint data = d_Data[pos] & mask;
    atomicAdd(s_WarpHist + data, 1);
  }

  // Merge per-warp histograms into per-block and write to global memory
  cg::sync(cta);

  uint numHistograms = cta.size()/tile.size();

  for (uint bin = threadIdx.x; bin < binNum; bin += cta.size()) {
    uint sum = 0;

    for (uint i = 0; i < numHistograms; i++) {
      sum += s_Hist[bin + i * binNum];
    }

    d_PartialHistograms[blockIdx.x * binNum + bin] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram256() output
// Run one threadblock per bin; each threadblock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram256
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergeHistogramBinNumKernel(uint *d_Histogram,
                                        uint *d_PartialHistograms,
                                        uint histogramCount, uint binNum) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  uint sum = 0;

  for (uint i = threadIdx.x; i < histogramCount; i += cta.size()) {
    sum += d_PartialHistograms[blockIdx.x + i * binNum];
  }
  
  // cg::this_thread_block().size() elements of shared memory
  extern __shared__ uint data[];
  data[threadIdx.x] = sum;

  for (uint stride = cta.size() / 2; stride > 0; stride >>= 1) {
    cg::sync(cta);

    if (threadIdx.x < stride) {
      data[threadIdx.x] += data[threadIdx.x + stride];
    }
  }

  if (threadIdx.x == 0) {
    d_Histogram[blockIdx.x] = data[0];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////
// histogram256kernel() intermediate results buffer
static const uint PARTIAL_HISTOGRAM256_COUNT = 1024;
static const uint BLOCK_SIZE = 256;

static uint *d_PartialHistograms;

// Internal memory allocation
extern "C" void initHistogramBinNum(uint binNum) {
  checkCudaErrors(cudaMalloc(
      (void **)&d_PartialHistograms,
      PARTIAL_HISTOGRAM256_COUNT * binNum * sizeof(uint)));
}

// Internal memory deallocation
extern "C" void closeHistogramBinNum(void) {
  checkCudaErrors(cudaFree(d_PartialHistograms));
}

extern "C" void histogramBinNum(uint *d_Histogram, void *d_Data, uint byteCount, uint binNum, uint Wc) {
  assert(byteCount % sizeof(uint) == 0);
  uint warps_per_block = BLOCK_SIZE/32;
  uint s_mem_bytes = (warps_per_block/Wc) * binNum * sizeof(uint);
  printf("Shared mem usage %d\n", s_mem_bytes);
  printf("PARTIAL_HISTOGRAM256_COUNT %d\n", PARTIAL_HISTOGRAM256_COUNT);
  printf("Blocksize %d\n", BLOCK_SIZE);
  getLastCudaError("BEFORE histogramBinNumKernel() execution failed\n");
  histogramBinNumKernel<<<PARTIAL_HISTOGRAM256_COUNT, BLOCK_SIZE, s_mem_bytes>>>(
      d_PartialHistograms, (uint *)d_Data, byteCount / sizeof(uint), binNum, Wc);
  getLastCudaError("histogramBinNumKernel() execution failed\n");

  mergeHistogramBinNumKernel<<<binNum, BLOCK_SIZE, BLOCK_SIZE*sizeof(uint)>>>(
      d_Histogram, d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT, binNum);
  getLastCudaError("mergeHistogramBinNumKernel() execution failed\n");
}