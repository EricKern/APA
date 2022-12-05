/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
* This sample implements 64-bin histogram calculation
* of arbitrary-sized 8-bit data array
*/

// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

// project include
#include "histogram_common.h"

bool isPowerof2(unsigned int x) { return ((x & (x - 1)) == 0); }
void histogramBinNumCPU(uint *h_Histogram, uint *h_Data,
                                   uint byteCount, uint binNum) {
  for (uint i = 0; i < binNum; ++i) h_Histogram[i] = 0;
                                                      // from samples helpers
  assert(sizeof(uint) == 4 && (byteCount % 4) == 0 && isPowerof2(binNum) == true);

  uint mask = binNum-1;
  for (uint i = 0; i < (byteCount / 4); i++) {
    uint data = h_Data[i] & mask;
    h_Histogram[data]++;
  }
}

const int numRuns = 16;
const static char *sSDKsample = "[histogram]\0";

int main(int argc, char **argv) {
  uchar *h_Data;
  uint *h_Data_int;
  uint *h_HistogramCPU, *h_HistogramGPU, *h_HistoBinNumCPU, *h_HistoBinNumGPU;
  uchar *d_Data;
  uint *d_Histogram, *d_HistoBinNum;
  StopWatchInterface *hTimer = NULL;
  int PassFailFlag = 1;
  uint byteCount = 64 * 1048576;
  uint uiSizeMult = 1;

  cudaDeviceProp deviceProp;
  deviceProp.major = 0;
  deviceProp.minor = 0;

  // set logfile name and start logs
  printf("[%s] - Starting...\n", sSDKsample);

  // Use command-line specified CUDA device, otherwise use device with highest
  // Gflops/s
  int dev = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

  int smem_bytes;
  cudaDeviceGetAttribute(&smem_bytes, cudaDevAttrMaxSharedMemoryPerBlock, dev);
  printf("Max Shared Mem per Block: %d MB\n", smem_bytes/1024);

  printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
         deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major,
         deviceProp.minor);

  // new CL parameter
  bool useMyHist = false;
  int Wc, binNum;
  if (checkCmdLineFlag(argc, (const char **)argv, "Wc")) {
    Wc = getCmdLineArgumentInt(argc, (const char **)argv, "Wc");
    printf("\nMax colliding warps: %d\n", Wc);
  }
  if (checkCmdLineFlag(argc, (const char **)argv, "binNum")) {
    useMyHist = true;
    binNum = getCmdLineArgumentInt(argc, (const char **)argv, "binNum");
    printf("Nr bins: %d\n\n", binNum);
  }

  sdkCreateTimer(&hTimer);

  // Optional Command-line multiplier to increase size of array to histogram
  if (checkCmdLineFlag(argc, (const char **)argv, "sizemult")) {
    uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    uiSizeMult = MAX(1, MIN(uiSizeMult, 10));
    byteCount *= uiSizeMult;
  }

  printf("Initializing data...\n");
  printf("...allocating CPU memory.\n");
  h_Data = (uchar *)malloc(byteCount);
  h_Data_int = (uint *)malloc(byteCount);
  h_HistoBinNumCPU = (uint *)malloc(binNum * sizeof(uint));
  h_HistoBinNumGPU = (uint *)malloc(binNum * sizeof(uint));
  h_HistogramCPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
  h_HistogramGPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

  printf("...generating input data\n");
  srand(2009);

  for (uint i = 0; i < byteCount; ++i) {
    h_Data[i] = rand() % 256;
  }
  for (uint i = 0; i < byteCount/sizeof(uint); ++i) {
    h_Data_int[i] = rand() % 8192;
  }

  printf("...allocating GPU memory and copying input data\n\n");
  checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount));
  if(useMyHist){
    checkCudaErrors(
        cudaMalloc((void **)&d_HistoBinNum, binNum * sizeof(uint)));
    checkCudaErrors(
        cudaMemcpy(d_Data, h_Data_int, byteCount, cudaMemcpyHostToDevice));

  } else{
    checkCudaErrors(
      cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    checkCudaErrors(
      cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));
  }

  if(useMyHist){
    printf("Starting up binNum histogram...\n\n");
    initHistogramBinNum(binNum);

    printf("Running %d-bin GPU histogram for %u bytes (%u runs)...\n\n",
            binNum, byteCount, numRuns);

    for (int iter = -1; iter < numRuns; iter++) {
      // iter == -1 -- warmup iteration
      if (iter == 0) {
        cudaDeviceSynchronize();
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);
      }

      histogramBinNum(d_HistoBinNum, d_Data, byteCount, binNum, Wc);
    }
    cudaDeviceSynchronize();
    sdkStopTimer(&hTimer);
    double dAvgSecs =
        1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
    printf("histogramBinNum() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs,
          ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf(
        "histogramBinNum, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, "
        "NumDevsUsed = %u, Workgroup = %u\n",
        (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1,
        HISTOGRAM64_THREADBLOCK_SIZE);

    printf("\nValidating GPU results...\n");
    printf(" ...reading back GPU results\n");
    checkCudaErrors(cudaMemcpy(h_HistoBinNumGPU, d_HistoBinNum,
                              binNum * sizeof(uint),
                              cudaMemcpyDeviceToHost));
    printf(" ...histogramBinNumCPU()\n");
    histogramBinNumCPU(h_HistoBinNumCPU, h_Data_int, byteCount, binNum);

    printf(" ...comparing the results...\n");
    for (uint i = 0; i < binNum; i++)
      if (h_HistoBinNumGPU[i] != h_HistoBinNumCPU[i]) {
        PassFailFlag = 0;
      }

    printf(PassFailFlag ? " ...%d-bin histograms match\n\n"
                        : " ***%d-bin histograms do not match!!!***\n\n", binNum, binNum);

    printf("Shutting down %d-bin histogram...\n\n\n", binNum);
    closeHistogramBinNum();


  } else{

    {
      printf("Starting up 64-bin histogram...\n\n");
      initHistogram64();

      printf("Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n",
            byteCount, numRuns);

      for (int iter = -1; iter < numRuns; iter++) {
        // iter == -1 -- warmup iteration
        if (iter == 0) {
          cudaDeviceSynchronize();
          sdkResetTimer(&hTimer);
          sdkStartTimer(&hTimer);
        }

        histogram64(d_Histogram, d_Data, byteCount);
      }

      cudaDeviceSynchronize();
      sdkStopTimer(&hTimer);
      double dAvgSecs =
          1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
      printf("histogram64() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs,
            ((double)byteCount * 1.0e-6) / dAvgSecs);
      printf(
          "histogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, "
          "NumDevsUsed = %u, Workgroup = %u\n",
          (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1,
          HISTOGRAM64_THREADBLOCK_SIZE);

      printf("\nValidating GPU results...\n");
      printf(" ...reading back GPU results\n");
      checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram,
                                HISTOGRAM64_BIN_COUNT * sizeof(uint),
                                cudaMemcpyDeviceToHost));

      printf(" ...histogram64CPU()\n");
      histogram64CPU(h_HistogramCPU, h_Data, byteCount);

      printf(" ...comparing the results...\n");

      for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_HistogramCPU[i]) {
          PassFailFlag = 0;
        }

      printf(PassFailFlag ? " ...64-bin histograms match\n\n"
                          : " ***64-bin histograms do not match!!!***\n\n");

      printf("Shutting down 64-bin histogram...\n\n\n");
      closeHistogram64();
    }

    {
      printf("Initializing 256-bin histogram...\n");
      initHistogram256();

      printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n",
            byteCount, numRuns);

      for (int iter = -1; iter < numRuns; iter++) {
        // iter == -1 -- warmup iteration
        if (iter == 0) {
          checkCudaErrors(cudaDeviceSynchronize());
          sdkResetTimer(&hTimer);
          sdkStartTimer(&hTimer);
        }

        histogram256(d_Histogram, d_Data, byteCount);
      }

      cudaDeviceSynchronize();
      sdkStopTimer(&hTimer);
      double dAvgSecs =
          1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
      printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n",
            dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
      printf(
          "histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, "
          "NumDevsUsed = %u, Workgroup = %u\n",
          (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1,
          HISTOGRAM256_THREADBLOCK_SIZE);

      printf("\nValidating GPU results...\n");
      printf(" ...reading back GPU results\n");
      checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram,
                                HISTOGRAM256_BIN_COUNT * sizeof(uint),
                                cudaMemcpyDeviceToHost));

      printf(" ...histogram256CPU()\n");
      histogram256CPU(h_HistogramCPU, h_Data, byteCount);

      printf(" ...comparing the results\n");

      for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_HistogramCPU[i]) {
          PassFailFlag = 0;
        }

      printf(PassFailFlag ? " ...256-bin histograms match\n\n"
                          : " ***256-bin histograms do not match!!!***\n\n");

      printf("Shutting down 256-bin histogram...\n\n\n");
      closeHistogram256();
    }
  }

  printf("Shutting down...\n");
  sdkDeleteTimer(&hTimer);

  if (useMyHist){
    checkCudaErrors(cudaFree(d_HistoBinNum));
  } else{
    checkCudaErrors(cudaFree(d_Histogram));
  }
  checkCudaErrors(cudaFree(d_Data));
  free(h_HistogramGPU);
  free(h_HistogramCPU);
  free(h_Data);
  free(h_HistoBinNumGPU);
  free(h_HistoBinNumCPU);
  free(h_Data_int);

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  printf("%s - Test Summary\n", sSDKsample);

  // pass or fail (for both 64 bit and 256 bit histograms)
  if (!PassFailFlag) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
