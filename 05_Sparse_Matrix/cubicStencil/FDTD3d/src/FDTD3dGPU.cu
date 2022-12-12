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

#include <helper_cuda.h>
#include <helper_functions.h>

#include <algorithm>
#include <iostream>

#include "FDTD3d.h"
#include "FDTD3dGPU.h"
#include "FDTD3dGPUKernel.cuh"

#define GPU_PROFILING

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc,
                                  const char **argv) {
  int deviceCount = 0;
  int targetDevice = 0;
  size_t memsize = 0;

  // Get the number of CUDA enabled GPU devices
  printf(" cudaGetDeviceCount\n");
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, (const char **)argv);

  // Query target device for maximum memory allocation
  printf(" cudaGetDeviceProperties\n");
  struct cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

  memsize = deviceProp.totalGlobalMem;

  // Save the result
  *result = (memsize_t)memsize;
  return true;
}

bool fdtdGPU(float *output, const float *input, const float *coeff,
             const int dimx, const int dimy, const int dimz, const int radius,
             const int timesteps, const int argc, const char **argv) {
  const int outerDimx = dimx + 2 * radius;
  const int outerDimy = dimy + 2 * radius;
  const int outerDimz = dimz + 2 * radius;
  const size_t volumeSize = outerDimx * outerDimy * outerDimz;
  int deviceCount = 0;
  int targetDevice = 0;
  float *bufferOut = 0;
  float *bufferIn = 0;
  dim3 dimBlock;
  dim3 dimGrid;

  // Ensure that the inner data starts on a 128B boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING
  cudaEvent_t profileStart = 0;
  cudaEvent_t profileEnd = 0;
  const int profileTimesteps = timesteps - 1;

  if (profileTimesteps < 1) {
    printf(
        " cannot profile with fewer than two timesteps (timesteps=%d), "
        "profiling is disabled.\n",
        timesteps);
  }

#endif

  // Check the radius is valid
  if (k_radius_min > radius && radius > k_radius_max) {
    printf("radius is invalid, must between %d and %d \n", k_radius_min,
           k_radius_max);
    exit(EXIT_FAILURE);
  }

  // Get the number of CUDA enabled GPU devices
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaSetDevice(targetDevice));

  // Allocate memory buffers
  checkCudaErrors(
      cudaMalloc((void **)&bufferOut, paddedVolumeSize * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&bufferIn, paddedVolumeSize * sizeof(float)));

  // Check for a command-line specified block size
  int userBlockSize;

  if (checkCmdLineFlag(argc, (const char **)argv, "block-size")) {
    userBlockSize = getCmdLineArgumentInt(argc, argv, "block-size");
    // Constrain to a multiple of k_blockDimX
    userBlockSize = (userBlockSize / k_blockDimX * k_blockDimX);

    // Constrain within allowed bounds
    userBlockSize = MIN(MAX(userBlockSize, k_blockSizeMin), k_blockSizeMax);
  } else {
    userBlockSize = k_blockSizeMax;
  }

  // Check the device limit on the number of threads
  struct cudaFuncAttributes funcAttrib;
  checkCudaErrors(
      cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<4>));

  userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

  // Set the block size
  dimBlock.x = k_blockDimX;
  // Visual Studio 2005 does not like std::min
  //    dimBlock.y = std::min<size_t>(userBlockSize / k_blockDimX,
  //    (size_t)k_blockDimMaxY);
  dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY)
                   ? (userBlockSize / k_blockDimX)
                   : (size_t)k_blockDimMaxY;
  dimGrid.x = (unsigned int)ceil((float)dimx / dimBlock.x);
  dimGrid.y = (unsigned int)ceil((float)dimy / dimBlock.y);
  printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
  printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

  // Check the block size is valid
  if (dimBlock.x < radius || dimBlock.y < radius) {
    printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n",
           dimBlock.x, dimBlock.y, radius);
    exit(EXIT_FAILURE);
  }

  // Copy the input to the device input buffer
  checkCudaErrors(cudaMemcpy(bufferIn + padding, input,
                             volumeSize * sizeof(float),
                             cudaMemcpyHostToDevice));

  // Copy the input to the device output buffer (actually only need the halo)
  checkCudaErrors(cudaMemcpy(bufferOut + padding, input,
                             volumeSize * sizeof(float),
                             cudaMemcpyHostToDevice));

  // Copy the coefficients to the device coefficient buffer
  bool use_kernel2 = false;
  if (checkCmdLineFlag(argc, (const char **)argv, "kernel2")) {
    use_kernel2 = true;
  }

  if (!use_kernel2) {
    checkCudaErrors(cudaMemcpyToSymbol(stencil, (void *)coeff,
                                       (radius + 1) * sizeof(float)));
  } else {
    int sten_dim = 2 * k_radius_max + 1;
    int sten_dim_len = sten_dim * sten_dim * sten_dim;
    checkCudaErrors(cudaMemcpyToSymbol(stencil2, (void *)coeff,
                                       sten_dim_len * sizeof(float)));
  }

#ifdef GPU_PROFILING

  // Create the events
  checkCudaErrors(cudaEventCreate(&profileStart));
  checkCudaErrors(cudaEventCreate(&profileEnd));

#endif

  // Execute the FDTD
  float *bufferSrc = bufferIn + padding;
  float *bufferDst = bufferOut + padding;
  printf(" GPU FDTD loop\n");

#ifdef GPU_PROFILING
  // Enqueue start event
  checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif
  int shMeme_size = (dimBlock.x + 2 * radius) * (dimBlock.y + 2 * radius);
  shMeme_size *= sizeof(float);
  for (int it = 0; it < timesteps; it++) {
    printf("\tt = %d ", it);

    // Launch the kernel
    if (use_kernel2) {
      printf("launch kernel2\n");
      switch (radius) {
        case 1:
          FiniteDifferencesKernel2<1><<<dimGrid, dimBlock, shMeme_size>>>(
              bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 2:
          FiniteDifferencesKernel2<2><<<dimGrid, dimBlock, shMeme_size>>>(
              bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 3:
          FiniteDifferencesKernel2<3><<<dimGrid, dimBlock, shMeme_size>>>(
              bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 4:
          FiniteDifferencesKernel2<4><<<dimGrid, dimBlock, shMeme_size>>>(
              bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 5:
          FiniteDifferencesKernel2<5><<<dimGrid, dimBlock, shMeme_size>>>(
              bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 6:
          FiniteDifferencesKernel2<6><<<dimGrid, dimBlock, shMeme_size>>>(
              bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 7:
          FiniteDifferencesKernel2<7><<<dimGrid, dimBlock, shMeme_size>>>(
              bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 8:
          FiniteDifferencesKernel2<8><<<dimGrid, dimBlock, shMeme_size>>>(
              bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 9:
          FiniteDifferencesKernel2<9><<<dimGrid, dimBlock, shMeme_size>>>(
              bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 10:
          FiniteDifferencesKernel2<10><<<dimGrid, dimBlock, shMeme_size>>>(
              bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        default:
          break;
      }
    } else {
      printf("launch kernel1\n");
      switch (radius) {
        case 1:
          FiniteDifferencesKernel<1>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 2:
          FiniteDifferencesKernel<2>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 3:
          FiniteDifferencesKernel<3>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 4:
          FiniteDifferencesKernel<4>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 5:
          FiniteDifferencesKernel<5>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 6:
          FiniteDifferencesKernel<6>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 7:
          FiniteDifferencesKernel<7>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 8:
          FiniteDifferencesKernel<8>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 9:
          FiniteDifferencesKernel<9>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        case 10:
          FiniteDifferencesKernel<10>
              <<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
          break;
        default:
          break;
      }
    }

    // Toggle the buffers
    // Visual Studio 2005 does not like std::swap
    //    std::swap<float *>(bufferSrc, bufferDst);
    float *tmp = bufferDst;
    bufferDst = bufferSrc;
    bufferSrc = tmp;
  }

  printf("\n");

#ifdef GPU_PROFILING
  // Enqueue end event
  checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif

  // Wait for the kernel to complete
  checkCudaErrors(cudaDeviceSynchronize());

  // Read the result back, result is in bufferSrc (after final toggle)
  checkCudaErrors(cudaMemcpy(output, bufferSrc, volumeSize * sizeof(float),
                             cudaMemcpyDeviceToHost));

// Report time
#ifdef GPU_PROFILING
  float elapsedTimeMS = 0;

  if (profileTimesteps > 0) {
    checkCudaErrors(
        cudaEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
  }

  if (profileTimesteps > 0) {
    // Convert milliseconds to seconds
    double elapsedTime = elapsedTimeMS * 1.0e-3;
    double avgElapsedTime = elapsedTime / (double)profileTimesteps;
    // Determine number of computations per timestep
    size_t pointsComputed = dimx * dimy * dimz;
    // Determine throughput
    double throughputM = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
    printf(
        "FDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %lu "
        "Points, "
        "NumDevsUsed = %u, Blocksize = %u\n",
        throughputM, avgElapsedTime, pointsComputed, 1,
        dimBlock.x * dimBlock.y);
  }

#endif

  // Cleanup
  if (bufferIn) {
    checkCudaErrors(cudaFree(bufferIn));
  }

  if (bufferOut) {
    checkCudaErrors(cudaFree(bufferOut));
  }

#ifdef GPU_PROFILING

  if (profileStart) {
    checkCudaErrors(cudaEventDestroy(profileStart));
  }

  if (profileEnd) {
    checkCudaErrors(cudaEventDestroy(profileEnd));
  }

#endif
  return true;
}

bool fdtdGPU_BandedMat(float *output, const float *input,
             const float **coeff_buffers,
             const int dimx, const int dimy, const int dimz, const int radius,
             const int timesteps, const int argc, const char **argv) {
  const int outerDimx = dimx + 2 * radius;
  const int outerDimy = dimy + 2 * radius;
  const int outerDimz = dimz + 2 * radius;
  const size_t volumeSize = outerDimx * outerDimy * outerDimz;
  int deviceCount = 0;
  int targetDevice = 0;
  float *bufferOut = 0;
  float *bufferIn = 0;
  dim3 dimBlock;
  dim3 dimGrid;

  // Ensure that the inner data starts on a 128B boundary
  const int padding = (128 / sizeof(float)) - radius;
  const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING
  cudaEvent_t profileStart = 0;
  cudaEvent_t profileEnd = 0;
  const int profileTimesteps = timesteps - 1;

  if (profileTimesteps < 1) {
    printf(
        " cannot profile with fewer than two timesteps (timesteps=%d), "
        "profiling is disabled.\n",
        timesteps);
  }

#endif

  // Check the radius is valid
  if (k_radius_min > radius && radius > k_radius_max) {
    printf("radius is invalid, must between %d and %d \n", k_radius_min,
           k_radius_max);
    exit(EXIT_FAILURE);
  }

  // Get the number of CUDA enabled GPU devices
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaSetDevice(targetDevice));

  // Allocate memory buffers
  checkCudaErrors(
      cudaMalloc((void **)&bufferOut, paddedVolumeSize * sizeof(float)));
  checkCudaErrors(
      cudaMalloc((void **)&bufferIn, paddedVolumeSize * sizeof(float)));

  // Check for a command-line specified block size
  int userBlockSize;

  if (checkCmdLineFlag(argc, (const char **)argv, "block-size")) {
    userBlockSize = getCmdLineArgumentInt(argc, argv, "block-size");
    // Constrain to a multiple of k_blockDimX
    userBlockSize = (userBlockSize / k_blockDimX * k_blockDimX);

    // Constrain within allowed bounds
    userBlockSize = MIN(MAX(userBlockSize, k_blockSizeMin), k_blockSizeMax);
  } else {
    userBlockSize = k_blockSizeMax;
  }

  // Check the device limit on the number of threads
  struct cudaFuncAttributes funcAttrib;
  checkCudaErrors(
      cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel<4>));

  userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

  // Set the block size
  dimBlock.x = k_blockDimX;
  // Visual Studio 2005 does not like std::min
  //    dimBlock.y = std::min<size_t>(userBlockSize / k_blockDimX,
  //    (size_t)k_blockDimMaxY);
  dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY)
                   ? (userBlockSize / k_blockDimX)
                   : (size_t)k_blockDimMaxY;
  dimGrid.x = (unsigned int)ceil((float)dimx / dimBlock.x);
  dimGrid.y = (unsigned int)ceil((float)dimy / dimBlock.y);
  printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
  printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

  // Check the block size is valid
  if (dimBlock.x < radius || dimBlock.y < radius) {
    printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n",
           dimBlock.x, dimBlock.y, radius);
    exit(EXIT_FAILURE);
  }

  // Allocate coeff buffers on GPU
  int num_coeff = 2*radius + 1;
  num_coeff = pow(num_coeff, 3);
  float **d_buffers = 0;
  checkCudaErrors(
      cudaMalloc((void **)&d_buffers, num_coeff * sizeof(float*)));

  // temporary store the device addresses of the buffers
  float **hd_buffers;
  hd_buffers = (float**)calloc(num_coeff, sizeof(float*));

  size_t innerVolumeSize = dimx * dimy * dimz;
  for (int i = 0; i < num_coeff; ++i) {
      checkCudaErrors(
        cudaMalloc((void **)&(hd_buffers[i]), innerVolumeSize * sizeof(float)));
  }
  
  // Copy the coeff buffer pointers to the device
  checkCudaErrors(cudaMemcpy(d_buffers, hd_buffers,
                             num_coeff * sizeof(float*),
                             cudaMemcpyHostToDevice));

  for (int i = 0; i < num_coeff; ++i) {
    // Copy the actual coeff buffers to the device
    checkCudaErrors(cudaMemcpy(hd_buffers[i], coeff_buffers[i],
                              innerVolumeSize * sizeof(float),
                              cudaMemcpyHostToDevice));
  }

  // Copy the input to the device input buffer
  checkCudaErrors(cudaMemcpy(bufferIn + padding, input,
                             volumeSize * sizeof(float),
                             cudaMemcpyHostToDevice));

  // Copy the input to the device output buffer (actually only need the halo)
  checkCudaErrors(cudaMemcpy(bufferOut + padding, input,
                             volumeSize * sizeof(float),
                             cudaMemcpyHostToDevice));


#ifdef GPU_PROFILING

  // Create the events
  checkCudaErrors(cudaEventCreate(&profileStart));
  checkCudaErrors(cudaEventCreate(&profileEnd));

#endif

  // Execute the FDTD
  float *bufferSrc = bufferIn + padding;
  float *bufferDst = bufferOut + padding;
  printf(" GPU FDTD loop\n");

#ifdef GPU_PROFILING
  // Enqueue start event
  checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif
  int shMeme_size = (dimBlock.x + 2 * radius) * (dimBlock.y + 2 * radius);
  shMeme_size *= sizeof(float);
  for (int it = 0; it < timesteps; it++) {
    printf("\tt = %d ", it);

    // Launch the kernel
    printf("launch kernel2\n");
    switch (radius) {
      case 1:
        FiniteDifferencesKernel3<1><<<dimGrid, dimBlock, shMeme_size>>>(
            bufferDst, bufferSrc, dimx, dimy, dimz, d_buffers);
        break;
      case 2:
        FiniteDifferencesKernel3<2><<<dimGrid, dimBlock, shMeme_size>>>(
            bufferDst, bufferSrc, dimx, dimy, dimz, d_buffers);
        break;
      case 3:
        FiniteDifferencesKernel3<3><<<dimGrid, dimBlock, shMeme_size>>>(
            bufferDst, bufferSrc, dimx, dimy, dimz, d_buffers);
        break;
      case 4:
        FiniteDifferencesKernel3<4><<<dimGrid, dimBlock, shMeme_size>>>(
            bufferDst, bufferSrc, dimx, dimy, dimz, d_buffers);
        break;
      case 5:
        FiniteDifferencesKernel3<5><<<dimGrid, dimBlock, shMeme_size>>>(
            bufferDst, bufferSrc, dimx, dimy, dimz, d_buffers);
        break;
      case 6:
        FiniteDifferencesKernel3<6><<<dimGrid, dimBlock, shMeme_size>>>(
            bufferDst, bufferSrc, dimx, dimy, dimz, d_buffers);
        break;
      case 7:
        FiniteDifferencesKernel3<7><<<dimGrid, dimBlock, shMeme_size>>>(
            bufferDst, bufferSrc, dimx, dimy, dimz, d_buffers);
        break;
      case 8:
        FiniteDifferencesKernel3<8><<<dimGrid, dimBlock, shMeme_size>>>(
            bufferDst, bufferSrc, dimx, dimy, dimz, d_buffers);
        break;
      case 9:
        FiniteDifferencesKernel3<9><<<dimGrid, dimBlock, shMeme_size>>>(
            bufferDst, bufferSrc, dimx, dimy, dimz, d_buffers);
        break;
      case 10:
        FiniteDifferencesKernel3<10><<<dimGrid, dimBlock, shMeme_size>>>(
            bufferDst, bufferSrc, dimx, dimy, dimz, d_buffers);
        break;
      default:
        break;
    }

    // Toggle the buffers
    // Visual Studio 2005 does not like std::swap
    //    std::swap<float *>(bufferSrc, bufferDst);
    float *tmp = bufferDst;
    bufferDst = bufferSrc;
    bufferSrc = tmp;
  }

  printf("\n");

#ifdef GPU_PROFILING
  // Enqueue end event
  checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif

  // Wait for the kernel to complete
  checkCudaErrors(cudaDeviceSynchronize());

  // Read the result back, result is in bufferSrc (after final toggle)
  checkCudaErrors(cudaMemcpy(output, bufferSrc, volumeSize * sizeof(float),
                             cudaMemcpyDeviceToHost));

// Report time
#ifdef GPU_PROFILING
  float elapsedTimeMS = 0;

  if (profileTimesteps > 0) {
    checkCudaErrors(
        cudaEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
  }

  if (profileTimesteps > 0) {
    // Convert milliseconds to seconds
    double elapsedTime = elapsedTimeMS * 1.0e-3;
    double avgElapsedTime = elapsedTime / (double)profileTimesteps;
    // Determine number of computations per timestep
    size_t pointsComputed = dimx * dimy * dimz;
    // Determine throughput
    double throughputM = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
    printf(
        "FDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %lu "
        "Points, "
        "NumDevsUsed = %u, Blocksize = %u\n",
        throughputM, avgElapsedTime, pointsComputed, 1,
        dimBlock.x * dimBlock.y);
  }

#endif

  // Cleanup
  if (bufferIn) {
    checkCudaErrors(cudaFree(bufferIn));
  }

  if (bufferOut) {
    checkCudaErrors(cudaFree(bufferOut));
  }

  if (d_buffers){
    for (int i = 0; i < num_coeff; ++i) {
      if(hd_buffers[i]){
        checkCudaErrors(cudaFree(hd_buffers[i]));
      }
    }
    checkCudaErrors(cudaFree(d_buffers));
  }


#ifdef GPU_PROFILING

  if (profileStart) {
    checkCudaErrors(cudaEventDestroy(profileStart));
  }

  if (profileEnd) {
    checkCudaErrors(cudaEventDestroy(profileEnd));
  }

#endif
  return true;
}