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

#include "FDTD3dGPU.h"
#include "FDTD3d.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Note: If you change the RADIUS, you should also change the unrolling below
//#define RADIUS 4

__constant__ float stencil[k_radius_max + 1];

template <int RADIUS>
__global__ void FiniteDifferencesKernel(float *output, const float *input,
                                        const int dimx, const int dimy,
                                        const int dimz) {
  bool validr = true;
  bool validw = true;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int ltidx = threadIdx.x;
  const int ltidy = threadIdx.y;
  const int workx = blockDim.x;
  const int worky = blockDim.y;
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];

  const int stride_y = dimx + 2 * RADIUS;
  const int stride_z = stride_y * (dimy + 2 * RADIUS);

  int inputIndex = 0;
  int outputIndex = 0;

  // Advance inputIndex to start of inner volume
  inputIndex += RADIUS * stride_y + RADIUS;

  // Advance inputIndex to target element
  inputIndex += gtidy * stride_y + gtidx;

  float infront[RADIUS];
  float behind[RADIUS];
  float current;

  const int tx = ltidx + RADIUS;
  const int ty = ltidy + RADIUS;

  // Check in bounds
  if ((gtidx >= dimx + RADIUS) || (gtidy >= dimy + RADIUS)) validr = false;

  if ((gtidx >= dimx) || (gtidy >= dimy)) validw = false;

  // Preload the "infront" and "behind" data
  for (int i = RADIUS - 2; i >= 0; i--) {
    if (validr) behind[i] = input[inputIndex];

    inputIndex += stride_z;
  }

  if (validr) current = input[inputIndex];

  outputIndex = inputIndex;
  inputIndex += stride_z;

  for (int i = 0; i < RADIUS; i++) {
    if (validr) infront[i] = input[inputIndex];

    inputIndex += stride_z;
  }

// Step through the xy-planes
#pragma unroll 9

  for (int iz = 0; iz < dimz; iz++) {
    // Advance the slice (move the thread-front)
    for (int i = RADIUS - 1; i > 0; i--) behind[i] = behind[i - 1];

    behind[0] = current;
    current = infront[0];
#pragma unroll RADIUS

    for (int i = 0; i < RADIUS - 1; i++) infront[i] = infront[i + 1];

    if (validr) infront[RADIUS - 1] = input[inputIndex];

    inputIndex += stride_z;
    outputIndex += stride_z;
    cg::sync(cta);

    // Note that for the work items on the boundary of the problem, the
    // supplied index when reading the halo (below) may wrap to the
    // previous/next row or even the previous/next xy-plane. This is
    // acceptable since a) we disable the output write for these work
    // items and b) there is at least one xy-plane before/after the
    // current plane, so the access will be within bounds.

    // Update the data slice in the local tile
    // Halo above & below
    if (ltidy < RADIUS) {
      tile[ltidy][tx] = input[outputIndex - RADIUS * stride_y];
    }
    if (ltidy >= blockDim.y - RADIUS) {
      tile[ltidy + RADIUS + RADIUS][tx] = input[outputIndex + RADIUS * stride_y];
    }

    // Halo left & right
    if (ltidx < RADIUS) {
      tile[ty][ltidx] = input[outputIndex - RADIUS];
    }
    if (ltidx >= blockDim.x - RADIUS){
      tile[ty][ltidx + RADIUS + RADIUS] = input[outputIndex + RADIUS];
    }
    tile[ty][tx] = current;
    cg::sync(cta);

    // Compute the output value
    float value = stencil[0] * current;
#pragma unroll RADIUS

    for (int i = 1; i <= RADIUS; i++) {
      value +=
          stencil[i] * (infront[i - 1] + behind[i - 1] + tile[ty - i][tx] +
                        tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
    }

    // Store the output value
    if (validw) output[outputIndex] = value;
  }
}

//========================================
// Output caching for cubic stencil mask
//========================================
__constant__ float stencil2[2* k_radius_max + 1][2* k_radius_max + 1][2* k_radius_max + 1];

template <int RADIUS>
__global__ void FiniteDifferencesKernel2(float *output, const float *input,
                                        const int dimx, const int dimy,
                                        const int dimz) {
  bool validr = true;
  bool validw = true;
  const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
  const int ltidx = threadIdx.x;
  const int ltidy = threadIdx.y;
  const int workx = blockDim.x;
  const int worky = blockDim.y;
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];

  const int stride_y = dimx + 2 * RADIUS;
  const int stride_z = stride_y * (dimy + 2 * RADIUS);

  int inputIndex = 0;
  int outputIndex = 0;

  // Advance inputIndex to start of inner square at lowest slice
  inputIndex += RADIUS * stride_y + RADIUS;

  // Advance inputIndex to target element
  inputIndex += gtidy * stride_y + gtidx;

  float out_buf[2*RADIUS + 1];

  const int tx = ltidx + RADIUS;
  const int ty = ltidy + RADIUS;

  outputIndex = inputIndex;
  // outputIndex is -RADIUS layers behind current input layer
  outputIndex -= RADIUS * stride_z;

  // Check in bounds
  if ((gtidx >= dimx) || (gtidy >= dimy)) validr = false;

  if ((gtidx >= dimx) || (gtidy >= dimy)) validw = false;

  // Init output buffer
  for (int i = 0; i < 2*RADIUS+1; i++){
    out_buf[i] = 0;
  }
  


  // if (validr) infront[i] = input[inputIndex];
  // inputIndex += stride_z;
  // // Advance the slice (move the thread-front)
  // for (int i = RADIUS - 1; i > 0; i--) behind[i] = behind[i - 1];

  // behind[0] = current;
  // current = infront[0];

// Step through the xy-planes

  for (int iz = 0; iz < dimz + 2*RADIUS; iz++) {
    // Update the data slice in the local tile
    // Halo above
    if(validr){
      if (ltidy < RADIUS) {
        tile[ltidy][tx] = input[inputIndex - RADIUS * stride_y];
      }
      // Halo below
      if (ltidy >= blockDim.y - RADIUS) {
        tile[ltidy + 2*RADIUS][tx] = input[inputIndex + RADIUS * stride_y];
      }
      // Halo left
      if (ltidx < RADIUS) {
        tile[ty][ltidx] = input[inputIndex - RADIUS];
      }
      // Halo right
      if (ltidx >= blockDim.x - RADIUS){
        tile[ty][ltidx + 2*RADIUS] = input[inputIndex + RADIUS];
      }

      // Corners
      // top left
      if (ltidy < RADIUS && ltidx < RADIUS) {
        tile[ltidy][ltidx] = input[inputIndex - RADIUS * stride_y - RADIUS];
      }
      // top right
      if (ltidy < RADIUS && ltidx >= blockDim.x - RADIUS) {
        tile[ltidy][ltidx + 2*RADIUS] = input[inputIndex - RADIUS * stride_y + RADIUS];
      }
      // bottom left
      if (ltidy >= blockDim.y - RADIUS && ltidx < RADIUS) {
        tile[ltidy + 2*RADIUS][ltidx] = input[inputIndex + RADIUS * stride_y - RADIUS];
      }
      // bottom right
      if (ltidy >= blockDim.y - RADIUS && ltidx >= blockDim.x - RADIUS) {
        tile[ltidy + 2*RADIUS][ltidx + 2*RADIUS] = input[inputIndex + RADIUS * stride_y + RADIUS];
      }

      tile[ty][tx] = input[inputIndex];
    }
    cg::sync(cta);

    // with every layer we want to calculate partial results of max 2*RADIUS + 1
    // other layers
    // 
    // iterate over 3d cube stencil mask
    for (int stencil_z = -RADIUS; stencil_z <= RADIUS; stencil_z++){
      // if out_buffer fill phase
      if (iz-stencil_z >= RADIUS){
        for (int stencil_y = -RADIUS; stencil_y <= RADIUS; stencil_y++){
          for (int stencil_x = -RADIUS; stencil_x <= RADIUS; stencil_x++){
              out_buf[RADIUS+stencil_z] += tile[ty+stencil_y][tx+stencil_y]
                        * stencil2[RADIUS+stencil_z][RADIUS+stencil_y][RADIUS+stencil_x];
            
          }
        }
      }
    }

    // if out_buffer is no longer in fill phase
    if (iz >= 2*RADIUS){
      if (validw) output[outputIndex] = out_buf[2*RADIUS];
    }
    // cycle elements
    for (int i = 2*RADIUS; i > 0; --i){
      out_buf[i] = out_buf[i-1];
    }
    out_buf[0] = 0;
    
    inputIndex += stride_z;
    outputIndex += stride_z;
    
    cg::sync(cta);

  }
}

