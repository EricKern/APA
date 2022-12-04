
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <stdio.h>

__device__
bool mine_is_min(float mine, float min, float epsilon=0.001){
  if (fabs(mine-min) < epsilon){
    return true;
  }else{
    return false;
  }
}

__device__
float my_min(float a, float b){
  return a<b ? a : b;
}

__global__
void reduce_intrin1(float *data, size_t len, float* lowest, int* min) {
  int laneId = threadIdx.x & 0x1f;
  float my_val = data[laneId];
  float my_lowest = my_val;
  
  // Use XOR mode to perform butterfly reduction
  for (int i=16; i>=1; i/=2){
    float partner_val = __shfl_xor_sync(0xffffffff, my_lowest, i, 32);
    my_lowest = my_min(my_lowest, partner_val);
  }
  // for integers use __reduce_max_sync()
  
  unsigned min_mask;
  min_mask = __ballot_sync(0xffffffff, mine_is_min(my_val, my_lowest));

  int ones = __popc(min_mask);          // how many ones are set
  int position = __ffs(min_mask) - 1;   // in which position first bit is set
  // return min postion and min value
  if (threadIdx.x == 0){
    *lowest = my_lowest;
    *min = position;
  }
}


int main(int argc, char *argv[]) {

  float *data;
  size_t data_len = 32;
  int *min_pos;
  float *min;
  cudaMallocManaged(&data, sizeof(float)*data_len);
  cudaMallocManaged(&min_pos, sizeof(int));
  cudaMallocManaged(&min, sizeof(float));
  for(int i = 0; i < data_len; ++i){
    if (i < data_len/2){
      data[i]=data_len-i;
    } else{
      data[i]=32;
    }
  }
  // std::iota(data, data+data_len, 0);
  float* min_elem = std::min_element(data, data+data_len);
  int cpu_min_pos = min_elem - data;

  dim3 grid, block;
  grid.x = 1;
  block.x = 32;
  reduce_intrin1<<<grid,block>>>(data, data_len, min, min_pos);
  cudaDeviceSynchronize();

  if (cpu_min_pos != *min_pos){
    throw std::runtime_error("Min positions don't match");
  }
  
  printf("End reached min is %f at index %d\n", *min, *min_pos);
  
  return 0;
}

