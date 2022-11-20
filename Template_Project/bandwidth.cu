#include <cuda_runtime.h>

#include <nvbench/nvbench.cuh>
#include "helper.hpp"



template <typename DType>
void memcopyd2d(nvbench::state &state, nvbench::type_list<DType>) {
  DType *d_idata = NULL;
  DType *d_odata = NULL;

  size_t size = 100 * 1024 * 1024;
  size_t elements = size / sizeof(DType);

  checkCudaErrors(cudaMalloc((void **)&d_idata, size));
  checkCudaErrors(cudaMalloc((void **)&d_odata, size));

  state.add_element_count(elements, "NumElements");
  state.add_global_memory_reads<DType>(elements, "DataSize");
  state.add_global_memory_writes<DType>(elements);

  state.exec([&d_idata, &d_odata, &size](nvbench::launch &launch) {
    checkCudaErrors(cudaMemcpyAsync(d_odata, d_idata, size,
                                    cudaMemcpyDeviceToDevice,
                                    launch.get_stream()));
  });

  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));
}


template <typename T, typename U>
__global__ void copy_kernel(const T *in, U *out, std::size_t n) {
  const auto init = blockIdx.x * blockDim.x + threadIdx.x;
  const auto step = blockDim.x * gridDim.x;

  for (auto i = init; i < n; i += step) {
    out[i] = static_cast<U>(in[i]);
  }
}

template <typename DType>
void copyd2d(nvbench::state &state, nvbench::type_list<DType>) {
  DType *d_idata = NULL;
  DType *d_odata = NULL;

  size_t size = 100 * 1024 * 1024;
  size_t elements = size / sizeof(DType);

  checkCudaErrors(cudaMalloc((void **)&d_idata, size));
  checkCudaErrors(cudaMalloc((void **)&d_odata, size));

  // cudaHostGetDevicePointer((void **)&d_a, (void *)h_data, 0);
  const auto elem_per_thread = state.get_int64("elem_per_thread");

  size_t total_threads = (elements + elem_per_thread) / elem_per_thread;
  size_t threads_per_block = 1024;
  size_t blocks = (total_threads + threads_per_block) / threads_per_block;

  state.add_element_count(elements, "NumElements");
  state.add_global_memory_reads<DType>(elements, "DataSize");
  state.add_global_memory_writes<DType>(elements);

  state.exec([&blocks, &threads_per_block, &d_idata, &d_odata,
              &elements](nvbench::launch &launch) {
    copy_kernel<<<blocks, threads_per_block, 0, launch.get_stream()>>>(
        d_idata, d_odata, elements);
  });
  checkCudaErrors(cudaFree(d_idata));
  checkCudaErrors(cudaFree(d_odata));
}

using my_types = nvbench::type_list<int, int2, int4>;

NVBENCH_BENCH_TYPES(memcopyd2d, NVBENCH_TYPE_AXES(my_types))
    .set_type_axes_names({"T"})
    .add_int64_power_of_two_axis("elem_per_thread", nvbench::range(0, 3, 1))
    .set_timeout(1);

NVBENCH_BENCH_TYPES(copyd2d, NVBENCH_TYPE_AXES(my_types))
    .set_type_axes_names({"T"})
    .add_int64_power_of_two_axis("elem_per_thread", nvbench::range(0, 3, 1))
    .set_timeout(1);
