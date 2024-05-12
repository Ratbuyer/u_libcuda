#include <cuda_fp16.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

#include "../headers/device/descriptor.cuh"
#include "../headers/host/kernel.cuh"

const int threads_per_block = 32 * 4; // 4 warps
const int blocks = 1;

const int iteration = 9999999;

__global__ void cuda_core_work(int *result)
{

  float sum = 0;

  for (int i = 0; i < iteration; i++)
  {
    sum = fma(1.0f, 1.0f, sum);
    sum = fma(1.1f, 1.1f, sum);
    sum = fma(1.2f, 1.2f, sum);
    sum = fma(1.3f, 1.3f, sum);
    sum = fma(1.4f, 1.4f, sum);
    sum = fma(1.5f, 1.5f, sum);
    sum = fma(1.6f, 1.6f, sum);
    sum = fma(1.7f, 1.7f, sum);
    sum = fma(1.8f, 1.8f, sum);
    sum = fma(1.9f, 1.9f, sum);
    sum = fma(1.0f, 1.0f, sum);
  }

  result[0] = sum;
}

__global__ void tensor_core_work(int *result)
{
  const int M = 64;
  const int N = 8;
  const int K = 16;

  __align__(16) __shared__ half A_shared[M * K];
  __align__(16) __shared__ half B_shared[K * N];

  GmmaDescriptor desc_a = make_desc_a(A_shared);
  GmmaDescriptor desc_b = make_desc_b(B_shared);

  int c[2] = {};

  asm volatile("wgmma.fence.sync.aligned; \n");

  for (int i = 0; i < iteration * 3; i++)
  {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
                 "{%0, %1}, "
                 "%2, %3, "
                 "1, "
                 "1, 1, "
                 "0, 0;"
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b));
  }

  asm volatile("wgmma.commit_group.sync.aligned; \n");

  asm volatile("wgmma.wait_group.sync.aligned 0; \n");

  asm volatile("wgmma.fence.sync.aligned; \n");

  result[0] = c[0] + c[1];
}

__global__ void overlap(int *result)
{
  const int M = 64;
  const int N = 8;
  const int K = 16;

  __align__(16) __shared__ half A_shared[M * K];
  __align__(16) __shared__ half B_shared[K * N];

  GmmaDescriptor desc_a = make_desc_a(A_shared);
  GmmaDescriptor desc_b = make_desc_b(B_shared);

  int c[2] = {};

  asm volatile("wgmma.fence.sync.aligned; \n");

  for (int i = 0; i < iteration * 3; i++)
  {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 "
                 "{%0, %1}, "
                 "%2, %3, "
                 "1, "
                 "1, 1, "
                 "0, 0;"
                 : "+r"(c[0]), "+r"(c[1])
                 : "l"(desc_a), "l"(desc_b));
  }

  asm volatile("wgmma.commit_group.sync.aligned; \n");

  float sum = 0;

  for (int i = 0; i < iteration; i++)
  {
    sum = fma(1.0f, 1.0f, sum);
    sum = fma(1.1f, 1.1f, sum);
    sum = fma(1.2f, 1.2f, sum);
    sum = fma(1.3f, 1.3f, sum);
    sum = fma(1.4f, 1.4f, sum);
    sum = fma(1.5f, 1.5f, sum);
    sum = fma(1.6f, 1.6f, sum);
    sum = fma(1.7f, 1.7f, sum);
    sum = fma(1.8f, 1.8f, sum);
    sum = fma(1.9f, 1.9f, sum);
    sum = fma(1.0f, 1.0f, sum);
  }

  asm volatile("wgmma.wait_group.sync.aligned 0; \n");

  asm volatile("wgmma.fence.sync.aligned; \n");

  result[0] = c[0] + c[1];
  result[0] += reinterpret_cast<int &>(sum);
}

int main()
{

  int *d_result, h_result;
  cudaMalloc(&d_result, sizeof(int));

  cuda_timer timer;

  timer.start_timer();

  cuda_core_work<<<blocks, threads_per_block>>>(d_result);

  timer.stop_timer();

  cuda_check_error();

  printf("Cuda core time: %f\n", timer.get_time());

  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  printf("Result: %d\n", h_result);

  // tensor core work

  timer.start_timer();

  tensor_core_work<<<blocks, threads_per_block>>>(d_result);

  timer.stop_timer();

  cuda_check_error();

  printf("Tensor core time: %f\n", timer.get_time());

  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  printf("Result: %d\n", h_result);

  // overlap

  timer.start_timer();

  overlap<<<blocks, threads_per_block>>>(d_result);

  timer.stop_timer();

  cuda_check_error();

  printf("Overlap time: %f\n", timer.get_time());

  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  printf("Result: %d\n", h_result);

  return 0;
}