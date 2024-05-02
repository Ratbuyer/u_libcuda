#include <stdio.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <random>

void print_matrix(half *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      printf("%f ", __half2float(matrix[i * cols + j]));
    }
    printf("\n");
  }
  printf("\n");
}

void fill_random(half *matrix, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      float value = 5.0f * rand() / RAND_MAX;
      matrix[i * cols + j] = __float2half(value);
    }
  }
}

void fill_fixed(half *matrix, int rows, int cols, float value)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      matrix[i * cols + j] = __float2half(value);
    }
  }
}

void CPU_gemm(half *A, half *B, half *C, int M, int N, int K)
{
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      C[i * N + j] = 0;
      for (int k = 0; k < K; k++)
      {
        float a = __half2float(A[i * K + k]);
        float b = __half2float(B[k * N + j]);
        float c = __half2float(C[i * N + j]);
        float new_c = a * b + c;
        C[i * N + j] = __float2half(new_c);
      }
    }
  }
}

void compare_matrices(half *A, half *B, int rows, int cols)
{
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      float a = __half2float(A[i * cols + j]);
      float b = __half2float(B[i * cols + j]);
      if (a != b || i * rows + j < 10)
      {
        printf("Error at (%d, %d) : %f != %f\n", i, j, a, b);
      }
    }
  }
}