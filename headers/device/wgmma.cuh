#pragma once

__device__ void wgmma_fense()
{
  asm volatile("wgmma.fence.sync.aligned; \n");
}

__device__ void wgmma_commit_group()
{
  asm volatile("wgmma.commit_group.sync.aligned; \n");
}

template <int N>
__device__ void wgmma_wait()
{
  static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}