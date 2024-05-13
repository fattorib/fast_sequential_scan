#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define FLOAT4_SIZE 4
#define STAGES 2

#define bf16 __nv_bfloat16
#define bf162 __nv_bfloat162

typedef struct __align__(16) {
  bf16 x;
  bf16 y;
  bf16 z;
  bf16 w;
  bf16 xx;
  bf16 yy;
  bf16 zz;
  bf16 ww;
}
half8;

inline __device__ void readFromGlobal(const int loadPerThread, bf16 *global,
                                      bf16 *rmem, const int offset,
                                      const int loadWidth,
                                      const int regLoadWidth) {

  bf16 *rmem_ptr = &rmem[0];
  bf16 *gmem_ptr = &global[offset];

  for (int loadIdx = 0; loadIdx < loadPerThread; loadIdx++) {
    *((half8 *)(rmem_ptr + (loadIdx * regLoadWidth))) =
        *((half8 *)(gmem_ptr + (loadIdx * loadWidth)));
  }
}

inline __device__ void storeToGlobal(const int storePerThread, bf16 *global,
                                     bf16 *rmem, const int offset,
                                     const int storeWidth,
                                     const int regStoreWidth) {
  bf16 *rmem_ptr = &rmem[0];
  bf16 *gmem_ptr = &global[offset];

  for (int storeIdx = 0; storeIdx < storePerThread; storeIdx++) {
    *((half8 *)(gmem_ptr + (storeWidth * storeIdx))) =
        *((half8 *)(rmem_ptr + (storeIdx * regStoreWidth)));
  }
}

template <const int dModel, const int numThread>
__global__ void sequential_scan_forward_half(bf16 *alpha, bf16 *beta, bf16 *out,
                                             const int numContext) {
  const u_int tx = threadIdx.x;

  const u_int batchIdx = blockIdx.x;
  const u_int batchStride = dModel * numContext;
  const u_int seqStride = dModel;

  // offset pointers to start of batch
  alpha += batchIdx * batchStride;
  beta += batchIdx * batchStride;
  out += batchIdx * batchStride;

  const int loadPerThread = (dModel / numThread / FLOAT4_SIZE / sizeof(bf16));

  int loadWidth = numThread * FLOAT4_SIZE * sizeof(bf16);
  const int offset = tx * FLOAT4_SIZE * sizeof(bf16);
  const int regLoadWidth = FLOAT4_SIZE * sizeof(bf16);

  const int storePerThread = loadPerThread;
  int storeWidth = loadWidth;
  const int storeOffset = offset;
  const int regStoreWidth = regLoadWidth;

  constexpr int compPerThread = (dModel / numThread);
  bf16 h[compPerThread];
  bf16 alphaReg[STAGES][compPerThread];
  bf16 betaReg[STAGES][compPerThread];

  int nc = 0;
  int nx;

  readFromGlobal(loadPerThread, beta, betaReg[nc], offset, loadWidth,
                 regLoadWidth);
  __syncthreads();

  // t = 0, read from beta to recurrent state
#pragma unroll
  for (int Idx = 0; Idx < compPerThread; Idx++) {
    h[Idx] = betaReg[nc][Idx];
  }
  __syncthreads();

  storeToGlobal(storePerThread, out, h, storeOffset, storeWidth, regStoreWidth);

  __syncthreads();

  alpha += seqStride;
  beta += seqStride;

  readFromGlobal(loadPerThread, alpha, alphaReg[nc], offset, loadWidth,
                 regLoadWidth);
  readFromGlobal(loadPerThread, beta, betaReg[nc], offset, loadWidth,
                 regLoadWidth);

  __syncthreads();

  for (int c = 2; c < numContext; c++) {
    nx = nc ^ 1;

    // update pointers
    alpha += seqStride;
    beta += seqStride;
    out += seqStride;

    readFromGlobal(loadPerThread, alpha, alphaReg[nx], offset, loadWidth,
                   regLoadWidth);
    readFromGlobal(loadPerThread, beta, betaReg[nx], offset, loadWidth,
                   regLoadWidth);

    // __syncthreads();

// recurrence computation in registers
#pragma unroll
    for (int Idx = 0; Idx < compPerThread; Idx += 2) {
      bf162 alphaReg2 = *((bf162 *)(&alphaReg[nc][Idx]));
      bf162 h2 = *((bf162 *)(&h[Idx]));
      bf162 betaReg2 = *((bf162 *)(&betaReg[nc][Idx]));
      bf162 val = __hfma2(alphaReg2, h2, betaReg2);
      *((bf162 *)(&h[Idx])) = val;
    }
    // __syncthreads();
    storeToGlobal(storePerThread, out, h, storeOffset, storeWidth,
                  regStoreWidth);
    nc ^= 1;

    __syncthreads();
  }

  // early exit loop for final compute and store

#pragma unroll
  for (int Idx = 0; Idx < compPerThread; Idx += 2) {
    bf162 alphaReg2 = *((bf162 *)(&alphaReg[nc][Idx]));
    bf162 h2 = *((bf162 *)(&h[Idx]));
    bf162 betaReg2 = *((bf162 *)(&betaReg[nc][Idx]));
    bf162 val = __hfma2(alphaReg2, h2, betaReg2);
    *((bf162 *)(&h[Idx])) = val;
  }
  __syncthreads();
  out += seqStride;
  storeToGlobal(storePerThread, out, h, storeOffset, storeWidth, regStoreWidth);
}

template <const int dModel, const int numThread>
__global__ void sequential_scan_backward_half(bf16 *alpha_saved, bf16 *h_saved,
                                              bf16 *grad_out, bf16 *grad_alpha,
                                              bf16 *grad_beta,
                                              const int numContext) {

  const u_int tx = threadIdx.x;
  const u_int batchIdx = blockIdx.x;
  const u_int batchStride = dModel * numContext;
  const u_int seqStride = dModel;

  // number of FLOAT4_SIZE vector loads to issue per thread
  const int loadPerThread = (dModel / numThread / FLOAT4_SIZE / sizeof(bf16));
  int loadWidth = numThread * FLOAT4_SIZE * sizeof(bf16);
  const int offset = tx * FLOAT4_SIZE * sizeof(bf16);
  const int regLoadWidth = FLOAT4_SIZE * sizeof(bf16);

  const int storePerThread = loadPerThread;
  int storeWidth = loadWidth;
  const int storeOffset = offset;
  const int regStoreWidth = regLoadWidth;

  // number of array dimensions we compute per thread (required to init regs)
  const int compPerThread = (dModel / numThread);

  // store all variables in registers
  bf16 hRecGrad[compPerThread];
  bf16 alphaGrad[compPerThread];

  bf16 alphaReg[STAGES][compPerThread];
  bf16 hReg[STAGES][compPerThread];
  bf16 outGrad[STAGES][compPerThread];

  int nc = 0;
  int nx;

  // offset all pointers to start of batch
  alpha_saved += (batchIdx * batchStride) + (numContext * seqStride);
  h_saved += (batchIdx * batchStride) + ((numContext - 2) * seqStride);
  grad_out += (batchIdx * batchStride) + ((numContext - 1) * seqStride);
  grad_alpha += (batchIdx * batchStride) + ((numContext - 1) * seqStride);
  grad_beta += (batchIdx * batchStride) + ((numContext - 1) * seqStride);

  // (t=T)
  readFromGlobal(loadPerThread, grad_out, outGrad[nc], offset, loadWidth,
                 regLoadWidth);
  readFromGlobal(loadPerThread, h_saved, hReg[nc], offset, loadWidth,
                 regLoadWidth);
  __syncthreads();

  for (int Idx = 0; Idx < compPerThread; Idx += 2) {
    bf162 outGrad2 = *((bf162 *)(&outGrad[nc][Idx]));
    bf162 hReg2 = *((bf162 *)(&hReg[nc][Idx]));
    bf162 alphaGrad2 = __hmul2(outGrad2, hReg2);

    *((bf162 *)(&hRecGrad[Idx])) = outGrad2;
    *((bf162 *)(&alphaGrad[Idx])) = alphaGrad2;
  }

  storeToGlobal(storePerThread, grad_beta, hRecGrad, storeOffset, storeWidth,
                regStoreWidth);
  storeToGlobal(storePerThread, grad_alpha, alphaGrad, storeOffset, storeWidth,
                regStoreWidth);
  __syncthreads();

  alpha_saved -= seqStride;
  h_saved -= seqStride;
  grad_out -= seqStride;

  readFromGlobal(loadPerThread, alpha_saved, alphaReg[nc], offset, loadWidth,
                 regLoadWidth);
  readFromGlobal(loadPerThread, grad_out, outGrad[nc], offset, loadWidth,
                 regLoadWidth);
  readFromGlobal(loadPerThread, h_saved, hReg[nc], offset, loadWidth,
                 regLoadWidth);

  // inner loop: t \in (T, 0)
  for (int c = 3; c < numContext; c++) {
    nx = nc ^ 1;

    // decrease pointers
    alpha_saved -= seqStride;
    h_saved -= seqStride;
    grad_out -= seqStride;
    grad_alpha -= seqStride;
    grad_beta -= seqStride;

    // load alpha
    readFromGlobal(loadPerThread, alpha_saved, alphaReg[nx], offset, loadWidth,
                   regLoadWidth);

    // load grad_out
    readFromGlobal(loadPerThread, grad_out, outGrad[nx], offset, loadWidth,
                   regLoadWidth);

    // load h
    readFromGlobal(loadPerThread, h_saved, hReg[nx], offset, loadWidth,
                   regLoadWidth);

    // actual grad computation
    for (int Idx = 0; Idx < compPerThread; Idx += 2) {
      bf162 alphaReg2 = *((bf162 *)(&alphaReg[nc][Idx]));
      bf162 hRecGrad2 = *((bf162 *)(&hRecGrad[Idx]));
      bf162 outGrad2 = *((bf162 *)(&outGrad[nc][Idx]));
      bf162 hReg2 = *((bf162 *)(&hReg[nc][Idx]));

      hRecGrad2 = __hfma2(alphaReg2, hRecGrad2, outGrad2);
      bf162 alphaGrad2 = __hmul2(hRecGrad2, hReg2);

      *((bf162 *)(&hRecGrad[Idx])) = hRecGrad2;
      *((bf162 *)(&alphaGrad[Idx])) = alphaGrad2;
    }

    storeToGlobal(storePerThread, grad_alpha, alphaGrad, storeOffset,
                  storeWidth, regStoreWidth);
    storeToGlobal(storePerThread, grad_beta, hRecGrad, storeOffset, storeWidth,
                  regStoreWidth);

    nc ^= 1;

    __syncthreads();
  }

  nx = nc ^ 1;

  // decrease pointers
  alpha_saved -= seqStride;
  grad_out -= seqStride;
  grad_alpha -= seqStride;
  grad_beta -= seqStride;

  // load alpha
  readFromGlobal(loadPerThread, alpha_saved, alphaReg[nx], offset, loadWidth,
                 regLoadWidth);

  // load grad_out
  readFromGlobal(loadPerThread, grad_out, outGrad[nx], offset, loadWidth,
                 regLoadWidth);

  __syncthreads();

  // actual grad computation
  for (int Idx = 0; Idx < compPerThread; Idx += 2) {
    bf162 alphaReg2 = *((bf162 *)(&alphaReg[nc][Idx]));
    bf162 hRecGrad2 = *((bf162 *)(&hRecGrad[Idx]));
    bf162 outGrad2 = *((bf162 *)(&outGrad[nc][Idx]));
    bf162 hReg2 = *((bf162 *)(&hReg[nc][Idx]));

    hRecGrad2 = __hfma2(alphaReg2, hRecGrad2, outGrad2);
    bf162 alphaGrad2 = __hmul2(hRecGrad2, hReg2);

    *((bf162 *)(&hRecGrad[Idx])) = hRecGrad2;
    *((bf162 *)(&alphaGrad[Idx])) = alphaGrad2;
  }

  storeToGlobal(storePerThread, grad_alpha, alphaGrad, storeOffset, storeWidth,
                regStoreWidth);
  storeToGlobal(storePerThread, grad_beta, hRecGrad, storeOffset, storeWidth,
                regStoreWidth);

  nc ^= 1;

  // first grad: t=0
  grad_alpha -= seqStride;
  grad_beta -= seqStride;

  // actual grad computation
  for (int Idx = 0; Idx < compPerThread; Idx += 2) {
    bf162 alphaReg2 = *((bf162 *)(&alphaReg[nc][Idx]));
    bf162 hRecGrad2 = *((bf162 *)(&hRecGrad[Idx]));
    bf162 outGrad2 = *((bf162 *)(&outGrad[nc][Idx]));
    hRecGrad2 = __hfma2(alphaReg2, hRecGrad2, outGrad2);
    *((bf162 *)(&hRecGrad[Idx])) = hRecGrad2;
  }

  __syncthreads();

  // store grads to global
  half8 tmp;
  tmp.x = __float2bfloat16(0.0f);
  tmp.y = __float2bfloat16(0.0f);
  tmp.z = __float2bfloat16(0.0f);
  tmp.w = __float2bfloat16(0.0f);
  tmp.xx = __float2bfloat16(0.0f);
  tmp.yy = __float2bfloat16(0.0f);
  tmp.zz = __float2bfloat16(0.0f);
  tmp.ww = __float2bfloat16(0.0f);

  bf16 *gmem_ptr = &grad_alpha[offset];

  for (int loadIdx = 0; loadIdx < loadPerThread; loadIdx++) {
    *((half8 *)(gmem_ptr + (loadWidth * loadIdx))) = tmp;
  }

  storeToGlobal(loadPerThread, grad_beta, hRecGrad, offset, loadWidth,
                regStoreWidth);
}