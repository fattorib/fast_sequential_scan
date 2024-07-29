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

// rounds an array of single precision to an array of bf16 values
inline __device__ void float2halfreg(const int numElem, bf16 *rmem_half,
                                     float *rmem_single) {
#pragma unroll
  for (int i = 0; i < numElem; i++) {
    rmem_half[i] = __float2bfloat16(rmem_single[i]);
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
  float h[compPerThread];

  bf16 h_half[compPerThread];

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
    h[Idx] = __bfloat162float(betaReg[nc][Idx]);
  }
  __syncthreads();

  float2halfreg(compPerThread, h_half, h);
  storeToGlobal(storePerThread, out, h_half, storeOffset, storeWidth,
                regStoreWidth);

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
    for (int Idx = 0; Idx < compPerThread; Idx++) {
      float alphaReg2 = __bfloat162float(alphaReg[nc][Idx]);
      float betaReg2 = __bfloat162float(betaReg[nc][Idx]);
      h[Idx] = (alphaReg2 * h[Idx]) + betaReg2;
    }
    // __syncthreads();
    float2halfreg(compPerThread, h_half, h);
    storeToGlobal(storePerThread, out, h_half, storeOffset, storeWidth,
                  regStoreWidth);

    nc ^= 1;

    __syncthreads();
  }

  // early exit loop for final compute and store

#pragma unroll
  for (int Idx = 0; Idx < compPerThread; Idx++) {
    float alphaReg2 = __bfloat162float(alphaReg[nc][Idx]);
    float betaReg2 = __bfloat162float(betaReg[nc][Idx]);
    h[Idx] = (alphaReg2 * h[Idx]) + betaReg2;
  }
  __syncthreads();
  out += seqStride;

  float2halfreg(compPerThread, h_half, h);
  storeToGlobal(storePerThread, out, h_half, storeOffset, storeWidth,
                regStoreWidth);
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
  bf16 hRecGrad_half[compPerThread];
  bf16 alphaGrad_half[compPerThread];

  float hRecGrad[compPerThread];
  float alphaGrad[compPerThread];

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

  // for (int Idx = 0; Idx < compPerThread; Idx += 2) {
  //   bf162 outGrad2 = *((bf162 *)(&outGrad[nc][Idx]));
  //   bf162 hReg2 = *((bf162 *)(&hReg[nc][Idx]));
  //   bf162 alphaGrad2 = __hmul2(outGrad2, hReg2);

  //   *((bf162 *)(&hRecGrad[Idx])) = outGrad2;
  //   *((bf162 *)(&alphaGrad[Idx])) = alphaGrad2;
  // }

#pragma unroll
  for (int Idx = 0; Idx < compPerThread; Idx++) {
    float outGrad2 = __bfloat162float(outGrad[nc][Idx]);
    float hReg2 = __bfloat162float(hReg[nc][Idx]);
    float alphaGrad2 = outGrad2 * hReg2;
    hRecGrad[Idx] = outGrad2;
    alphaGrad[Idx] = alphaGrad2;
  }

  float2halfreg(compPerThread, hRecGrad_half, hRecGrad);
  float2halfreg(compPerThread, alphaGrad_half, alphaGrad);
  storeToGlobal(storePerThread, grad_beta, hRecGrad_half, storeOffset,
                storeWidth, regStoreWidth);
  storeToGlobal(storePerThread, grad_alpha, alphaGrad_half, storeOffset,
                storeWidth, regStoreWidth);
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
#pragma unroll
    for (int Idx = 0; Idx < compPerThread; Idx++) {
      float alphaReg2 = __bfloat162float(alphaReg[nc][Idx]);
      float hRecGrad2 = __bfloat162float(hRecGrad[Idx]);
      float outGrad2 = __bfloat162float(outGrad[nc][Idx]);
      float hReg2 = __bfloat162float(hReg[nc][Idx]);

      hRecGrad2 = (alphaReg2 * hRecGrad2) + outGrad2;
      float alphaGrad2 = hRecGrad2 * hReg2;

      hRecGrad[Idx] = hRecGrad2;
      alphaGrad[Idx] = alphaGrad2;
    }

    float2halfreg(compPerThread, hRecGrad_half, hRecGrad);
    float2halfreg(compPerThread, alphaGrad_half, alphaGrad);

    storeToGlobal(storePerThread, grad_alpha, alphaGrad_half, storeOffset,
                  storeWidth, regStoreWidth);
    storeToGlobal(storePerThread, grad_beta, hRecGrad_half, storeOffset,
                  storeWidth, regStoreWidth);

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
#pragma unroll
  for (int Idx = 0; Idx < compPerThread; Idx++) {
    float alphaReg2 = __bfloat162float(alphaReg[nc][Idx]);
    float hRecGrad2 = __bfloat162float(hRecGrad[Idx]);
    float outGrad2 = __bfloat162float(outGrad[nc][Idx]);
    float hReg2 = __bfloat162float(hReg[nc][Idx]);

    hRecGrad2 = (alphaReg2 * hRecGrad2) + outGrad2;
    float alphaGrad2 = hRecGrad2 * hReg2;

    hRecGrad[Idx] = hRecGrad2;
    alphaGrad[Idx] = alphaGrad2;
  }

  float2halfreg(compPerThread, hRecGrad_half, hRecGrad);
  float2halfreg(compPerThread, alphaGrad_half, alphaGrad);

  storeToGlobal(storePerThread, grad_alpha, alphaGrad_half, storeOffset,
                storeWidth, regStoreWidth);
  storeToGlobal(storePerThread, grad_beta, hRecGrad_half, storeOffset,
                storeWidth, regStoreWidth);

  nc ^= 1;

  // first grad: t=0
  grad_alpha -= seqStride;
  grad_beta -= seqStride;

// actual grad computation
#pragma unroll
  for (int Idx = 0; Idx < compPerThread; Idx++) {
    float alphaReg2 = __bfloat162float(alphaReg[nc][Idx]);
    float hRecGrad2 = __bfloat162float(hRecGrad[Idx]);
    float outGrad2 = __bfloat162float(outGrad[nc][Idx]);
    hRecGrad2 = (alphaReg2 * hRecGrad2) + outGrad2;
    hRecGrad[Idx] = hRecGrad2;
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

#pragma unroll
  for (int loadIdx = 0; loadIdx < loadPerThread; loadIdx++) {
    *((half8 *)(gmem_ptr + (loadWidth * loadIdx))) = tmp;
  }

  float2halfreg(compPerThread, hRecGrad_half, hRecGrad);
  storeToGlobal(loadPerThread, grad_beta, hRecGrad_half, offset, loadWidth,
                regStoreWidth);
}