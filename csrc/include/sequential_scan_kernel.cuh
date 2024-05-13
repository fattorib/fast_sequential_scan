#include <cuda.h>
#include <cuda_runtime.h>
#define FLOAT4_SIZE 4
#define STAGES 2

// reads from global memory to shared memory
inline __device__ void readFromGlobal(const int loadPerThread, float *global,
                                      float *rmem, const int offset,
                                      const int loadWidth,
                                      const int regLoadWidth) {
  float *rmem_ptr = &rmem[0];
  float *global_ptr = &global[offset];

  for (int loadIdx = 0; loadIdx < loadPerThread; loadIdx++) {
    *((float4 *)(rmem_ptr + (loadIdx * regLoadWidth))) =
        *((float4 *)(global_ptr + (loadIdx * loadWidth)));
  }
}

// stores to global memory from shared memory
inline __device__ void storeToGlobal(const int storePerThread, float *global,
                                     float *rmem, const int offset,
                                     const int storeWidth,
                                     const int regStoreWidth) {

  float *rmem_ptr = &rmem[0];
  float *global_ptr = &global[offset];

  for (int storeIdx = 0; storeIdx < storePerThread; storeIdx++) {
    *((float4 *)(global_ptr + (storeWidth * storeIdx))) =
        *((float4 *)(rmem_ptr + (storeIdx * regStoreWidth)));
  }
}

template <const int dModel, const int numThread>
__global__ void sequential_scan_forward(float *alpha, float *beta, float *out,
                                        const int numContext) {
  const u_int tx = threadIdx.x;

  const u_int batchIdx = blockIdx.x;
  const u_int batchStride = dModel * numContext;
  const u_int seqStride = dModel;

  // offset pointers to start of batch
  alpha += batchIdx * batchStride;
  beta += batchIdx * batchStride;
  out += batchIdx * batchStride;

  // number of FLOAT4_SIZE vector loads to issue per thread
  const int loadPerThread = (dModel / numThread / FLOAT4_SIZE);
  int loadWidth = numThread * FLOAT4_SIZE;
  const int offset = FLOAT4_SIZE * tx;
  const int regLoadWidth = FLOAT4_SIZE;

  const int storePerThread = loadPerThread;
  int storeWidth = loadWidth;
  const int storeOffset = offset;
  const int regStoreWidth = regLoadWidth;

  // number of array dimensions we compute per thread (required to init regs)
  constexpr int compPerThread = (dModel / numThread);

  float h[compPerThread];
  float alphaReg[STAGES][compPerThread];
  float betaReg[STAGES][compPerThread];

  int nc = 0;
  int nx;

  readFromGlobal(loadPerThread, beta, betaReg[nc], offset, loadWidth,
                 regLoadWidth);

  __syncthreads();

  // t = 0, read from beta to recurrent state
  for (int idx = 0; idx < compPerThread; idx++) {
    h[idx] = betaReg[nc][idx];
  }

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

// recurrence computation in registers
#pragma unroll
    for (int idx = 0; idx < compPerThread; idx++) {
      h[idx] = (alphaReg[nc][idx] * h[idx]) + betaReg[nc][idx];
    }

    storeToGlobal(storePerThread, out, h, storeOffset, storeWidth,
                  regStoreWidth);

    nc ^= 1;

    __syncthreads();
  }

  // early exit loop for final compute and store
#pragma unroll
  for (int idx = 0; idx < compPerThread; idx++) {
    h[idx] = (alphaReg[nc][idx] * h[idx]) + betaReg[nc][idx];
  }

  out += seqStride;
  storeToGlobal(storePerThread, out, h, storeOffset, storeWidth, regStoreWidth);
}

template <const int dModel, const int numThread>
__global__ void sequential_scan_backward(float *alpha_saved, float *h_saved,
                                         float *grad_out, float *grad_alpha,
                                         float *grad_beta,
                                         const int numContext) {

  const u_int tx = threadIdx.x;
  const u_int batchIdx = blockIdx.x;
  const u_int batchStride = dModel * numContext;
  const u_int seqStride = dModel;

  // number of FLOAT4_SIZE vector loads to issue per thread
  const int loadPerThread = (dModel / numThread / FLOAT4_SIZE);
  int loadWidth = numThread * FLOAT4_SIZE;
  const int offset = FLOAT4_SIZE * tx;
  const int regLoadWidth = FLOAT4_SIZE;

  const int storePerThread = loadPerThread;
  int storeWidth = loadWidth;
  const int storeOffset = offset;
  const int regStoreWidth = regLoadWidth;

  // number of array dimensions we compute per thread (required to init regs)
  const int compPerThread = (dModel / numThread);

  // store all variables in registers
  float hRecGrad[compPerThread];
  float alphaGrad[compPerThread];

  float alphaReg[STAGES][compPerThread];
  float hReg[STAGES][compPerThread];
  float outGrad[STAGES][compPerThread];

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

  for (int idx = 0; idx < compPerThread; idx++) {
    hRecGrad[idx] = outGrad[nc][idx];
    alphaGrad[idx] = outGrad[nc][idx] * hReg[nc][idx];
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

    // load h -> memory error here since we try and load this one too many
    // times!
    readFromGlobal(loadPerThread, h_saved, hReg[nx], offset, loadWidth,
                   regLoadWidth);

    // actual grad computation
    for (int idx = 0; idx < compPerThread; idx++) {
      hRecGrad[idx] = alphaReg[nc][idx] * hRecGrad[idx];
      hRecGrad[idx] += outGrad[nc][idx];
      alphaGrad[idx] = hRecGrad[idx] * hReg[nc][idx];
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

  // actual grad computation
  for (int idx = 0; idx < compPerThread; idx++) {
    hRecGrad[idx] = alphaReg[nc][idx] * hRecGrad[idx];
    hRecGrad[idx] += outGrad[nc][idx];
    alphaGrad[idx] = hRecGrad[idx] * hReg[nc][idx];
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
  for (int idx = 0; idx < compPerThread; idx++) {
    hRecGrad[idx] = alphaReg[nc][idx] * hRecGrad[idx];
    hRecGrad[idx] += outGrad[nc][idx];
  }

  __syncthreads();

  // store grads to global
  float4 tmp;
  tmp.x = 0.0f;
  tmp.y = 0.0f;
  tmp.z = 0.0f;
  tmp.w = 0.0f;

  float *gmem_ptr = &grad_alpha[storeOffset];

  for (int storeIdx = 0; storeIdx < storePerThread; storeIdx++) {
    *((float4 *)(gmem_ptr + (storeWidth * storeIdx))) = tmp;
  }

  storeToGlobal(storePerThread, grad_beta, hRecGrad, storeOffset, storeWidth,
                regStoreWidth);
}