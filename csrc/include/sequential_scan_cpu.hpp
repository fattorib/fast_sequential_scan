#pragma once

#include <vector>
// reference scan on a 3-dimensional array (bs, sq, d)
template <int batchSize, int numSeq, int dModel>
void reference_scan_fwd(std::vector<float> *alpha, std::vector<float> *beta,
                        std::vector<float> *h) {

  float carry[dModel] = {0.0f};

  // strides -> row major array
  constexpr int batchStride = numSeq * dModel;
  constexpr int seqStride = dModel;
  constexpr int dStride = 1;

#pragma omp parallel for
  for (int bsIdx = 0; bsIdx < batchSize; bsIdx++) {

    for (int sqIdx = 0; sqIdx < numSeq; sqIdx++) {

      for (int dimIdx = 0; dimIdx < dModel; dimIdx++) {
        int index = batchStride * bsIdx + sqIdx * seqStride + dimIdx * dStride;

        if (sqIdx == 0) {
          carry[dimIdx] = (*beta)[index];
          (*h)[index] = carry[dimIdx];
        } else {
          carry[dimIdx] = (*alpha)[index] * (carry[dimIdx]) + (*beta)[index];
          (*h)[index] = carry[dimIdx];
        }
      }
    }
  }
};

template <int batchSize, int numSeq, int dModel>
void reference_scan_bwd(std::vector<float> *alpha_s, std::vector<float> *h_s,
                        std::vector<float> *d_out, std::vector<float> *d_alpha,
                        std::vector<float> *d_beta) {

  float h_grad[dModel] = {0.0f}; // might be too slow!

  // strides -> row major array
  constexpr int batchStride = numSeq * dModel;
  constexpr int seqStride = dModel;

// TODO: parallelize
#pragma omp parallel for
  for (int bsIdx = 0; bsIdx < batchSize; bsIdx++) {

    // sequence-level pointers
    int batch_offs = batchStride * bsIdx;
    int alpha_offs = batch_offs + seqStride * numSeq;
    int d_out_offs = batch_offs + seqStride * (numSeq - 1);
    int h_offs = batch_offs + seqStride * (numSeq - 2);

    int d_alpha_offs = batch_offs + seqStride * (numSeq - 1);
    int d_beta_offs = batch_offs + seqStride * (numSeq - 1);

    // cover grads for t=T
    for (int dimIdx = 0; dimIdx < dModel; dimIdx++) {
      h_grad[dimIdx] = (*d_out)[d_out_offs + dimIdx];
      (*d_alpha)[d_alpha_offs + dimIdx] =
          h_grad[dimIdx] * (*h_s)[h_offs + dimIdx];

      (*d_beta)[d_beta_offs + dimIdx] = (*d_out)[d_out_offs + dimIdx];
    }

    // cover grads for (t = (T-1) to (1))
    for (int sqIdx = 2; sqIdx < numSeq; sqIdx++) {

      // decrease pointers
      d_alpha_offs -= seqStride;
      d_beta_offs -= seqStride;
      h_offs -= seqStride;
      d_out_offs -= seqStride;
      alpha_offs -= seqStride;

      for (int dimIdx = 0; dimIdx < dModel; dimIdx++) {

        // compute grad for recurrent state
        h_grad[dimIdx] = h_grad[dimIdx] * (*alpha_s)[alpha_offs + dimIdx];
        h_grad[dimIdx] += (*d_out)[d_out_offs + dimIdx];

        // compute dalpha
        (*d_alpha)[d_alpha_offs + dimIdx] =
            h_grad[dimIdx] * (*h_s)[h_offs + dimIdx];

        // compute dbeta
        (*d_beta)[d_beta_offs + dimIdx] = h_grad[dimIdx];
      }
    }

    // cover grads for t=0

    d_alpha_offs -= seqStride;
    d_beta_offs -= seqStride;
    h_offs -= seqStride;
    d_out_offs -= seqStride;
    alpha_offs -= seqStride;

    for (int dimIdx = 0; dimIdx < dModel; dimIdx++) {

      // compute grad for recurrent state
      h_grad[dimIdx] = h_grad[dimIdx] * (*alpha_s)[alpha_offs + dimIdx];
      h_grad[dimIdx] += (*d_out)[d_out_offs + dimIdx];

      // compute dalpha
      (*d_alpha)[d_alpha_offs + dimIdx] = 0.0f;

      // compute dbeta
      (*d_beta)[d_beta_offs + dimIdx] = h_grad[dimIdx];
    }
  }
};
