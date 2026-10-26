// Copyright © 2026 Precisit. E4M3-activation x NVFP4-weight sorted gathered matmul on NAX (modelopt-to-mlx F8 arm,
// PROTOCOL amendment 17). Needs Metal 4.1 (the fp8 tensor formats).
//
// y[m, n] = x_scale[m] / 2 * sum_b e4m3(S[n, b]) * sum_{k in b} x_q[m, k] * w2[n, k], x_q E4M3 with one float scale per
// row (max|x| / 448), w2 = 2 * e2m1(w) staged as its exact E4M3 bytes. The structure is fp_gather_qmm_rhs_nax_i8's
// (A8, f295ac5): schedule_row_tile, a BN x BK weight tile staged per K tile through a 256-entry byte table, block
// scales staged [NB][BN]. The block product is one e4m3 x e4m3 -> f32 matmul2d per NVFP4 block (K = 16, mode
// multiply) on tensors over device x and the staged tile, then rescaled per output column into an fp32
// accumulator. Every product is exact; the block sum is the hardware's fp32 summation (gate F-G2 bounds it).
// RELAXED is matmul2d's relaxed_precision flag; 0 (strict) is the default route, 1 is a reading.

#pragma once

#include "mlx/backend/metal/kernels/fp_quantized_i8_utils.h"

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 410)

template <int BM, int BN, int BK, int WM, int WN, bool RELAXED>
[[kernel]] void fp_gather_qmm_rhs_nax_f8(
    const device uint8_t* x [[buffer(0)]],
    const device float* x_scale [[buffer(1)]],
    const device uint32_t* w [[buffer(2)]],
    const device uint8_t* scales [[buffer(3)]],
    const device int32_t* offsets [[buffer(4)]],
    device float* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    const constant int& num_groups [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = BK + 16;
  constexpr int NB = BK / 16;
  constexpr int TPR = (WM * WN * SIMD_SIZE) / BN;
  constexpr int VPT = BK / TPR;
  static_assert(TPR >= 1 && BK % TPR == 0 && VPT % 32 == 0, "a thread stages whole 16-byte weight chunks");
  static_assert(BK_padded % 16 == 0, "16-byte aligned weight rows in threadgroup memory");

  alignas(16) threadgroup uint8_t Ws[BN * BK_padded];
  alignas(16) threadgroup float Ss[NB * BN];

  int y_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {
    return;
  }
  const int y_col = tid.x * BN;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short sgp_sm = short(clamp(int(tgp_bm) - tm, 0, int(SM)));
  const bool sg_active = sgp_sm > 0;
  // Rows read for this simdgroup: SM, except at the end of x (rows past the group are computed, never stored)
  const int x_rows = min(int(SM), M - (y_row + tm));

  const int nl = lid / TPR;
  const int kl = (lid % TPR) * VPT;
  const device uint8_t* wrow = (const device uint8_t*)w + (size_t(group) * N + y_col + nl) * size_t(K / 2);
  const device uint8_t* srow = scales + (size_t(group) * N + y_col + nl) * size_t(K / 16);
  threadgroup uint8_t* wdst = Ws + nl * BK_padded + kl;

  const device uint8_t* xn = x + size_t(y_row + tm) * K;

  typedef metal::metal_fp8_e4m3_format fp8_t;
  typedef metal::dextents<int32_t, 2> ext_t;
  typedef metal::tensor<device fp8_t, ext_t, metal::tensor_inline> x_t;
  typedef metal::tensor<threadgroup fp8_t, ext_t, metal::tensor_inline> w_t;
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      SM, SN, 16, false, true, RELAXED, mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto P = op.template get_destination_cooperative_tensor<x_t, w_t, float>();
  auto D = op.template get_destination_cooperative_tensor<x_t, w_t, float>();
  for (ushort i = 0; i < D.get_capacity(); i++) {
    D[i] = 0.0f;
  }

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const device uint4* p = (const device uint4*)(wrow + (k + kl) / 2);
      STEEL_PRAGMA_UNROLL
      for (int c = 0; c < VPT / 32; c++) {
        const uint4 wv = p[c];
        uint o[8];
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 4; i++) {
          o[2 * i] = uint(f8_e2m1_x2_lut2[wv[i] & 0xFF]) | (uint(f8_e2m1_x2_lut2[(wv[i] >> 8) & 0xFF]) << 16);
          o[2 * i + 1] = uint(f8_e2m1_x2_lut2[(wv[i] >> 16) & 0xFF]) | (uint(f8_e2m1_x2_lut2[wv[i] >> 24]) << 16);
        }
        threadgroup uint4* d4 = (threadgroup uint4*)(wdst + 32 * c);
        d4[0] = uint4(o[0], o[1], o[2], o[3]);
        d4[1] = uint4(o[4], o[5], o[6], o[7]);
      }
      STEEL_PRAGMA_UNROLL
      for (int b = 0; b < VPT / 16; b++) {
        Ss[(kl / 16 + b) * BN + nl] = i8_e4m3_to_float(srow[(k + kl) / 16 + b]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_active) {
      STEEL_PRAGMA_NO_UNROLL
      for (int kb = 0; kb < NB; kb++) {
        x_t X((device uchar*)(xn + k + kb * 16), ext_t(16, x_rows), metal::array<int32_t, 2>{1, K});
        w_t W((threadgroup uchar*)(Ws + tn * BK_padded + kb * 16), ext_t(16, SN), metal::array<int32_t, 2>{1, BK_padded});
        op.run(X, W, P);
        for (ushort i = 0; i < P.get_capacity(); i++) {
          if (P.is_valid_element(i)) {
            const auto pos = P.get_multidimensional_index(i);
            D[i] += P[i] * Ss[kb * BN + tn + int(pos[0])];
          }
        }
      }
    }
  }

  if (!sg_active) {
    return;
  }
  for (ushort i = 0; i < D.get_capacity(); i++) {
    if (D.is_valid_element(i)) {
      const auto pos = D.get_multidimensional_index(i);
      const int r_local = tm + int(pos[1]);
      if (r_local < tgp_bm) {
        const int row = y_row + r_local;
        y[size_t(row) * N + y_col + tn + int(pos[0])] = D[i] * x_scale[row] * 0.5f;
      }
    }
  }
}

#endif // __METAL_VERSION__ >= 410
