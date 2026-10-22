// Copyright © 2026 Precisit. int8-activation x NVFP4-weight gathered matmul (modelopt-to-mlx A8 arm).
//
// y[m, n] = x_scale[m] * 1/2 * sum_b e4m3(S[n, b]) * sum_{k in block b} x_q[m, k] * w2[n, k]
//
// x_q is int8 in [-127, 127] with one float scale per row; w2 = 2 * e2m1(w) is an exact integer in
// {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12}. Each 16-wide block partial is an exact int32 (|sum| <= 24384);
// the kernel rounds only when it scales a partial by its E4M3 block scale and accumulates in fp32. No
// global scale: the caller multiplies by the per-expert weight_scale_2, as the A16 path does.

#pragma once

#include "mlx/backend/metal/kernels/fp_quantized_i8_utils.h"

// The exact block partial p = sum_{k in block} x_q[k] * 2*e2m1(w[k]) for one 16-value block (its 8 weight bytes as
// one uint2), from x_q held in float. 2*e2m1 is built as an fp16 bit pattern: code s|e|m -> bits s<<15 | e<<10 |
// m<<9 is e2m1 * 2^-14 (e = 0 gives the fp16 subnormal m * 2^-15), so half(bits) * 2^15 = 2*e2m1 exactly. Every
// product x_q * e2m1 * 2^-14 and every partial sum is an integer multiple of 2^-15 below 2^24 of them, so the fp32
// dot is exact in any order, and the final * 2^15 gives the same integer as an int32 sum (speed lane step 2.3:
// on M5 Max this runs at the stock bf16 vector kernel's speed, against 2.3-4.5x for the integer form).
METAL_FUNC float i8_nvfp4_block_dot(const thread float* xf, uint2 wv) {
  float acc = 0.0f;
#pragma clang loop unroll(full)
  for (int j = 0; j < 4; j++) {
    const uint t = i8_spread_nibbles((j < 2 ? wv.x : wv.y) >> (16 * (j & 1)) & 0xFFFFu);
    const uint u0 = (t & 0xFFu) | ((t & 0xFF00u) << 8);
    const uint u1 = ((t >> 16) & 0xFFu) | ((t >> 8) & 0xFF0000u);
    const half2 h0 = as_type<half2>(((u0 & 0x00080008u) << 12) | ((u0 & 0x00070007u) << 9));
    const half2 h1 = as_type<half2>(((u1 & 0x00080008u) << 12) | ((u1 & 0x00070007u) << 9));
    acc = fma(xf[4 * j + 0], float(h0.x), acc);
    acc = fma(xf[4 * j + 1], float(h0.y), acc);
    acc = fma(xf[4 * j + 2], float(h1.x), acc);
    acc = fma(xf[4 * j + 3], float(h1.y), acc);
  }
  return acc * 32768.0f;
}

// One threadgroup = 2 simdgroups x 4 output rows each (8 rows, as the stock qmv), one x row (tid.x), one
// batch element (tid.z). Each thread walks the K blocks simd_lid, simd_lid + 32, ... so every thread's
// per-block partial is a whole NVFP4 block.
[[kernel]] void fp_gather_qmv_i8(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    const device int8_t* x [[buffer(2)]],
    const device float* x_scale [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device float* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& x_batch_ndims [[buffer(9)]],
    const constant int* x_shape [[buffer(10)]],
    const constant int64_t* x_strides [[buffer(11)]],
    const constant int& w_batch_ndims [[buffer(12)]],
    const constant int* w_shape [[buffer(13)]],
    const constant int64_t* w_strides [[buffer(14)]],
    const constant int64_t* s_strides [[buffer(15)]],
    const constant int& batch_ndims [[buffer(16)]],
    const constant int* batch_shape [[buffer(17)]],
    const constant int64_t* lhs_strides [[buffer(18)]],
    const constant int64_t* rhs_strides [[buffer(19)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int rows_per_simdgroup = 4;
  constexpr int num_simdgroups = 2;
  const int M = x_shape[x_batch_ndims];

  // gathered batch element -> (x matrix, expert)
  uint32_t x_idx, w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  int64_t x_off = (x_batch_ndims == 1) ? int64_t(x_idx) * x_strides[0]
                                       : int64_t(elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims));
  // x and x_scale are row-contiguous with the same batch layout: x_scale has one entry per x row.
  const device int8_t* xr = x + x_off + int64_t(tid.x) * K;
  const float xs = x_scale[x_off / K + tid.x];
  const device uint8_t* wbytes = (const device uint8_t*)w;
  if (w_batch_ndims == 1) {
    wbytes += int64_t(w_idx) * w_strides[0] * 4;
    scales += int64_t(w_idx) * s_strides[0];
  } else {
    ulong2 idx = elem_to_loc_broadcast(w_idx, w_shape, w_strides, s_strides, w_batch_ndims);
    wbytes += idx.x * 4;
    scales += idx.y;
  }
  y += int64_t(tid.z) * M * N + int64_t(tid.x) * N;

  const int out_row = tid.y * (num_simdgroups * rows_per_simdgroup) + simd_gid * rows_per_simdgroup;
  if (out_row >= N) {
    return;
  }
  const int n_blocks = K / 16;
  const int row_bytes = K / 2;
  float result[rows_per_simdgroup] = {0};
  thread float xf[16];
  for (int b = simd_lid; b < n_blocks; b += SIMD_SIZE) {
    // 16 x values as four 4-byte loads (row starts are 16-byte aligned: K % 16 == 0 and the host checks x's offset)
    const device char4* xv = (const device char4*)(xr + b * 16);
#pragma clang loop unroll(full)
    for (int j = 0; j < 4; j++) {
      const char4 c = xv[j];
#pragma clang loop unroll(full)
      for (int e = 0; e < 4; e++) {
        xf[4 * j + e] = float(c[e]);
      }
    }
#pragma clang loop unroll(full)
    for (int r = 0; r < rows_per_simdgroup; r++) {
      const int n = out_row + r;
      if (n < N) {
        const float p = i8_nvfp4_block_dot(xf, *(const device uint2*)(wbytes + int64_t(n) * row_bytes + b * 8));
        result[r] += p * i8_e4m3_to_float(scales[int64_t(n) * n_blocks + b]);
      }
    }
  }
#pragma clang loop unroll(full)
  for (int r = 0; r < rows_per_simdgroup; r++) {
    float v = simd_sum(result[r]);
    if (simd_lid == 0 && out_row + r < N) {
      y[out_row + r] = v * xs * 0.5f;
    }
  }
}

// y[r, :] *= x_scale[r] (the non-NAX sorted fallback: stock fp32 matmul on the exact int8 values, then this).
[[kernel]] void fp_i8_scale_rows(
    device float* y [[buffer(0)]],
    const device float* x_scale [[buffer(1)]],
    const constant int& N [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  y[size_t(gid.y) * N + gid.x] *= x_scale[gid.y];
}
