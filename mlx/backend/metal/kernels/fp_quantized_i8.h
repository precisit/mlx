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

METAL_FUNC int i8_nvfp4_block_dot(const thread int8_t* xq, const device uint8_t* wb) {
  // wb: the 8 bytes of one 16-value block, element 2j in the low nibble of byte j (fp_quantize order).
  int acc = 0;
#pragma clang loop unroll(full)
  for (int j = 0; j < 8; j++) {
    uint8_t b = wb[j];
    acc += int(xq[2 * j]) * i8_e2m1_x2(b & 0x0f) + int(xq[2 * j + 1]) * i8_e2m1_x2(b >> 4);
  }
  return acc;
}

// Four words of four int8 values 2*e2m1 (elements 0..15 of one block) from its 8 weight bytes, by unpack
// variant (speed lane step 2.3; MLX_GATHER_QMM_I8_QMV_UNPACK): 1 SWAR integer decode, 2 byte table,
// 8 no decode (raw codes; a timing bound with wrong numerics).
template <int U>
METAL_FUNC void i8_unpack_block(uint2 wv, thread uint* q) {
  if (U == 2) {
    q[0] = uint(i8_e2m1_x2_lut2[wv.x & 0xFF]) | (uint(i8_e2m1_x2_lut2[(wv.x >> 8) & 0xFF]) << 16);
    q[1] = uint(i8_e2m1_x2_lut2[(wv.x >> 16) & 0xFF]) | (uint(i8_e2m1_x2_lut2[wv.x >> 24]) << 16);
    q[2] = uint(i8_e2m1_x2_lut2[wv.y & 0xFF]) | (uint(i8_e2m1_x2_lut2[(wv.y >> 8) & 0xFF]) << 16);
    q[3] = uint(i8_e2m1_x2_lut2[(wv.y >> 16) & 0xFF]) | (uint(i8_e2m1_x2_lut2[wv.y >> 24]) << 16);
  } else {
    q[0] = i8_spread_nibbles(wv.x & 0xFFFF);
    q[1] = i8_spread_nibbles(wv.x >> 16);
    q[2] = i8_spread_nibbles(wv.y & 0xFFFF);
    q[3] = i8_spread_nibbles(wv.y >> 16);
    if (U == 1) {
#pragma clang loop unroll(full)
      for (int j = 0; j < 4; j++) {
        q[j] = i8_e2m1_x2_swar(q[j]);
      }
    }
  }
}

// One threadgroup = 2 simdgroups x 4 output rows each (8 rows, as the stock qmv), one x row (tid.x), one
// batch element (tid.z). Each thread walks the K blocks simd_lid, simd_lid + 32, ... so every thread's
// per-block partial is a whole NVFP4 block.
//
// Unpack variants U (speed lane step 2.3) change only how the exact integer block partial p is formed; the
// fp32 work (p * e4m3 scale, the accumulation order, simd_sum, the final scale) is the same in all of them.
// 0: scalar unpack (as scored). 1: one 8-byte weight load, SWAR decode, integer dot. 2: as 1 with a byte
// table. 3: fp16 bit construction (2*e2m1 = half(bits) * 2^15, exact) and an fp32 dot, every product and
// partial sum an exact multiple of 2^-15 below 2^24 ulps. 8: as 1 without the decode (timing bound only).
template <int U>
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
  thread int8_t xq[16];
  thread int xi[16];
  thread float xf[16];
  for (int b = simd_lid; b < n_blocks; b += SIMD_SIZE) {
    if (U == 0) {
#pragma clang loop unroll(full)
      for (int j = 0; j < 16; j++) {
        xq[j] = xr[b * 16 + j];
      }
    } else {
      // 16 x values as four 4-byte loads (row starts are 16-byte aligned: K % 16 == 0, host checks the offset)
      const device char4* xv = (const device char4*)(xr + b * 16);
#pragma clang loop unroll(full)
      for (int j = 0; j < 4; j++) {
        const char4 c = xv[j];
#pragma clang loop unroll(full)
        for (int e = 0; e < 4; e++) {
          xi[4 * j + e] = int(c[e]);
          xf[4 * j + e] = float(c[e]);
        }
      }
    }
#pragma clang loop unroll(full)
    for (int r = 0; r < rows_per_simdgroup; r++) {
      const int n = out_row + r;
      if (n < N) {
        const device uint8_t* wb = wbytes + int64_t(n) * row_bytes + b * 8;
        int p;
        if (U == 0) {
          p = i8_nvfp4_block_dot(xq, wb);
        } else if (U == 3) {
          const uint2 wv = *(const device uint2*)wb;
          float acc = 0.0f;
#pragma clang loop unroll(full)
          for (int j = 0; j < 4; j++) {
            const uint t = i8_spread_nibbles(j < 2 ? (j == 0 ? wv.x & 0xFFFF : wv.x >> 16) : (j == 2 ? wv.y & 0xFFFF : wv.y >> 16));
            const uint u0 = (t & 0xFFu) | ((t & 0xFF00u) << 8);
            const uint u1 = ((t >> 16) & 0xFFu) | ((t >> 8) & 0xFF0000u);
            const half2 h0 = as_type<half2>((((u0 & 0x00080008u) << 12) | ((u0 & 0x00070007u) << 9)));
            const half2 h1 = as_type<half2>((((u1 & 0x00080008u) << 12) | ((u1 & 0x00070007u) << 9)));
            acc = fma(xf[4 * j + 0], float(h0.x), acc);
            acc = fma(xf[4 * j + 1], float(h0.y), acc);
            acc = fma(xf[4 * j + 2], float(h1.x), acc);
            acc = fma(xf[4 * j + 3], float(h1.y), acc);
          }
          p = int(acc * 32768.0f);
        } else {
          thread uint q[4];
          i8_unpack_block<U>(*(const device uint2*)wb, q);
          int acc = 0;
#pragma clang loop unroll(full)
          for (int j = 0; j < 4; j++) {
            const char4 wc = as_type<char4>(q[j]);
#pragma clang loop unroll(full)
            for (int e = 0; e < 4; e++) {
              acc += xi[4 * j + e] * int(wc[e]);
            }
          }
          p = acc;
        }
        result[r] += float(p) * i8_e4m3_to_float(scales[int64_t(n) * n_blocks + b]);
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
