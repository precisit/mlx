// Copyright © 2026 Precisit. int8-activation x NVFP4-weight gathered matmul (modelopt-to-mlx A8 arm).
//
// y[m, n] = x_scale[m] * 1/2 * sum_b e4m3(S[n, b]) * sum_{k in block b} x_q[m, k] * w2[n, k]
//
// x_q is int8 in [-127, 127] with one float scale per row; w2 = 2 * e2m1(w) is an exact integer in
// {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12}. Each 16-wide block partial is an exact int32 (|sum| <= 24384);
// the kernel rounds only when it scales a partial by its E4M3 block scale and accumulates in fp32. No
// global scale: the caller multiplies by the per-expert weight_scale_2, as the A16 path does.

#pragma once

// 2 * e2m1(code) for the 3 magnitude bits; bit 3 is the sign.
constant constexpr int8_t i8_e2m1_x2[8] = {0, 1, 2, 3, 4, 6, 8, 12};

METAL_FUNC int i8_nvfp4_block_dot(const thread int8_t* xq, const device uint8_t* wb) {
  // wb: the 8 bytes of one 16-value block, element 2j in the low nibble of byte j (fp_quantize order).
  int acc = 0;
#pragma clang loop unroll(full)
  for (int j = 0; j < 8; j++) {
    uint8_t b = wb[j];
    uint8_t lo = b & 0x0f, hi = b >> 4;
    int vlo = i8_e2m1_x2[lo & 7];
    int vhi = i8_e2m1_x2[hi & 7];
    vlo = (lo & 8) ? -vlo : vlo;
    vhi = (hi & 8) ? -vhi : vhi;
    acc += int(xq[2 * j]) * vlo + int(xq[2 * j + 1]) * vhi;
  }
  return acc;
}

METAL_FUNC float i8_e4m3_to_float(uint8_t s) {
  return float(*(thread fp8_e4m3*)(&s));
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
  thread int8_t xq[16];
  for (int b = simd_lid; b < n_blocks; b += SIMD_SIZE) {
#pragma clang loop unroll(full)
    for (int j = 0; j < 16; j++) {
      xq[j] = xr[b * 16 + j];
    }
#pragma clang loop unroll(full)
    for (int r = 0; r < rows_per_simdgroup; r++) {
      const int n = out_row + r;
      if (n < N) {
        int p = i8_nvfp4_block_dot(xq, wbytes + int64_t(n) * row_bytes + b * 8);
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
