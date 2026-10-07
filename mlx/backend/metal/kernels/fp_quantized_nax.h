// Copyright © 2025-2026 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/fp4.h"
#include "mlx/backend/metal/kernels/fp8.h"

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int wsize = 8, int bits>
inline constexpr short get_pack_factor() {
  return wsize / bits;
}

template <int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  return wsize / 8;
}

template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    // Use nv scale
    return T(*(thread fp8_e4m3*)(&s));
  } else {
    return T(*(thread fp8_e8m0*)(&s));
  }
}

template <int bits>
struct Quantize {
  uint8_t operator()(float x) thread {
    if (bits == 8) {
      return fp8_e4m3(x).bits;
    } else {
      return fp4_e2m1(x).bits;
    }
  }
};

template <int bits, typename U = float>
struct Dequantize {
  U operator()(uint8_t x) thread {
    if constexpr (bits == 8) {
      return U(*(thread fp8_e4m3*)(&x));
    } else {
      return U(*(thread fp4_e2m1*)(&x));
    }
  }
};

template <typename U, int bits>
inline void dequantize(uint8_t w, float scale, threadgroup U* w_local) {
  if constexpr (bits == 4) {
    w_local[0] = static_cast<U>(scale * Dequantize<4, float>{}(w));
    w_local[1] = static_cast<U>(scale * Dequantize<4, float>{}(w >> 4));
  } else {
    w_local[0] = static_cast<U>(scale * Dequantize<8, float>{}(w));
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits,
    bool has_global_scale = false>
struct QuantizedBlockLoader {
  MLX_MTL_CONST short pack_factor = get_pack_factor<8, bits>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;

  MLX_MTL_CONST short n_reads_per_scale = (n_reads * pack_factor) <= group_size
      ? n_reads
      : (group_size / pack_factor);
  MLX_MTL_CONST short n_steps_per_read = n_reads / n_reads_per_scale;

  MLX_MTL_CONST short n_groups = BCOLS / group_size;

  const int src_ld;
  const int tile_stride;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  const short group_id;

  threadgroup T* dst;
  const device uint8_t* src;
  const device uint8_t* scales;
  // nvfp4 tensor scale, folded into the group scale as fp_dequantize does.
  // Kept in float: it is ~1e-5, so in fp16 small scales lose most bits.
  float inv_scale_enc = 1.0f;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device uint8_t* scales_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]],
      const device float* global_scale = nullptr) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        group_id((bj * pack_factor) / group_size),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size + group_id) {
    if constexpr (has_global_scale) {
      inv_scale_enc = *global_scale / (F8E4M3_MAX * F4E2M1_MAX);
    }
  }

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    int k = 0;
    for (int i = 0; i < n_steps_per_read; i++) {
      float scale =
          float(dequantize_scale<T, group_size>(scales[i])) * inv_scale_enc;
      for (int j = 0; j < n_reads_per_scale; j++) {
        dequantize<T, bits>(
            src[k * bytes_per_pack], scale, dst + k * pack_factor);
        k++;
      }
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    int k = 0;
    for (int i = 0; i < n_steps_per_read; i++) {
      float scale =
          float(dequantize_scale<T, group_size>(scales[i])) * inv_scale_enc;
      for (int j = 0; j < n_reads_per_scale; j++) {
        dequantize<T, bits>(
            src[k * bytes_per_pack], scale, dst + k * pack_factor);
        k++;
      }
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      scales += n_groups;
    } else {
      scales += group_stride;
    }
  }
};

using namespace mlx::steel;

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool has_global_scale = false,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
METAL_FUNC void fp_qmm_t_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    device T* y,
    threadgroup Wtype* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  // Instantiate Loader
  using loader_w_t = QuantizedBlockLoader<
      Wtype,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      has_global_scale>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid, global_scale);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(int(SN), N - (y_col + tn));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<Wtype, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<Wtype, BK_padded, 1>(Ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        loader_w.next();
      }

      // Store results to device memory
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool has_global_scale = false,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
METAL_FUNC void fp_qmm_n_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;
  (void)M;

  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      has_global_scale>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the x loader and mma operation
  // const short num_els = min(BM, M - y_row);
  // const short num_outs = min(BN, N - y_col);
  loader_w_t loader_w(wl, scales, K, Ws, simd_gid, simd_lid, global_scale);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  const short ldb_tgp = BN_padded;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = false;

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<Wtype, TK, TN> Btile;

      volatile int compiler_barrier;

      Atile.load(x + kk1, K);
      Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * ldb_tgp);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    x += BK;
    loader_w.next();
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);

  Dtile.store(y + tm * N + tn, N);
}

template <typename T, typename S>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device S*& scales,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
  }
  y += tid.z * output_stride;
}

template <bool has_global_scale, typename T, typename S>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device S*& scales,
    const device float*& global_scale,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
  }
  // One global scale per expert, contiguous over the batch dims of w.
  if constexpr (has_global_scale) {
    global_scale += w_idx;
  }
  y += tid.z * output_stride;
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    const bool has_global_scale = false,
    typename Wtype = bfloat>
[[kernel]] void fp_qmm_t_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  threadgroup Wtype Ws[BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets(
        x,
        w,
        scales,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        tid);
  }
  fp_qmm_t_impl<
      T,
      group_size,
      bits,
      aligned_N,
      has_global_scale,
      BM,
      BK,
      BN,
      WM,
      WN,
      Wtype>(
      w, scales, global_scale, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    typename Wtype = bfloat>
[[kernel]] void fp_qmm_n_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets(
        x,
        w,
        scales,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        tid);
  }

  fp_qmm_n_impl<T, group_size, bits, false, BM, BK, BN, WM, WN, Wtype>(
      w, scales, nullptr, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    const bool has_global_scale = false,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_t_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));

  threadgroup Wtype Ws[BN * BK_padded];

  adjust_matrix_offsets<has_global_scale>(
      x,
      w,
      scales,
      global_scale,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      tid);
  fp_qmm_t_impl<
      T,
      group_size,
      bits,
      aligned_N,
      has_global_scale,
      BM,
      BK,
      BN,
      WM,
      WN,
      Wtype>(
      w, scales, global_scale, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2,
    const bool has_global_scale = false,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_n_nax(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device T* x,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T* y,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets<has_global_scale>(
      x,
      w,
      scales,
      global_scale,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      tid);
  fp_qmm_n_impl<T, group_size, bits, false, BM, BK, BN, WM, WN, Wtype>(
      w, scales, nullptr, x, y, Xs, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    const int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose,
    bool has_global_scale = false,
    typename Wtype = bfloat>
[[kernel]] void fp_gather_qmm_rhs_nax(
    const device T* x,
    const device uint32_t* w,
    const device uint8_t* scales,
    const device float* global_scale,
    const device int32_t* offsets,
    device T* y,
    const constant int& M,
    const constant int& N,
    const constant int& K,
    const constant int& num_groups,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<8, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack();
  constexpr int BK_padded = (BK + 16 / sizeof(Wtype));
  constexpr int BN_padded = (BN + 16 / sizeof(Wtype));

  using loader_w_t = QuantizedBlockLoader<
      Wtype,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits,
      has_global_scale>;

  threadgroup Wtype Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  int y_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(
          offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {
    return;
  }
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm = short(clamp(int(tgp_bm) - tm, 0, int(SM)));
  const short sgp_sn = align_N ? SN : min(int(SN), max(0, (N - (y_col + tn))));

  const bool rows_in_bounds = y_row + tm + SM <= M;
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  const bool sg_active = sgp_sm > 0;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  const device T* xn = x + tm * K;

  // Prepare threadgroup loading operations
  thread loader_w_t loader_w(
      wl + group * stride_w,
      scales + group * stride_s,
      transpose ? K : N,
      Ws,
      simd_group_id,
      simd_lane_id,
      global_scale + group);

  dispatch_bool(rows_in_bounds, [&](auto kAlignedM) {
    dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K_it; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(
              transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<Wtype, BR, BC> Btile;

            volatile int compiler_barrier;

            if constexpr (kAlignedM.value) {
              Atile.load(xn + kk1, K);
            } else {
              Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
            }

            if constexpr (transpose) {
              Btile.template load<Wtype, BK_padded, 1>(
                  Ws + tn * BK_padded + kk1);
            } else {
              Btile.template load<Wtype, BN_padded, 1>(
                  Ws + tn + kk1 * BN_padded);
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }

        xn += BK;
        loader_w.next();
      }

      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_w.load_safe(tile_w);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<Wtype, BR, BC> Btile;

            volatile int compiler_barrier;

            const short psk = min(int(SK), max(0, (k_remain - kk1)));
            Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

            if constexpr (transpose) {
              Btile.template load<Wtype, BK_padded, 1>(
                  Ws + tn * BK_padded + kk1);
            } else {
              Btile.template load<Wtype, BN_padded, 1>(
                  Ws + tn + kk1 * BN_padded);
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Store results to device memory
      if constexpr (kAlignedN.value) {
        if (sgp_sm == SM) {
          Dtile.store(y + tm * N + tn, N);
        } else {
          Dtile.store_slice(
              y + tm * N + tn, N, short2(0, 0), short2(SN, sgp_sm));
        }
      } else {
        Dtile.store_slice(
            y + tm * N + tn, N, short2(0, 0), short2(sgp_sn, sgp_sm));
      }
    });
  });
}

// Staged fp8 qmm: x is rounded to fp8 (E4M3) with a power of two scale per row,
// w is moved to the largest scale of its row, the product is summed in float.

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 410)

// Quantize each row of x to fp8 with a power of two scale, one threadgroup per
// row. The output is stored as [M / 64][K / 64][64][64].
template <typename T>
[[kernel]] void fp_qmm_fp8_quantize_x(
    const device T* x [[buffer(0)]],
    device uint8_t* x8 [[buffer(1)]],
    device float* x_scale [[buffer(2)]],
    const constant int& K [[buffer(3)]],
    const constant int& M [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint n_simd [[simdgroups_per_threadgroup]]) {
  constexpr int BM = 64;
  constexpr int BK = 64;

  threadgroup float shared[33];

  const int row = tid.y;
  const int n_blocks = K / BK;
  const size_t tile = size_t(row / BM) * n_blocks + lid;
  device packed_uchar4* q =
      (device packed_uchar4*)(x8 + (tile * BM + row % BM) * BK);

  if (row >= M) {
    for (int i = 0; i < BK / 4; i++) {
      q[i] = uchar4(0);
    }
    return;
  }

  x += size_t(row) * K + lid * BK;

  float vals[BK];
  float block_max = 0.0f;
  for (int i = 0; i < BK; i++) {
    vals[i] = float(x[i]);
    block_max = max(block_max, fabs(vals[i]));
  }
  const float simd_row_max = simd_max(block_max);
  if (simd_lid == 0) {
    shared[simd_gid] = simd_row_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid == 0) {
    float row_max = shared[0];
    for (uint i = 1; i < n_simd; i++) {
      row_max = max(row_max, shared[i]);
    }

    // Smallest power of two sx with row_max / sx <= F8E4M3_MAX (1.75 * 2^8)
    const uint b = as_type<uint>(max(row_max, 1e-30f));
    const uint e = (b >> 23) - 8 + ((b & 0x7FFFFFu) > 0x600000u ? 1 : 0);
    shared[32] = as_type<float>((254u - e) << 23);
    x_scale[row] = as_type<float>(e << 23);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float inv = shared[32];

  for (int i = 0; i < BK / 4; i++) {
    const float4 v =
        float4(vals[4 * i], vals[4 * i + 1], vals[4 * i + 2], vals[4 * i + 3]);
    q[i] = metal::pack<metal::metal_fp8_e4m3_format>(v * inv).as_storage_type();
  }
}

// 4 fp8 codes times 2^-d, d >= 0, rounded to nearest even. Below the normal
// range the significand is shifted into the subnormals.
inline uint qmm_fp8_shift(uint w, int d) {
  const int4 c = int4(w & 255u, (w >> 8) & 255u, (w >> 16) & 255u, w >> 24);
  const int4 sign = c & 0x80;
  const int4 e = (c >> 3) & 15;
  const int4 m = c & 7;
  const int4 e1 = max(e, int4(1));
  const int4 sig = m + select(int4(0), int4(8), e > 0);
  const int4 ne = e1 - d;
  const int4 sh = clamp(d - e1 + 1, int4(0), int4(5));
  const int4 rounding = (int4(1) << sh) >> 1;
  const int4 r =
      select(sig, (sig + rounding - 1 + ((sig >> sh) & 1)) >> sh, sh > 0);
  const int4 v = sign | select(r, select(r, (ne << 3) | m, ne >= 1), sig >= 8);
  return as_type<uint>(uchar4(v));
}

// Stage 32 weights of one quantization group as fp8 at the scale of their
// row, which is d powers of two above the scale of the group
template <int bits>
inline void
qmm_fp8_stage(const device uint8_t* src, int d, threadgroup uint8_t* dst) {
  static_assert(
      bits == 4 || bits == 8, "Template undefined for bits not in {4, 8}");

  if constexpr (bits == 8) {
    uint4 wa = *(const device packed_uint4*)src;
    uint4 wb = *(const device packed_uint4*)(src + 16);

    // While every exponent field stays above 0 the shift is a subtraction in
    // that field
    const uint4 top = uint4(0x80808080u);
    const uint4 thr = uint4(uint(d + 1) * 0x08080808u);
    const bool plain = d < 15 &&
        all((((wa & 0x78787878u) + top - thr) & top) == top) &&
        all((((wb & 0x78787878u) + top - thr) & top) == top);
    if (plain || d == 0) {
      wa -= uint4(uint(d) * 0x08080808u);
      wb -= uint4(uint(d) * 0x08080808u);
    } else {
      for (int i = 0; i < 4; i++) {
        wa[i] = qmm_fp8_shift(wa[i], d);
        wb[i] = qmm_fp8_shift(wb[i], d);
      }
    }

    *(threadgroup packed_uint4*)dst = wa;
    *(threadgroup packed_uint4*)(dst + 16) = wb;
  } else {
    // Decode as fp4_e2m1 does, the scale is exact
    const float scale = 16384.0f * as_type<float>(uint(127 - min(d, 40)) << 23);

    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < 8; i++) {
      const uint p = *(const device uint16_t*)(src + 2 * i);
      const uint4 c = uint4(p & 15u, (p >> 4) & 15u, (p >> 8) & 15u, p >> 12);
      const float4 mag = float4(as_type<half4>(ushort4((c & 7u) << 9))) * scale;
      const float4 v = select(mag, -mag, (c & 8u) != 0u);
      *(threadgroup packed_uchar4*)(dst + 4 * i) =
          metal::pack<metal::metal_fp8_e4m3_format>(v).as_storage_type();
    }
  }
}

// x @ w.T in tiles of 64 by 64 with one fp8 matmul per simdgroup, the fp8
// operands are read from memory. With aligned, M and N are multiples of 64.
template <typename T, const int bits, const bool aligned>
[[kernel]] void fp_qmm_t_nax_fp8(
    const device uint32_t* w [[buffer(0)]],
    const device uint8_t* scales [[buffer(1)]],
    device uint8_t* x8 [[buffer(2)]],
    const device float* x_scale [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K_ [[buffer(5)]],
    const constant int& N_ [[buffer(6)]],
    const constant int& M_ [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 64;
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int group_size = 32;

  // Each thread stages VPT weights of one row, all in one quantization group
  constexpr int VPT = (BN * BK) / (WM * WN * SIMD_SIZE);
  static_assert(
      VPT == group_size && BK == 2 * VPT,
      "Two threads stage a row, one quantization group of the weights each");

  constexpr int SM = BM / WM;
  constexpr int SN = BN / WN;

  typedef metal::metal_fp8_e4m3_format fp8_t;
  typedef metal::dextents<int32_t, 2> extents_t;
  typedef metal::tensor<device fp8_t, extents_t, metal::tensor_inline> x_t;
  typedef metal::tensor<threadgroup fp8_t, extents_t, metal::tensor_inline> w_t;

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      SM,
      SN,
      BK,
      false,
      true,
      true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

  threadgroup uint Ws_storage[(BN * BK) / 4];
  threadgroup uint8_t* Ws = (threadgroup uint8_t*)Ws_storage;
  threadgroup uint8_t Ss[2 * BN];
  threadgroup float Wscales[BN];

  const int K = K_;
  const int N = N_;
  const int M = M_;
  (void)M;

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const int tm = SM * (simd_gid / WN);
  const int tn = SN * (simd_gid % WN);

  // The part of the weight tile this thread stages
  const int nl = lid / 2;
  const int kl = (lid % 2) * VPT;
  const int n = y_col + nl;
  const bool valid = aligned || n < N;

  threadgroup uint8_t* dst = Ws + nl * BK + kl;
  const device uint8_t* wl = (const device uint8_t*)w;
  if (valid) {
    wl += (size_t(n) * K + kl) * bits / 8;
    scales += size_t(n) * (K / group_size);
  } else {
    for (int i = 0; i < VPT / 4; i++) {
      *(threadgroup uint*)(dst + 4 * i) = 0;
    }
  }

  // The largest scale of a row of the weights, each of its two threads
  // looks at half of the row
  {
    uint8_t s = 0;
    if (valid) {
      const int n_scales = K / BK;
      const device uint8_t* sp = scales + (lid % 2) * n_scales;
      uchar4 s4 = uchar4(0);
      int g = 0;
      for (; g + 4 <= n_scales; g += 4) {
        s4 = max(s4, uchar4(*(const device packed_uchar4*)(sp + g)));
      }
      s = max(max(s4.x, s4.y), max(s4.z, s4.w));
      for (; g < n_scales; g++) {
        s = max(s, sp[g]);
      }
    }
    Ss[lid] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int row_scale = max(Ss[lid], Ss[lid ^ 1]);
  if (kl == 0) {
    Wscales[nl] = dequantize_scale<float, 32>(uint8_t(row_scale));
  }
  // The scales this thread stages with
  scales += kl / group_size;

  // x8 has one BM x BK block per row tile and BK columns of K
  x8 += (size_t(tid.y) * (K / BK) * BM + tm) * BK;

  w_t Wtile(Ws + tn * BK, extents_t(BK, SN));
  auto Dtile =
      gemm_op.template get_destination_cooperative_tensor<x_t, w_t, float>();
  for (ushort i = 0; i < Dtile.get_capacity(); i++) {
    Dtile[i] = 0.0f;
  }

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
      qmm_fp8_stage<bits>(
          wl + k * bits / 8, row_scale - int(scales[k / group_size]), dst);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    x_t Xtile(x8, extents_t(BK, SM));
    gemm_op.run(Xtile, Wtile, Dtile);

    x8 += BM * BK;
  }

  // A 64-bit index from the corner of the tile, a 32-bit one is slower
  device T* y_tile = y + (size_t(y_row + tm) * size_t(N) + size_t(y_col + tn));

  // A test on the stores of a cooperative tensor is slow, even one for the
  // whole tile, so there is none when the output has whole tiles only
  if constexpr (aligned) {
    for (ushort i = 0; i < Dtile.get_capacity(); i++) {
      if (Dtile.is_valid_element(i)) {
        const auto pos = Dtile.get_multidimensional_index(i);
        const int r = y_row + tm + int(pos[1]);
        const int c = tn + int(pos[0]);
        y_tile[size_t(pos[1]) * size_t(N) + size_t(pos[0])] =
            static_cast<T>(Dtile[i] * x_scale[r] * Wscales[c]);
      }
    }
  } else {
    const bool interior = (y_row + tm + SM <= M) && (y_col + tn + SN <= N);
    dispatch_bool(interior, [&](auto kInterior) {
      for (ushort i = 0; i < Dtile.get_capacity(); i++) {
        if (!Dtile.is_valid_element(i)) {
          continue;
        }
        const auto pos = Dtile.get_multidimensional_index(i);
        const int r = y_row + tm + int(pos[1]);
        const int c = tn + int(pos[0]);
        if constexpr (kInterior.value) {
          y_tile[size_t(pos[1]) * size_t(N) + size_t(pos[0])] =
              static_cast<T>(Dtile[i] * x_scale[r] * Wscales[c]);
        } else if (r < M && y_col + c < N) {
          y_tile[size_t(pos[1]) * size_t(N) + size_t(pos[0])] =
              static_cast<T>(Dtile[i] * x_scale[r] * Wscales[c]);
        }
      }
    });
  }
}

#endif // __METAL_VERSION__ >= 410
