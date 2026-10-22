// Copyright © 2026 Precisit. int8-activation x NVFP4-weight sorted gathered matmul on NAX (M5).
//
// Same arithmetic as fp_quantized_i8.h: y = x_scale/2 * sum_b e4m3(S[n,b]) * (sum_{k in b} x_q * 2*e2m1(w)).
// The NAX matmul2d fragment is M16 x N32 x K16, so one K16 step is exactly one NVFP4 block: each block's
// int8 x int8 -> int32 product goes into a cleared int32 tile, is converted to fp32, scaled per output
// column by its E4M3 block scale and added to the fp32 accumulator. Row scheduling (one expert per row
// tile, sorted assignments) is stock MLX's schedule_row_tile, as in fp_gather_qmm_rhs_nax.
//
// Speed (modelopt-to-mlx stage2/A8-KERNEL.md, speed lane step 2.2; variant 3 of the diagnostic branch
// nvfp4-gather-qmm-i8-diag, measured bit-identical there): weights unpack through a 16-entry table, and block
// scales are staged [NB][BN] so that each fragment column group reads its four scales as one float4. Both
// change only how values are fetched: every product and every addition is the same, in the same order.

#pragma once

#include "mlx/backend/metal/kernels/fp_quantized_i8_utils.h"

constant constexpr int8_t i8_e2m1_x2_lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// Unpack variants U (speed lane step 2.3; MLX_GATHER_QMM_I8_NAX_UNPACK) change only how the int8 weight tile is
// staged: 0 as scored (one byte load and two 16-entry table reads per weight byte, byte stores); 1 one 16-byte load
// per 32 weights, SWAR decode, 16-byte stores; 2 as 1 with a 256-entry byte table; 8 as 1 without the decode
// (timing bound only, wrong numerics). The staged int8 values, and so every product and sum, are the same in 0-2.
template <int BM, int BN, int BK, int WM, int WN, int U>
[[kernel]] void fp_gather_qmm_rhs_nax_i8(
    const device int8_t* x [[buffer(0)]],
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
  constexpr int NB = BK / 16;                             // NVFP4 blocks per K tile
  constexpr int TPR = (WM * WN * SIMD_SIZE) / BN;         // staging threads per weight row
  constexpr int VPT = BK / TPR;                           // weight values staged per thread
  static_assert(TPR >= 1 && BK % TPR == 0 && VPT % 16 == 0, "a thread stages whole NVFP4 blocks");
  static_assert(U == 0 || (VPT % 32 == 0 && BK_padded % 16 == 0), "wide staging: whole 16-byte loads and stores");

  alignas(16) threadgroup int8_t Ws[BN * BK_padded];
  alignas(16) threadgroup float Ss[NB * BN];   // [NB][BN]: block kb's scales for the tile's BN columns

  int y_row;
  int group;
  short tgp_bm;
  if (!schedule_row_tile<BM>(offsets, num_groups, M, tid.y, simd_lane_id, y_row, group, tgp_bm)) {
    return;
  }
  const int y_col = tid.x * BN;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short sgp_sm = short(clamp(int(tgp_bm) - tm, 0, int(SM)));
  const bool sg_active = sgp_sm > 0;

  // The part of the weight tile this thread stages (N % BN == 0 and K % BK == 0 by the host's routing)
  const int nl = lid / TPR;
  const int kl = (lid % TPR) * VPT;
  const device uint8_t* wrow = (const device uint8_t*)w + (size_t(group) * N + y_col + nl) * size_t(K / 2);
  const device uint8_t* srow = scales + (size_t(group) * N + y_col + nl) * size_t(K / 16);
  threadgroup int8_t* wdst = Ws + nl * BK_padded + kl;

  const device int8_t* xn = x + size_t(y_row + tm) * K;

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      const device uint8_t* p = wrow + (k + kl) / 2;
      if (U == 0) {
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < VPT / 2; j++) {
          const uint8_t b = p[j];
          wdst[2 * j] = i8_e2m1_x2_lut[b & 0x0f];
          wdst[2 * j + 1] = i8_e2m1_x2_lut[b >> 4];
        }
      } else {
        // 32 weights = 16 bytes = one uint4 (16-byte aligned: K % 64 == 0 and k, kl multiples of 32)
        STEEL_PRAGMA_UNROLL
        for (int c = 0; c < VPT / 32; c++) {
          const uint4 wv = ((const device uint4*)p)[c];
          uint o[8];
          STEEL_PRAGMA_UNROLL
          for (int i = 0; i < 4; i++) {
            if (U == 2) {
              o[2 * i] = uint(i8_e2m1_x2_lut2[wv[i] & 0xFF]) | (uint(i8_e2m1_x2_lut2[(wv[i] >> 8) & 0xFF]) << 16);
              o[2 * i + 1] = uint(i8_e2m1_x2_lut2[(wv[i] >> 16) & 0xFF]) | (uint(i8_e2m1_x2_lut2[wv[i] >> 24]) << 16);
            } else {
              o[2 * i] = i8_spread_nibbles(wv[i] & 0xFFFF);
              o[2 * i + 1] = i8_spread_nibbles(wv[i] >> 16);
              if (U == 1) {
                o[2 * i] = i8_e2m1_x2_swar(o[2 * i]);
                o[2 * i + 1] = i8_e2m1_x2_swar(o[2 * i + 1]);
              }
            }
          }
          threadgroup uint4* d4 = (threadgroup uint4*)(wdst + 32 * c);
          d4[0] = uint4(o[0], o[1], o[2], o[3]);
          d4[1] = uint4(o[4], o[5], o[6], o[7]);
        }
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
        NAXTile<int8_t, TM, 1> Atile;
        NAXTile<int8_t, TN, 1> Btile;
        NAXTile<int32_t, TM, TN> Ptile;
        Ptile.clear();

        volatile int compiler_barrier;

        if (sgp_sm == SM) {
          Atile.load(xn + k + kb * 16, K);
        } else {
          Atile.load_safe(xn + k + kb * 16, K, short2(16, sgp_sm));
        }
        Btile.template load<int8_t, BK_padded, 1>(Ws + tn * BK_padded + kb * 16);

        tile_matmad_nax(
            Ptile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});

        // A thread's fragment elements e and e + 4 share column fn + e % 4 (BaseNAXFrag::get_coord): one float4 of
        // scales per fragment column group. Each df[e] gets the same product as a per-element scale read would give.
        const short fn = BaseNAXFrag::get_coord(0).x;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j++) {
          const float4 sc = *(const threadgroup float4*)(Ss + kb * BN + tn + j * BaseNAXFrag::kFragCols + fn);
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < TM; i++) {
            const auto pf = Ptile.frag_at(i, j);
            thread auto& df = Dtile.frag_at(i, j);
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < BaseNAXFrag::kElemsPerFrag; e++) {
              df[e] += float(pf[e]) * sc[e % 4];
            }
          }
        }
        (void)compiler_barrier;
      }
    }
  }

  if (!sg_active) {
    return;
  }
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < TM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < TN; j++) {
      const auto df = Dtile.frag_at(i, j);
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < BaseNAXFrag::kElemsPerFrag; e++) {
        const short2 pos = BaseNAXFrag::get_coord(e);
        const int r_local = tm + i * BaseNAXFrag::kFragRows + pos.y;
        if (r_local < tgp_bm) {
          const int row = y_row + r_local;
          const int c = y_col + tn + j * BaseNAXFrag::kFragCols + pos.x;
          y[size_t(row) * N + c] = df[e] * x_scale[row] * 0.5f;
        }
      }
    }
  }
}
