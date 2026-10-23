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
// Speed lane step 2.3 (diagnostic branch nvfp4-gather-qmm-i8-unpack-diag, variant 2, measured byte-identical there):
// each staging thread loads 32 weights as one uint4, unpacks them through a 256-entry byte table (one read per byte
// gives both values) and stores them as two uint4. The staged int8 values are the same.

#pragma once

#include "mlx/backend/metal/kernels/fp_quantized_i8_utils.h"

// Speed lane step 2.4 diagnostic (branch nvfp4-gather-qmm-i8-rescale-diag, A8-KERNEL.md): the block product as one
// matmul2d per fragment pair, as tile_matmad_nax does for TN % 2 == 0, with MULT selecting mode::multiply (the op
// writes P = A * B, so no cleared tile and no copy of C into the op) instead of multiply_accumulate on a cleared tile.
// Integer products and sums either way, so P is the same.
template <bool MULT, class CTile, class ATile, class BTile>
METAL_FUNC void i8_block_matmul_nax(thread CTile& C, thread ATile& A, thread BTile& B) {
  constexpr short TM = CTile::kTileRows;
  constexpr short TN = CTile::kTileCols;
  constexpr short EPF = CTile::kElemsPerFrag;
  static_assert(TN % 2 == 0 && ATile::kTileRows == TM && ATile::kTileCols == 1 && BTile::kTileRows == TN &&
                BTile::kTileCols == 1, "one K16 block, B transposed, column fragments in pairs");
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, true, true,
      MULT ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply
           : mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  if constexpr (!MULT) {
    C.clear();
  }
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < TM; ++mm) {
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < TN; nn += 2) {
      mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
      auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
      auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
      auto ct_c = op.template get_destination_cooperative_tensor<
          metal::remove_addrspace_t<decltype(ct_a)>, metal::remove_addrspace_t<decltype(ct_b)>, int32_t>();
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < EPF; i++) {
        ct_a[i] = A.frag_at(mm, 0)[i];
        ct_b[i] = B.frag_at(nn, 0)[i];
        ct_b[EPF + i] = B.frag_at(nn + 1, 0)[i];
      }
      if constexpr (!MULT) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < EPF; i++) {
          ct_c[i] = C.frag_at(mm, nn)[i];
          ct_c[EPF + i] = C.frag_at(mm, nn + 1)[i];
        }
      }
      op.run(ct_a, ct_b, ct_c);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < EPF; i++) {
        C.frag_at(mm, nn)[i] = ct_c[i];
        C.frag_at(mm, nn + 1)[i] = ct_c[EPF + i];
      }
    }
  }
}

// RS (diagnostic): 0 the gated kernel; 1 mode::multiply; 2 block loop unrolled; 4 two-stage pipeline (block kb + 1's
// product issued before block kb's rescale; implies 2); 8 no rescale (int32 sum over all K: wrong numerics, timing
// bound). 1, 2 and 4 keep every fp32 multiply and add, in the same order per output element.
template <int BM, int BN, int BK, int WM, int WN, int RS>
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
  static_assert(TPR >= 1 && BK % TPR == 0 && VPT % 32 == 0, "a thread stages whole 16-byte weight chunks");
  static_assert(BK_padded % 16 == 0, "16-byte aligned weight rows in threadgroup memory");

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
  NAXTile<int32_t, TM, TN> Itile;   // RS 8 only: one int32 sum over all K (cleared once)
  Itile.clear();

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      // 32 weights = 16 bytes = one uint4 (16-byte aligned: K % 64 == 0, k and kl multiples of 32, host checks w)
      const device uint4* p = (const device uint4*)(wrow + (k + kl) / 2);
      STEEL_PRAGMA_UNROLL
      for (int c = 0; c < VPT / 32; c++) {
        const uint4 wv = p[c];
        uint o[8];
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 4; i++) {
          o[2 * i] = uint(i8_e2m1_x2_lut2[wv[i] & 0xFF]) | (uint(i8_e2m1_x2_lut2[(wv[i] >> 8) & 0xFF]) << 16);
          o[2 * i + 1] = uint(i8_e2m1_x2_lut2[(wv[i] >> 16) & 0xFF]) | (uint(i8_e2m1_x2_lut2[wv[i] >> 24]) << 16);
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
      constexpr bool MULT = RS & 1;
      constexpr bool PIPE = RS & 4;
      constexpr bool NORESCALE = RS & 8;
      static_assert(!(NORESCALE && (RS & 7)) && NB % 2 == 0, "diagnostic variants");
      const short fn = BaseNAXFrag::get_coord(0).x;
      // Block kb's product into P (A from device memory, B from the staged tile), as in the gated kernel
      auto block_product = [&](thread NAXTile<int32_t, TM, TN>& P, const int kb) {
        NAXTile<int8_t, TM, 1> Atile;
        NAXTile<int8_t, TN, 1> Btile;
        if (sgp_sm == SM) {
          Atile.load(xn + k + kb * 16, K);
        } else {
          Atile.load_safe(xn + k + kb * 16, K, short2(16, sgp_sm));
        }
        Btile.template load<int8_t, BK_padded, 1>(Ws + tn * BK_padded + kb * 16);
        if constexpr (RS == 0) {   // exactly the gated kernel's calls
          P.clear();
          tile_matmad_nax(P, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
        } else {
          i8_block_matmul_nax<MULT>(P, Atile, Btile);
        }
      };
      // A thread's fragment elements e and e + 4 share column fn + e % 4 (BaseNAXFrag::get_coord): one float4 of
      // scales per fragment column group. Each df[e] gets the same product as a per-element scale read would give.
      auto rescale = [&](thread const NAXTile<int32_t, TM, TN>& P, const int kb) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < TN; j++) {
          const float4 sc = *(const threadgroup float4*)(Ss + kb * BN + tn + j * BaseNAXFrag::kFragCols + fn);
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < TM; i++) {
            const auto pf = P.frag_at(i, j);
            thread auto& df = Dtile.frag_at(i, j);
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < BaseNAXFrag::kElemsPerFrag; e++) {
              df[e] += float(pf[e]) * sc[e % 4];
            }
          }
        }
      };
      if constexpr (NORESCALE) {
        STEEL_PRAGMA_NO_UNROLL
        for (int kb = 0; kb < NB; kb++) {
          NAXTile<int8_t, TM, 1> Atile;
          NAXTile<int8_t, TN, 1> Btile;
          volatile int compiler_barrier;
          if (sgp_sm == SM) {
            Atile.load(xn + k + kb * 16, K);
          } else {
            Atile.load_safe(xn + k + kb * 16, K, short2(16, sgp_sm));
          }
          Btile.template load<int8_t, BK_padded, 1>(Ws + tn * BK_padded + kb * 16);
          i8_block_matmul_nax<false>(Itile, Atile, Btile);   // (does not clear: see Itile's note)
          (void)compiler_barrier;
        }
      } else if constexpr (PIPE) {
        NAXTile<int32_t, TM, TN> P0, P1;
        block_product(P0, 0);
        STEEL_PRAGMA_UNROLL
        for (int kb = 0; kb < NB; kb += 2) {
          volatile int compiler_barrier;
          block_product(P1, kb + 1);
          rescale(P0, kb);
          if (kb + 2 < NB) {
            block_product(P0, kb + 2);
          }
          rescale(P1, kb + 1);
          (void)compiler_barrier;
        }
      } else if constexpr (RS & 2) {
        STEEL_PRAGMA_UNROLL
        for (int kb = 0; kb < NB; kb++) {
          NAXTile<int32_t, TM, TN> Ptile;
          volatile int compiler_barrier;
          block_product(Ptile, kb);
          rescale(Ptile, kb);
          (void)compiler_barrier;
        }
      } else {
        STEEL_PRAGMA_NO_UNROLL
        for (int kb = 0; kb < NB; kb++) {
          NAXTile<int32_t, TM, TN> Ptile;
          volatile int compiler_barrier;
          block_product(Ptile, kb);
          rescale(Ptile, kb);
          (void)compiler_barrier;
        }
      }
    }
  }

  if constexpr (RS & 8) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < TM; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < TN; j++) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < BaseNAXFrag::kElemsPerFrag; e++) {
          Dtile.frag_at(i, j)[e] = float(Itile.frag_at(i, j)[e]);
        }
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
