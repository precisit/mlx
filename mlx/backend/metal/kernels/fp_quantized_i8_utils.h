// Copyright © 2026 Precisit. Helpers shared by fp_quantized_i8.h and fp_quantized_nax_i8.h (no kernels here,
// so both .metal files can include it without duplicate symbols).
#pragma once

// 2 * e2m1 magnitude for the 3 low bits of a code: 0, 1, 2, 3, 4, 6, 8, 12.
METAL_FUNC int i8_e2m1_x2_mag(uint8_t c) {
  c &= 7;
  return c < 4 ? int(c) : ((4 << ((c - 4) >> 1)) * ((c & 1) ? 3 : 2)) / 2;
}

METAL_FUNC int i8_e2m1_x2(uint8_t code) {
  const int v = i8_e2m1_x2_mag(code);
  return (code & 8) ? -v : v;
}

METAL_FUNC float i8_e4m3_to_float(uint8_t s) {
  return float(*(thread fp8_e4m3*)(&s));
}
