use std::f32::consts::{FRAC_PI_2, LN_2, LOG2_E, PI};

use cranelift::prelude::*;
use cranexpr_ast::BinOp;

use crate::context::{FunctionCx, VEC_TYPE};

pub(crate) fn splat_f32(fx: &mut FunctionCx<'_, '_>, val: f32) -> Value {
  let scalar = fx.bcx.ins().f32const(val);
  fx.bcx.ins().splat(VEC_TYPE, scalar)
}

/// Splats an `i32` constant across an `I32X4` vector.
pub(crate) fn splat_i32(fx: &mut FunctionCx<'_, '_>, val: i64) -> Value {
  let scalar = fx.bcx.ins().iconst(types::I32, val);
  fx.bcx.ins().splat(types::I32X4, scalar)
}

/// Converts an `I32X4` of zero-extended IEEE-754 binary16 bit patterns into an
/// `F32X4`.
///
/// Based on <https://gist.github.com/rygorous/2156668>
pub(crate) fn half_to_float_simd(fx: &mut FunctionCx<'_, '_>, h: Value) -> Value {
  let no_flags = MemFlagsData::new();

  let mant_exp_mask = splat_i32(fx, 0x7fff);
  let mant_exp = fx.bcx.ins().band(h, mant_exp_mask);
  let shifted = fx.bcx.ins().ishl_imm_s(mant_exp, 13);
  let as_f = fx.bcx.ins().bitcast(VEC_TYPE, no_flags, shifted);

  let magic = splat_f32(fx, f32::from_bits(0x7780_0000));
  let scaled = fx.bcx.ins().fmul(as_f, magic);

  let was_infnan = splat_f32(fx, f32::from_bits(0x4780_0000));
  let infnan_mask = fx
    .bcx
    .ins()
    .fcmp(FloatCC::GreaterThanOrEqual, scaled, was_infnan);
  let scaled_bits = fx.bcx.ins().bitcast(types::I32X4, no_flags, scaled);
  let inf_exp = splat_i32(fx, 255 << 23);
  let with_inf = fx.bcx.ins().bor(scaled_bits, inf_exp);
  let merged = fx.bcx.ins().bitselect(infnan_mask, with_inf, scaled_bits);

  let sign_mask = splat_i32(fx, 0x8000);
  let sign = fx.bcx.ins().band(h, sign_mask);
  let sign = fx.bcx.ins().ishl_imm_s(sign, 16);
  let result = fx.bcx.ins().bor(merged, sign);
  fx.bcx.ins().bitcast(VEC_TYPE, no_flags, result)
}

/// Converts an `F32X4` into an `I32X4` of IEEE-754 binary16 bit patterns, held
/// in the low 16 bits of each lane.
///
/// Based on <https://gist.github.com/rygorous/2156668>
pub(crate) fn float_to_half_simd(fx: &mut FunctionCx<'_, '_>, f: Value) -> Value {
  let no_flags = MemFlagsData::new();
  let f_bits = fx.bcx.ins().bitcast(types::I32X4, no_flags, f);

  let sign_mask = splat_i32(fx, i64::from(0x8000_0000_u32 as i32));
  let sign = fx.bcx.ins().band(f_bits, sign_mask);
  let abs_mask = splat_i32(fx, 0x7fff_ffff);
  let abs = fx.bcx.ins().band(f_bits, abs_mask);

  // Inf/NaN: NaN -> 0x7e00, Inf -> 0x7c00.
  let f16max = splat_i32(fx, 0x4780_0000);
  let is_infnan = fx
    .bcx
    .ins()
    .icmp(IntCC::UnsignedGreaterThanOrEqual, abs, f16max);
  let f32infty = splat_i32(fx, 0x7f80_0000);
  let is_nan = fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, abs, f32infty);
  let qnan = splat_i32(fx, 0x7e00);
  let inf = splat_i32(fx, 0x7c00);
  let infnan_o = fx.bcx.ins().bitselect(is_nan, qnan, inf);

  // Subnormal/zero: align the mantissa with a magic add, then subtract bias.
  let sub_thresh = splat_i32(fx, 113 << 23);
  let is_subnormal = fx.bcx.ins().icmp(IntCC::UnsignedLessThan, abs, sub_thresh);
  let abs_f = fx.bcx.ins().bitcast(VEC_TYPE, no_flags, abs);
  let denorm_magic = splat_f32(fx, f32::from_bits(0x3f00_0000));
  let sub_f = fx.bcx.ins().fadd(abs_f, denorm_magic);
  let sub_bits = fx.bcx.ins().bitcast(types::I32X4, no_flags, sub_f);
  let denorm_magic_bits = splat_i32(fx, 0x3f00_0000);
  let sub_o = fx.bcx.ins().isub(sub_bits, denorm_magic_bits);

  // Normalized: bias the exponent and round to nearest even.
  let shifted13 = fx.bcx.ins().ushr_imm_s(abs, 13);
  let one = splat_i32(fx, 1);
  let mant_odd = fx.bcx.ins().band(shifted13, one);
  let bias = splat_i32(fx, i64::from(0xc800_0fff_u32 as i32));
  let norm = fx.bcx.ins().iadd(abs, bias);
  let norm = fx.bcx.ins().iadd(norm, mant_odd);
  let norm_o = fx.bcx.ins().ushr_imm_s(norm, 13);

  let o = fx.bcx.ins().bitselect(is_subnormal, sub_o, norm_o);
  let o = fx.bcx.ins().bitselect(is_infnan, infnan_o, o);
  let sign_h = fx.bcx.ins().ushr_imm_s(sign, 16);
  fx.bcx.ins().bor(o, sign_h)
}

/// Converts a vector boolean mask to F32X4, where true lanes become 1.0 and
/// false lanes become 0.0.
pub(crate) fn bool_to_float_simd(fx: &mut FunctionCx<'_, '_>, mask: Value) -> Value {
  let one_bits = fx.bcx.ins().iconst(types::I32, i64::from(1.0f32.to_bits()));
  let one_bits_splat = fx.bcx.ins().splat(types::I32X4, one_bits);
  let anded = fx.bcx.ins().band(mask, one_bits_splat);
  fx.bcx.ins().bitcast(VEC_TYPE, MemFlagsData::new(), anded)
}

/// Picks lanes from `true_val` where `mask` is all-1s, and from `false_val`
/// where `mask` is all-0s.
pub(crate) fn vselect_f32x4(
  fx: &mut FunctionCx<'_, '_>,
  mask: Value,
  true_val: Value,
  false_val: Value,
) -> Value {
  let t_bits = fx
    .bcx
    .ins()
    .bitcast(types::I32X4, MemFlagsData::new(), true_val);
  let f_bits = fx
    .bcx
    .ins()
    .bitcast(types::I32X4, MemFlagsData::new(), false_val);
  let selected = fx.bcx.ins().bitselect(mask, t_bits, f_bits);
  fx.bcx
    .ins()
    .bitcast(VEC_TYPE, MemFlagsData::new(), selected)
}

/// Vectorized maximum.
pub(crate) fn fmax_simd(fx: &mut FunctionCx<'_, '_>, lhs: Value, rhs: Value) -> Value {
  // Benchmarked as faster than Cranelift's `fmax` since we shouldn't need its
  // full IEEE-754 semantics.
  let cmp = fx.bcx.ins().fcmp(FloatCC::LessThan, rhs, lhs);
  let mask = fx.bcx.ins().bitcast(VEC_TYPE, MemFlagsData::new(), cmp);
  fx.bcx.ins().bitselect(mask, lhs, rhs)
}

/// Vectorized minimum.
pub(crate) fn fmin_simd(fx: &mut FunctionCx<'_, '_>, lhs: Value, rhs: Value) -> Value {
  // Benchmarked as faster than Cranelift's `fmin` since we shouldn't need its
  // full IEEE-754 semantics.
  let cmp = fx.bcx.ins().fcmp(FloatCC::LessThan, lhs, rhs);
  let mask = fx.bcx.ins().bitcast(VEC_TYPE, MemFlagsData::new(), cmp);
  fx.bcx.ins().bitselect(mask, lhs, rhs)
}

/// Returns a value with the magnitude of `mag` and the sign of `sig`.
fn copysign_simd(fx: &mut FunctionCx<'_, '_>, mag: Value, sig: Value) -> Value {
  let sign_mask = fx
    .bcx
    .ins()
    .iconst(types::I32, i64::from(0x8000_0000_u32 as i32));
  let sign_mask = fx.bcx.ins().splat(types::I32X4, sign_mask);
  let mag_bits = fx.bcx.ins().bitcast(types::I32X4, MemFlagsData::new(), mag);
  let sig_bits = fx.bcx.ins().bitcast(types::I32X4, MemFlagsData::new(), sig);
  let not_sign_mask = fx.bcx.ins().bnot(sign_mask);
  let mag_abs = fx.bcx.ins().band(mag_bits, not_sign_mask);
  let sig_sign = fx.bcx.ins().band(sig_bits, sign_mask);
  let combined = fx.bcx.ins().bor(mag_abs, sig_sign);
  fx.bcx
    .ins()
    .bitcast(VEC_TYPE, MemFlagsData::new(), combined)
}

#[derive(Clone, Copy)]
pub(crate) enum SinCos {
  Sin,
  Cos,
  Tan,
}

/// Vectorized minimax polynomial approximation of `sin(x)`, `cos(x)`, or
/// `tan(x)` over F32X4.
///
/// Refer:
///   Software Manual for the Elementary Functions
///   Cody, W.J. and Waite, W.M.C.
pub(crate) fn sin_cos_simd(fx: &mut FunctionCx<'_, '_>, x: Value, mode: SinCos) -> Value {
  let vf32_from_bits = |fx: &mut FunctionCx<'_, '_>, bits: u32| -> Value {
    let scalar = fx.bcx.ins().iconst(types::I32, i64::from(bits));
    let splat = fx.bcx.ins().splat(types::I32X4, scalar);
    fx.bcx.ins().bitcast(VEC_TYPE, MemFlagsData::new(), splat)
  };
  let vi32_from_bits = |fx: &mut FunctionCx<'_, '_>, bits: u32| -> Value {
    let scalar = fx.bcx.ins().iconst(types::I32, i64::from(bits));
    fx.bcx.ins().splat(types::I32X4, scalar)
  };

  let absmask = vi32_from_bits(fx, 0x7fff_ffff);
  let float_invpi = vf32_from_bits(fx, 0x3ea2_f983);
  let float_pi1 = vf32_from_bits(fx, 0x4049_0000);
  let float_pi2 = vf32_from_bits(fx, 0x3a7d_a000);
  let float_pi3 = vf32_from_bits(fx, 0x3422_2000);
  let float_pi4 = vf32_from_bits(fx, 0x2cb4_611a);

  // Range-reduce |x| to [-pi/2, pi/2]:
  //   n = round(|x|/pi)
  //   reduced = |x| - n*pi
  let abs_x = fx.bcx.ins().fabs(x);
  let n_float = fx.bcx.ins().fmul(abs_x, float_invpi);
  let n_rounded = fx.bcx.ins().nearest(n_float);
  let n_int = fx.bcx.ins().fcvt_to_sint_sat(types::I32X4, n_rounded);

  let odd_n_sign = fx.bcx.ins().ishl_imm_s(n_int, 31);

  let sign = match mode {
    SinCos::Sin => {
      let x_bits = fx.bcx.ins().bitcast(types::I32X4, MemFlagsData::new(), x);
      let not_absmask = fx.bcx.ins().bnot(absmask);
      let input_sign = fx.bcx.ins().band(x_bits, not_absmask);
      fx.bcx.ins().bxor(input_sign, odd_n_sign)
    }
    SinCos::Cos => odd_n_sign,
    SinCos::Tan => {
      let x_bits = fx.bcx.ins().bitcast(types::I32X4, MemFlagsData::new(), x);
      let not_absmask = fx.bcx.ins().bnot(absmask);
      fx.bcx.ins().band(x_bits, not_absmask)
    }
  };
  let n_float = fx.bcx.ins().fcvt_from_sint(VEC_TYPE, n_int);

  // Subtract n*pi in four increasing-precision steps.
  let neg_pi1 = fx.bcx.ins().fneg(float_pi1);
  let neg_pi2 = fx.bcx.ins().fneg(float_pi2);
  let neg_pi3 = fx.bcx.ins().fneg(float_pi3);
  let neg_pi4 = fx.bcx.ins().fneg(float_pi4);
  let reduced = fx.bcx.ins().fma(n_float, neg_pi1, abs_x);
  let reduced = fx.bcx.ins().fma(n_float, neg_pi2, reduced);
  let reduced = fx.bcx.ins().fma(n_float, neg_pi3, reduced);
  let reduced = fx.bcx.ins().fma(n_float, neg_pi4, reduced);

  let sin_poly = |fx: &mut FunctionCx<'_, '_>, reduced: Value| -> Value {
    // Minimax polynomial for sin(x) in [-pi/2, pi/2]:
    //   x + x * x^2 * (C3 + x^2 * (C5 + x^2 * (C7 + x^2 * C9)))
    let float_sinc3 = vf32_from_bits(fx, 0xbe2a_aaa6);
    let float_sinc5 = vf32_from_bits(fx, 0x3c08_876a);
    let float_sinc7 = vf32_from_bits(fx, 0xb94f_b7ff);
    let float_sinc9 = vf32_from_bits(fx, 0x362e_def8);
    let xsq = fx.bcx.ins().fmul(reduced, reduced);
    let poly = fx.bcx.ins().fma(xsq, float_sinc9, float_sinc7);
    let poly = fx.bcx.ins().fma(poly, xsq, float_sinc5);
    let poly = fx.bcx.ins().fma(poly, xsq, float_sinc3);
    let poly = fx.bcx.ins().fmul(poly, xsq);
    let poly = fx.bcx.ins().fmul(poly, reduced);
    fx.bcx.ins().fadd(reduced, poly)
  };
  let cos_poly = |fx: &mut FunctionCx<'_, '_>, reduced: Value| -> Value {
    // Minimax polynomial for cos(x) in [-pi/2, pi/2]:
    //   1 + x^2 * (C2 + x^2 * (C4 + x^2 * (C6 + x^2 * C8)))
    let float_cosc2 = vf32_from_bits(fx, 0xbeff_ffe2);
    let float_cosc4 = vf32_from_bits(fx, 0x3d2a_a73c);
    let float_cosc6 = vf32_from_bits(fx, 0xbab5_8d50);
    let float_cosc8 = vf32_from_bits(fx, 0x37c1_ad76);
    let xsq = fx.bcx.ins().fmul(reduced, reduced);
    let one = splat_f32(fx, 1.0);
    let poly = fx.bcx.ins().fma(xsq, float_cosc8, float_cosc6);
    let poly = fx.bcx.ins().fma(poly, xsq, float_cosc4);
    let poly = fx.bcx.ins().fma(poly, xsq, float_cosc2);
    fx.bcx.ins().fma(poly, xsq, one)
  };

  let result = match mode {
    SinCos::Sin => sin_poly(fx, reduced),
    SinCos::Cos => cos_poly(fx, reduced),
    SinCos::Tan => {
      let s = sin_poly(fx, reduced);
      let c = cos_poly(fx, reduced);
      fx.bcx.ins().fdiv(s, c)
    }
  };

  // Apply the accumulated sign bit.
  let result_bits = fx
    .bcx
    .ins()
    .bitcast(types::I32X4, MemFlagsData::new(), result);
  let signed = fx.bcx.ins().bxor(sign, result_bits);
  fx.bcx.ins().bitcast(VEC_TYPE, MemFlagsData::new(), signed)
}

/// Vectorized minimax-polynomial approximation of `exp(x)` over F32X4.
///
/// Refer:
///   Software Manual for the Elementary Functions
///   Cody, W.J. and Waite, W.M.C.
#[allow(clippy::excessive_precision)]
pub(crate) fn exp_simd(fx: &mut FunctionCx<'_, '_>, x: Value) -> Value {
  // Argument reduction:
  //   exp(x) = 2^k * exp(s)
  //   s = x - k*ln(2)
  //   k = round(x / ln(2))
  // <https://github.com/shibatch/sleef/blob/3.9.0/src/common/misc.h#L113>
  let log2e = splat_f32(fx, LOG2_E); // 1 / ln(2)
  let l2u_f = splat_f32(fx, 0.693_145_751_953_125); // hi part of ln(2)
  let l2l_f = splat_f32(fx, 1.428_606_765_330_187e-6); // lo part of ln(2)
  let one = splat_f32(fx, 1.0);

  // Minimax polynomial coefficients for exp(s) on s in [-ln2/2, ln2/2],
  // Reconstruction is:
  //   exp(s) ~= 1 + s + s^2 * poly(s)
  //   poly(s) approximates (exp(s) - 1 - s) / s^2.
  let exp_p0 = splat_f32(fx, 0.000_198_527_617_612_853_646_278_381);
  let exp_p1 = splat_f32(fx, 0.001_393_043_552_525_341_510_772_71);
  let exp_p2 = splat_f32(fx, 0.008_333_360_776_305_198_669_433_59);
  let exp_p3 = splat_f32(fx, 0.041_666_485_369_205_474_853_515_6);
  let exp_p4 = splat_f32(fx, 0.166_666_671_633_720_397_949_219);
  let exp_p5 = splat_f32(fx, 0.5);

  // Clamp the input so the exponent reconstruction never overflows the field.
  let hi = splat_f32(fx, 104.0);
  let lo = splat_f32(fx, -104.0);
  let x = fmin_simd(fx, x, hi);
  let x = fmax_simd(fx, x, lo);

  // k = round_nearest_even(x * log2e)
  let kf = fx.bcx.ins().fmul(x, log2e);
  let kf = fx.bcx.ins().nearest(kf);
  let q = fx.bcx.ins().fcvt_to_sint_sat(types::I32X4, kf);

  // s = x - kf * L2U - kf * L2L
  let neg_kf = fx.bcx.ins().fneg(kf);
  let s = fx.bcx.ins().fma(neg_kf, l2u_f, x);
  let s = fx.bcx.ins().fma(neg_kf, l2l_f, s);

  // Horner evaluation of poly(s), then reconstruct 1 + s + s^2 * poly.
  let s2 = fx.bcx.ins().fmul(s, s);
  let y = fx.bcx.ins().fma(exp_p0, s, exp_p1);
  let y = fx.bcx.ins().fma(y, s, exp_p2);
  let y = fx.bcx.ins().fma(y, s, exp_p3);
  let y = fx.bcx.ins().fma(y, s, exp_p4);
  let y = fx.bcx.ins().fma(y, s, exp_p5);
  let y = fx.bcx.ins().fma(y, s2, s);
  let y = fx.bcx.ins().fadd(y, one);

  // Build 2^k as the product of two half-exponent injections.
  let bias = splat_i32(fx, 0x7f);
  let q_hi = fx.bcx.ins().sshr_imm_s(q, 1);
  let q_lo = fx.bcx.ins().isub(q, q_hi);
  let biased_hi = fx.bcx.ins().iadd(q_hi, bias);
  let biased_lo = fx.bcx.ins().iadd(q_lo, bias);
  let emm_hi = fx.bcx.ins().ishl_imm_s(biased_hi, 23);
  let emm_lo = fx.bcx.ins().ishl_imm_s(biased_lo, 23);
  let scale_hi = fx.bcx.ins().bitcast(VEC_TYPE, MemFlagsData::new(), emm_hi);
  let scale_lo = fx.bcx.ins().bitcast(VEC_TYPE, MemFlagsData::new(), emm_lo);
  let r = fx.bcx.ins().fmul(y, scale_hi);
  fx.bcx.ins().fmul(r, scale_lo)
}

/// Vectorized approximation of `log(x)` over F32X4.
#[allow(clippy::excessive_precision, clippy::cast_precision_loss)]
pub(crate) fn log_simd(fx: &mut FunctionCx<'_, '_>, x: Value) -> Value {
  let zero = splat_f32(fx, 0.0);
  let one = splat_f32(fx, 1.0);
  let four_thirds = splat_f32(fx, 4.0 / 3.0);
  let ln2 = splat_f32(fx, LN_2);

  // <https://github.com/shibatch/sleef/blob/3.9.0/src/libm/sleefsimdsp.c#L2008-L2012>
  let p0 = splat_f32(fx, 0.239_282_846_450_805_664_062_5);
  let p1 = splat_f32(fx, 0.285_182_118_415_832_519_531_25);
  let p2 = splat_f32(fx, 0.400_005_877_017_974_853_515_625);
  let p3 = splat_f32(fx, 0.666_666_686_534_881_591_796_875);
  let two = splat_f32(fx, 2.0);

  let flt_min = splat_f32(fx, f32::MIN_POSITIVE);
  let scale_2_64 = splat_f32(fx, (1u64 << 32) as f32 * (1u64 << 32) as f32);
  let is_subnormal = fx.bcx.ins().fcmp(FloatCC::LessThan, x, flt_min);
  let scaled = fx.bcx.ins().fmul(x, scale_2_64);
  let d = vselect_f32x4(fx, is_subnormal, scaled, x);

  let d_scaled = fx.bcx.ins().fmul(d, four_thirds);
  let d_scaled_bits = fx
    .bcx
    .ins()
    .bitcast(types::I32X4, MemFlagsData::new(), d_scaled);
  let exp_mask = fx.bcx.ins().iconst(types::I32, 0xff);
  let exp_mask_splat = fx.bcx.ins().splat(types::I32X4, exp_mask);
  let biased_exp = fx.bcx.ins().ushr_imm_s(d_scaled_bits, 23);
  let biased_exp = fx.bcx.ins().band(biased_exp, exp_mask_splat);
  let bias = fx.bcx.ins().iconst(types::I32, 0x7f);
  let bias_splat = fx.bcx.ins().splat(types::I32X4, bias);
  let e_int = fx.bcx.ins().isub(biased_exp, bias_splat);

  let sixty_four = fx.bcx.ins().iconst(types::I32, 64);
  let sixty_four_splat = fx.bcx.ins().splat(types::I32X4, sixty_four);
  let e_adj = fx.bcx.ins().isub(e_int, sixty_four_splat);
  let e_int = fx.bcx.ins().bitselect(is_subnormal, e_adj, e_int);
  let e = fx.bcx.ins().fcvt_from_sint(VEC_TYPE, e_int);

  let d_bits = fx.bcx.ins().bitcast(types::I32X4, MemFlagsData::new(), d);
  let e_shifted = fx.bcx.ins().ishl_imm_s(e_int, 23);
  let m_bits = fx.bcx.ins().isub(d_bits, e_shifted);
  let m = fx.bcx.ins().bitcast(VEC_TYPE, MemFlagsData::new(), m_bits);

  let m_minus_1 = fx.bcx.ins().fsub(m, one);
  let m_plus_1 = fx.bcx.ins().fadd(m, one);
  let u = fx.bcx.ins().fdiv(m_minus_1, m_plus_1);
  let u2 = fx.bcx.ins().fmul(u, u);

  let t = fx.bcx.ins().fma(p0, u2, p1);
  let t = fx.bcx.ins().fma(t, u2, p2);
  let t = fx.bcx.ins().fma(t, u2, p3);
  let t = fx.bcx.ins().fma(t, u2, two);

  let e_ln2 = fx.bcx.ins().fmul(e, ln2);
  let r = fx.bcx.ins().fma(u, t, e_ln2);

  let neg_inf = splat_f32(fx, f32::NEG_INFINITY);
  let pos_inf = splat_f32(fx, f32::INFINITY);
  let nan = splat_f32(fx, f32::NAN);
  let is_zero = fx.bcx.ins().fcmp(FloatCC::Equal, x, zero);
  let is_neg = fx.bcx.ins().fcmp(FloatCC::LessThan, x, zero);
  let is_nan = fx.bcx.ins().fcmp(FloatCC::NotEqual, x, x); // x != x
  let is_pinf = fx.bcx.ins().fcmp(FloatCC::Equal, x, pos_inf);
  let r = vselect_f32x4(fx, is_pinf, pos_inf, r);
  let invalid = fx.bcx.ins().bor(is_neg, is_nan);
  let r = vselect_f32x4(fx, invalid, nan, r);
  vselect_f32x4(fx, is_zero, neg_inf, r)
}

pub(crate) fn fpow_by_squaring(fx: &mut FunctionCx<'_, '_>, base: Value, exp: u32) -> Value {
  debug_assert!(exp >= 3);
  let bits = exp.bit_width();
  let mut result = base;
  for i in (0..bits - 1).rev() {
    result = fx.bcx.ins().fmul(result, result);
    if (exp >> i) & 1 == 1 {
      result = fx.bcx.ins().fmul(result, base);
    }
  }
  result
}

/// Vectorized minimax-polynomial approximation of `atan(x)` over F32X4.
#[allow(clippy::excessive_precision)]
fn atan_simd(fx: &mut FunctionCx<'_, '_>, x: Value) -> Value {
  // Coefficients for a minimax polynomial of atan on [-1, 1], evaluated as
  // `atan(z) ~= z + z * z^2 * poly(z^2)`.
  // <https://github.com/shibatch/sleef/blob/3.9.0/src/libm/sleefsp.c#L942-L949>
  let c0 = splat_f32(fx, 0.002_823_638_962_581_753_730_77);
  let c1 = splat_f32(fx, -0.015_956_902_876_496_315_002_4);
  let c2 = splat_f32(fx, 0.042_504_988_610_744_476_318_4);
  let c3 = splat_f32(fx, -0.074_890_092_015_266_418_457_0);
  let c4 = splat_f32(fx, 0.106_347_933_411_598_205_566);
  let c5 = splat_f32(fx, -0.142_027_363_181_114_196_777);
  let c6 = splat_f32(fx, 0.199_926_957_488_059_997_558);
  let c7 = splat_f32(fx, -0.333_331_018_686_294_555_664);

  let one = splat_f32(fx, 1.0);
  let half_pi = splat_f32(fx, FRAC_PI_2);

  // z = min(|x|, 1/|x|)
  let abs_x = fx.bcx.ins().fabs(x);
  let flip = fx.bcx.ins().fcmp(FloatCC::GreaterThan, abs_x, one);
  let recip = fx.bcx.ins().fdiv(one, abs_x);
  let z = vselect_f32x4(fx, flip, recip, abs_x);

  // Horner evaluation of poly(z^2), then reconstruct z + z * z^2 * poly.
  let z2 = fx.bcx.ins().fmul(z, z);
  let p = fx.bcx.ins().fma(c0, z2, c1);
  let p = fx.bcx.ins().fma(p, z2, c2);
  let p = fx.bcx.ins().fma(p, z2, c3);
  let p = fx.bcx.ins().fma(p, z2, c4);
  let p = fx.bcx.ins().fma(p, z2, c5);
  let p = fx.bcx.ins().fma(p, z2, c6);
  let p = fx.bcx.ins().fma(p, z2, c7);
  let p = fx.bcx.ins().fmul(p, z2);
  let p = fx.bcx.ins().fma(p, z, z);

  // If we flipped, recover atan(|x|) as pi/2 - atan(1/|x|).
  let flipped = fx.bcx.ins().fsub(half_pi, p);
  let mag = vselect_f32x4(fx, flip, flipped, p);

  // Re-apply the sign of the original input.
  copysign_simd(fx, mag, x)
}

/// Vectorized `atan2(y, x)` over F32X4.
pub(crate) fn atan2_simd(fx: &mut FunctionCx<'_, '_>, y: Value, x: Value) -> Value {
  // Constants.
  let zero = splat_f32(fx, 0.0);
  let pi = splat_f32(fx, PI);
  let half_pi = splat_f32(fx, FRAC_PI_2);

  // atan(y/x)
  let y_div_x = fx.bcx.ins().fdiv(y, x);
  let atan_y_div_x = atan_simd(fx, y_div_x);

  // If x is negative, the result will be atan(y/x), then:
  //   If y is positive, add pi.
  //   If y is negative, subtract pi.
  let signed_pi = copysign_simd(fx, pi, y);
  let negative_x_result = fx.bcx.ins().fadd(atan_y_div_x, signed_pi);

  // If x is zero, the result will be pi/2 with the sign of y.
  let zero_x_result = copysign_simd(fx, half_pi, y);

  // If x > 0: atan(y/x)
  // Else if x < 0: negative_x_result
  // Else (x = 0): zero_x_result
  let x_gt_0 = fx.bcx.ins().fcmp(FloatCC::GreaterThan, x, zero);
  let x_lt_0 = fx.bcx.ins().fcmp(FloatCC::LessThan, x, zero);
  let inner = vselect_f32x4(fx, x_gt_0, atan_y_div_x, zero_x_result);
  let result = vselect_f32x4(fx, x_lt_0, negative_x_result, inner);

  // atan2(0, 0) is undefined.
  let x_is_zero = fx.bcx.ins().fcmp(FloatCC::Equal, x, zero);
  let y_is_zero = fx.bcx.ins().fcmp(FloatCC::Equal, y, zero);
  let both_zero = fx.bcx.ins().band(x_is_zero, y_is_zero);
  vselect_f32x4(fx, both_zero, zero, result)
}

pub(crate) fn codegen_float_binop_simd(
  fx: &mut FunctionCx<'_, '_>,
  op: BinOp,
  lhs: Value,
  rhs: Value,
) -> Value {
  match op {
    BinOp::Add => fx.bcx.ins().fadd(lhs, rhs),
    BinOp::Sub => fx.bcx.ins().fsub(lhs, rhs),
    BinOp::Mul => fx.bcx.ins().fmul(lhs, rhs),
    BinOp::Div => fx.bcx.ins().fdiv(lhs, rhs),
    BinOp::Rem => {
      // x - trunc(x / y) * y
      let quot = fx.bcx.ins().fdiv(lhs, rhs);
      let trunc_quot = fx.bcx.ins().trunc(quot);
      let prod = fx.bcx.ins().fmul(trunc_quot, rhs);
      fx.bcx.ins().fsub(lhs, prod)
    }
    BinOp::Gt | BinOp::Gte | BinOp::Lt | BinOp::Lte | BinOp::Eq => {
      let float_cc = match op {
        BinOp::Gt => FloatCC::GreaterThan,
        BinOp::Gte => FloatCC::GreaterThanOrEqual,
        BinOp::Lt => FloatCC::LessThan,
        BinOp::Lte => FloatCC::LessThanOrEqual,
        BinOp::Eq => FloatCC::Equal,
        _ => unreachable!(),
      };
      let result = fx.bcx.ins().fcmp(float_cc, lhs, rhs);
      bool_to_float_simd(fx, result)
    }
    BinOp::Max => fmax_simd(fx, lhs, rhs),
    BinOp::Min => fmin_simd(fx, lhs, rhs),
    BinOp::And | BinOp::Or | BinOp::Xor => {
      let zero = splat_f32(fx, 0.0);
      let lhs_bool = fx.bcx.ins().fcmp(FloatCC::GreaterThan, lhs, zero);
      let rhs_bool = fx.bcx.ins().fcmp(FloatCC::GreaterThan, rhs, zero);
      let result = match op {
        BinOp::And => fx.bcx.ins().band(lhs_bool, rhs_bool),
        BinOp::Or => fx.bcx.ins().bor(lhs_bool, rhs_bool),
        BinOp::Xor => fx.bcx.ins().bxor(lhs_bool, rhs_bool),
        _ => unreachable!(),
      };
      bool_to_float_simd(fx, result)
    }
    BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
      let lhs_rounded = fx.bcx.ins().nearest(lhs);
      let rhs_rounded = fx.bcx.ins().nearest(rhs);
      let lhs_i = fx.bcx.ins().fcvt_to_sint_sat(types::I32X4, lhs_rounded);
      let rhs_i = fx.bcx.ins().fcvt_to_sint_sat(types::I32X4, rhs_rounded);
      let res_i = match op {
        BinOp::BitAnd => fx.bcx.ins().band(lhs_i, rhs_i),
        BinOp::BitOr => fx.bcx.ins().bor(lhs_i, rhs_i),
        BinOp::BitXor => fx.bcx.ins().bxor(lhs_i, rhs_i),
        _ => unreachable!(),
      };
      fx.bcx.ins().fcvt_from_sint(VEC_TYPE, res_i)
    }
    BinOp::Pow => unreachable!("pow is handled by pow_simd"),
    BinOp::Atan2 => unreachable!("atan2 is handled by atan2_simd"),
  }
}
