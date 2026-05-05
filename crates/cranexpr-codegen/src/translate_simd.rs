use std::f32::consts::{LN_2, LOG2_E};

use cranelift::codegen::ir::{ConstantData, Endianness};
use cranelift::prelude::*;
use cranexpr_ast::{BinOp, BoundaryMode, Expr, TernaryOp, UnOp};

use crate::compiler::{FunctionCx, SRC_MEMFLAGS};
use crate::component_type::ComponentType;
use crate::errors::CodegenError;
use crate::simd_plan::SIMD_LANES;
use crate::translate::{codegen_boundary_mode, codegen_pixel_offset, resolve_clip_name};

const VEC_TYPE: types::Type = types::F32X4;

pub(crate) fn translate_expr_simd(
  fx: &mut FunctionCx<'_, '_>,
  expr: &Expr,
) -> Result<Value, CodegenError> {
  match expr {
    Expr::Lit(literal) => {
      let scalar = fx.bcx.ins().f32const(*literal);
      Ok(fx.bcx.ins().splat(VEC_TYPE, scalar))
    }

    Expr::Prop(name, prop) => {
      let clip_idx = resolve_clip_name(name, &fx.src_types)?;
      let variable = fx
        .variables
        .get(&format!("prop_{clip_idx}_{prop}"))
        .expect("frame property variable not defined");
      let scalar = fx.bcx.use_var(*variable);
      Ok(fx.bcx.ins().splat(VEC_TYPE, scalar))
    }

    Expr::Binary(BinOp::Pow, base, exponent) => pow_simd(fx, base, exponent),
    Expr::Binary(op, lhs, rhs) => {
      let lhs = translate_expr_simd(fx, lhs)?;
      let rhs = translate_expr_simd(fx, rhs)?;
      Ok(codegen_float_binop_simd(fx, *op, lhs, rhs))
    }

    Expr::Ternary(op, a, b, c) => {
      let a = translate_expr_simd(fx, a)?;
      let b = translate_expr_simd(fx, b)?;
      let c = translate_expr_simd(fx, c)?;
      Ok(match op {
        TernaryOp::Clip => {
          let min_val = b;
          let max_val = c;
          let min_result = fx.bcx.ins().fmin(a, max_val);
          fx.bcx.ins().fmax(min_result, min_val)
        }
      })
    }

    Expr::Unary(op, x) => {
      let x = translate_expr_simd(fx, x)?;
      Ok(match op {
        UnOp::Abs => fx.bcx.ins().fabs(x),
        UnOp::Floor => fx.bcx.ins().floor(x),
        UnOp::Round => fx.bcx.ins().nearest(x),
        UnOp::Neg => fx.bcx.ins().fneg(x),
        UnOp::Sqrt => fx.bcx.ins().sqrt(x),
        UnOp::Trunc => fx.bcx.ins().trunc(x),
        UnOp::BitNot => {
          let rounded = fx.bcx.ins().nearest(x);
          let i = fx.bcx.ins().fcvt_to_sint_sat(types::I32X4, rounded);
          let res = fx.bcx.ins().bnot(i);
          fx.bcx.ins().fcvt_from_sint(VEC_TYPE, res)
        }
        UnOp::Not => {
          let zero = splat_f32(fx, 0.0);
          let mask = fx.bcx.ins().fcmp(FloatCC::LessThanOrEqual, x, zero);
          bool_to_float_simd(fx, mask)
        }
        UnOp::Sign => {
          let zero = splat_f32(fx, 0.0);
          let one = splat_f32(fx, 1.0);
          let neg_one = splat_f32(fx, -1.0);
          let is_neg = fx.bcx.ins().fcmp(FloatCC::LessThan, x, zero);
          let is_pos = fx.bcx.ins().fcmp(FloatCC::GreaterThan, x, zero);
          let pos_or_zero = vselect_f32x4(fx, is_pos, one, zero);
          vselect_f32x4(fx, is_neg, neg_one, pos_or_zero)
        }
        UnOp::Sine => sin_cos_simd(fx, x, SinCos::Sin),
        UnOp::Cosine => sin_cos_simd(fx, x, SinCos::Cos),
        UnOp::Exp => exp_simd(fx, x),
        UnOp::Log => log_simd(fx, x),
        UnOp::Tangent => {
          unreachable!("op {op:?} should have been rejected by SIMD eligibility check")
        }
      })
    }

    Expr::Ident(name) => {
      if name == "X"
        && let Some(var) = fx.variables.get("simd_X")
      {
        return Ok(fx.bcx.use_var(*var));
      }

      let variable = if let Ok(clip_idx) = resolve_clip_name(name, &fx.src_types) {
        fx.variables.get(&format!("simd_src{clip_idx}"))
      } else {
        fx.variables.get(name)
      }
      .ok_or_else(|| CodegenError::UndefinedVariable(name.clone()))?;

      let val = fx.bcx.use_var(*variable);
      let ty = fx.bcx.func.dfg.value_type(val);
      if ty == VEC_TYPE {
        Ok(val)
      } else {
        // Scalar identifiers splat to all lanes.
        let scalar = if ty.is_int() {
          fx.bcx.ins().fcvt_from_uint(types::F32, val)
        } else {
          val
        };
        Ok(fx.bcx.ins().splat(VEC_TYPE, scalar))
      }
    }

    Expr::Load(name) => {
      let simd_key = format!("simd_{name}");
      let variable = fx
        .user_variables
        .get(&simd_key)
        .ok_or_else(|| CodegenError::UndefinedVariable(name.clone()))?;
      Ok(fx.bcx.use_var(*variable))
    }

    Expr::Store(name, inner) => {
      let value = translate_expr_simd(fx, inner)?;
      let simd_key = format!("simd_{name}");
      let variable = if let Some(variable) = fx.user_variables.get(&simd_key) {
        *variable
      } else {
        let var = fx.bcx.declare_var(VEC_TYPE);
        fx.user_variables.insert(simd_key, var);
        var
      };
      fx.bcx.def_var(variable, value);
      Ok(value)
    }

    Expr::RelAccess {
      clip,
      rel_x,
      rel_y,
      boundary_mode,
    } => {
      let cache_key = format!("simd_{clip}[{rel_x},{rel_y}]");
      if let Some(existing) = fx.variables.get(&cache_key) {
        return Ok(fx.bcx.use_var(*existing));
      }

      let clip_idx = resolve_clip_name(clip, &fx.src_types)?;
      let boundary_mode = boundary_mode.unwrap_or(fx.boundary_mode);

      let val = translate_rel_access_simd(fx, clip_idx, *rel_x, *rel_y, boundary_mode)?;

      let var = fx.bcx.declare_var(VEC_TYPE);
      fx.variables.insert(cache_key, var);
      fx.bcx.def_var(var, val);

      Ok(fx.bcx.use_var(var))
    }

    Expr::IfElse(condition, then_body, else_body) => {
      let cond = translate_expr_simd(fx, condition)?;
      let then_val = translate_expr_simd(fx, then_body)?;
      let else_val = translate_expr_simd(fx, else_body)?;
      let zero = splat_f32(fx, 0.0);
      let mask = fx.bcx.ins().fcmp(FloatCC::GreaterThan, cond, zero);
      Ok(vselect_f32x4(fx, mask, then_val, else_val))
    }

    Expr::AbsAccess {
      clip,
      x,
      y,
      boundary_mode,
    } => {
      let clip_idx = resolve_clip_name(clip, &fx.src_types)?;
      let boundary_mode = boundary_mode.unwrap_or(fx.boundary_mode);
      let x_vec = translate_expr_simd(fx, x)?;
      let y_vec = translate_expr_simd(fx, y)?;
      Ok(translate_abs_access_simd(
        fx,
        clip_idx,
        x_vec,
        y_vec,
        boundary_mode,
      ))
    }
  }
}

fn splat_f32(fx: &mut FunctionCx<'_, '_>, val: f32) -> Value {
  let scalar = fx.bcx.ins().f32const(val);
  fx.bcx.ins().splat(VEC_TYPE, scalar)
}

/// Per-lane x offsets used to synthesize the SIMD X coordinate vector.
///
/// Recall that in the scalar implementation, each pixel has one X coordinate.
/// For SIMD, a single iteration will be processing 4 adjacent pixels, so the X
/// variable has to be a vector of four coordinates: `[X, X+1, X+2, X+3]`.
pub(crate) fn simd_lane_offsets_f32x4(fx: &mut FunctionCx<'_, '_>) -> Value {
  const LANE_OFFSETS: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
  let mut bytes = [0u8; 16];
  for (i, f) in LANE_OFFSETS.iter().enumerate() {
    bytes[i * 4..(i + 1) * 4].copy_from_slice(&f.to_le_bytes());
  }
  let handle = fx.bcx.func.dfg.constants.insert(bytes.as_slice().into());
  fx.bcx.ins().vconst(VEC_TYPE, handle)
}

/// Converts a vector boolean mask to F32X4, where true lanes become 1.0 and
/// false lanes become 0.0.
fn bool_to_float_simd(fx: &mut FunctionCx<'_, '_>, mask: Value) -> Value {
  let one_bits = fx.bcx.ins().iconst(types::I32, i64::from(1.0f32.to_bits()));
  let one_bits_splat = fx.bcx.ins().splat(types::I32X4, one_bits);
  let anded = fx.bcx.ins().band(mask, one_bits_splat);
  fx.bcx.ins().bitcast(VEC_TYPE, MemFlags::new(), anded)
}

/// Picks lanes from `true_val` where `mask` is all-1s, and from `false_val`
/// where `mask` is all-0s.
fn vselect_f32x4(
  fx: &mut FunctionCx<'_, '_>,
  mask: Value,
  true_val: Value,
  false_val: Value,
) -> Value {
  let t_bits = fx
    .bcx
    .ins()
    .bitcast(types::I32X4, MemFlags::new(), true_val);
  let f_bits = fx
    .bcx
    .ins()
    .bitcast(types::I32X4, MemFlags::new(), false_val);
  let selected = fx.bcx.ins().bitselect(mask, t_bits, f_bits);
  fx.bcx.ins().bitcast(VEC_TYPE, MemFlags::new(), selected)
}

#[derive(Clone, Copy)]
enum SinCos {
  Sin,
  Cos,
}

/// Vectorized minimax polynomial approximation of `sin(x)` or `cos(x)` over
/// F32X4.
///
/// Refer:
///   Software Manual for the Elementary Functions
///   Cody, W.J. and Waite, W.M.C.
fn sin_cos_simd(fx: &mut FunctionCx<'_, '_>, x: Value, mode: SinCos) -> Value {
  let vf32_from_bits = |fx: &mut FunctionCx<'_, '_>, bits: u32| -> Value {
    let scalar = fx.bcx.ins().iconst(types::I32, i64::from(bits));
    let splat = fx.bcx.ins().splat(types::I32X4, scalar);
    fx.bcx.ins().bitcast(VEC_TYPE, MemFlags::new(), splat)
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

  let odd_n_sign = fx.bcx.ins().ishl_imm(n_int, 31);

  let sign = match mode {
    SinCos::Sin => {
      let x_bits = fx.bcx.ins().bitcast(types::I32X4, MemFlags::new(), x);
      let not_absmask = fx.bcx.ins().bnot(absmask);
      let input_sign = fx.bcx.ins().band(x_bits, not_absmask);
      fx.bcx.ins().bxor(input_sign, odd_n_sign)
    }
    SinCos::Cos => odd_n_sign,
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

  let result = match mode {
    SinCos::Sin => {
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
    }
    SinCos::Cos => {
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
    }
  };

  // Apply the accumulated sign bit.
  let result_bits = fx.bcx.ins().bitcast(types::I32X4, MemFlags::new(), result);
  let signed = fx.bcx.ins().bxor(sign, result_bits);
  fx.bcx.ins().bitcast(VEC_TYPE, MemFlags::new(), signed)
}

/// Vectorized minimax-polynomial approximation of `exp(x)` over F32X4.
///
/// Refer:
///   Software Manual for the Elementary Functions
///   Cody, W.J. and Waite, W.M.C.
#[allow(clippy::excessive_precision)]
fn exp_simd(fx: &mut FunctionCx<'_, '_>, x: Value) -> Value {
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

  // Build 2^k by injecting (k + 127) into the exponent field.
  let bias = fx.bcx.ins().iconst(types::I32, 0x7f);
  let bias_splat = fx.bcx.ins().splat(types::I32X4, bias);
  let biased = fx.bcx.ins().iadd(q, bias_splat);
  let emm0 = fx.bcx.ins().ishl_imm(biased, 23);
  let scale = fx.bcx.ins().bitcast(VEC_TYPE, MemFlags::new(), emm0);
  let r = fx.bcx.ins().fmul(y, scale);

  let over_thresh = splat_f32(fx, 104.0);
  let under_thresh = splat_f32(fx, -104.0);
  let zero = splat_f32(fx, 0.0);
  let pos_inf = splat_f32(fx, f32::INFINITY);
  let over = fx.bcx.ins().fcmp(FloatCC::GreaterThan, x, over_thresh);
  let under = fx.bcx.ins().fcmp(FloatCC::LessThan, x, under_thresh);
  let r = vselect_f32x4(fx, under, zero, r);
  vselect_f32x4(fx, over, pos_inf, r)
}

/// Vectorized approximation of `log(x)` over F32X4.
#[allow(clippy::excessive_precision, clippy::cast_precision_loss)]
fn log_simd(fx: &mut FunctionCx<'_, '_>, x: Value) -> Value {
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
    .bitcast(types::I32X4, MemFlags::new(), d_scaled);
  let exp_mask = fx.bcx.ins().iconst(types::I32, 0xff);
  let exp_mask_splat = fx.bcx.ins().splat(types::I32X4, exp_mask);
  let biased_exp = fx.bcx.ins().ushr_imm(d_scaled_bits, 23);
  let biased_exp = fx.bcx.ins().band(biased_exp, exp_mask_splat);
  let bias = fx.bcx.ins().iconst(types::I32, 0x7f);
  let bias_splat = fx.bcx.ins().splat(types::I32X4, bias);
  let e_int = fx.bcx.ins().isub(biased_exp, bias_splat);

  let sixty_four = fx.bcx.ins().iconst(types::I32, 64);
  let sixty_four_splat = fx.bcx.ins().splat(types::I32X4, sixty_four);
  let e_adj = fx.bcx.ins().isub(e_int, sixty_four_splat);
  let e_int = fx.bcx.ins().bitselect(is_subnormal, e_adj, e_int);
  let e = fx.bcx.ins().fcvt_from_sint(VEC_TYPE, e_int);

  let d_bits = fx.bcx.ins().bitcast(types::I32X4, MemFlags::new(), d);
  let e_shifted = fx.bcx.ins().ishl_imm(e_int, 23);
  let m_bits = fx.bcx.ins().isub(d_bits, e_shifted);
  let m = fx.bcx.ins().bitcast(VEC_TYPE, MemFlags::new(), m_bits);

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

fn pow_simd(
  fx: &mut FunctionCx<'_, '_>,
  base: &Expr,
  exponent: &Expr,
) -> Result<Value, CodegenError> {
  let base_val = translate_expr_simd(fx, base)?;

  #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
  if let Expr::Lit(exp) = exponent {
    if *exp == 0.0 {
      return Ok(splat_f32(fx, 1.0));
    }
    if *exp == 1.0 {
      return Ok(base_val);
    }
    if *exp == 2.0 {
      return Ok(fx.bcx.ins().fmul(base_val, base_val));
    }
    if *exp == 0.5 {
      return Ok(fx.bcx.ins().sqrt(base_val));
    }
    let exp_u32 = *exp as u32;
    if *exp == exp_u32 as f32 && (3..=8).contains(&exp_u32) {
      let x2 = fx.bcx.ins().fmul(base_val, base_val);
      return Ok(match exp_u32 {
        3 => fx.bcx.ins().fmul(x2, base_val),
        4 => fx.bcx.ins().fmul(x2, x2),
        5 => {
          let x4 = fx.bcx.ins().fmul(x2, x2);
          fx.bcx.ins().fmul(x4, base_val)
        }
        6 => {
          let x3 = fx.bcx.ins().fmul(x2, base_val);
          fx.bcx.ins().fmul(x3, x3)
        }
        7 => {
          let x3 = fx.bcx.ins().fmul(x2, base_val);
          let x6 = fx.bcx.ins().fmul(x3, x3);
          fx.bcx.ins().fmul(x6, base_val)
        }
        8 => {
          let x4 = fx.bcx.ins().fmul(x2, x2);
          fx.bcx.ins().fmul(x4, x4)
        }
        _ => unreachable!(),
      });
    }
  }

  let exp_val = translate_expr_simd(fx, exponent)?;
  let log_x = log_simd(fx, base_val);
  let prod = fx.bcx.ins().fmul(log_x, exp_val);
  Ok(exp_simd(fx, prod))
}

fn codegen_float_binop_simd(
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
    BinOp::Max => fx.bcx.ins().fmax(lhs, rhs),
    BinOp::Min => fx.bcx.ins().fmin(lhs, rhs),
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
    BinOp::Rem | BinOp::Atan2 => {
      unreachable!("op {op:?} should have been rejected by SIMD eligibility check")
    }
  }
}

fn translate_rel_access_simd(
  fx: &mut FunctionCx<'_, '_>,
  clip_idx: usize,
  rel_x: i32,
  rel_y: i32,
  boundary_mode: BoundaryMode,
) -> Result<Value, CodegenError> {
  let x_var = fx
    .variables
    .get("X")
    .ok_or_else(|| CodegenError::UndefinedVariable("X".to_string()))?;
  let y_var = fx
    .variables
    .get("Y")
    .ok_or_else(|| CodegenError::UndefinedVariable("Y".to_string()))?;
  let x_base = fx.bcx.use_var(*x_var);
  let y_base = fx.bcx.use_var(*y_var);

  let target_y = if rel_y == 0 {
    y_base
  } else {
    let rel_y_val = fx.bcx.ins().iconst(types::I64, i64::from(rel_y));
    let shifted_y = fx.bcx.ins().iadd(y_base, rel_y_val);
    codegen_boundary_mode(fx, shifted_y, fx.height, boundary_mode)
  };

  if rel_x == 0 {
    let (src_ptr, pixel_offset) =
      codegen_pixel_offset(fx, clip_idx, x_base, target_y, ComponentType::F32);
    return Ok(src_ptr.offset_value(fx, pixel_offset).load(
      fx,
      VEC_TYPE,
      SRC_MEMFLAGS.with_aligned(),
    ));
  }

  match boundary_mode {
    BoundaryMode::Clamp => Ok(translate_rel_access_clamp_x(
      fx, clip_idx, rel_x, target_y, x_base,
    )),
    BoundaryMode::Mirror => Ok(translate_rel_access_mirror_x(
      fx, clip_idx, rel_x, target_y, x_base,
    )),
  }
}

fn translate_rel_access_clamp_x(
  fx: &mut FunctionCx<'_, '_>,
  clip_idx: usize,
  rel_x: i32,
  target_y: Value,
  x_base: Value,
) -> Value {
  let width = fx.width;
  let one = fx.bcx.ins().iconst(types::I64, 1);
  let zero = fx.bcx.ins().iconst(types::I64, 0);
  let width_minus_one = fx.bcx.ins().isub(width, one);
  let rel_x_val = fx.bcx.ins().iconst(types::I64, i64::from(rel_x));
  let shifted_x = fx.bcx.ins().iadd(x_base, rel_x_val);

  // clamped_x = clamp(x_base + rel_x, 0, width - 1)
  let too_large = fx
    .bcx
    .ins()
    .icmp(IntCC::SignedGreaterThan, shifted_x, width_minus_one);
  let clamped_hi = fx.bcx.ins().select(too_large, width_minus_one, shifted_x);
  let too_small = fx.bcx.ins().icmp(IntCC::SignedLessThan, clamped_hi, zero);
  let clamped_x = fx.bcx.ins().select(too_small, zero, clamped_hi);

  let (src_ptr, pixel_offset) =
    codegen_pixel_offset(fx, clip_idx, clamped_x, target_y, ComponentType::F32);
  let raw = src_ptr
    .offset_value(fx, pixel_offset)
    .load(fx, VEC_TYPE, SRC_MEMFLAGS);

  if rel_x < 0 {
    apply_left_swizzle_fixup(fx, raw, x_base, rel_x)
  } else {
    apply_right_swizzle_fixup(fx, raw, clamped_x, width)
  }
}

fn apply_left_swizzle_fixup(
  fx: &mut FunctionCx<'_, '_>,
  raw: Value,
  x_base: Value,
  rel_x: i32,
) -> Value {
  let abs_rel = i64::from(rel_x.unsigned_abs());
  let lanes = SIMD_LANES;
  let mut case_x: Vec<i64> = Vec::new();
  let mut case_masks: Vec<[u8; 4]> = Vec::new();
  let mut i: i64 = 0;
  while i < abs_rel {
    let mut mask = [0u8; 4];
    for j in 0..lanes {
      let src_lane = ((i + j + i64::from(rel_x)).max(0)) % lanes;
      mask[j as usize] = src_lane as u8;
    }
    case_x.push(i);
    case_masks.push(mask);
    i += lanes;
  }
  emit_swizzle_switch(fx, raw, x_base, &case_x, &case_masks, None)
}

fn apply_right_swizzle_fixup(
  fx: &mut FunctionCx<'_, '_>,
  raw: Value,
  clamped_x: Value,
  width: Value,
) -> Value {
  let lanes = SIMD_LANES;
  let lanes_val = fx.bcx.ins().iconst(types::I64, lanes);
  let x_plus_lanes = fx.bcx.ins().iadd(clamped_x, lanes_val);
  let dist = fx.bcx.ins().isub(x_plus_lanes, width);

  let zero = fx.bcx.ins().iconst(types::I64, 0);
  let in_bounds = fx.bcx.ins().icmp(IntCC::SignedLessThanOrEqual, dist, zero);

  let in_bounds_block = fx.bcx.create_block();
  let fixup_block = fx.bcx.create_block();
  let merge_block = fx.bcx.create_block();
  fx.bcx.append_block_param(merge_block, VEC_TYPE);
  fx.bcx
    .ins()
    .brif(in_bounds, in_bounds_block, &[], fixup_block, &[]);

  fx.bcx.switch_to_block(in_bounds_block);
  fx.bcx.seal_block(in_bounds_block);
  fx.bcx.ins().jump(merge_block, &[raw.into()]);

  fx.bcx.switch_to_block(fixup_block);
  fx.bcx.seal_block(fixup_block);

  let mut case_vals: Vec<i64> = Vec::new();
  let mut case_masks: Vec<[u8; 4]> = Vec::new();
  for i in 1..lanes {
    let mut mask = [0u8; 4];
    let mut last: u8 = 0;
    for j in 0..lanes {
      if j + i < lanes {
        mask[j as usize] = j as u8;
        last = j as u8;
      } else {
        mask[j as usize] = last;
      }
    }
    case_vals.push(i);
    case_masks.push(mask);
  }

  let default_mask = [0u8; 4];
  let fixed = emit_swizzle_switch(fx, raw, dist, &case_vals, &case_masks, Some(default_mask));
  fx.bcx.ins().jump(merge_block, &[fixed.into()]);

  fx.bcx.switch_to_block(merge_block);
  fx.bcx.seal_block(merge_block);
  fx.bcx.block_params(merge_block)[0]
}

fn emit_swizzle_switch(
  fx: &mut FunctionCx<'_, '_>,
  raw: Value,
  selector: Value,
  case_vals: &[i64],
  case_masks: &[[u8; 4]],
  default_mask: Option<[u8; 4]>,
) -> Value {
  assert_eq!(case_vals.len(), case_masks.len());
  if case_vals.is_empty() && default_mask.is_none() {
    return raw;
  }

  let merge_block = fx.bcx.create_block();
  fx.bcx.append_block_param(merge_block, VEC_TYPE);

  for (case_val, mask) in case_vals.iter().zip(case_masks.iter()) {
    let case_body = fx.bcx.create_block();
    let next_test = fx.bcx.create_block();
    let case_const = fx.bcx.ins().iconst(types::I64, *case_val);
    let matches = fx.bcx.ins().icmp(IntCC::Equal, selector, case_const);
    fx.bcx.ins().brif(matches, case_body, &[], next_test, &[]);

    fx.bcx.switch_to_block(case_body);
    fx.bcx.seal_block(case_body);
    let shuffled = apply_shuffle(fx, raw, *mask);
    fx.bcx.ins().jump(merge_block, &[shuffled.into()]);

    fx.bcx.switch_to_block(next_test);
    fx.bcx.seal_block(next_test);
  }

  let fallthrough = default_mask.map_or(raw, |mask| apply_shuffle(fx, raw, mask));
  fx.bcx.ins().jump(merge_block, &[fallthrough.into()]);

  fx.bcx.switch_to_block(merge_block);
  fx.bcx.seal_block(merge_block);
  fx.bcx.block_params(merge_block)[0]
}

fn apply_shuffle(fx: &mut FunctionCx<'_, '_>, raw: Value, lane_mask: [u8; 4]) -> Value {
  let mut bytes = [0u8; 16];
  for (j, &src_lane) in lane_mask.iter().enumerate() {
    let base = src_lane as usize * 4;
    for k in 0..4 {
      bytes[j * 4 + k] = (base + k) as u8;
    }
  }
  let imm = fx
    .bcx
    .func
    .dfg
    .immediates
    .push(ConstantData::from(bytes.as_slice()));
  let byte_flags = MemFlags::new().with_endianness(Endianness::Little);
  let raw_i8 = fx.bcx.ins().bitcast(types::I8X16, byte_flags, raw);
  let shuffled_i8 = fx.bcx.ins().shuffle(raw_i8, raw_i8, imm);
  fx.bcx.ins().bitcast(VEC_TYPE, byte_flags, shuffled_i8)
}

fn translate_rel_access_mirror_x(
  fx: &mut FunctionCx<'_, '_>,
  clip_idx: usize,
  rel_x: i32,
  target_y: Value,
  x_base: Value,
) -> Value {
  let lanes = SIMD_LANES;
  let rel_x_val = fx.bcx.ins().iconst(types::I64, i64::from(rel_x));

  let zero = fx.bcx.ins().f32const(0.0f32);
  let mut vec = fx.bcx.ins().splat(VEC_TYPE, zero);
  for lane in 0..lanes {
    let lane_off = fx.bcx.ins().iconst(types::I64, lane);
    let x_unshifted = fx.bcx.ins().iadd(x_base, lane_off);
    let x_shifted = fx.bcx.ins().iadd(x_unshifted, rel_x_val);
    let x_mirrored = codegen_boundary_mode(fx, x_shifted, fx.width, BoundaryMode::Mirror);
    let (src_ptr, pixel_offset) =
      codegen_pixel_offset(fx, clip_idx, x_mirrored, target_y, ComponentType::F32);
    let scalar = src_ptr
      .offset_value(fx, pixel_offset)
      .load(fx, types::F32, SRC_MEMFLAGS);
    vec = fx.bcx.ins().insertlane(vec, scalar, lane as u8);
  }
  vec
}

fn translate_abs_access_simd(
  fx: &mut FunctionCx<'_, '_>,
  clip_idx: usize,
  x_vec: Value,
  y_vec: Value,
  boundary_mode: BoundaryMode,
) -> Value {
  let x_rounded = fx.bcx.ins().nearest(x_vec);
  let y_rounded = fx.bcx.ins().nearest(y_vec);
  let x_i32x4 = fx.bcx.ins().fcvt_to_sint_sat(types::I32X4, x_rounded);
  let y_i32x4 = fx.bcx.ins().fcvt_to_sint_sat(types::I32X4, y_rounded);

  let zero = fx.bcx.ins().f32const(0.0f32);
  let mut vec = fx.bcx.ins().splat(VEC_TYPE, zero);
  for lane in 0..SIMD_LANES {
    let x_lane_i32 = fx.bcx.ins().extractlane(x_i32x4, lane as u8);
    let y_lane_i32 = fx.bcx.ins().extractlane(y_i32x4, lane as u8);
    let x_lane = fx.bcx.ins().sextend(types::I64, x_lane_i32);
    let y_lane = fx.bcx.ins().sextend(types::I64, y_lane_i32);
    let clamped_x = codegen_boundary_mode(fx, x_lane, fx.width, boundary_mode);
    let clamped_y = codegen_boundary_mode(fx, y_lane, fx.height, boundary_mode);
    let (src_ptr, pixel_offset) =
      codegen_pixel_offset(fx, clip_idx, clamped_x, clamped_y, ComponentType::F32);
    let scalar = src_ptr
      .offset_value(fx, pixel_offset)
      .load(fx, types::F32, SRC_MEMFLAGS);
    vec = fx.bcx.ins().insertlane(vec, scalar, lane as u8);
  }
  vec
}
