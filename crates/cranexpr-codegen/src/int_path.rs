//! Exact 16-bit integer evaluation for 8-bit expressions.
//!
//! When every intermediate value is provably within the `i16` range, the result
//! can be computed on [`types::I16X8`] vectors with twice the pixels per
//! instruction and no int/float conversions.

use cranelift::codegen::ir::Endianness;
use cranelift::prelude::*;
use cranexpr_ast::{BinOp, BoundaryMode, Expr, UnOp};

use crate::component_type::ComponentType;
use crate::context::{FunctionCx, SRC_MEMFLAGS};
use crate::pixel_access::resolve_clip_name;
use crate::pointer::Pointer;

/// Number of pixels processed per integer iteration.
pub(crate) const INT_LANES: i64 = 8;

const INT_VEC_TYPE: types::Type = types::I16X8;

/// An expression that can be evaluated exactly on [`types::I16X8`] vectors as
/// `round_half_even(body / 2^shift)`.
pub(crate) struct IntPlan<'a> {
  body: &'a Expr,
  shift: u32,
}

impl<'a> IntPlan<'a> {
  /// Returns a plan if `ast` is a single integer-exact expression over 8-bit
  /// clips with an 8-bit destination.
  pub(crate) fn new(
    ast: &'a [Expr],
    dst_type: ComponentType,
    src_types: &[ComponentType],
  ) -> Option<Self> {
    let [expr] = ast else {
      return None;
    };
    if dst_type != ComponentType::U8 {
      return None;
    }

    let (body, shift) = match expr {
      Expr::Binary(BinOp::Div, body, divisor) => {
        let Expr::Lit(divisor) = divisor.as_ref() else {
          return None;
        };
        let divisor = exact_int(*divisor)?;
        if divisor <= 0 || divisor.count_ones() != 1 {
          return None;
        }
        (body.as_ref(), divisor.trailing_zeros())
      }
      _ => (expr, 0),
    };

    let (lo, hi) = value_range(body, src_types)?;
    let bias = if shift == 0 { 0 } else { 1 << (shift - 1) };
    (fits_i16(lo) && fits_i16(hi + bias)).then_some(Self { body, shift })
  }

  /// Emits the body for the `INT_LANES` pixels starting at `x`, storing to
  /// `dst_row`. Every relative access must be in bounds.
  pub(crate) fn codegen(&self, fx: &mut FunctionCx<'_, '_>, dst_row: Value, x: Value) {
    let value = translate_int(fx, self.body, x);
    let rounded = if self.shift == 0 {
      value
    } else {
      // round_half_even(v / 2^k) = (v + 2^(k-1) - 1 + ((v >> k) & 1)) >> k
      let shift = fx.bcx.ins().iconst(types::I32, i64::from(self.shift));
      let quot = fx.bcx.ins().sshr(value, shift);
      let one = splat_i16(fx, 1);
      let odd = fx.bcx.ins().band(quot, one);
      let bias = splat_i16(fx, (1 << (self.shift - 1)) - 1);
      let biased = fx.bcx.ins().iadd(value, bias);
      let biased = fx.bcx.ins().iadd(biased, odd);
      fx.bcx.ins().sshr(biased, shift)
    };

    // Unsigned saturating narrow clamps to [0, 255].
    let narrowed = fx.bcx.ins().unarrow(rounded, rounded);
    let byte_flags = MemFlagsData::new().with_endianness(Endianness::Little);
    let as_i64x2 = fx.bcx.ins().bitcast(types::I64X2, byte_flags, narrowed);
    let low = fx.bcx.ins().extractlane(as_i64x2, 0);
    Pointer::new(dst_row)
      .offset_value(fx, x)
      .store(fx, low, MemFlagsData::trusted());
  }
}

const fn fits_i16(v: i64) -> bool {
  v >= i16::MIN as i64 && v <= i16::MAX as i64
}

#[allow(clippy::cast_possible_truncation, clippy::float_cmp)]
fn exact_int(v: f32) -> Option<i64> {
  (v.trunc() == v && v.abs() <= 32768.0).then_some(v as i64)
}

fn is_u8_clip(clip: &str, src_types: &[ComponentType]) -> bool {
  resolve_clip_name(clip, src_types).is_ok_and(|idx| src_types[idx] == ComponentType::U8)
}

/// Returns the inclusive range of `expr` if it is integer-valued and every
/// intermediate value fits in `i16`.
fn value_range(expr: &Expr, src_types: &[ComponentType]) -> Option<(i64, i64)> {
  let range = match expr {
    Expr::Lit(v) => {
      let v = exact_int(*v)?;
      (v, v)
    }
    Expr::Ident(clip) | Expr::RelAccess { clip, .. } if is_u8_clip(clip, src_types) => (0, 255),
    Expr::Unary(UnOp::Neg, x) => {
      let (lo, hi) = value_range(x, src_types)?;
      (-hi, -lo)
    }
    Expr::Binary(op @ (BinOp::Add | BinOp::Sub | BinOp::Mul), lhs, rhs) => {
      let (a_lo, a_hi) = value_range(lhs, src_types)?;
      let (b_lo, b_hi) = value_range(rhs, src_types)?;
      match op {
        BinOp::Add => (a_lo + b_lo, a_hi + b_hi),
        BinOp::Sub => (a_lo - b_hi, a_hi - b_lo),
        BinOp::Mul => {
          let products = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi];
          (
            *products.iter().min().unwrap(),
            *products.iter().max().unwrap(),
          )
        }
        _ => unreachable!(),
      }
    }
    _ => return None,
  };
  (fits_i16(range.0) && fits_i16(range.1)).then_some(range)
}

fn splat_i16(fx: &mut FunctionCx<'_, '_>, v: i64) -> Value {
  let scalar = fx.bcx.ins().iconst(types::I16, v);
  fx.bcx.ins().splat(INT_VEC_TYPE, scalar)
}

fn load_u8x8(fx: &mut FunctionCx<'_, '_>, row: Value, x: Value, rel_x: i32) -> Value {
  let addr = fx.bcx.ins().iadd(row, x);
  fx.bcx.ins().uload8x8(SRC_MEMFLAGS, addr, rel_x)
}

fn translate_int(fx: &mut FunctionCx<'_, '_>, expr: &Expr, x: Value) -> Value {
  match expr {
    Expr::Lit(v) => splat_i16(fx, exact_int(*v).unwrap()),
    Expr::Ident(clip) => {
      let clip_idx = resolve_clip_name(clip, &fx.src_types).unwrap();
      let row = fx.row_ptrs[&(clip_idx, 0, BoundaryMode::Clamp)];
      load_u8x8(fx, row, x, 0)
    }
    Expr::RelAccess {
      clip,
      rel_x,
      rel_y,
      boundary_mode,
    } => {
      let clip_idx = resolve_clip_name(clip, &fx.src_types).unwrap();
      let mode = boundary_mode.unwrap_or(fx.boundary_mode);
      let row = fx.row_ptrs[&(clip_idx, *rel_y, mode)];
      load_u8x8(fx, row, x, *rel_x)
    }
    Expr::Unary(UnOp::Neg, inner) => {
      let inner = translate_int(fx, inner, x);
      fx.bcx.ins().ineg(inner)
    }
    Expr::Binary(op, lhs, rhs) => {
      // Scale by powers of two with a shift.
      if *op == BinOp::Mul {
        for (value, factor) in [(lhs, rhs), (rhs, lhs)] {
          if let Expr::Lit(factor) = factor.as_ref()
            && let Some(factor) = exact_int(*factor)
            && factor > 0
            && factor.count_ones() == 1
          {
            let value = translate_int(fx, value, x);
            let shift = fx
              .bcx
              .ins()
              .iconst(types::I32, i64::from(factor.trailing_zeros()));
            return fx.bcx.ins().ishl(value, shift);
          }
        }
      }
      let lhs = translate_int(fx, lhs, x);
      let rhs = translate_int(fx, rhs, x);
      match op {
        BinOp::Add => fx.bcx.ins().iadd(lhs, rhs),
        BinOp::Sub => fx.bcx.ins().isub(lhs, rhs),
        BinOp::Mul => fx.bcx.ins().imul(lhs, rhs),
        _ => unreachable!(),
      }
    }
    _ => unreachable!(),
  }
}
