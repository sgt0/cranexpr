use std::ptr::NonNull;

use cranelift::prelude::*;
use cranexpr_ast::{BinOp, Expr, TernaryOp, UnOp};

use crate::context::{FunctionCx, VEC_TYPE};
use crate::errors::CodegenError;
use crate::math_simd::{
  SinCos, atan2_simd, bool_to_float_simd, codegen_float_binop_simd, exp_simd, fmax_simd, fmin_simd,
  fpow_by_squaring, log_simd, sin_cos_simd, splat_f32, vselect_f32x4,
};
use crate::pixel_access::{
  resolve_clip_name, translate_abs_access_simd, translate_rel_access_simd,
};

pub(crate) fn translate_expr_simd(
  fx: &mut FunctionCx<'_, '_>,
  expr: &Expr,
) -> Result<Value, CodegenError> {
  let cache_key = NonNull::from(expr);
  if let Some(&cached) = fx.cache.get(&cache_key) {
    return Ok(cached);
  }
  let result = translate_expr_simd_inner(fx, expr)?;
  fx.cache.insert(cache_key, result);
  Ok(result)
}

fn translate_expr_simd_inner(
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
    Expr::Binary(BinOp::Atan2, y, x) => {
      let y = translate_expr_simd(fx, y)?;
      let x = translate_expr_simd(fx, x)?;
      Ok(atan2_simd(fx, y, x))
    }
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
          let min_result = fmin_simd(fx, a, max_val);
          fmax_simd(fx, min_result, min_val)
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
        UnOp::Tangent => sin_cos_simd(fx, x, SinCos::Tan),
        UnOp::Exp => exp_simd(fx, x),
        UnOp::Log => log_simd(fx, x),
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
    if *exp == exp_u32 as f32 && exp_u32 >= 3 {
      return Ok(fpow_by_squaring(fx, base_val, exp_u32));
    }
    let exp_i32 = *exp as i32;
    if *exp == exp_i32 as f32 && exp_i32 <= -1 {
      let abs_exp = exp_i32.unsigned_abs();
      let pow_abs = if abs_exp == 1 {
        base_val
      } else if abs_exp == 2 {
        fx.bcx.ins().fmul(base_val, base_val)
      } else {
        fpow_by_squaring(fx, base_val, abs_exp)
      };
      let one = splat_f32(fx, 1.0);
      return Ok(fx.bcx.ins().fdiv(one, pow_abs));
    }
  }

  let exp_val = translate_expr_simd(fx, exponent)?;
  let log_x = log_simd(fx, base_val);
  let prod = fx.bcx.ins().fmul(log_x, exp_val);
  Ok(exp_simd(fx, prod))
}
