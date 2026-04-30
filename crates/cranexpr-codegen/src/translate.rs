use std::f32::consts::PI;

use cranelift::{
  codegen::ir::{BlockArg, immediates::Offset32},
  prelude::*,
};
use cranelift_module::{Linkage, Module};
use cranexpr_ast::{BinOp, BoundaryMode, Expr, TernaryOp, UnOp};

use crate::{
  compiler::{FunctionCx, SRC_MEMFLAGS},
  component_type::ComponentType,
  errors::CodegenError,
  pointer::Pointer,
};

pub(crate) fn translate_expr(
  fx: &mut FunctionCx<'_, '_>,
  expr: &Expr,
) -> Result<Value, CodegenError> {
  match expr {
    Expr::Lit(literal) => Ok(fx.bcx.ins().f32const(*literal)),
    Expr::Prop(name, prop) => {
      let clip_idx = resolve_clip_name(name, &fx.src_types)?;
      let variable = fx
        .variables
        .get(&format!("prop_{clip_idx}_{prop}"))
        .expect("frame property variable not defined");
      Ok(fx.bcx.use_var(*variable))
    }
    Expr::Binary(BinOp::Pow, base, exponent) => Ok(codegen_pow(fx, base, exponent)?),
    Expr::Binary(op, lhs, rhs) => {
      let lhs = translate_expr(fx, lhs)?;
      let rhs = translate_expr(fx, rhs)?;
      Ok(codegen_float_binop(fx, *op, lhs, rhs))
    }
    Expr::Ternary(op, a, b, c) => {
      let a = translate_expr(fx, a)?;
      let b = translate_expr(fx, b)?;
      let c = translate_expr(fx, c)?;
      Ok(match op {
        TernaryOp::Clip => {
          // clip(x, min, max) is equivalent to `max(min(x, max), min)`.
          let min_val = b;
          let max_val = c;
          let min_result = fx.bcx.ins().fmin(a, max_val);
          fx.bcx.ins().fmax(min_result, min_val)
        }
      })
    }
    Expr::Unary(op, x) => {
      let x = translate_expr(fx, x)?;
      Ok(match op {
        UnOp::Abs => fx.bcx.ins().fabs(x),
        UnOp::Cosine => translate_float_intrinsic_call(fx, "cosf", &[x]),
        UnOp::Exp => translate_float_intrinsic_call(fx, "expf", &[x]),
        UnOp::Floor => fx.bcx.ins().floor(x),
        UnOp::Round => translate_float_intrinsic_call(fx, "roundf", &[x]),
        UnOp::Log => translate_float_intrinsic_call(fx, "logf", &[x]),
        UnOp::Neg => fx.bcx.ins().fneg(x),
        UnOp::BitNot => {
          let rounded = fx.bcx.ins().nearest(x);
          let i = fx.bcx.ins().fcvt_to_sint(types::I32, rounded);
          let res = fx.bcx.ins().bnot(i);
          fx.bcx.ins().fcvt_from_sint(types::F32, res)
        }
        UnOp::Not => {
          // A value is considered truthy if and only if it is greater than 0.
          // Therefore, logical inversion is equivalent to a value being less
          // than or equal to 0.
          let zero = fx.bcx.ins().f32const(0.0);
          let condition = fx.bcx.ins().fcmp(FloatCC::LessThanOrEqual, x, zero);
          bool_to_float(fx, condition)
        }
        UnOp::Sign => {
          let zero = fx.bcx.ins().f32const(0.0);
          let one = fx.bcx.ins().f32const(1.0);

          let is_zero = fx.bcx.ins().fcmp(FloatCC::Equal, x, zero);
          let sign = fx.bcx.ins().fcopysign(one, x);
          fx.bcx.ins().select(is_zero, zero, sign)
        }
        UnOp::Tangent => translate_float_intrinsic_call(fx, "tanf", &[x]),
        UnOp::Sine => translate_float_intrinsic_call(fx, "sinf", &[x]),
        UnOp::Sqrt => fx.bcx.ins().sqrt(x),
        UnOp::Trunc => fx.bcx.ins().trunc(x),
      })
    }

    // Globals.
    Expr::Ident(name) => {
      let variable = if let Ok(clip_idx) = resolve_clip_name(name, &fx.src_types) {
        fx.variables.get(&format!("src{clip_idx}"))
      } else {
        fx.variables.get(name)
      }
      .ok_or_else(|| CodegenError::UndefinedVariable(name.clone()))?;

      let val = fx.bcx.use_var(*variable);
      Ok(if fx.bcx.func.dfg.value_type(val) == types::I64 {
        fx.bcx.ins().fcvt_from_uint(types::F32, val)
      } else {
        val
      })
    }

    // User variables.
    Expr::Load(name) => {
      let variable = fx
        .user_variables
        .get(name)
        .ok_or_else(|| CodegenError::UndefinedVariable(name.clone()))?;
      Ok(fx.bcx.use_var(*variable))
    }

    Expr::IfElse(condition, then_body, else_body) => {
      translate_if_else(fx, condition, then_body, else_body)
    }
    Expr::Store(name, expr) => {
      let value = translate_expr(fx, expr)?;
      let variable = if let Some(variable) = fx.user_variables.get(name) {
        *variable
      } else {
        let var = fx.bcx.declare_var(types::F32);
        fx.user_variables.insert(name.clone(), var);
        var
      };
      fx.bcx.def_var(variable, value);
      Ok(value)
    }
    Expr::AbsAccess {
      clip,
      x,
      y,
      boundary_mode,
    } => {
      let x_val = translate_expr(fx, x)?;
      let y_val = translate_expr(fx, y)?;

      // Resolve clip name to clip index.
      let clip_idx = resolve_clip_name(clip, &fx.src_types)?;
      let src_type = fx.src_types[clip_idx];

      // Round and convert to integer.
      let x_nearest = fx.bcx.ins().nearest(x_val);
      let y_nearest = fx.bcx.ins().nearest(y_val);
      let x_int = fx.bcx.ins().fcvt_to_sint(types::I64, x_nearest);
      let y_int = fx.bcx.ins().fcvt_to_sint(types::I64, y_nearest);

      // Boundary handling.
      let boundary_mode = boundary_mode.unwrap_or(fx.boundary_mode);
      let clamped_x = codegen_boundary_mode(fx, x_int, fx.width, boundary_mode);
      let clamped_y = codegen_boundary_mode(fx, y_int, fx.height, boundary_mode);

      let (src_ptr, pixel_offset) =
        codegen_pixel_offset(fx, clip_idx, clamped_x, clamped_y, src_type);

      // Load pixel value.
      let val = src_ptr
        .offset_value(fx, pixel_offset)
        .load(fx, src_type.into(), SRC_MEMFLAGS);

      // Convert to float.
      let val = match src_type {
        ComponentType::U8 | ComponentType::U16 => fx.bcx.ins().fcvt_from_uint(types::F32, val),
        ComponentType::F32 => val,
      };

      Ok(val)
    }
    Expr::RelAccess {
      clip,
      rel_x,
      rel_y,
      boundary_mode,
    } => {
      if let Some(existing) = fx.variables.get(&format!("{clip}[{rel_x},{rel_y}]")) {
        return Ok(fx.bcx.use_var(*existing));
      }

      // Resolve clip name to clip index
      let clip_idx = resolve_clip_name(clip, &fx.src_types)?;
      let src_type = fx.src_types[clip_idx];

      let x_var = fx
        .variables
        .get("X")
        .ok_or_else(|| CodegenError::UndefinedVariable("X".to_string()))?;
      let y_var = fx
        .variables
        .get("Y")
        .ok_or_else(|| CodegenError::UndefinedVariable("Y".to_string()))?;

      let x_coord = fx.bcx.use_var(*x_var);
      let y_coord = fx.bcx.use_var(*y_var);

      // Target coordinates.
      let rel_x_val = fx.bcx.ins().iconst(types::I64, i64::from(*rel_x));
      let rel_y_val = fx.bcx.ins().iconst(types::I64, i64::from(*rel_y));
      let target_x = fx.bcx.ins().iadd(x_coord, rel_x_val);
      let target_y = fx.bcx.ins().iadd(y_coord, rel_y_val);

      // Boundary handling.
      let boundary_mode = boundary_mode.unwrap_or(fx.boundary_mode);
      let clamped_x = codegen_boundary_mode(fx, target_x, fx.width, boundary_mode);
      let clamped_y = codegen_boundary_mode(fx, target_y, fx.height, boundary_mode);

      let (src_ptr, pixel_offset) =
        codegen_pixel_offset(fx, clip_idx, clamped_x, clamped_y, src_type);

      // Load pixel value.
      let val = src_ptr
        .offset_value(fx, pixel_offset)
        .load(fx, src_type.into(), SRC_MEMFLAGS);

      // Convert to float.
      let val = match src_type {
        ComponentType::U8 | ComponentType::U16 => fx.bcx.ins().fcvt_from_uint(types::F32, val),
        ComponentType::F32 => val,
      };

      let var = fx.bcx.declare_var(types::F32);
      fx.variables.insert(format!("{clip}[{rel_x},{rel_y}]"), var);
      fx.bcx.def_var(var, val);

      Ok(fx.bcx.use_var(var))
    }
  }
}

/// Computes the byte offset for a pixel at (x, y).
fn codegen_pixel_offset(
  fx: &mut FunctionCx<'_, '_>,
  clip_idx: usize,
  x: Value,
  y: Value,
  src_type: ComponentType,
) -> (Pointer, Value) {
  // Load source pointer for this clip.
  let pointer_size = fx.pointer_type.bytes() as i32;
  let src_ptr_val = fx
    .src_clips
    .offset(fx, Offset32::new(clip_idx as i32 * pointer_size))
    .load(fx, fx.pointer_type, SRC_MEMFLAGS);
  let src_ptr = Pointer::new(src_ptr_val);

  // Load stride for this clip.
  let src_stride = fx
    .src_strides
    .offset(fx, Offset32::new(clip_idx as i32 * 8))
    .load(fx, types::I64, SRC_MEMFLAGS);

  // pixel_offset = y * stride + x * bytes_per_sample
  let row_offset = fx.bcx.ins().imul(y, src_stride);
  let col_offset = fx.bcx.ins().imul_imm(x, src_type.bytes() as i64);
  let pixel_offset = fx.bcx.ins().iadd(row_offset, col_offset);

  (src_ptr, pixel_offset)
}

/// Resolves a clip name (e.g., "x", "y", "src0") to a clip index.
fn resolve_clip_name(clip: &str, src_types: &[ComponentType]) -> Result<usize, CodegenError> {
  // Check if it's a shorthand (x, y, z, a, b, ...)
  if clip.len() == 1 {
    let ch = clip.chars().next().unwrap();
    if ('x'..='z').contains(&ch) {
      let idx = (ch as u8 - b'x') as usize;
      if idx < src_types.len() {
        return Ok(idx);
      }
    } else if ch.is_ascii_lowercase() {
      let idx = (ch as u8 - b'a' + 3) as usize;
      if idx < src_types.len() {
        return Ok(idx);
      }
    }
  }

  // Check if it's srcN format
  if let Some(stripped) = clip.strip_prefix("src")
    && let Ok(idx) = stripped.parse::<usize>()
    && idx < src_types.len()
  {
    return Ok(idx);
  }

  Err(CodegenError::UndefinedVariable(clip.to_string()))
}

fn translate_if_else(
  fx: &mut FunctionCx<'_, '_>,
  condition: &Expr,
  then_body: &Expr,
  else_body: &Expr,
) -> Result<Value, CodegenError> {
  // A value is considered truthy if and only if it is greater than 0.
  let zero = fx.bcx.ins().f32const(0.0);
  let mut condition_value = translate_expr(fx, condition)?;
  condition_value = fx
    .bcx
    .ins()
    .fcmp(FloatCC::GreaterThan, condition_value, zero);

  let then_block = fx.bcx.create_block();
  let else_block = fx.bcx.create_block();
  let merge_block = fx.bcx.create_block();

  fx.bcx.append_block_param(merge_block, types::F32);

  // Test the condition and then conditionally branch.
  fx.bcx
    .ins()
    .brif(condition_value, then_block, &[], else_block, &[]);

  fx.bcx.switch_to_block(then_block);
  fx.bcx.seal_block(then_block);
  let then_return = translate_expr(fx, then_body)?;

  // Jump to the merge block, passing it the block return value.
  fx.bcx
    .ins()
    .jump(merge_block, &[BlockArg::Value(then_return)]);

  fx.bcx.switch_to_block(else_block);
  fx.bcx.seal_block(else_block);
  let else_return = translate_expr(fx, else_body)?;

  // Jump to the merge block, passing it the block return value.
  fx.bcx
    .ins()
    .jump(merge_block, &[BlockArg::Value(else_return)]);

  // Switch to the merge block for subsequent statements.
  fx.bcx.switch_to_block(merge_block);

  // We've now seen all the predecessors of the merge block.
  fx.bcx.seal_block(merge_block);

  // Read the value of the if-else by reading the merge block
  // parameter.
  Ok(fx.bcx.block_params(merge_block)[0])
}

fn translate_float_intrinsic_call(
  fx: &mut FunctionCx<'_, '_>,
  name: &str,
  args: &[Value],
) -> Value {
  let mut sig = fx.module.make_signature();
  sig
    .params
    .extend(args.iter().map(|_| AbiParam::new(types::F32)));
  sig.returns.push(AbiParam::new(types::F32));

  let callee = fx
    .module
    .declare_function(name, Linkage::Import, &sig)
    .unwrap();
  let local_callee = fx.module.declare_func_in_func(callee, fx.bcx.func);
  let call = fx.bcx.ins().call(local_callee, args);
  fx.bcx.inst_results(call)[0]
}

fn codegen_float_binop(fx: &mut FunctionCx<'_, '_>, op: BinOp, lhs: Value, rhs: Value) -> Value {
  match op {
    BinOp::Add => fx.bcx.ins().fadd(lhs, rhs),
    BinOp::Sub => fx.bcx.ins().fsub(lhs, rhs),
    BinOp::Mul => fx.bcx.ins().fmul(lhs, rhs),
    BinOp::Div => fx.bcx.ins().fdiv(lhs, rhs),
    BinOp::Rem => translate_float_intrinsic_call(fx, "fmodf", &[lhs, rhs]),
    BinOp::Gt | BinOp::Gte | BinOp::Lt | BinOp::Lte | BinOp::Eq => {
      let float_cc = match op {
        BinOp::Gt => FloatCC::GreaterThan,
        BinOp::Gte => FloatCC::GreaterThanOrEqual,
        BinOp::Lt => FloatCC::LessThan,
        BinOp::Lte => FloatCC::LessThanOrEqual,
        BinOp::Eq => FloatCC::Equal,
        _ => unreachable!("{:?}({:?}, {:?})", op, lhs, rhs),
      };
      let result = fx.bcx.ins().fcmp(float_cc, lhs, rhs);
      bool_to_float(fx, result)
    }
    BinOp::Max => fx.bcx.ins().fmax(lhs, rhs),
    BinOp::Min => fx.bcx.ins().fmin(lhs, rhs),
    BinOp::Atan2 => codegen_atan2(fx, lhs, rhs),
    BinOp::And => {
      let zero = fx.bcx.ins().f32const(0.0);
      let lhs_bool = fx.bcx.ins().fcmp(FloatCC::GreaterThan, lhs, zero);
      let rhs_bool = fx.bcx.ins().fcmp(FloatCC::GreaterThan, rhs, zero);
      let result = fx.bcx.ins().band(lhs_bool, rhs_bool);
      bool_to_float(fx, result)
    }
    BinOp::Or => {
      let zero = fx.bcx.ins().f32const(0.0);
      let lhs_bool = fx.bcx.ins().fcmp(FloatCC::GreaterThan, lhs, zero);
      let rhs_bool = fx.bcx.ins().fcmp(FloatCC::GreaterThan, rhs, zero);
      let result = fx.bcx.ins().bor(lhs_bool, rhs_bool);
      bool_to_float(fx, result)
    }
    BinOp::Xor => {
      let zero = fx.bcx.ins().f32const(0.0);
      let lhs_bool = fx.bcx.ins().fcmp(FloatCC::GreaterThan, lhs, zero);
      let rhs_bool = fx.bcx.ins().fcmp(FloatCC::GreaterThan, rhs, zero);
      let result = fx.bcx.ins().bxor(lhs_bool, rhs_bool);
      bool_to_float(fx, result)
    }
    BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
      let lhs_rounded = fx.bcx.ins().nearest(lhs);
      let rhs_rounded = fx.bcx.ins().nearest(rhs);
      let lhs_i = fx.bcx.ins().fcvt_to_sint(types::I32, lhs_rounded);
      let rhs_i = fx.bcx.ins().fcvt_to_sint(types::I32, rhs_rounded);
      let res_i = match op {
        BinOp::BitAnd => fx.bcx.ins().band(lhs_i, rhs_i),
        BinOp::BitOr => fx.bcx.ins().bor(lhs_i, rhs_i),
        BinOp::BitXor => fx.bcx.ins().bxor(lhs_i, rhs_i),
        _ => unreachable!(),
      };
      fx.bcx.ins().fcvt_from_sint(types::F32, res_i)
    }
    BinOp::Pow => unreachable!("pow is handled by codegen_pow"),
  }
}

fn bool_to_float(fx: &mut FunctionCx<'_, '_>, value: Value) -> Value {
  let one = fx.bcx.ins().f32const(1.0);
  let zero = fx.bcx.ins().f32const(0.0);
  fx.bcx.ins().select(value, one, zero)
}

/// Strength-reduce `pow` for small non-negative integer exponents and 0.5;
/// fall back to the `powf` libcall otherwise.
fn codegen_pow(
  fx: &mut FunctionCx<'_, '_>,
  base: &Expr,
  exponent: &Expr,
) -> Result<Value, CodegenError> {
  let base_val = translate_expr(fx, base)?;

  #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
  if let Expr::Lit(exp) = exponent {
    if *exp == 0.0 {
      return Ok(fx.bcx.ins().f32const(1.0));
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

    // For small non-negative integer exponents (3..=8), emit a chain of
    // multiplications. Reuse intermediary squares to keep the chain short.
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

  let rhs_val = translate_expr(fx, exponent)?;
  Ok(translate_float_intrinsic_call(
    fx,
    "powf",
    &[base_val, rhs_val],
  ))
}

// Benchmarked to be faster than calling the `atan2f` intrinsic.
fn codegen_atan2(fx: &mut FunctionCx<'_, '_>, y: Value, x: Value) -> Value {
  // Constants.
  let zero = fx.bcx.ins().f32const(0.0);
  let pi = fx.bcx.ins().f32const(PI);
  let half_pi = fx.bcx.ins().f32const(PI / 2.0);

  // atan(y/x)
  let y_div_x = codegen_float_binop(fx, BinOp::Div, y, x);
  let atan_y_div_x = translate_float_intrinsic_call(fx, "atanf", &[y_div_x]);

  // If x is negative, the result will be atan(y/x), then:
  //   If y is positive, add pi.
  //   If y is negative, subtract pi.
  let signed_pi = fx.bcx.ins().fcopysign(pi, y);
  let negative_x_result = codegen_float_binop(fx, BinOp::Add, atan_y_div_x, signed_pi);

  // If x is zero, the result will be pi/2 with the sign of y.
  let zero_x_result = fx.bcx.ins().fcopysign(half_pi, y);

  // If x > 0: atan(y/x)
  // Else if x < 0: negative_x_result
  // Else (x = 0): zero_x_result
  let x_gt_0 = fx.bcx.ins().fcmp(FloatCC::GreaterThan, x, zero);
  let x_lt_0 = fx.bcx.ins().fcmp(FloatCC::LessThan, x, zero);
  let inner_select = fx.bcx.ins().select(x_gt_0, atan_y_div_x, zero_x_result);
  let result = fx.bcx.ins().select(x_lt_0, negative_x_result, inner_select);

  // atan2(0, 0) is undefined.
  let x_is_zero = fx.bcx.ins().fcmp(FloatCC::Equal, x, zero);
  let y_is_zero = fx.bcx.ins().fcmp(FloatCC::Equal, y, zero);
  let both_zero = fx.bcx.ins().band(x_is_zero, y_is_zero);

  fx.bcx.ins().select(both_zero, zero, result)
}

/// Applies boundary mode to a coordinate value.
pub(crate) fn codegen_boundary_mode(
  fx: &mut FunctionCx<'_, '_>,
  coord: Value,
  max: Value,
  mode: BoundaryMode,
) -> Value {
  let zero = fx.bcx.ins().iconst(types::I64, 0);
  let one = fx.bcx.ins().iconst(types::I64, 1);

  match mode {
    // Clamp: clamp(coord, 0, max - 1)
    BoundaryMode::Clamp => {
      let max_minus_one = fx.bcx.ins().isub(max, one);

      // First clamp to max-1
      let is_too_large = fx
        .bcx
        .ins()
        .icmp(IntCC::SignedGreaterThan, coord, max_minus_one);
      let clamped_max = fx.bcx.ins().select(is_too_large, max_minus_one, coord);

      // Then clamp to 0
      let is_too_small = fx.bcx.ins().icmp(IntCC::SignedLessThan, clamped_max, zero);
      fx.bcx.ins().select(is_too_small, zero, clamped_max)
    }

    // Mirror: reflect at edges
    // If coord < 0: mirror = -coord - 1
    // If coord >= max: mirror = 2 * max - coord - 1
    // Else: mirror = coord
    BoundaryMode::Mirror => {
      let is_negative = fx.bcx.ins().icmp(IntCC::SignedLessThan, coord, zero);
      let is_too_large = fx
        .bcx
        .ins()
        .icmp(IntCC::SignedGreaterThanOrEqual, coord, max);

      // Calculate negative mirror: -coord - 1
      let neg_mirror = fx.bcx.ins().isub(zero, coord);
      let neg_mirror = fx.bcx.ins().isub(neg_mirror, one);

      // Calculate positive mirror: 2 * max - coord - 1
      let two = fx.bcx.ins().iconst(types::I64, 2);
      let two_max = fx.bcx.ins().imul(max, two);
      let pos_mirror = fx.bcx.ins().isub(two_max, coord);
      let pos_mirror = fx.bcx.ins().isub(pos_mirror, one);

      // Select based on conditions: first handle negative, then too large
      let result = fx.bcx.ins().select(is_negative, neg_mirror, coord);
      fx.bcx.ins().select(is_too_large, pos_mirror, result)
    }
  }
}
