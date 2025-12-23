use std::f32::consts::PI;

use cranelift::{
  codegen::ir::{BlockArg, immediates::Offset32},
  prelude::*,
};
use cranelift_module::{Linkage, Module};

use crate::{
  codegen::{
    compiler::{FunctionCx, SRC_MEMFLAGS, apply_boundary_mode},
    pixel_type::PixelType,
    pointer::Pointer,
  },
  errors::CranexprError,
  parser::ast::{BinOp, Expr, TernaryOp, UnOp},
};

pub(crate) fn translate_expr(
  fx: &mut FunctionCx<'_, '_>,
  expr: &Expr,
) -> Result<Value, CranexprError> {
  match expr {
    Expr::Lit(literal) => Ok(fx.bcx.ins().f32const(*literal)),
    Expr::Prop(name, prop) => {
      let variable = fx
        .variables
        .get(&format!("{name}.{prop}"))
        .expect("frame property variable not defined");
      Ok(fx.bcx.use_var(*variable))
    }
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
      })
    }

    // TODO: globals and user variables should reference different underlying
    // storage.
    Expr::Ident(name) | Expr::Load(name) => {
      let variable = fx
        .variables
        .get(name)
        .ok_or_else(|| CranexprError::UndefinedVariable(name.clone()))?;
      let val = fx.bcx.use_var(*variable);
      Ok(if fx.bcx.func.dfg.value_type(val) == types::I64 {
        fx.bcx.ins().fcvt_from_uint(types::F32, val)
      } else {
        val
      })
    }

    Expr::IfElse(condition, then_body, else_body) => {
      translate_if_else(fx, condition, then_body, else_body)
    }
    Expr::Store(name, expr) => {
      let value = translate_expr(fx, expr)?;
      let variable = if let Some(variable) = fx.variables.get(name) {
        *variable
      } else {
        let var = fx.bcx.declare_var(types::F32);
        fx.variables.insert(name.clone(), var);
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
      let clamped_x = apply_boundary_mode(fx, x_int, fx.width, boundary_mode);
      let clamped_y = apply_boundary_mode(fx, y_int, fx.height, boundary_mode);

      // idx = y * width + x
      let y_times_width = fx.bcx.ins().imul(clamped_y, fx.width);
      let target_idx = fx.bcx.ins().iadd(y_times_width, clamped_x);

      // Load source pointer for this clip.
      let pointer_size = fx.pointer_type.bytes() as i32;
      let src_ptr_val = fx
        .src_clips
        .offset(fx, Offset32::new(clip_idx as i32 * pointer_size))
        .load(fx, fx.pointer_type, SRC_MEMFLAGS);
      let src_ptr = Pointer::new(src_ptr_val);

      // Calculate byte offset for the pixel.
      let pixel_offset = fx.bcx.ins().imul_imm(target_idx, src_type.bytes() as i64);
      let pixel_ptr = src_ptr.offset_value(fx, pixel_offset);

      // Load pixel value.
      let val = pixel_ptr.load(fx, src_type.into(), SRC_MEMFLAGS);

      // Convert to float.
      let val = match src_type {
        PixelType::U8 | PixelType::U16 => fx.bcx.ins().fcvt_from_uint(types::F32, val),
        PixelType::F32 => val,
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
        .ok_or_else(|| CranexprError::UndefinedVariable("X".to_string()))?;
      let y_var = fx
        .variables
        .get("Y")
        .ok_or_else(|| CranexprError::UndefinedVariable("Y".to_string()))?;

      let x_coord = fx.bcx.use_var(*x_var);
      let y_coord = fx.bcx.use_var(*y_var);

      // Target coordinates.
      let rel_x_val = fx.bcx.ins().iconst(types::I64, i64::from(*rel_x));
      let rel_y_val = fx.bcx.ins().iconst(types::I64, i64::from(*rel_y));
      let target_x = fx.bcx.ins().iadd(x_coord, rel_x_val);
      let target_y = fx.bcx.ins().iadd(y_coord, rel_y_val);

      // Boundary handling.
      let boundary_mode = boundary_mode.unwrap_or(fx.boundary_mode);
      let clamped_x = apply_boundary_mode(fx, target_x, fx.width, boundary_mode);
      let clamped_y = apply_boundary_mode(fx, target_y, fx.height, boundary_mode);

      // idx = y * width + x
      let y_times_width = fx.bcx.ins().imul(clamped_y, fx.width);
      let target_idx = fx.bcx.ins().iadd(y_times_width, clamped_x);

      // Load source pointer for this clip.
      let pointer_size = fx.pointer_type.bytes() as i32;
      let src_ptr_val = fx
        .src_clips
        .offset(fx, Offset32::new(clip_idx as i32 * pointer_size))
        .load(fx, fx.pointer_type, SRC_MEMFLAGS);
      let src_ptr = Pointer::new(src_ptr_val);

      // Calculate byte offset for the pixel.
      let pixel_offset = fx.bcx.ins().imul_imm(target_idx, src_type.bytes() as i64);
      let pixel_ptr = src_ptr.offset_value(fx, pixel_offset);

      // Load pixel value.
      let val = pixel_ptr.load(fx, src_type.into(), SRC_MEMFLAGS);

      // Convert to float.
      let val = match src_type {
        PixelType::U8 | PixelType::U16 => fx.bcx.ins().fcvt_from_uint(types::F32, val),
        PixelType::F32 => val,
      };

      let var = fx.bcx.declare_var(types::F32);
      fx.variables.insert(format!("{clip}[{rel_x},{rel_y}]"), var);
      fx.bcx.def_var(var, val);

      Ok(fx.bcx.use_var(var))
    }
  }
}

/// Resolves a clip name (e.g., "x", "y", "src0") to a clip index.
fn resolve_clip_name(
  clip: &str,
  src_types: &[crate::codegen::pixel_type::PixelType],
) -> Result<usize, CranexprError> {
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

  Err(CranexprError::UndefinedVariable(clip.to_string()))
}

fn translate_if_else(
  fx: &mut FunctionCx<'_, '_>,
  condition: &Expr,
  then_body: &Expr,
  else_body: &Expr,
) -> Result<Value, CranexprError> {
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
    BinOp::Pow => translate_float_intrinsic_call(fx, "powf", &[lhs, rhs]),
    BinOp::Rem => translate_float_intrinsic_call(fx, "fmodf", &[lhs, rhs]),
    BinOp::Gt | BinOp::Lt | BinOp::Eq => {
      let float_cc = match op {
        BinOp::Gt => FloatCC::GreaterThan,
        BinOp::Lt => FloatCC::LessThan,
        BinOp::Eq => FloatCC::Equal,
        _ => unreachable!("{:?}({:?}, {:?})", op, lhs, rhs),
      };
      let result = fx.bcx.ins().fcmp(float_cc, lhs, rhs);
      bool_to_float(fx, result)
    }
    BinOp::Max => fx.bcx.ins().fmax(lhs, rhs),
    BinOp::Min => fx.bcx.ins().fmin(lhs, rhs),
    BinOp::Atan2 => codegen_atan2(fx, lhs, rhs),
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
  }
}

fn bool_to_float(fx: &mut FunctionCx<'_, '_>, value: Value) -> Value {
  let one = fx.bcx.ins().f32const(1.0);
  let zero = fx.bcx.ins().f32const(0.0);
  fx.bcx.ins().select(value, one, zero)
}

// Benchmarked to be faster than calling the `atan2f` intrinsic.
fn codegen_atan2(fx: &mut FunctionCx<'_, '_>, y: Value, x: Value) -> Value {
  // Constants.
  let zero = fx.bcx.ins().f32const(0.0);
  let pi = fx.bcx.ins().f32const(PI);
  let half_pi = fx.bcx.ins().f32const(PI / 2.0);

  // atan(y/x)
  let y_div_x = fx.bcx.ins().fdiv(y, x);
  let atan_y_div_x = translate_float_intrinsic_call(fx, "atanf", &[y_div_x]);

  // If x is negative, the result will be atan(y/x), then:
  //   If y is positive, add pi.
  //   If y is negative, subtract pi.
  let signed_pi = fx.bcx.ins().fcopysign(pi, y);
  let negative_x_result = fx.bcx.ins().fadd(atan_y_div_x, signed_pi);

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
