use cranelift::{codegen::ir::BlockArg, prelude::*};
use cranelift_module::{Linkage, Module};

use crate::{
  codegen::compiler::FunctionCx,
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
    Expr::Ident(name) => {
      let variable = fx
        .variables
        .get(name)
        .ok_or_else(|| CranexprError::UndefinedVariable(name.clone()))?;
      Ok(fx.bcx.use_var(*variable))
    }
    Expr::IfElse(condition, then_body, else_body) => {
      translate_if_else(fx, condition, then_body, else_body)
    }
  }
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
    BinOp::Gt | BinOp::Lt => {
      let float_cc = match op {
        BinOp::Gt => FloatCC::GreaterThan,
        BinOp::Lt => FloatCC::LessThan,
        _ => unreachable!("{:?}({:?}, {:?})", op, lhs, rhs),
      };
      let result = fx.bcx.ins().fcmp(float_cc, lhs, rhs);
      bool_to_float(fx, result)
    }
    BinOp::Max => fx.bcx.ins().fmax(lhs, rhs),
    BinOp::Min => fx.bcx.ins().fmin(lhs, rhs),
  }
}

fn bool_to_float(fx: &mut FunctionCx<'_, '_>, value: Value) -> Value {
  let one = fx.bcx.ins().f32const(1.0);
  let zero = fx.bcx.ins().f32const(0.0);
  fx.bcx.ins().select(value, one, zero)
}
