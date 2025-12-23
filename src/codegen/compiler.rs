use std::collections::HashMap;
use std::sync::Arc;

use cranelift::codegen::ir::immediates::Offset32;
use cranelift::codegen::isa::TargetIsa;
use cranelift::codegen::print_errors::pretty_error;
use cranelift::prelude::*;
use cranelift::{codegen::Context, prelude::FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, ModuleError};
use nanoid::nanoid;

use crate::BoundaryMode;
use crate::codegen::pointer::Pointer;
use crate::codegen::translate::translate_expr;
use crate::codegen::{MainFunction, pixel_type::PixelType};
use crate::errors::{CranexprError, CranexprResult};
use crate::parser::ast::Expr;
use crate::parser::{self};

pub(crate) const SRC_MEMFLAGS: MemFlags = MemFlags::trusted().with_readonly().with_can_move();

pub(crate) struct FunctionCx<'m, 'clif> {
  pub(crate) module: &'m mut dyn Module,
  pub(crate) pointer_type: Type,

  pub(crate) bcx: FunctionBuilder<'clif>,
  pub(crate) variables: HashMap<String, Variable>,

  #[allow(dead_code)]
  pub(crate) dst_type: PixelType,
  #[allow(dead_code)]
  pub(crate) src_types: Vec<PixelType>,

  pub(crate) boundary_mode: BoundaryMode,

  // For relative pixel access
  pub(crate) src_clips: Pointer,
  pub(crate) width: Value,
  pub(crate) height: Value,
}

fn build_isa() -> Arc<dyn TargetIsa + 'static> {
  let mut flag_builder = settings::builder();
  let enable_verifier = if cfg!(debug_assertions) {
    "true"
  } else {
    "false"
  };
  flag_builder
    .set("enable_verifier", enable_verifier)
    .unwrap();
  flag_builder
    .set("regalloc_checker", enable_verifier)
    .unwrap();
  flag_builder.set("opt_level", "speed_and_size").unwrap();

  let flags = settings::Flags::new(flag_builder);

  // TODO: may need to support non-native targets.
  let isa_builder = cranelift_native::builder_with_options(true).unwrap();

  match isa_builder.finish(flags) {
    Ok(target_isa) => target_isa,
    Err(err) => panic!("failed to build TargetIsa: {err}"),
  }
}

fn create_jit_module() -> JITModule {
  let isa = build_isa();
  let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
  JITModule::new(jit_builder)
}

pub(crate) fn compile_jit(
  input: &str,
  dst_type: PixelType,
  src_types: &[PixelType],
  boundary_mode: Option<BoundaryMode>,
) -> Result<MainFunction, CranexprError> {
  let ast = parser::parse_expr(input)?;

  let mut jit_module = create_jit_module();
  let main_func_id = create_entry_fn(&mut jit_module, &ast, dst_type, src_types, boundary_mode)?;
  jit_module.finalize_definitions().unwrap();

  let finalized_main = jit_module.get_finalized_function(main_func_id);
  Ok(MainFunction::from_ptr(finalized_main))
}

fn create_entry_fn(
  m: &mut dyn Module,
  ast: &[Expr],
  dst_type: PixelType,
  src_types: &[PixelType],
  boundary_mode: Option<BoundaryMode>,
) -> CranexprResult<FuncId> {
  let pointer_type = m.target_config().pointer_type();
  let pointer_size = pointer_type.bytes() as i32;

  let main_sig = Signature {
    params: vec![
      AbiParam::new(pointer_type), // Destination buffer pointer.
      AbiParam::new(types::I64),   // Destination buffer length.
      AbiParam::new(pointer_type), // Sources buffer pointer.
      AbiParam::new(types::I64),   // Sources buffer length.
      AbiParam::new(types::I64),   // Destination plane width.
      AbiParam::new(types::I64),   // Destination plane height.
    ],
    returns: vec![],
    call_conv: m.target_config().default_call_conv,
  };
  let main_func_id = m
    .declare_function(
      &format!("__CRANEXPR_{}", nanoid!()),
      Linkage::Export,
      &main_sig,
    )
    .unwrap();

  let mut ctx = Context::new();
  ctx.func.signature = main_sig;
  {
    let mut func_ctx = FunctionBuilderContext::new();
    let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

    let block = bcx.create_block();
    bcx.switch_to_block(block);
    bcx.append_block_params_for_function_params(block);

    let (dest_ptr, dest_len, src_ptrs, _src_len, width, height) = {
      let params = bcx.block_params(block);
      (
        params[0], params[1], params[2], params[3], params[4], params[5],
      )
    };
    let dest_ptr = Pointer::new(dest_ptr);

    let start_idx = bcx.ins().iconst(types::I64, 0);
    let step = bcx.ins().iconst(types::I64, 1);

    let pointer_type = m.target_config().pointer_type();
    let mut fx = FunctionCx {
      module: m,
      bcx,
      variables: HashMap::new(),
      dst_type,
      src_types: src_types.to_vec(),
      pointer_type,
      boundary_mode: boundary_mode.unwrap_or(BoundaryMode::Clamp),
      src_clips: Pointer::new(src_ptrs),
      width,
      height,
    };

    // Constants.
    codegen_variable(&mut fx, "width", width);
    codegen_variable(&mut fx, "height", height);

    codegen_for_loop(&mut fx, start_idx, dest_len, step, |fx, idx| {
      // Set up loop-variant variables.

      // X = idx % width, Y = idx / width
      let x_coord = fx.bcx.ins().urem(idx, width);
      let y_coord = fx.bcx.ins().udiv(idx, width);
      codegen_variable(fx, "X", x_coord);
      codegen_variable(fx, "Y", y_coord);

      for (var_idx, src_type) in src_types.iter().enumerate() {
        // srcN
        let src_ptr_val = Pointer::new(src_ptrs)
          .offset(fx, Offset32::new(var_idx as i32 * pointer_size))
          .load(fx, pointer_type, SRC_MEMFLAGS);
        let src_ptr = Pointer::new(src_ptr_val);

        // srcN[idx]
        let offset = fx.bcx.ins().imul_imm(idx, src_type.bytes() as i64);
        let pixel_ptr = src_ptr.offset_value(fx, offset);
        let val = pixel_ptr.load(fx, (*src_type).into(), SRC_MEMFLAGS);

        // Convert to float.
        let val = match src_type {
          PixelType::U8 | PixelType::U16 => fx.bcx.ins().fcvt_from_uint(types::F32, val),
          PixelType::F32 => val,
        };

        // Store it in a variable.
        let var = fx.bcx.declare_var(types::F32);
        fx.bcx.def_var(var, val);
        fx.variables.insert(format!("src{var_idx}"), var);
        if let Some(shorthand) = clip_idx_to_shorthand(var_idx as u8) {
          fx.variables.insert(shorthand.to_string(), var);
        }
      }

      // Evaluate expressions.

      let expr_val = if let Some((last_expr, preceding_exprs)) = ast.split_last() {
        // Evaluate all preceding expressions first for any side effects.
        for expr in preceding_exprs {
          translate_expr(fx, expr)?;
        }

        // Last expression is expected to evaluate to the final result.
        translate_expr(fx, last_expr)?
      } else {
        return Err(CranexprError::ExpressionEvaluatesToNothing);
      };

      // Convert output floats to integers if necessary.
      let expr_val = match dst_type {
        PixelType::U8 | PixelType::U16 => {
          let zero = fx.bcx.ins().f32const(0.0);
          let peak_value = fx.bcx.ins().f32const(dst_type.peak_value());

          let mut clamped = fx.bcx.ins().fmin(expr_val, peak_value);
          clamped = fx.bcx.ins().fmax(clamped, zero);

          fx.bcx.ins().fcvt_to_uint_sat(types::I32, clamped)
        }
        PixelType::F32 => expr_val,
      };

      let dest_offset = fx.bcx.ins().imul_imm(idx, dst_type.bytes() as i64);
      dest_ptr
        .offset_value(fx, dest_offset)
        .store(fx, expr_val, MemFlags::new());

      Ok(())
    })?;

    fx.bcx.ins().return_(&[]);
    fx.bcx.seal_all_blocks();
    fx.bcx.finalize();
  }

  if let Err(err) = m.define_function(main_func_id, &mut ctx) {
    let err_msg = match err {
      ModuleError::Compilation(source) => pretty_error(&ctx.func, source),
      _ => err.to_string(),
    };
    return Err(CranexprError::CompilationError(err_msg));
  }

  // println!("{}", ctx.func.display());

  Ok(main_func_id)
}

fn codegen_for_loop(
  fx: &mut FunctionCx<'_, '_>,
  start_idx: Value,
  stop_idx: Value,
  step: Value,
  mut codegen_loop_body: impl FnMut(&mut FunctionCx<'_, '_>, Value) -> CranexprResult<()>,
) -> CranexprResult<()> {
  let current_block = fx.bcx.current_block().unwrap();
  let loop_header_block = fx.bcx.create_block();
  let loop_body_block = fx.bcx.create_block();
  let continue_block = fx.bcx.create_block();

  fx.bcx.ensure_inserted_block();
  fx.bcx.insert_block_after(loop_header_block, current_block);
  fx.bcx
    .insert_block_after(loop_body_block, loop_header_block);
  fx.bcx.insert_block_after(continue_block, loop_body_block);

  // Current block: jump to the loop header block with index = start_idx.
  fx.bcx.ins().jump(loop_header_block, &[start_idx.into()]);

  // Loop header block: check if we're done, then jump to either the continue
  // block or the loop body block.
  fx.bcx.switch_to_block(loop_header_block);
  fx.bcx
    .append_block_param(loop_header_block, fx.pointer_type);
  let idx = fx.bcx.block_params(loop_header_block)[0];
  let done = fx
    .bcx
    .ins()
    .icmp(IntCC::UnsignedGreaterThanOrEqual, idx, stop_idx);
  fx.bcx
    .ins()
    .brif(done, continue_block, &[], loop_body_block, &[]);

  // Loop body block: do main logic, compute the next index, and then jump back
  // to the loop header block.
  fx.bcx.switch_to_block(loop_body_block);
  codegen_loop_body(fx, idx)?;
  let next_idx = fx.bcx.ins().iadd(idx, step);
  fx.bcx.ins().jump(loop_header_block, &[next_idx.into()]);

  // Continue.
  fx.bcx.switch_to_block(continue_block);
  fx.bcx.seal_block(loop_header_block);
  fx.bcx.seal_block(loop_body_block);
  fx.bcx.seal_block(continue_block);

  Ok(())
}

const fn clip_idx_to_shorthand(n: u8) -> Option<char> {
  match n {
    0..=2 => Some((b'x' + n) as char),
    3..=25 => Some((b'a' + (n - 3)) as char),
    _ => None,
  }
}

fn codegen_variable<K: Into<String>>(fx: &mut FunctionCx<'_, '_>, name: K, data: Value) {
  let src_type = fx.bcx.func.dfg.value_type(data);
  let var = fx.bcx.declare_var(src_type);
  fx.bcx.def_var(var, data);
  fx.variables.insert(name.into(), var);
}

/// Applies boundary mode to a coordinate value.
pub(crate) fn apply_boundary_mode(
  fx: &mut FunctionCx<'_, '_>,
  coord: Value,
  max: Value,
  mode: BoundaryMode,
) -> Value {
  match mode {
    BoundaryMode::Clamp => {
      // Clamp: clamp(coord, 0, max - 1)
      let zero = fx.bcx.ins().iconst(types::I64, 0);
      let one = fx.bcx.ins().iconst(types::I64, 1);
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
    BoundaryMode::Mirror => {
      // Mirror: reflect at edges
      // If coord < 0: mirror = -coord - 1
      // If coord >= max: mirror = 2 * max - coord - 1
      // Else: mirror = coord
      let zero = fx.bcx.ins().iconst(types::I64, 0);
      let one = fx.bcx.ins().iconst(types::I64, 1);

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

#[cfg(test)]
mod tests {
  use std::{f32::consts::PI, slice};

  use approx::assert_relative_eq;
  use rstest::rstest;

  use super::*;

  fn run_expr(expr: &str) -> f32 {
    let compiled = compile_jit(expr, PixelType::F32, &[], None).expect("should compile expr");
    let mut dst = vec![0.0];
    unsafe { compiled.invoke(&mut dst, &[&[] as &[u8]], 100, 50) };
    dst[0]
  }

  #[rstest]
  #[case("1 2 +", 3.0)]
  #[case("2 1 +", 3.0)]
  #[case("25 25 +", 50.0)]
  fn test_addition(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("2 9 -", -7.0)]
  #[case("9 2 -", 7.0)]
  #[case("100 20 50 - -", 130.0)]
  #[case("0 7 -", -7.0)]
  fn test_subtraction(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("5 2 - 3 *", 9.0)]
  #[case("-5 2 * 10 -", -20.0)]
  fn test_multiplication(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("10 20 5 - /", 2.0 / 3.0)]
  #[case("8 -2 / -5 -", 1.0)]
  fn test_division(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("1 10 20 ?", 10.0)]
  #[case("0.5 10 20 ?", 10.0)]
  #[case("0 10 20 ?", 20.0)]
  #[case("-1 10 20 ?", 20.0)]
  #[case("-0.5 10 20 ?", 20.0)]
  #[case("50 0 -10 ?", 0.0)]
  #[case("50 -10 0 ?", -10.0)]
  #[case("5 3 - 10 20 ?", 10.0)]
  #[case("3 5 - 10 20 ?", 20.0)]
  #[case("1 2 3 + 4 5 + ?", 5.0)]
  #[case("0 2 3 + 4 5 + ?", 9.0)]
  #[case("10 5 3 > 100 200 ? +", 110.0)]
  #[case("1 10 -1 100 200 ? ?", 10.0)]
  #[case("-1 10 1 100 200 ? ?", 100.0)]
  fn test_ternary(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case(0.0, 0.0f32.sin())]
  #[case(0.5, 0.5f32.sin())]
  #[case(1.0, 1.0f32.sin())]
  fn test_sin(#[case] input: f32, #[case] expected: f32) {
    assert_relative_eq!(run_expr(&format!("{input} sin")), expected);
  }

  #[rstest]
  #[case(0.0, 0.0f32.cos())]
  #[case(0.5, 0.5f32.cos())]
  #[case(1.0, 1.0f32.cos())]
  fn test_cos(#[case] input: f32, #[case] expected: f32) {
    assert_relative_eq!(run_expr(&format!("{input} cos")), expected);
  }

  #[rstest]
  #[case(0.0, 0.0f32.tan())]
  #[case(0.5, 0.5f32.tan())]
  #[case(1.0, 1.0f32.tan())]
  fn test_tan(#[case] input: f32, #[case] expected: f32) {
    assert_relative_eq!(run_expr(&format!("{input} tan")), expected);
  }

  #[rstest]
  #[case("1 exp", 1f32.exp())]
  #[case("0 exp", 0f32.exp())]
  #[case("-1 exp", (-1f32).exp())]
  #[case("0.5 exp", 0.5f32.exp())]
  #[case("2 3 + exp", 148.41316)]
  #[case("2 exp 3 +", 2f32.exp() + 3f32)]
  #[case("3 2 exp *", 22.16717)]
  fn test_exp(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  fn test_variables() {
    let compiled = compile_jit(
      "x y +",
      PixelType::F32,
      &[PixelType::F32, PixelType::F32],
      None,
    )
    .expect("should compile expr");

    let mut actual = [0.0f32];
    let x = [3.0f32];
    let y = [17.0f32];
    unsafe {
      compiled.invoke(
        &mut actual,
        &[
          slice::from_raw_parts(
            x.as_ptr().cast::<u8>(),
            x.len() * types::F32.bytes() as usize,
          ),
          slice::from_raw_parts(
            y.as_ptr().cast::<u8>(),
            y.len() * types::F32.bytes() as usize,
          ),
        ],
        1,
        1,
      );
    };
    assert_relative_eq!(actual[0], 20.0);
  }

  #[rstest]
  #[case("-1.0 abs", 1.0)]
  #[case("-1.0 abs", 1.0)]
  #[case("-0.0 abs", 0.0)]
  #[case("0.0 abs", 0.0)]
  fn test_abs(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("1.0 log", 1.0f32.ln())]
  #[case("2.718281828459045 log", 1.0)]
  fn test_log(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("1.2 floor", 1.0)]
  #[case("1.8 floor", 1.0)]
  #[case("-1.2 floor", -2.0)]
  #[case("-1.8 floor", -2.0)]
  #[case("3.0 floor", 3.0)]
  fn test_floor(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("5 2 %", 1.0)]
  #[case("1.2 1.0 %", 0.2)]
  #[case("-5 2 %", -1.0)]
  #[case("5 -2 %", 1.0)]
  #[case("-5 -2 %", -1.0)]
  #[case("0 5 %", 0.0)]
  fn test_modulo(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("2 3 pow", 8.0)]
  #[case("3 2 pow", 9.0)]
  fn test_pow(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("2 sqrt", 2f32.sqrt())]
  #[case("3 sqrt", 3f32.sqrt())]
  fn test_sqrt(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("40 not", 0.0)]
  #[case("-40 not", 1.0)]
  #[case("0 not", 1.0)]
  #[case("-0 not", 1.0)]
  fn test_not(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("1.2 round", 1.0)]
  #[case("1.8 round", 2.0)]
  #[case("-1.2 round", -1.0)]
  #[case("-1.8 round", -2.0)]
  #[case("3.0 round", 3.0)]
  fn test_round(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("-20 sgn", -1.0)]
  #[case("0 sgn", 0.0)]
  #[case("20 sgn", 1.0)]
  fn test_sign(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("4 dup0 * sqrt", 4.0)]
  #[case("5 dup * dup *", 625.0)]
  #[case("10 20 dup1 + /", 1.0 / 3.0)]
  #[case("8 3 2 dup2 * + /", 0.42105263)]
  #[case("10 4 dup swap / -", 9.0)]
  #[case("100 5 dup1 / swap /", 0.0005)]
  #[case("10 20 30 40 swap3 / swap - /", -2.3529413)]
  fn test_stack_manipulation(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("10 0 20 clip", 10.0)]
  #[case("-10 0 20 clip", 0.0)]
  #[case("30 0 20 clip", 20.0)]
  #[case("0 0 20 clip", 0.0)]
  #[case("20 0 20 clip", 20.0)]
  fn test_clip(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("0 1 atan2", 0.0)]
  #[case("1 1 atan2", PI / 4.0)]
  #[case("1 0 atan2", PI / 2.0)]
  #[case("1 -1 atan2", 3.0 * PI / 4.0)]
  #[case("0 -1 atan2", PI)]
  #[case("-1 -1 atan2", -3.0 * PI / 4.0)]
  #[case("-1 0 atan2", -PI / 2.0)]
  #[case("-1 1 atan2", -PI / 4.0)]
  fn test_atan2(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  fn test_integer_format_clamp() {
    let x = [33839u16];

    let compiled = compile_jit(
      "x 32768 / 0.86 pow 65535 *",
      PixelType::U16,
      &[PixelType::U16],
      None,
    )
    .expect("should compile expr");

    let mut actual = [0u16];

    unsafe {
      compiled.invoke(
        &mut actual,
        &[slice::from_raw_parts(x.as_ptr().cast::<u8>(), x.len())],
        1,
        1,
      );
    };
    assert_eq!(actual[0], 65535);
  }

  #[rstest]
  #[case("5 3 bitand", 1.0)] // 101 & 011 = 001 (1)
  #[case("5.2 2.8 bitand", 1.0)] // 5 & 3 = 1
  #[case("6 3 bitor", 7.0)] // 110 | 011 = 111 (7)
  #[case("6 3 bitxor", 5.0)] // 110 ^ 011 = 101 (5)
  #[case("0 bitnot", -1.0)] // ~0 = -1
  #[case("-1 bitnot", 0.0)] // ~-1 = 0
  fn test_bitwise(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  fn test_abs_access() {
    // 10 11 12
    // 13 14 15
    // 16 17 18
    let x = [10u8, 11, 12, 13, 14, 15, 16, 17, 18];

    let compiled =
      compile_jit("1 1 x[]", PixelType::U8, &[PixelType::U8], None).expect("should compile expr");

    let mut actual = [0u8; 9];
    unsafe {
      compiled.invoke(
        &mut actual,
        &[slice::from_raw_parts(x.as_ptr(), x.len())],
        3,
        3,
      );
    };
    for v in actual {
      assert_eq!(v, 14);
    }
  }

  #[rstest]
  fn test_horizontal_flip_u16() {
    // 1 2 3
    // 4 5 6
    // 7 8 9
    let x = [1u16, 2, 3, 4, 5, 6, 7, 8, 9];

    // 3 2 1
    // 6 5 4
    // 9 8 7
    let expected = [3u16, 2, 1, 6, 5, 4, 9, 8, 7];

    let expr = "width X - 1 - Y x[]";

    let compiled =
      compile_jit(expr, PixelType::U16, &[PixelType::U16], None).expect("should compile expr");

    let mut actual = [0u16; 9];
    unsafe {
      compiled.invoke(
        &mut actual,
        &[slice::from_raw_parts(x.as_ptr().cast::<u8>(), x.len() * 2)],
        3,
        3,
      );
    };

    assert_eq!(actual, expected);
  }
}
