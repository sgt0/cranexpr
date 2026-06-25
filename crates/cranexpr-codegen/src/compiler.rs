use std::collections::HashMap;
use std::f32::consts::PI;
use std::ptr::NonNull;
use std::sync::Arc;

use cranelift::codegen::ir::immediates::Offset32;
use cranelift::codegen::isa::TargetIsa;
use cranelift::codegen::print_errors::pretty_error;
use cranelift::codegen::write::decorate_function;
use cranelift::prelude::*;
use cranelift::{codegen::Context, prelude::FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, ModuleError};
use cranexpr_ast::{BoundaryMode, Expr};
use cranexpr_transforms::simplify;
use nanoid::nanoid;

use crate::MainFunction;
use crate::SelectFunction;
use crate::comment_writer::CommentWriter;
use crate::component_type::ComponentType;
use crate::errors::{CodegenError, CodegenResult};
use crate::pointer::Pointer;
use crate::translate_simd::{
  SIMD_LANES, codegen_pixel_offset, load_pixel_vec_f32x4, simd_lane_offsets_f32x4,
  store_pixel_vec_f32x4, translate_expr_simd,
};

pub(crate) const SRC_MEMFLAGS: MemFlagsData =
  MemFlagsData::trusted().with_readonly().with_can_move();
const FRAME_PROP_MEMFLAGS: MemFlagsData = MemFlagsData::trusted().with_readonly().with_can_move();

pub(crate) struct FunctionCx<'m, 'clif> {
  #[allow(dead_code)]
  pub(crate) module: &'m mut dyn Module,
  pub(crate) pointer_type: Type,

  pub(crate) bcx: FunctionBuilder<'clif>,
  pub(crate) variables: HashMap<String, Variable>,

  /// User-defined variables created by the `var!` store expression.
  pub(crate) user_variables: HashMap<String, Variable>,

  #[allow(dead_code)]
  pub(crate) dst_type: ComponentType,
  pub(crate) src_types: Vec<ComponentType>,

  pub(crate) boundary_mode: BoundaryMode,

  // For relative pixel access
  pub(crate) src_clips: Pointer,
  pub(crate) src_strides: Pointer,
  pub(crate) width: Value,
  pub(crate) height: Value,

  pub(crate) comments: CommentWriter,

  /// Memoization cache.
  pub(crate) cache: HashMap<NonNull<Expr>, Value>,
}

struct BuiltEntryFn {
  func_id: FuncId,
  ctx: Context,
  comments: CommentWriter,
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

/// Compiles an expression AST into a JIT-compiled function.
///
/// # Errors
///
/// Returns a [`CodegenError`] if the expression cannot be compiled.
///
/// # Panics
///
/// Panics if JIT finalization fails.
pub fn compile_jit(
  ast: &[Expr],
  dst_type: ComponentType,
  src_types: &[ComponentType],
  boundary_mode: Option<BoundaryMode>,
  required_frame_props: &[(usize, String)],
) -> Result<MainFunction, CodegenError> {
  let ast: Vec<Expr> = ast.iter().map(simplify).collect();
  let mut jit_module = create_jit_module();
  let mut main_func = build_entry_fn(
    &mut jit_module,
    &ast,
    dst_type,
    src_types,
    boundary_mode,
    required_frame_props,
  )?;

  if let Err(err) = jit_module.define_function(main_func.func_id, &mut main_func.ctx) {
    let err_msg = match err {
      ModuleError::Compilation(source) => pretty_error(&main_func.ctx.func, source),
      _ => err.to_string(),
    };
    return Err(CodegenError::CompilationError(err_msg));
  }

  jit_module.finalize_definitions().unwrap();

  let finalized_main = jit_module.get_finalized_function(main_func.func_id);
  Ok(MainFunction::from_ptr(finalized_main))
}

/// Compiles an expression AST and returns the pretty-printed Cranelift IR with
/// comments.
///
/// # Errors
///
/// Returns a [`CodegenError`] if the expression cannot be compiled.
pub fn compile_clif(
  ast: &[Expr],
  dst_type: ComponentType,
  src_types: &[ComponentType],
  boundary_mode: Option<BoundaryMode>,
  required_frame_props: &[(usize, String)],
) -> Result<String, CodegenError> {
  let ast: Vec<Expr> = ast.iter().map(simplify).collect();
  let mut jit_module = create_jit_module();
  let mut built = build_entry_fn(
    &mut jit_module,
    &ast,
    dst_type,
    src_types,
    boundary_mode,
    required_frame_props,
  )?;

  let mut output = String::new();
  decorate_function(&mut built.comments, &mut output, &built.ctx.func)
    .map_err(|err| CodegenError::CompilationError(err.to_string()))?;
  Ok(output)
}

/// Compiles an expression AST and returns the native disassembly of the
/// generated machine code.
///
/// # Errors
///
/// Returns a [`CodegenError`] if the expression cannot be compiled.
pub fn compile_disasm(
  ast: &[Expr],
  dst_type: ComponentType,
  src_types: &[ComponentType],
  boundary_mode: Option<BoundaryMode>,
  required_frame_props: &[(usize, String)],
) -> Result<String, CodegenError> {
  let ast: Vec<Expr> = ast.iter().map(simplify).collect();
  let mut jit_module = create_jit_module();
  let mut built = build_entry_fn(
    &mut jit_module,
    &ast,
    dst_type,
    src_types,
    boundary_mode,
    required_frame_props,
  )?;

  built.ctx.set_disasm(true);
  if let Err(err) = jit_module.define_function(built.func_id, &mut built.ctx) {
    let err_msg = match err {
      ModuleError::Compilation(source) => pretty_error(&built.ctx.func, source),
      _ => err.to_string(),
    };
    return Err(CodegenError::CompilationError(err_msg));
  }

  let disasm = built
    .ctx
    .compiled_code()
    .and_then(|code| code.vcode.as_deref())
    .ok_or_else(|| CodegenError::CompilationError("disassembly unavailable".to_string()))?;
  Ok(disasm.to_string())
}

/// Compiles an expression AST into a JIT-compiled function that evaluates the
/// expression once.
///
/// The expression must not reference any pixel-access identifiers, relative or
/// absolute pixel access, nor per-pixel context variables. Callers should
/// validate this via something like [`PixelAccessVisitor`] prior to compilation.
///
/// The `num_prop_clips` argument is the number of input clips from which
/// frame properties may be drawn.
///
/// # Errors
///
/// Returns a [`CodegenError`] if the expression cannot be compiled.
///
/// # Panics
///
/// Panics if JIT finalization fails.
pub fn compile_jit_select(
  ast: &[Expr],
  num_prop_clips: usize,
  required_frame_props: &[(usize, String)],
) -> Result<SelectFunction, CodegenError> {
  let ast: Vec<Expr> = ast.iter().map(simplify).collect();
  let mut jit_module = create_jit_module();
  let mut built = build_select_fn(&mut jit_module, &ast, num_prop_clips, required_frame_props)?;

  if let Err(err) = jit_module.define_function(built.func_id, &mut built.ctx) {
    let err_msg = match err {
      ModuleError::Compilation(source) => pretty_error(&built.ctx.func, source),
      _ => err.to_string(),
    };
    return Err(CodegenError::CompilationError(err_msg));
  }

  jit_module.finalize_definitions().unwrap();

  let finalized = jit_module.get_finalized_function(built.func_id);
  Ok(SelectFunction::from_ptr(finalized))
}

fn build_entry_fn(
  m: &mut dyn Module,
  ast: &[Expr],
  dst_type: ComponentType,
  src_types: &[ComponentType],
  boundary_mode: Option<BoundaryMode>,
  required_frame_props: &[(usize, String)],
) -> CodegenResult<BuiltEntryFn> {
  let pointer_type = m.target_config().pointer_type();

  let main_sig = Signature {
    params: vec![
      AbiParam::new(pointer_type), // Destination buffer pointer.
      AbiParam::new(types::I64),   // Destination stride.
      AbiParam::new(pointer_type), // Sources buffer pointer.
      AbiParam::new(pointer_type), // Source strides array pointer.
      AbiParam::new(types::I64),   // Destination plane width.
      AbiParam::new(types::I64),   // Destination plane height.
      AbiParam::new(types::I64),   // Current frame number (N).
      AbiParam::new(pointer_type), // Frame properties array pointer.
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
  let comments;
  {
    let mut func_ctx = FunctionBuilderContext::new();
    let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

    let block = bcx.create_block();
    bcx.switch_to_block(block);
    bcx.append_block_params_for_function_params(block);

    let (dest_ptr, dst_stride, src_ptrs, src_strides_ptr, width, height, n, props_ptr) = {
      let params = bcx.block_params(block);
      (
        params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
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
      user_variables: HashMap::new(),
      dst_type,
      src_types: src_types.to_vec(),
      pointer_type,
      boundary_mode: boundary_mode.unwrap_or(BoundaryMode::Clamp),
      src_clips: Pointer::new(src_ptrs),
      src_strides: Pointer::new(src_strides_ptr),
      width,
      height,
      comments: CommentWriter::new(),
      cache: HashMap::new(),
    };

    // Constants.
    codegen_variable(&mut fx, "width", width);
    codegen_variable(&mut fx, "height", height);
    codegen_variable(&mut fx, "N", n);

    let pi_val = fx.bcx.ins().f32const(PI);
    codegen_variable(&mut fx, "pi", pi_val);

    // Frame properties.
    let props_ptr = Pointer::new(props_ptr);
    for (i, (clip_idx, prop_name)) in required_frame_props.iter().enumerate() {
      let offset = Offset32::new((i * size_of::<f32>()) as i32);
      let val = props_ptr
        .offset(&mut fx, offset)
        .load(&mut fx, types::F32, FRAME_PROP_MEMFLAGS);

      let var = fx.bcx.declare_var(types::F32);
      fx.bcx.def_var(var, val);

      fx.variables
        .insert(format!("prop_{clip_idx}_{prop_name}"), var);
    }

    let (last_expr, preceding_exprs) = ast
      .split_last()
      .ok_or(cranexpr_parser::ParseError::ExpressionEvaluatesToNothing)?;

    let lane_offsets_val = simd_lane_offsets_f32x4(&mut fx);
    let src_types_owned = src_types.to_vec();
    let simd_pixel_body =
      |fx: &mut FunctionCx<'_, '_>, y_coord: Value, x_coord: Value| -> CodegenResult<()> {
        codegen_variable(fx, "X", x_coord);

        for (var_idx, src_type) in src_types_owned.iter().enumerate() {
          let (src_ptr, pixel_offset) =
            codegen_pixel_offset(fx, var_idx, x_coord, y_coord, *src_type);
          let val = load_pixel_vec_f32x4(fx, src_ptr, pixel_offset, *src_type, true);

          let var = fx.bcx.declare_var(types::F32X4);
          fx.bcx.def_var(var, val);
          fx.variables.insert(format!("simd_src{var_idx}"), var);
        }

        let x_f32 = fx.bcx.ins().fcvt_from_uint(types::F32, x_coord);
        let x_splat = fx.bcx.ins().splat(types::F32X4, x_f32);
        let x_vec = fx.bcx.ins().fadd(x_splat, lane_offsets_val);
        let x_var = fx.bcx.declare_var(types::F32X4);
        fx.bcx.def_var(x_var, x_vec);
        fx.variables.insert("simd_X".to_string(), x_var);

        for expr in preceding_exprs {
          translate_expr_simd(fx, expr)?;
        }
        let expr_val = translate_expr_simd(fx, last_expr)?;

        let dest_row_offset = fx.bcx.ins().imul(y_coord, dst_stride);
        let dest_col_offset = fx.bcx.ins().imul_imm(x_coord, dst_type.bytes() as i64);
        let dest_offset = fx.bcx.ins().iadd(dest_row_offset, dest_col_offset);
        store_pixel_vec_f32x4(fx, dest_ptr, dest_offset, expr_val, dst_type);

        fx.variables
          .retain(|k, _| !k.starts_with("simd_") && !k.starts_with("src"));
        fx.user_variables.clear();
        fx.cache.clear();

        Ok(())
      };

    let lanes_val = fx.bcx.ins().iconst(types::I64, SIMD_LANES);
    codegen_for_loop(&mut fx, start_idx, height, step, |fx, y_coord| {
      codegen_variable(fx, "Y", y_coord);
      codegen_for_loop(fx, start_idx, width, lanes_val, |fx, x_coord| {
        simd_pixel_body(fx, y_coord, x_coord)
      })
    })?;

    fx.bcx.ins().return_(&[]);
    fx.bcx.seal_all_blocks();
    fx.bcx.finalize();
    comments = fx.comments;
  }

  Ok(BuiltEntryFn {
    func_id: main_func_id,
    ctx,
    comments,
  })
}

fn build_select_fn(
  m: &mut dyn Module,
  ast: &[Expr],
  num_prop_clips: usize,
  required_frame_props: &[(usize, String)],
) -> CodegenResult<BuiltEntryFn> {
  let pointer_type = m.target_config().pointer_type();

  let main_sig = Signature {
    params: vec![
      AbiParam::new(types::I64),   // Current frame number (N).
      AbiParam::new(types::I64),   // Clip width.
      AbiParam::new(types::I64),   // Clip height.
      AbiParam::new(pointer_type), // Frame properties array pointer.
    ],
    returns: vec![AbiParam::new(types::F32)],
    call_conv: m.target_config().default_call_conv,
  };
  let main_func_id = m
    .declare_function(
      &format!("__CRANEXPR_SELECT_{}", nanoid!()),
      Linkage::Export,
      &main_sig,
    )
    .unwrap();

  let mut ctx = Context::new();
  ctx.func.signature = main_sig;
  let comments;
  {
    let mut func_ctx = FunctionBuilderContext::new();
    let mut bcx = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

    let block = bcx.create_block();
    bcx.switch_to_block(block);
    bcx.append_block_params_for_function_params(block);

    let (n, width, height, props_ptr) = {
      let params = bcx.block_params(block);
      (params[0], params[1], params[2], params[3])
    };

    // The actual types are not relevant here because pixel access is not
    // permitted anyways.
    let src_types = vec![ComponentType::F32; num_prop_clips];

    let mut fx = FunctionCx {
      module: m,
      bcx,
      variables: HashMap::new(),
      user_variables: HashMap::new(),
      dst_type: ComponentType::F32,
      src_types,
      pointer_type,
      boundary_mode: BoundaryMode::Clamp,
      src_clips: Pointer::new(props_ptr),
      src_strides: Pointer::new(props_ptr),
      width,
      height,
      comments: CommentWriter::new(),
      cache: HashMap::new(),
    };

    // Constants.
    codegen_variable(&mut fx, "N", n);
    codegen_variable(&mut fx, "width", width);
    codegen_variable(&mut fx, "height", height);
    let pi_val = fx.bcx.ins().f32const(PI);
    codegen_variable(&mut fx, "pi", pi_val);

    // Frame properties.
    let props_ptr = Pointer::new(props_ptr);
    for (i, (clip_idx, prop_name)) in required_frame_props.iter().enumerate() {
      let offset = Offset32::new((i * size_of::<f32>()) as i32);
      let val = props_ptr
        .offset(&mut fx, offset)
        .load(&mut fx, types::F32, FRAME_PROP_MEMFLAGS);

      let var = fx.bcx.declare_var(types::F32);
      fx.bcx.def_var(var, val);

      fx.variables
        .insert(format!("prop_{clip_idx}_{prop_name}"), var);
    }

    // Evaluate expressions.
    let expr_val = if let Some((last_expr, preceding_exprs)) = ast.split_last() {
      for expr in preceding_exprs {
        translate_expr_simd(&mut fx, expr)?;
      }
      translate_expr_simd(&mut fx, last_expr)?
    } else {
      return Err(cranexpr_parser::ParseError::ExpressionEvaluatesToNothing.into());
    };

    let scalar = fx.bcx.ins().extractlane(expr_val, 0);
    fx.bcx.ins().return_(&[scalar]);
    fx.bcx.seal_all_blocks();
    fx.bcx.finalize();
    comments = fx.comments;
  }

  Ok(BuiltEntryFn {
    func_id: main_func_id,
    ctx,
    comments,
  })
}

fn codegen_for_loop(
  fx: &mut FunctionCx<'_, '_>,
  start_idx: Value,
  stop_idx: Value,
  step: Value,
  mut codegen_loop_body: impl FnMut(&mut FunctionCx<'_, '_>, Value) -> CodegenResult<()>,
) -> CodegenResult<()> {
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

fn codegen_variable<K: Into<String>>(fx: &mut FunctionCx<'_, '_>, name: K, data: Value) {
  let src_type = fx.bcx.func.dfg.value_type(data);
  let var = fx.bcx.declare_var(src_type);
  fx.bcx.def_var(var, data);
  fx.variables.insert(name.into(), var);
}

#[cfg(test)]
mod tests {
  use std::f32::consts::{E, PI};
  use std::slice;

  use aligned_vec::{AVec, ConstAlign};
  use approx::assert_relative_eq;
  use ndarray::{Array2, array};
  use rstest::rstest;

  use super::*;

  /// Row alignment that VapourSynth guarantees for every plane.
  const ROW_ALIGNMENT: usize = 32;

  type AlignedBytes = AVec<u8, ConstAlign<ROW_ALIGNMENT>>;

  /// Allocates a zero-filled, 32-byte-aligned plane buffer with row stride
  /// padded up to `ROW_ALIGNMENT` bytes. Returns the buffer and its byte
  /// stride.
  fn alloc_aligned_plane(row_bytes: usize, rows: usize) -> (AlignedBytes, usize) {
    let stride = row_bytes.next_multiple_of(ROW_ALIGNMENT).max(ROW_ALIGNMENT);
    let size = stride
      .checked_mul(rows)
      .expect("plane size overflow")
      .max(1);
    let mut buf = AVec::<u8, ConstAlign<ROW_ALIGNMENT>>::new(ROW_ALIGNMENT);
    buf.resize(size, 0);
    (buf, stride)
  }

  /// Copies `rows` rows of `width` elements of type `T` from packed, row-major
  /// `src` into `dst` at the given byte `stride`.
  fn fill_plane<T: Copy>(dst: &mut [u8], stride: usize, src: &[T], width: usize, rows: usize) {
    assert_eq!(src.len(), rows * width);
    let row_bytes = width * std::mem::size_of::<T>();
    for y in 0..rows {
      let src_row = &src[y * width..y * width + width];
      // SAFETY: `T: Copy` and the source slice covers exactly `row_bytes` bytes.
      let src_bytes = unsafe { slice::from_raw_parts(src_row.as_ptr().cast::<u8>(), row_bytes) };
      dst[y * stride..y * stride + row_bytes].copy_from_slice(src_bytes);
    }
  }

  /// Copies `rows` rows of `width` elements of type `T` out of a strided plane
  /// buffer into a packed `Vec<T>`.
  fn unpack_plane<T: Copy + Default>(
    src: &[u8],
    stride: usize,
    width: usize,
    rows: usize,
  ) -> Vec<T> {
    let mut out = vec![T::default(); rows * width];
    let row_bytes = width * std::mem::size_of::<T>();
    for y in 0..rows {
      let dst_row = &mut out[y * width..y * width + width];
      let src_bytes = &src[y * stride..y * stride + row_bytes];
      // SAFETY: `T: Copy` and the destination row holds exactly `row_bytes` bytes.
      let dst_bytes =
        unsafe { slice::from_raw_parts_mut(dst_row.as_mut_ptr().cast::<u8>(), row_bytes) };
      dst_bytes.copy_from_slice(src_bytes);
    }
    out
  }

  fn run_expr_ndarray<T>(expr: &str, src: &Array2<T>) -> Array2<T>
  where
    T: Into<ComponentType> + Default + Copy,
  {
    let ast = cranexpr_parser::parse_expr(expr).unwrap();
    let (height, width) = src.dim();
    let pixel = T::default().into();

    let compiled = compile_jit(&ast, pixel, &[pixel], None, &[]).expect("should compile expr");

    let row_bytes = width * std::mem::size_of::<T>();
    let (mut src_buf, src_stride) = alloc_aligned_plane(row_bytes, height);
    fill_plane(
      src_buf.as_mut_slice(),
      src_stride,
      src.as_slice_memory_order().expect("src not contiguous"),
      width,
      height,
    );
    let (mut dst_buf, dst_stride) = alloc_aligned_plane(row_bytes, height);

    let src_ptrs = [src_buf.as_ptr()];
    let src_strides = [src_stride as i64];

    unsafe {
      compiled.invoke(
        dst_buf.as_mut_slice(),
        dst_stride as i64,
        &src_ptrs,
        &src_strides,
        width as i32,
        height as i32,
        0,
        &[],
      );
    }

    let packed = unpack_plane::<T>(dst_buf.as_slice(), dst_stride, width, height);
    Array2::from_shape_vec((height, width), packed).expect("shape matches")
  }

  fn run_expr(expr: &str) -> f32 {
    let src = array![[0.0_f32; 4]];
    let dst = run_expr_ndarray(expr, &src);
    dst[[0, 0]]
  }

  #[rstest]
  #[case("N", 0.0)]
  #[case("N 1 +", 1.0)]
  fn test_n(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
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
  #[case("pi", PI)]
  #[case("pi 2 *", PI * 2.0)]
  fn test_pi(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case(0.0)]
  #[case(0.5)]
  #[case(1.0)]
  #[case(-1.5)]
  #[case(-0.25)]
  #[case(0.25)]
  #[case(1.5)]
  #[case(PI / 2.0)]
  #[case(PI)]
  #[case(3.0 * PI / 2.0)]
  #[case(2.0 * PI)]
  #[case(-PI)]
  #[case(10.0)]
  fn test_sin(#[case] input: f32) {
    let expected = input.sin();
    assert_relative_eq!(run_expr(&format!("{input} sin")), expected, epsilon = 5e-6);
  }

  #[rstest]
  #[case(0.0)]
  #[case(0.5)]
  #[case(1.0)]
  #[case(-1.5)]
  #[case(-0.25)]
  #[case(0.25)]
  #[case(1.5)]
  #[case(PI / 2.0)]
  #[case(PI)]
  #[case(3.0 * PI / 2.0)]
  #[case(2.0 * PI)]
  #[case(-PI)]
  #[case(10.0)]
  fn test_cos(#[case] input: f32) {
    let expected = input.cos();
    assert_relative_eq!(run_expr(&format!("{input} cos")), expected, epsilon = 5e-6);
  }

  #[rstest]
  fn test_sin_cos_per_lane() {
    let src: &[&[f32]] = &[&[0.0, 0.5, 1.0, 2.0]];
    let sin_out = run_expr_padded("x sin", src, None);
    let cos_out = run_expr_padded("x cos", src, None);
    for (i, &v) in src[0].iter().enumerate() {
      assert_relative_eq!(sin_out[0][i], v.sin(), epsilon = 5e-6);
      assert_relative_eq!(cos_out[0][i], v.cos(), epsilon = 5e-6);
    }
  }

  #[rstest]
  #[case(0.0)]
  #[case(0.5)]
  #[case(1.0)]
  #[case(-1.0)]
  #[case(-0.25)]
  #[case(0.25)]
  #[case(PI / 4.0)]
  #[case(-PI / 4.0)]
  #[case(PI)]
  #[case(-PI)]
  #[case(2.0 * PI)]
  fn test_tan(#[case] input: f32) {
    let expected = input.tan();
    assert_relative_eq!(run_expr(&format!("{input} tan")), expected, epsilon = 5e-6);
  }

  #[rstest]
  fn test_tan_per_lane() {
    let src: &[&[f32]] = &[&[0.0, PI / 4.0, 1.0, -0.5]];
    let out = run_expr_padded("x tan", src, None);
    for (i, &v) in src[0].iter().enumerate() {
      assert_relative_eq!(out[0][i], v.tan(), epsilon = 5e-6);
    }
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
  fn test_exp_per_lane() {
    let src: &[&[f32]] = &[&[-2.0, -0.5, 0.5, 2.0]];
    let out = run_expr_padded("x exp", src, None);
    for (i, &v) in src[0].iter().enumerate() {
      assert_relative_eq!(out[0][i], v.exp(), epsilon = 5e-6);
    }
  }

  #[rstest]
  fn test_variables() {
    let ast = cranexpr_parser::parse_expr("x y +").unwrap();
    let compiled = compile_jit(
      &ast,
      ComponentType::F32,
      &[ComponentType::F32, ComponentType::F32],
      None,
      &[],
    )
    .expect("should compile expr");

    let (mut x_buf, x_stride) = alloc_aligned_plane(std::mem::size_of::<f32>(), 1);
    fill_plane(x_buf.as_mut_slice(), x_stride, &[3.0f32], 1, 1);
    let (mut y_buf, y_stride) = alloc_aligned_plane(std::mem::size_of::<f32>(), 1);
    fill_plane(y_buf.as_mut_slice(), y_stride, &[17.0f32], 1, 1);
    let (mut dst_buf, dst_stride) = alloc_aligned_plane(std::mem::size_of::<f32>(), 1);

    let src_ptrs = [x_buf.as_ptr(), y_buf.as_ptr()];
    let src_strides = [x_stride as i64, y_stride as i64];

    unsafe {
      compiled.invoke(
        dst_buf.as_mut_slice(),
        dst_stride as i64,
        &src_ptrs,
        &src_strides,
        1,
        1,
        0,
        &[],
      );
    };
    let actual = unpack_plane::<f32>(dst_buf.as_slice(), dst_stride, 1, 1);
    assert_relative_eq!(actual[0], 20.0);
  }

  #[rstest]
  fn test_properties() {
    let ast = cranexpr_parser::parse_expr("x.PlaneStatsAverage y.PlaneStatsAverage + x *").unwrap();
    let required_props = vec![
      (0, "PlaneStatsAverage".to_string()),
      (1, "PlaneStatsAverage".to_string()),
    ];
    let compiled = compile_jit(
      &ast,
      ComponentType::F32,
      &[ComponentType::F32, ComponentType::F32],
      None,
      &required_props,
    )
    .expect("should compile expr");

    let (mut x_buf, x_stride) = alloc_aligned_plane(std::mem::size_of::<f32>(), 1);
    fill_plane(x_buf.as_mut_slice(), x_stride, &[3.0f32], 1, 1);
    let (mut y_buf, y_stride) = alloc_aligned_plane(std::mem::size_of::<f32>(), 1);
    fill_plane(y_buf.as_mut_slice(), y_stride, &[17.0f32], 1, 1);
    let (mut dst_buf, dst_stride) = alloc_aligned_plane(std::mem::size_of::<f32>(), 1);
    let frame_props = [0.5f32, 1.5f32];

    let src_ptrs = [x_buf.as_ptr(), y_buf.as_ptr()];
    let src_strides = [x_stride as i64, y_stride as i64];

    unsafe {
      compiled.invoke(
        dst_buf.as_mut_slice(),
        dst_stride as i64,
        &src_ptrs,
        &src_strides,
        1,
        1,
        0,
        &frame_props,
      );
    };
    let actual = unpack_plane::<f32>(dst_buf.as_slice(), dst_stride, 1, 1);
    assert_relative_eq!(actual[0], 6.0);
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
  fn test_log_per_lane() {
    let src: &[&[f32]] = &[&[0.5, 1.0, E, 10.0]];
    let out = run_expr_padded("x log", src, None);
    for (i, &v) in src[0].iter().enumerate() {
      assert_relative_eq!(out[0][i], v.ln(), epsilon = 5e-6);
    }
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
  #[case("1.2 trunc", 1.0)]
  #[case("1.8 trunc", 1.0)]
  #[case("-1.2 trunc", -1.0)]
  #[case("-1.8 trunc", -1.0)]
  #[case("3.0 trunc", 3.0)]
  fn test_trunc(#[case] expr: &str, #[case] expected: f32) {
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
  #[allow(clippy::cast_precision_loss)]
  fn test_modulo_per_lane() {
    let src: &[&[f32]] = &[&[5.0, -5.0, 1.2, 7.5]];
    let out = run_expr_padded("x 2.0 %", src, None);
    for (i, &v) in src[0].iter().enumerate() {
      assert_relative_eq!(out[0][i], v % 2.0, epsilon = 1e-6);
    }

    let src2: &[&[f32]] = &[&[10.0, -10.0, 13.5, -13.5]];
    let out2 = run_expr_padded("x -3.0 %", src2, None);
    for (i, &v) in src2[0].iter().enumerate() {
      assert_relative_eq!(out2[0][i], v % -3.0, epsilon = 1e-6);
    }

    let src3: &[&[f32]] = &[&[10.0, 11.0, 12.5, -7.25]];
    let out3 = run_expr_padded("x X 1 + %", src3, None);
    for (i, &v) in src3[0].iter().enumerate() {
      let d = (i as f32) + 1.0;
      assert_relative_eq!(out3[0][i], v % d, epsilon = 1e-6);
    }
  }

  #[rstest]
  #[case("2 3 pow", 8.0)]
  #[case("3 2 pow", 9.0)]
  fn test_pow(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  fn test_pow_simd_literal_exponents() {
    let src: &[&[f32]] = &[&[0.5, 1.5, 2.5, 3.5]];
    for (expr, exponent) in [
      ("x 0 pow", 0.0_f32),
      ("x 1 pow", 1.0),
      ("x 2 pow", 2.0),
      ("x 0.5 pow", 0.5),
      ("x 3 pow", 3.0),
      ("x 4 pow", 4.0),
      ("x 5 pow", 5.0),
      ("x 6 pow", 6.0),
      ("x 7 pow", 7.0),
      ("x 8 pow", 8.0),
      ("x 9 pow", 9.0),
      ("x 11 pow", 11.0),
      ("x 13 pow", 13.0),
      ("x 15 pow", 15.0),
      ("x 17 pow", 17.0),
      ("x 19 pow", 19.0),
    ] {
      let out = run_expr_padded(expr, src, None);
      for (i, &v) in src[0].iter().enumerate() {
        let expected = v.powf(exponent);
        assert_relative_eq!(out[0][i], expected, epsilon = 5e-5);
      }
    }
  }

  #[rstest]
  fn test_pow_negative_base_large_exponent() {
    let src: &[&[f32]] = &[&[-0.5, -1.5, -2.0, -0.25]];
    for exp in [9, 11, 13, 15, 17, 19] {
      let expr = format!("x {exp} pow");
      let out = run_expr_padded(&expr, src, None);
      for (i, &v) in src[0].iter().enumerate() {
        let expected = v.powi(exp);
        assert_relative_eq!(out[0][i], expected, epsilon = 5e-3);
      }
    }
  }

  #[rstest]
  fn test_pow_simd_general() {
    let src: &[&[f32]] = &[&[0.25, 0.75, 1.5, 4.0]];
    let out = run_expr_padded("x 0.86 pow", src, None);
    for (i, &v) in src[0].iter().enumerate() {
      let expected = v.powf(0.86);
      assert_relative_eq!(out[0][i], expected, epsilon = 5e-5);
    }
  }

  #[rstest]
  fn test_pow_simd_dynamic_exponent() {
    let src: &[&[f32]] = &[&[2.0, 3.0, 4.0, 5.0]];
    let out = run_expr_padded("x Y 1 + pow", src, None);
    let expected_exp = 1.0_f32;
    for (i, &v) in src[0].iter().enumerate() {
      let expected = v.powf(expected_exp);
      assert_relative_eq!(out[0][i], expected, epsilon = 5e-5);
    }
  }

  #[rstest]
  fn test_pow_negative_integer_exponent() {
    let src: &[&[f32]] = &[&[0.5, 2.0, 3.0, 4.0]];
    for exp in [-1, -2, -3, -5, -8, -127] {
      let expr = format!("x {exp} pow");
      let out = run_expr_padded(&expr, src, None);
      for (i, &v) in src[0].iter().enumerate() {
        let expected = v.powi(exp);
        assert_relative_eq!(out[0][i], expected, epsilon = 1e-4);
      }
    }
  }

  #[rstest]
  fn test_pow_negative_exponent_zero_base() {
    assert_relative_eq!(run_expr("0 2 -127 pow /"), 0.0);
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
  #[case("0.5 round", 0.0)]
  #[case("1.5 round", 2.0)]
  #[case("2.5 round", 2.0)]
  #[case("-0.5 round", 0.0)]
  #[case("-1.5 round", -2.0)]
  #[case("-2.5 round", -2.0)]
  fn test_round(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  fn test_round_per_lane() {
    let src: &[&[f32]] = &[&[0.5, 1.5, -0.5, -1.5, 1.2, 1.8, -1.2, -1.8]];
    let out = run_expr_padded("x round", src, None);
    let expected = [0.0, 2.0, 0.0, -2.0, 1.0, 2.0, -1.0, -2.0];
    for (i, &e) in expected.iter().enumerate() {
      assert_relative_eq!(out[0][i], e);
    }
  }

  #[rstest]
  #[case("-20 sgn", -1.0)]
  #[case("0 sgn", 0.0)]
  #[case("20 sgn", 1.0)]
  fn test_sign(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  fn test_sign_per_lane() {
    let src: &[&[f32]] = &[&[-2.0, -0.5, -0.0, 0.0, 0.5, 2.0, -100.0, 100.0]];
    let out = run_expr_padded("x sgn", src, None);
    let expected = [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0];
    for (i, &e) in expected.iter().enumerate() {
      assert_relative_eq!(out[0][i], e);
    }
  }

  #[rstest]
  #[case("3 1 2 sort3 + +", 6.0)]
  #[case("3 1 2 sort3 drop drop", 3.0)]
  #[case("3 1 2 sort3 drop2", 3.0)]
  #[case("5 3 sort2 -", 2.0)]
  #[case("5 3 sort2 /", 5.0 / 3.0)]
  #[case("4 1 3 2 sort4 + + +", 10.0)]
  #[case("4 1 3 2 sort4 drop drop drop", 4.0)]
  fn test_sort(#[case] expr: &str, #[case] expected: f32) {
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
  fn test_store_does_not_affect_stack_indices() {
    let src = array![[0.2f32]];
    let dst = run_expr_ndarray(
      "7 6 5 4 3 2 1 0 dup4 max_val! dup3 min_val! drop8 x min_val@ max_val@ clamp",
      &src,
    );
    assert_relative_eq!(dst[[0, 0]], 3.0);
  }

  #[rstest]
  fn test_rel_access_with_var_store_load() {
    let src = array![
      [0.0f32, 0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6, 0.7],
      [0.8, 0.9, 0.1, 0.2]
    ];
    let dst = run_expr_ndarray(
      "x[-1,-1] x[0,-1] x[1,-1] x[-1,0] x[1,0] x[-1,1] x[0,1] x[1,1] dup4 max_val! dup3 min_val! drop8 x min_val@ max_val@ clamp",
      &src,
    );
    assert_relative_eq!(dst[[1, 0]], 0.5);
    assert_relative_eq!(dst[[1, 1]], 0.6);
  }

  #[rstest]
  fn test_strided_pixel_access() {
    let width = 3i32;
    let height = 2i32;
    let stride: i64 = 16;

    let src_data: [f32; 8] = [1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0, 99.0];
    let mut dst_data: [f32; 8] = [0.0; 8];

    let ast = cranexpr_parser::parse_expr("x").unwrap();
    let compiled = compile_jit(&ast, ComponentType::F32, &[ComponentType::F32], None, &[])
      .expect("should compile");

    let src_ptrs = [src_data.as_ptr().cast::<u8>()];
    let src_strides = [stride];

    unsafe {
      compiled.invoke(
        &mut dst_data,
        stride,
        &src_ptrs,
        &src_strides,
        width,
        height,
        0,
        &[],
      );
    }

    assert_relative_eq!(dst_data[0], 1.0);
    assert_relative_eq!(dst_data[1], 2.0);
    assert_relative_eq!(dst_data[2], 3.0);
    // dst_data[3] is padding.
    assert_relative_eq!(dst_data[4], 4.0);
    assert_relative_eq!(dst_data[5], 5.0);
    assert_relative_eq!(dst_data[6], 6.0);
  }

  #[rstest]
  fn test_strided_rel_access() {
    let width = 3i32;
    let height = 3i32;
    let stride: i64 = 16;

    // [1.0, 2.0, 3.0, _]
    // [4.0, 5.0, 6.0, _]
    // [7.0, 8.0, 9.0, _]
    let src_data: [f32; 12] = [
      1.0, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0, 99.0, 7.0, 8.0, 9.0, 99.0,
    ];
    let mut dst_data: [f32; 12] = [0.0; 12];

    let ast = cranexpr_parser::parse_expr("x[1,0]").unwrap();
    let compiled = compile_jit(&ast, ComponentType::F32, &[ComponentType::F32], None, &[])
      .expect("should compile");

    let src_ptrs = [src_data.as_ptr().cast::<u8>()];
    let src_strides = [stride];

    unsafe {
      compiled.invoke(
        &mut dst_data,
        stride,
        &src_ptrs,
        &src_strides,
        width,
        height,
        0,
        &[],
      );
    }

    assert_relative_eq!(dst_data[0], 2.0);
    assert_relative_eq!(dst_data[1], 3.0);
    assert_relative_eq!(dst_data[2], 3.0);
    assert_relative_eq!(dst_data[4], 5.0);
    assert_relative_eq!(dst_data[5], 6.0);
  }

  #[rstest]
  #[case("3 5 max", 5.0)]
  #[case("5 3 max", 5.0)]
  #[case("5 5 max", 5.0)]
  #[case("-2 -7 max", -2.0)]
  #[case("3 5 min", 3.0)]
  #[case("5 3 min", 3.0)]
  #[case("5 5 min", 5.0)]
  #[case("-2 -7 min", -7.0)]
  fn test_max_min(#[case] expr: &str, #[case] expected: f32) {
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
  #[case("1 1 =", 1.0)]
  #[case("1 0 =", 0.0)]
  #[case("5 5 =", 1.0)]
  #[case("5 6 =", 0.0)]
  fn test_eq(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("5 3 >=", 1.0)]
  #[case("3 5 >=", 0.0)]
  #[case("5 5 >=", 1.0)]
  fn test_gte(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("3 5 <=", 1.0)]
  #[case("5 3 <=", 0.0)]
  #[case("5 5 <=", 1.0)]
  fn test_lte(#[case] expr: &str, #[case] expected: f32) {
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
  fn test_atan2_per_lane() {
    let src: &[&[f32]] = &[&[0.0, 1.0, -1.0, 0.5]];
    let out = run_expr_padded("x 1 x - atan2", src, None);
    for (i, &v) in src[0].iter().enumerate() {
      let expected = v.atan2(1.0 - v);
      assert_relative_eq!(out[0][i], expected, epsilon = 5e-6);
    }

    let src: &[&[f32]] = &[&[0.0, 1.0, -1.0, 2.0]];
    let out = run_expr_padded("x x -1 * atan2", src, None);
    for (i, &v) in src[0].iter().enumerate() {
      let expected = if v == 0.0 { 0.0 } else { v.atan2(-v) };
      assert_relative_eq!(out[0][i], expected, epsilon = 5e-6);
    }
  }

  #[rstest]
  fn test_integer_format_clamp() {
    let ast = cranexpr_parser::parse_expr("x 32768 / 0.86 pow 65535 *").unwrap();

    let compiled = compile_jit(&ast, ComponentType::U16, &[ComponentType::U16], None, &[])
      .expect("should compile expr");

    let (mut x_buf, x_stride) = alloc_aligned_plane(std::mem::size_of::<u16>(), 1);
    fill_plane(x_buf.as_mut_slice(), x_stride, &[33839u16], 1, 1);
    let (mut dst_buf, dst_stride) = alloc_aligned_plane(std::mem::size_of::<u16>(), 1);

    let src_ptrs = [x_buf.as_ptr()];
    let src_strides = [x_stride as i64];

    unsafe {
      compiled.invoke(
        dst_buf.as_mut_slice(),
        dst_stride as i64,
        &src_ptrs,
        &src_strides,
        1,
        1,
        0,
        &[],
      );
    };
    let actual = unpack_plane::<u16>(dst_buf.as_slice(), dst_stride, 1, 1);
    assert_eq!(actual[0], 65535);
  }

  #[rstest]
  fn test_integer_format_round() {
    let ast = cranexpr_parser::parse_expr("1.65").unwrap();

    let compiled = compile_jit(&ast, ComponentType::U8, &[ComponentType::U8], None, &[])
      .expect("should compile expr");

    let (mut x_buf, x_stride) = alloc_aligned_plane(std::mem::size_of::<u8>(), 1);
    fill_plane(x_buf.as_mut_slice(), x_stride, &[0u8], 1, 1);
    let (mut dst_buf, dst_stride) = alloc_aligned_plane(std::mem::size_of::<u8>(), 1);

    let src_ptrs = [x_buf.as_ptr()];
    let src_strides = [x_stride as i64];

    unsafe {
      compiled.invoke(
        dst_buf.as_mut_slice(),
        dst_stride as i64,
        &src_ptrs,
        &src_strides,
        1,
        1,
        0,
        &[],
      );
    };
    let actual = unpack_plane::<u8>(dst_buf.as_slice(), dst_stride, 1, 1);
    assert_eq!(actual[0], 2);
  }

  #[rstest]
  #[case("1 1 and", 1.0)]
  #[case("1 0 and", 0.0)]
  #[case("0 1 and", 0.0)]
  #[case("0 0 and", 0.0)]
  #[case("-1 1 and", 0.0)]
  #[case("5 3 and", 1.0)]
  fn test_and(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("1 1 or", 1.0)]
  #[case("1 0 or", 1.0)]
  #[case("0 1 or", 1.0)]
  #[case("0 0 or", 0.0)]
  #[case("-1 1 or", 1.0)]
  #[case("-1 0 or", 0.0)]
  fn test_or(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
  }

  #[rstest]
  #[case("1 1 xor", 0.0)]
  #[case("1 0 xor", 1.0)]
  #[case("0 1 xor", 1.0)]
  #[case("0 0 xor", 0.0)]
  #[case("-1 1 xor", 1.0)]
  #[case("5 3 xor", 0.0)]
  fn test_xor(#[case] expr: &str, #[case] expected: f32) {
    assert_relative_eq!(run_expr(expr), expected);
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
    let x = array![[10u8, 11, 12], [13, 14, 15], [16, 17, 18]];

    let actual = run_expr_ndarray("1 1 x[]", &x);

    assert_eq!(actual, Array2::from_elem((3, 3), 14u8));
  }

  #[rstest]
  fn test_horizontal_flip_u16() {
    // 1 2 3
    // 4 5 6
    // 7 8 9
    let x = array![[1u16, 2, 3], [4, 5, 6], [7, 8, 9]];

    // 3 2 1
    // 6 5 4
    // 9 8 7
    let expected = array![[3, 2, 1], [6, 5, 4], [9, 8, 7]];

    let expr = "width X - 1 - Y x[]";

    let actual = run_expr_ndarray(expr, &x);

    assert_eq!(actual, expected);
  }

  fn run_expr_padded(
    expr: &str,
    src_rows: &[&[f32]],
    boundary_mode: Option<BoundaryMode>,
  ) -> Vec<Vec<f32>> {
    let height = src_rows.len();
    let width = src_rows[0].len();
    assert!(src_rows.iter().all(|r| r.len() == width));

    let row_bytes = std::mem::size_of_val(src_rows[0]);
    let src_flat: Vec<f32> = src_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let (mut src_buf, src_stride) = alloc_aligned_plane(row_bytes, height);
    fill_plane(src_buf.as_mut_slice(), src_stride, &src_flat, width, height);
    let (mut dst_buf, dst_stride) = alloc_aligned_plane(row_bytes, height);

    let ast = cranexpr_parser::parse_expr(expr).unwrap();
    let compiled = compile_jit(
      &ast,
      ComponentType::F32,
      &[ComponentType::F32],
      boundary_mode,
      &[],
    )
    .expect("should compile expr");

    let src_ptrs = [src_buf.as_ptr()];
    let src_strides = [src_stride as i64];

    unsafe {
      compiled.invoke(
        dst_buf.as_mut_slice(),
        dst_stride as i64,
        &src_ptrs,
        &src_strides,
        width as i32,
        height as i32,
        0,
        &[],
      );
    }

    let packed = unpack_plane::<f32>(dst_buf.as_slice(), dst_stride, width, height);
    (0..height)
      .map(|y| packed[y * width..y * width + width].to_vec())
      .collect()
  }

  #[rstest]
  fn test_simd_rel_access_left_shift_clamp() {
    let src: &[&[f32]] = &[&[10.0, 20.0, 30.0, 40.0]];
    let out = run_expr_padded("x[-1,0]", src, None);
    assert_eq!(out, vec![vec![10.0, 10.0, 20.0, 30.0]]);
  }

  #[rstest]
  fn test_simd_rel_access_right_shift_clamp() {
    let src: &[&[f32]] = &[&[10.0, 20.0, 30.0, 40.0]];
    let out = run_expr_padded("x[1,0]", src, None);
    assert_eq!(out, vec![vec![20.0, 30.0, 40.0, 40.0]]);
  }

  #[rstest]
  fn test_simd_rel_access_left_shift_two_clamp() {
    let src: &[&[f32]] = &[&[10.0, 20.0, 30.0, 40.0]];
    let out = run_expr_padded("x[-2,0]", src, None);
    assert_eq!(out, vec![vec![10.0, 10.0, 10.0, 20.0]]);
  }

  #[rstest]
  fn test_simd_rel_access_non_multiple_width_past_tail() {
    let src: &[&[f32]] = &[&[1.0, 2.0, 3.0]];
    let out = run_expr_padded("x 10 *", src, None);
    assert_eq!(out, vec![vec![10.0, 20.0, 30.0]]);
  }

  #[rstest]
  fn test_simd_rel_access_vertical_clamp() {
    let src: &[&[f32]] = &[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]];
    let out = run_expr_padded("x[0,-1]", src, None);
    assert_eq!(
      out,
      vec![vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0, 4.0]]
    );
  }

  #[rstest]
  fn test_simd_rel_access_mirror_horizontal() {
    let src: &[&[f32]] = &[&[10.0, 20.0, 30.0, 40.0]];
    let out = run_expr_padded("x[-1,0]", src, Some(BoundaryMode::Mirror));
    assert_eq!(out, vec![vec![10.0, 10.0, 20.0, 30.0]]);
  }

  #[rstest]
  fn test_simd_rel_access_right_shift_non_multiple_width() {
    let src: &[&[f32]] = &[&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]];
    let out = run_expr_padded("x[1,0]", src, None);
    assert_eq!(out, vec![vec![20.0, 30.0, 40.0, 50.0, 60.0, 60.0]]);
  }

  #[rstest]
  fn test_simd_if_else_per_lane_blend() {
    let src: &[&[f32]] = &[&[10.0, 20.0, 30.0, 40.0]];
    let out = run_expr_padded("x 25 > 100 200 ?", src, None);
    assert_eq!(out, vec![vec![200.0, 200.0, 100.0, 100.0]]);
  }

  #[rstest]
  fn test_simd_if_else_truthy_threshold() {
    let src: &[&[f32]] = &[&[-1.0, 0.0, 0.5, 1.0]];
    let out = run_expr_padded("x 7 8 ?", src, None);
    assert_eq!(out, vec![vec![8.0, 8.0, 7.0, 7.0]]);
  }

  #[rstest]
  fn test_simd_if_else_nested() {
    let src: &[&[f32]] = &[&[-1.0, 0.0, 0.5, 1.0]];
    let out = run_expr_padded("x 1 x -1 0 ? ?", src, None);
    assert_eq!(out, vec![vec![0.0, 0.0, 1.0, 1.0]]);
  }

  #[rstest]
  fn test_simd_if_else_with_arith_arms() {
    let src: &[&[f32]] = &[&[1.0, 2.0, 3.0, 4.0]];
    let out = run_expr_padded("x 2 > x 10 * x 100 + ?", src, None);
    assert_eq!(out, vec![vec![101.0, 102.0, 30.0, 40.0]]);
  }

  #[rstest]
  fn test_simd_abs_access_constant_coords() {
    let src: &[&[f32]] = &[&[10.0, 20.0, 30.0, 40.0]];
    let out = run_expr_padded("1 0 x[]", src, None);
    assert_eq!(out, vec![vec![20.0, 20.0, 20.0, 20.0]]);
  }

  #[rstest]
  fn test_simd_abs_access_horizontal_flip() {
    let src: &[&[f32]] = &[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]];
    let out = run_expr_padded("width X - 1 - Y x[]", src, None);
    assert_eq!(
      out,
      vec![vec![4.0, 3.0, 2.0, 1.0], vec![8.0, 7.0, 6.0, 5.0]]
    );
  }

  #[rstest]
  fn test_simd_abs_access_clamp_boundary() {
    let src: &[&[f32]] = &[&[10.0, 20.0, 30.0, 40.0]];
    let out = run_expr_padded("X 2 - 0 x[]", src, None);
    assert_eq!(out, vec![vec![10.0, 10.0, 10.0, 20.0]]);
  }

  #[rstest]
  fn test_simd_abs_access_mirror_boundary() {
    let src: &[&[f32]] = &[&[10.0, 20.0, 30.0, 40.0]];
    let out = run_expr_padded("-1 0 x[]", src, Some(BoundaryMode::Mirror));
    assert_eq!(out, vec![vec![10.0, 10.0, 10.0, 10.0]]);
  }

  fn run_expr_u8_padded(expr: &str, src_rows: &[&[u8]]) -> Vec<Vec<u8>> {
    let height = src_rows.len();
    let width = src_rows[0].len();
    assert!(src_rows.iter().all(|r| r.len() == width));

    let src_flat: Vec<u8> = src_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let (mut src_buf, src_stride) = alloc_aligned_plane(width, height);
    fill_plane(src_buf.as_mut_slice(), src_stride, &src_flat, width, height);
    let (mut dst_buf, dst_stride) = alloc_aligned_plane(width, height);

    let ast = cranexpr_parser::parse_expr(expr).unwrap();
    let compiled = compile_jit(&ast, ComponentType::U8, &[ComponentType::U8], None, &[])
      .expect("should compile expr");

    let src_ptrs = [src_buf.as_ptr()];
    let src_strides = [src_stride as i64];

    unsafe {
      compiled.invoke(
        dst_buf.as_mut_slice(),
        dst_stride as i64,
        &src_ptrs,
        &src_strides,
        width as i32,
        height as i32,
        0,
        &[],
      );
    }

    let packed = unpack_plane::<u8>(dst_buf.as_slice(), dst_stride, width, height);
    (0..height)
      .map(|y| packed[y * width..y * width + width].to_vec())
      .collect()
  }

  fn run_expr_u16_padded(expr: &str, src_rows: &[&[u16]]) -> Vec<Vec<u16>> {
    let height = src_rows.len();
    let width = src_rows[0].len();
    assert!(src_rows.iter().all(|r| r.len() == width));

    let row_bytes = std::mem::size_of_val(src_rows[0]);
    let src_flat: Vec<u16> = src_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let (mut src_buf, src_stride) = alloc_aligned_plane(row_bytes, height);
    fill_plane(src_buf.as_mut_slice(), src_stride, &src_flat, width, height);
    let (mut dst_buf, dst_stride) = alloc_aligned_plane(row_bytes, height);

    let ast = cranexpr_parser::parse_expr(expr).unwrap();
    let compiled = compile_jit(&ast, ComponentType::U16, &[ComponentType::U16], None, &[])
      .expect("should compile expr");

    let src_ptrs = [src_buf.as_ptr()];
    let src_strides = [src_stride as i64];

    unsafe {
      compiled.invoke(
        dst_buf.as_mut_slice(),
        dst_stride as i64,
        &src_ptrs,
        &src_strides,
        width as i32,
        height as i32,
        0,
        &[],
      );
    }

    let packed = unpack_plane::<u16>(dst_buf.as_slice(), dst_stride, width, height);
    (0..height)
      .map(|y| packed[y * width..y * width + width].to_vec())
      .collect()
  }

  #[rstest]
  fn test_simd_u8_identity() {
    let src: &[&[u8]] = &[&[0, 1, 2, 3, 10, 20, 30, 40]];
    let out = run_expr_u8_padded("x", src);
    assert_eq!(out, vec![vec![0, 1, 2, 3, 10, 20, 30, 40]]);
  }

  #[rstest]
  fn test_simd_u8_arithmetic() {
    let src: &[&[u8]] = &[&[0, 1, 2, 3, 4, 5, 6, 7]];
    let out = run_expr_u8_padded("x 2 *", src);
    assert_eq!(out, vec![vec![0, 2, 4, 6, 8, 10, 12, 14]]);
  }

  #[rstest]
  fn test_simd_u8_saturating_store() {
    let src: &[&[u8]] = &[&[0, 50, 100, 150, 200, 250, 255, 255]];
    let out = run_expr_u8_padded("x 2 *", src);
    assert_eq!(out, vec![vec![0, 100, 200, 255, 255, 255, 255, 255]]);
  }

  #[rstest]
  fn test_simd_u8_negative_saturation() {
    let src: &[&[u8]] = &[&[0, 5, 9, 10, 15, 20, 100, 200]];
    let out = run_expr_u8_padded("x 10 -", src);
    assert_eq!(out, vec![vec![0, 0, 0, 0, 5, 10, 90, 190]]);
  }

  #[rstest]
  fn test_simd_u8_rel_access_right() {
    let src: &[&[u8]] = &[&[10, 20, 30, 40, 50, 60, 70, 80]];
    let out = run_expr_u8_padded("x[1,0]", src);
    assert_eq!(out, vec![vec![20, 30, 40, 50, 60, 70, 80, 80]]);
  }

  #[rstest]
  fn test_simd_u8_non_multiple_width_tail() {
    let src: &[&[u8]] = &[&[1, 2, 3, 4, 5, 6, 7]];
    let out = run_expr_u8_padded("x 3 +", src);
    assert_eq!(out, vec![vec![4, 5, 6, 7, 8, 9, 10]]);
  }

  #[rstest]
  fn test_simd_u16_identity() {
    let src: &[&[u16]] = &[&[0, 1, 2, 3, 1000, 2000, 30000, 60000]];
    let out = run_expr_u16_padded("x", src);
    assert_eq!(out, vec![vec![0, 1, 2, 3, 1000, 2000, 30000, 60000]]);
  }

  #[rstest]
  fn test_simd_u16_arithmetic_with_saturation() {
    let src: &[&[u16]] = &[&[0, 100, 1000, 10000, 20000, 30000, 40000, 60000]];
    let out = run_expr_u16_padded("x 3 *", src);
    assert_eq!(
      out,
      vec![vec![0, 300, 3000, 30000, 60000, 65535, 65535, 65535]]
    );
  }

  #[rstest]
  fn test_simd_u16_rel_access_left() {
    let src: &[&[u16]] = &[&[10, 20, 30, 40, 50, 60, 70, 80]];
    let out = run_expr_u16_padded("x[-1,0]", src);
    assert_eq!(out, vec![vec![10, 10, 20, 30, 40, 50, 60, 70]]);
  }

  #[rstest]
  fn test_simd_mixed_sources_u8_u16() {
    let x_rows: &[u8] = &[10, 20, 30, 40, 50, 60, 70, 80];
    let y_rows: &[u16] = &[100, 200, 300, 400, 500, 600, 700, 800];
    let width = 8usize;
    let height = 1usize;
    let row_padded_u8 = 32usize;
    let row_padded_u16 = 16usize;

    let mut x_flat = vec![0u8; row_padded_u8 * height];
    x_flat[..width].copy_from_slice(x_rows);
    let mut y_flat = vec![0u16; row_padded_u16 * height];
    y_flat[..width].copy_from_slice(y_rows);
    let mut dst_flat = vec![0u16; row_padded_u16 * height];

    let ast = cranexpr_parser::parse_expr("x y +").unwrap();
    let compiled = compile_jit(
      &ast,
      ComponentType::U16,
      &[ComponentType::U8, ComponentType::U16],
      None,
      &[],
    )
    .expect("should compile expr");

    let src_ptrs = [x_flat.as_ptr(), y_flat.as_ptr().cast::<u8>()];
    let src_strides = [row_padded_u8 as i64, (row_padded_u16 * 2) as i64];
    let dst_stride = (row_padded_u16 * 2) as i64;

    unsafe {
      compiled.invoke(
        &mut dst_flat[..],
        dst_stride,
        &src_ptrs,
        &src_strides,
        width as i32,
        height as i32,
        0,
        &[],
      );
    }

    let out: Vec<u16> = dst_flat[..width].to_vec();
    assert_eq!(out, vec![110, 220, 330, 440, 550, 660, 770, 880]);
  }

  fn run_select(expr: &str, n: i32) -> f32 {
    let ast = cranexpr_parser::parse_expr(expr).unwrap();
    let compiled = compile_jit_select(&ast, 0, &[]).expect("should compile select");
    unsafe { compiled.invoke(n, 1, 1, &[]) }
  }

  fn run_select_with_props(
    expr: &str,
    n: i32,
    num_prop_clips: usize,
    required_props: &[(usize, String)],
    frame_props: &[f32],
  ) -> f32 {
    let ast = cranexpr_parser::parse_expr(expr).unwrap();
    let compiled =
      compile_jit_select(&ast, num_prop_clips, required_props).expect("should compile select");
    unsafe { compiled.invoke(n, 1, 1, frame_props) }
  }

  #[rstest]
  #[case("0", 0, 0.0)]
  #[case("1", 0, 1.0)]
  #[case("N", 0, 0.0)]
  #[case("N", 7, 7.0)]
  #[case("N 1 +", 5, 6.0)]
  #[case("1 2 +", 0, 3.0)]
  #[case("pi", 0, PI)]
  fn test_select_basic(#[case] expr: &str, #[case] n: i32, #[case] expected: f32) {
    assert_relative_eq!(run_select(expr, n), expected);
  }

  #[rstest]
  fn test_select_frame_props() {
    let required = vec![
      (0, "PlaneStatsAverage".to_string()),
      (1, "PlaneStatsAverage".to_string()),
    ];
    let expr = "x.PlaneStatsAverage y.PlaneStatsAverage >";
    let result = run_select_with_props(expr, 0, 2, &required, &[0.7, 0.3]);
    assert_relative_eq!(result, 1.0);
    let result = run_select_with_props(expr, 0, 2, &required, &[0.1, 0.9]);
    assert_relative_eq!(result, 0.0);
  }

  #[rstest]
  fn test_select_width_height() {
    let ast = cranexpr_parser::parse_expr("width height *").unwrap();
    let compiled = compile_jit_select(&ast, 0, &[]).expect("should compile select");
    let result = unsafe { compiled.invoke(0, 640, 480, &[]) };
    assert_relative_eq!(result, 640.0 * 480.0);
  }

  #[rstest]
  fn test_select_width_height_individually() {
    let ast = cranexpr_parser::parse_expr("width").unwrap();
    let compiled = compile_jit_select(&ast, 0, &[]).expect("should compile select");
    let result = unsafe { compiled.invoke(0, 1920, 1080, &[]) };
    assert_relative_eq!(result, 1920.0);

    let ast = cranexpr_parser::parse_expr("height").unwrap();
    let compiled = compile_jit_select(&ast, 0, &[]).expect("should compile select");
    let result = unsafe { compiled.invoke(0, 1920, 1080, &[]) };
    assert_relative_eq!(result, 1080.0);
  }
}
