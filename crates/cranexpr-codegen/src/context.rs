use std::collections::HashMap;
use std::ptr::NonNull;

use cranelift::prelude::*;
use cranelift_module::Module;
use cranexpr_ast::{BoundaryMode, Expr};

use crate::comment_writer::CommentWriter;
use crate::component_type::ComponentType;
use crate::pointer::Pointer;

pub(crate) const VEC_TYPE: types::Type = types::F32X4;

/// Number of pixels processed per SIMD iteration.
pub(crate) const SIMD_LANES: i64 = 4;

pub(crate) const SRC_MEMFLAGS: MemFlagsData =
  MemFlagsData::trusted().with_readonly().with_can_move();

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
