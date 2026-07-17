//! LUT compilation strategy.
//!
//! When an expression is a pure function of a single integer clip's current
//! pixel, the whole expression can be evaluated once per possible input value
//! at compile time. Runtime evaluation then boils down to a table lookup.

use cranexpr_ast::Expr;
use cranexpr_transforms::{LutVisitor, Visitor};

use crate::MainFunction;
use crate::component_type::ComponentType;
use crate::translate_simd::resolve_clip_name;

/// A compiled expression as a lookup table.
///
/// The table is indexed by the raw source sample and stores raw destination
/// samples.
#[derive(Debug)]
pub struct LutFunction {
  clip_idx: usize,
  src_type: ComponentType,
  table: LookupTable,
}

#[derive(Debug)]
enum LookupTable {
  U8(Vec<u8>),
  U16(Vec<u16>),
  F32(Vec<f32>),
}

impl LutFunction {
  /// Builds a LUT for `ast` when it qualifies by evaluating `main` over the
  /// whole input domain of the single referenced clip.
  ///
  /// Returns `None` when the expression is not a pure function of one integer
  /// clip's current pixel.
  pub(crate) fn try_build(
    ast: &[Expr],
    main: &MainFunction,
    dst_type: ComponentType,
    src_types: &[ComponentType],
    required_frame_props: &[(usize, String)],
  ) -> Option<Self> {
    if !required_frame_props.is_empty() {
      return None;
    }

    let mut visitor = LutVisitor::new();
    for node in ast {
      visitor.visit_expr(node);
    }
    if !visitor.eligible {
      return None;
    }
    let clip_idx = resolve_clip_name(visitor.clip?, src_types).ok()?;
    let src_type = src_types[clip_idx];
    if !matches!(src_type, ComponentType::U8 | ComponentType::U16) {
      return None;
    }

    // Cover every possible sample value of the source.
    let src: Vec<u8> = match src_type {
      ComponentType::U8 => (0..=u8::MAX).collect(),
      ComponentType::U16 => (0..=u16::MAX).flat_map(u16::to_le_bytes).collect(),
      ComponentType::F16 | ComponentType::F32 => unreachable!(),
    };
    let domain = 1usize << (src_type.bytes() * 8);
    debug_assert_eq!(src.len(), domain * src_type.bytes());

    // The expression only dereferences its single referenced clip, but the
    // compiled function receives one pointer per input.
    let src_ptrs = vec![src.as_ptr(); src_types.len()];
    let src_strides = vec![src.len() as i64; src_types.len()];

    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let width = domain as i32;
    macro_rules! run {
      ($t:ty) => {{
        let mut table = vec![<$t>::default(); domain];
        unsafe {
          main.invoke(
            &mut table,
            (domain * size_of::<$t>()) as i64,
            &src_ptrs,
            &src_strides,
            width,
            1,
            0,
            &[],
          );
        }
        table
      }};
    }
    let table = match dst_type {
      ComponentType::U8 => LookupTable::U8(run!(u8)),
      ComponentType::U16 | ComponentType::F16 => LookupTable::U16(run!(u16)),
      ComponentType::F32 => LookupTable::F32(run!(f32)),
    };

    Some(Self {
      clip_idx,
      src_type,
      table,
    })
  }

  /// Applies the lookup table.
  ///
  /// # Safety
  ///
  /// The caller must ensure that `dst` and `srcs` are valid and that the types
  /// match those used during compilation.
  #[allow(clippy::too_many_arguments)]
  pub(crate) unsafe fn invoke<D>(
    &self,
    dst: &mut [D],
    dst_stride: i64,
    srcs: &[*const u8],
    src_strides: &[i64],
    width: i32,
    height: i32,
  ) {
    let src_ptr = srcs[self.clip_idx];
    let src_stride = src_strides[self.clip_idx];
    let dst_ptr = dst.as_mut_ptr().cast::<u8>();
    #[allow(clippy::cast_sign_loss)]
    let (width, height) = (width as usize, height as usize);
    let src_u16 = matches!(self.src_type, ComponentType::U16);

    unsafe {
      match (&self.table, src_u16) {
        (LookupTable::U8(t), false) => {
          apply_lut::<u8, u8>(t, src_ptr, src_stride, dst_ptr, dst_stride, width, height);
        }
        (LookupTable::U8(t), true) => {
          apply_lut::<u16, u8>(t, src_ptr, src_stride, dst_ptr, dst_stride, width, height);
        }
        (LookupTable::U16(t), false) => {
          apply_lut::<u8, u16>(t, src_ptr, src_stride, dst_ptr, dst_stride, width, height);
        }
        (LookupTable::U16(t), true) => {
          apply_lut::<u16, u16>(t, src_ptr, src_stride, dst_ptr, dst_stride, width, height);
        }
        (LookupTable::F32(t), false) => {
          apply_lut::<u8, f32>(t, src_ptr, src_stride, dst_ptr, dst_stride, width, height);
        }
        (LookupTable::F32(t), true) => {
          apply_lut::<u16, f32>(t, src_ptr, src_stride, dst_ptr, dst_stride, width, height);
        }
      }
    }
  }
}

/// Applies `table` row by row.
unsafe fn apply_lut<S, D>(
  table: &[D],
  src_ptr: *const u8,
  src_stride: i64,
  dst_ptr: *mut u8,
  dst_stride: i64,
  width: usize,
  height: usize,
) where
  S: Copy,
  usize: From<S>,
  D: Copy,
{
  for y in 0..height {
    unsafe {
      let src_row = src_ptr.offset(y as isize * src_stride as isize).cast::<S>();
      let dst_row = dst_ptr.offset(y as isize * dst_stride as isize).cast::<D>();
      for x in 0..width {
        let s = usize::from(src_row.add(x).read());
        dst_row.add(x).write(*table.get_unchecked(s));
      }
    }
  }
}
