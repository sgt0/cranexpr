//! LUT compilation strategy.
//!
//! When an expression is a pure function of the current pixel of a single
//! integer clip, or of two 8-bit clips, the whole expression can be evaluated
//! once per possible combination of input values at compile time. Runtime
//! evaluation then boils down to a table lookup.

use cranexpr_ast::Expr;
use cranexpr_transforms::{LutVisitor, Visitor};

use crate::MainFunction;
use crate::component_type::ComponentType;
use crate::pixel_access::resolve_clip_name;

/// A compiled expression as a lookup table.
///
/// The table is indexed by the raw source sample(s) and stores raw destination
/// samples.
#[derive(Debug)]
pub struct LutFunction {
  inputs: LutInputs,
  table: LookupTable,
}

#[derive(Debug)]
enum LutInputs {
  /// One integer clip. The table is indexed by its sample.
  One {
    clip_idx: usize,
    src_type: ComponentType,
  },
  /// Two 8-bit clips. The table is indexed by `(hi << 8) | lo`.
  Two { lo: usize, hi: usize },
}

#[derive(Debug)]
enum LookupTable {
  U8(Vec<u8>),
  U16(Vec<u16>),
  F32(Vec<f32>),
}

impl LutFunction {
  /// Builds a LUT for `ast` when it qualifies by evaluating `main` over the
  /// whole input domain of the referenced clips.
  ///
  /// Returns `None` when the expression is not a pure function of the current
  /// pixel of one integer clip or two 8-bit clips.
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
    // Note that different names may refer to the same clip, e.g. `x` and
    // `src0`.
    let mut clip_idxs = Vec::with_capacity(visitor.clips.len());
    for name in visitor.clips {
      let idx = resolve_clip_name(name, src_types).ok()?;
      if !clip_idxs.contains(&idx) {
        clip_idxs.push(idx);
      }
    }

    // Each referenced clip's synthetic input as (clip index, samples, stride).
    let (inputs, width, height, srcs) = match *clip_idxs.as_slice() {
      [clip_idx] => {
        let src_type = src_types[clip_idx];
        // Cover every possible sample value of the source in one row.
        let src: Vec<u8> = match src_type {
          ComponentType::U8 => (0..=u8::MAX).collect(),
          ComponentType::U16 => (0..=u16::MAX).flat_map(u16::to_le_bytes).collect(),
          ComponentType::F16 | ComponentType::F32 => return None,
        };
        let domain: usize = 1 << (src_type.bytes() * 8);
        let stride = src.len() as i64;
        (
          LutInputs::One { clip_idx, src_type },
          domain,
          1,
          vec![(clip_idx, src, stride)],
        )
      }
      [lo, hi] => {
        if src_types[lo] != ComponentType::U8 || src_types[hi] != ComponentType::U8 {
          return None;
        }
        // A 256x256 grid where `lo` is the column and `hi` the row.
        let lo_src: Vec<u8> = (0..=u8::MAX).collect();
        let hi_src: Vec<u8> = (0..=u8::MAX).flat_map(|v| [v; 256]).collect();
        (
          LutInputs::Two { lo, hi },
          256,
          256,
          vec![(lo, lo_src, 0), (hi, hi_src, 256)],
        )
      }
      _ => return None,
    };

    // The expression only dereferences its referenced clips, but the compiled
    // function receives one pointer per input.
    let mut src_ptrs = vec![srcs[0].1.as_ptr(); src_types.len()];
    let mut src_strides = vec![srcs[0].2; src_types.len()];
    for (idx, src, stride) in &srcs {
      src_ptrs[*idx] = src.as_ptr();
      src_strides[*idx] = *stride;
    }

    macro_rules! run {
      ($t:ty) => {{
        let mut table = vec![<$t>::default(); width * height];
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        unsafe {
          main.invoke(
            &mut table,
            (width * size_of::<$t>()) as i64,
            &src_ptrs,
            &src_strides,
            width as i32,
            height as i32,
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

    Some(Self { inputs, table })
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
    let dst_ptr = dst.as_mut_ptr().cast::<u8>();
    #[allow(clippy::cast_sign_loss)]
    let (width, height) = (width as usize, height as usize);

    match self.inputs {
      LutInputs::One { clip_idx, src_type } => {
        let src_ptr = srcs[clip_idx];
        let src_stride = src_strides[clip_idx];
        let src_u16 = matches!(src_type, ComponentType::U16);
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
      LutInputs::Two { lo, hi } => {
        let lo = (srcs[lo], src_strides[lo]);
        let hi = (srcs[hi], src_strides[hi]);
        unsafe {
          match &self.table {
            LookupTable::U8(t) => apply_lut2(t, lo, hi, dst_ptr, dst_stride, width, height),
            LookupTable::U16(t) => apply_lut2(t, lo, hi, dst_ptr, dst_stride, width, height),
            LookupTable::F32(t) => apply_lut2(t, lo, hi, dst_ptr, dst_stride, width, height),
          }
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

/// Applies a two-clip `table` row by row, indexing it by `(hi << 8) | lo`.
unsafe fn apply_lut2<D: Copy>(
  table: &[D],
  (lo_ptr, lo_stride): (*const u8, i64),
  (hi_ptr, hi_stride): (*const u8, i64),
  dst_ptr: *mut u8,
  dst_stride: i64,
  width: usize,
  height: usize,
) {
  debug_assert_eq!(table.len(), 1 << 16);
  for y in 0..height {
    unsafe {
      let lo_row = lo_ptr.offset(y as isize * lo_stride as isize);
      let hi_row = hi_ptr.offset(y as isize * hi_stride as isize);
      let dst_row = dst_ptr.offset(y as isize * dst_stride as isize).cast::<D>();
      for x in 0..width {
        let s = usize::from(lo_row.add(x).read()) | (usize::from(hi_row.add(x).read()) << 8);
        dst_row.add(x).write(*table.get_unchecked(s));
      }
    }
  }
}
