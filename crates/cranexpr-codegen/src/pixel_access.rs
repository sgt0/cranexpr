use cranelift::codegen::ir::immediates::Offset32;
use cranelift::codegen::ir::{ConstantData, Endianness};
use cranelift::prelude::*;
use cranexpr_ast::BoundaryMode;

use crate::component_type::ComponentType;
use crate::context::{FunctionCx, SIMD_LANES, SRC_MEMFLAGS, VEC_TYPE};
use crate::errors::CodegenError;
use crate::math_simd::{
  float_to_half_simd, fmax_simd, fmin_simd, half_to_float_simd, splat_f32, splat_i32,
};
use crate::pointer::Pointer;

/// Loads 4 adjacent pixels of arbitrary component type at `src_ptr + offset`,
/// promoting them to an `F32X4` vector.
pub(crate) fn load_pixel_vec_f32x4(
  fx: &mut FunctionCx<'_, '_>,
  src_ptr: Pointer,
  offset: Value,
  src_type: ComponentType,
  aligned: bool,
) -> Value {
  let ptr = src_ptr.offset_value(fx, offset);
  match src_type {
    ComponentType::F16 => {
      let addr = ptr.get_addr(fx);
      let halfbits = fx.bcx.ins().uload16x4(SRC_MEMFLAGS, addr, 0);
      half_to_float_simd(fx, halfbits)
    }
    ComponentType::F32 => {
      let flags = if aligned {
        SRC_MEMFLAGS.with_aligned()
      } else {
        SRC_MEMFLAGS
      };
      ptr.load(fx, VEC_TYPE, flags)
    }
    ComponentType::U16 => {
      let addr = ptr.get_addr(fx);
      let i32x4 = fx.bcx.ins().uload16x4(SRC_MEMFLAGS, addr, 0);
      fx.bcx.ins().fcvt_from_uint(VEC_TYPE, i32x4)
    }
    ComponentType::U8 => {
      let addr = ptr.get_addr(fx);
      let i16x8 = fx.bcx.ins().uload8x8(SRC_MEMFLAGS, addr, 0);
      let i32x4 = fx.bcx.ins().uwiden_low(i16x8);
      fx.bcx.ins().fcvt_from_uint(VEC_TYPE, i32x4)
    }
  }
}

/// Clamps, rounds, converts, and narrows an `F32X4` output vector to the
/// destination pixel format, then stores 4 pixels at `dest + offset`.
pub(crate) fn store_pixel_vec_f32x4(
  fx: &mut FunctionCx<'_, '_>,
  dest_ptr: Pointer,
  offset: Value,
  value: Value,
  dst_type: ComponentType,
) {
  let ptr = dest_ptr.offset_value(fx, offset);
  match dst_type {
    ComponentType::F16 => {
      let halfbits = float_to_half_simd(fx, value);
      // Narrow the 16-bit half patterns down to I16X8.
      let narrowed = fx.bcx.ins().unarrow(halfbits, halfbits);
      let byte_flags = MemFlagsData::new().with_endianness(Endianness::Little);
      let as_i64x2 = fx.bcx.ins().bitcast(types::I64X2, byte_flags, narrowed);
      let low = fx.bcx.ins().extractlane(as_i64x2, 0);
      ptr.store(fx, low, MemFlagsData::trusted());
    }
    ComponentType::F32 => {
      // VapourSynth guarantees row strides are aligned to at least 32 bytes.
      ptr.store(fx, value, MemFlagsData::trusted().with_aligned());
    }
    ComponentType::U8 | ComponentType::U16 => {
      let zero = splat_f32(fx, 0.0);
      let peak = splat_f32(fx, dst_type.peak_value());
      let clamped = fmax_simd(fx, value, zero);
      let clamped = fmin_simd(fx, clamped, peak);

      // `clamped` is within [0, 2^16), so adding 2^23 places each lane's
      // integer value in the low mantissa bits. Masking them out is cheaper
      // than doing `nearest` + `fcvt_to_sint_sat`.
      let magic = splat_f32(fx, 2f32.powi(23));
      let biased = fx.bcx.ins().fadd(clamped, magic);
      let bits = fx
        .bcx
        .ins()
        .bitcast(types::I32X4, MemFlagsData::new(), biased);
      let mantissa_mask = splat_i32(fx, 0xFFFF);
      let i32x4 = fx.bcx.ins().band(bits, mantissa_mask);
      let byte_flags = MemFlagsData::new().with_endianness(Endianness::Little);
      match dst_type {
        ComponentType::U16 => {
          // Narrow I32X4 -> I16X8 with unsigned saturation.
          let narrowed = fx.bcx.ins().unarrow(i32x4, i32x4);
          let as_i64x2 = fx.bcx.ins().bitcast(types::I64X2, byte_flags, narrowed);
          let low = fx.bcx.ins().extractlane(as_i64x2, 0);
          ptr.store(fx, low, MemFlagsData::trusted());
        }
        ComponentType::U8 => {
          // Narrow I32X4 -> I16X8 (unsigned saturating),
          // then narrow I16X8 -> I8X16 (unsigned saturating).
          let n16 = fx.bcx.ins().unarrow(i32x4, i32x4);
          let n8 = fx.bcx.ins().unarrow(n16, n16);
          let as_i32x4 = fx.bcx.ins().bitcast(types::I32X4, byte_flags, n8);
          let low = fx.bcx.ins().extractlane(as_i32x4, 0);
          ptr.store(fx, low, MemFlagsData::trusted());
        }
        ComponentType::F16 | ComponentType::F32 => unreachable!(),
      }
    }
  }
}

/// Scalar counterpart of [`half_to_float_simd`]: converts a zero-extended
/// `I32` binary16 bit pattern into an `F32`.
fn half_to_float_scalar(fx: &mut FunctionCx<'_, '_>, h: Value) -> Value {
  let no_flags = MemFlagsData::new();

  let mant_exp_mask = fx.bcx.ins().iconst(types::I32, 0x7fff);
  let mant_exp = fx.bcx.ins().band(h, mant_exp_mask);
  let shifted = fx.bcx.ins().ishl_imm_s(mant_exp, 13);
  let as_f = fx.bcx.ins().bitcast(types::F32, no_flags, shifted);

  let magic = fx.bcx.ins().f32const(f32::from_bits(0x7780_0000));
  let scaled = fx.bcx.ins().fmul(as_f, magic);

  let was_infnan = fx.bcx.ins().f32const(f32::from_bits(0x4780_0000));
  let infnan = fx
    .bcx
    .ins()
    .fcmp(FloatCC::GreaterThanOrEqual, scaled, was_infnan);
  let scaled_bits = fx.bcx.ins().bitcast(types::I32, no_flags, scaled);
  let inf_exp = fx.bcx.ins().iconst(types::I32, 255 << 23);
  let with_inf = fx.bcx.ins().bor(scaled_bits, inf_exp);
  let merged = fx.bcx.ins().select(infnan, with_inf, scaled_bits);

  let sign_mask = fx.bcx.ins().iconst(types::I32, 0x8000);
  let sign = fx.bcx.ins().band(h, sign_mask);
  let sign = fx.bcx.ins().ishl_imm_s(sign, 16);
  let result = fx.bcx.ins().bor(merged, sign);
  fx.bcx.ins().bitcast(types::F32, no_flags, result)
}

/// Per-lane x offsets used to synthesize the SIMD X coordinate vector.
///
/// Recall that in the scalar implementation, each pixel has one X coordinate.
/// For SIMD, a single iteration will be processing 4 adjacent pixels, so the X
/// variable has to be a vector of four coordinates: `[X, X+1, X+2, X+3]`.
pub(crate) fn simd_lane_offsets_f32x4(fx: &mut FunctionCx<'_, '_>) -> Value {
  const LANE_OFFSETS: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
  let mut bytes = [0u8; 16];
  for (i, f) in LANE_OFFSETS.iter().enumerate() {
    bytes[i * 4..(i + 1) * 4].copy_from_slice(&f.to_le_bytes());
  }
  let handle = fx.bcx.func.dfg.constants.insert(bytes.as_slice().into());
  fx.bcx.ins().vconst(VEC_TYPE, handle)
}

pub(crate) fn translate_rel_access_simd(
  fx: &mut FunctionCx<'_, '_>,
  clip_idx: usize,
  rel_x: i32,
  rel_y: i32,
  boundary_mode: BoundaryMode,
) -> Result<Value, CodegenError> {
  let x_var = fx
    .variables
    .get("X")
    .ok_or_else(|| CodegenError::UndefinedVariable("X".to_string()))?;
  let y_var = fx
    .variables
    .get("Y")
    .ok_or_else(|| CodegenError::UndefinedVariable("Y".to_string()))?;
  let x_base = fx.bcx.use_var(*x_var);
  let y_base = fx.bcx.use_var(*y_var);

  let src_type = fx.src_types[clip_idx];

  let target_y = if rel_y == 0 {
    y_base
  } else {
    let rel_y_val = fx.bcx.ins().iconst(types::I64, i64::from(rel_y));
    let shifted_y = fx.bcx.ins().iadd(y_base, rel_y_val);
    codegen_boundary_mode(fx, shifted_y, fx.height, boundary_mode)
  };

  if rel_x == 0 {
    let (src_ptr, pixel_offset) = codegen_pixel_offset(fx, clip_idx, x_base, target_y, src_type);
    return Ok(load_pixel_vec_f32x4(
      fx,
      src_ptr,
      pixel_offset,
      src_type,
      true,
    ));
  }

  match boundary_mode {
    BoundaryMode::Clamp => Ok(translate_rel_access_clamp_x(
      fx, clip_idx, rel_x, target_y, x_base,
    )),
    BoundaryMode::Mirror => Ok(translate_rel_access_mirror_x(
      fx, clip_idx, rel_x, target_y, x_base,
    )),
  }
}

fn translate_rel_access_clamp_x(
  fx: &mut FunctionCx<'_, '_>,
  clip_idx: usize,
  rel_x: i32,
  target_y: Value,
  x_base: Value,
) -> Value {
  let src_type = fx.src_types[clip_idx];
  let width = fx.width;
  let one = fx.bcx.ins().iconst(types::I64, 1);
  let zero = fx.bcx.ins().iconst(types::I64, 0);
  let width_minus_one = fx.bcx.ins().isub(width, one);
  let rel_x_val = fx.bcx.ins().iconst(types::I64, i64::from(rel_x));
  let shifted_x = fx.bcx.ins().iadd(x_base, rel_x_val);

  // clamped_x = clamp(x_base + rel_x, 0, width - 1)
  let too_large = fx
    .bcx
    .ins()
    .icmp(IntCC::SignedGreaterThan, shifted_x, width_minus_one);
  let clamped_hi = fx.bcx.ins().select(too_large, width_minus_one, shifted_x);
  let too_small = fx.bcx.ins().icmp(IntCC::SignedLessThan, clamped_hi, zero);
  let clamped_x = fx.bcx.ins().select(too_small, zero, clamped_hi);

  let (src_ptr, pixel_offset) = codegen_pixel_offset(fx, clip_idx, clamped_x, target_y, src_type);
  let raw = load_pixel_vec_f32x4(fx, src_ptr, pixel_offset, src_type, false);

  if rel_x < 0 {
    apply_left_swizzle_fixup(fx, raw, x_base, rel_x)
  } else {
    apply_right_swizzle_fixup(fx, raw, clamped_x, width)
  }
}

fn apply_left_swizzle_fixup(
  fx: &mut FunctionCx<'_, '_>,
  raw: Value,
  x_base: Value,
  rel_x: i32,
) -> Value {
  let abs_rel = i64::from(rel_x.unsigned_abs());
  let lanes = SIMD_LANES;
  let mut case_x: Vec<i64> = Vec::new();
  let mut case_masks: Vec<[u8; 4]> = Vec::new();
  let mut i: i64 = 0;
  while i < abs_rel {
    let mut mask = [0u8; 4];
    for j in 0..lanes {
      let src_lane = ((i + j + i64::from(rel_x)).max(0)) % lanes;
      mask[j as usize] = src_lane as u8;
    }
    case_x.push(i);
    case_masks.push(mask);
    i += lanes;
  }
  emit_swizzle_switch(fx, raw, x_base, &case_x, &case_masks, None)
}

fn apply_right_swizzle_fixup(
  fx: &mut FunctionCx<'_, '_>,
  raw: Value,
  clamped_x: Value,
  width: Value,
) -> Value {
  let lanes = SIMD_LANES;
  let lanes_val = fx.bcx.ins().iconst(types::I64, lanes);
  let x_plus_lanes = fx.bcx.ins().iadd(clamped_x, lanes_val);
  let dist = fx.bcx.ins().isub(x_plus_lanes, width);

  let zero = fx.bcx.ins().iconst(types::I64, 0);
  let in_bounds = fx.bcx.ins().icmp(IntCC::SignedLessThanOrEqual, dist, zero);

  let in_bounds_block = fx.bcx.create_block();
  let fixup_block = fx.bcx.create_block();
  let merge_block = fx.bcx.create_block();
  fx.bcx.append_block_param(merge_block, VEC_TYPE);
  fx.bcx
    .ins()
    .brif(in_bounds, in_bounds_block, &[], fixup_block, &[]);

  fx.bcx.switch_to_block(in_bounds_block);
  fx.bcx.seal_block(in_bounds_block);
  fx.bcx.ins().jump(merge_block, &[raw.into()]);

  fx.bcx.switch_to_block(fixup_block);
  fx.bcx.seal_block(fixup_block);

  let mut case_vals: Vec<i64> = Vec::new();
  let mut case_masks: Vec<[u8; 4]> = Vec::new();
  for i in 1..lanes {
    let mut mask = [0u8; 4];
    let mut last: u8 = 0;
    for j in 0..lanes {
      if j + i < lanes {
        mask[j as usize] = j as u8;
        last = j as u8;
      } else {
        mask[j as usize] = last;
      }
    }
    case_vals.push(i);
    case_masks.push(mask);
  }

  let default_mask = [0u8; 4];
  let fixed = emit_swizzle_switch(fx, raw, dist, &case_vals, &case_masks, Some(default_mask));
  fx.bcx.ins().jump(merge_block, &[fixed.into()]);

  fx.bcx.switch_to_block(merge_block);
  fx.bcx.seal_block(merge_block);
  fx.bcx.block_params(merge_block)[0]
}

fn emit_swizzle_switch(
  fx: &mut FunctionCx<'_, '_>,
  raw: Value,
  selector: Value,
  case_vals: &[i64],
  case_masks: &[[u8; 4]],
  default_mask: Option<[u8; 4]>,
) -> Value {
  assert_eq!(case_vals.len(), case_masks.len());
  if case_vals.is_empty() && default_mask.is_none() {
    return raw;
  }

  let merge_block = fx.bcx.create_block();
  fx.bcx.append_block_param(merge_block, VEC_TYPE);

  for (case_val, mask) in case_vals.iter().zip(case_masks.iter()) {
    let case_body = fx.bcx.create_block();
    let next_test = fx.bcx.create_block();
    let case_const = fx.bcx.ins().iconst(types::I64, *case_val);
    let matches = fx.bcx.ins().icmp(IntCC::Equal, selector, case_const);
    fx.bcx.ins().brif(matches, case_body, &[], next_test, &[]);

    fx.bcx.switch_to_block(case_body);
    fx.bcx.seal_block(case_body);
    let shuffled = apply_shuffle(fx, raw, *mask);
    fx.bcx.ins().jump(merge_block, &[shuffled.into()]);

    fx.bcx.switch_to_block(next_test);
    fx.bcx.seal_block(next_test);
  }

  let fallthrough = default_mask.map_or(raw, |mask| apply_shuffle(fx, raw, mask));
  fx.bcx.ins().jump(merge_block, &[fallthrough.into()]);

  fx.bcx.switch_to_block(merge_block);
  fx.bcx.seal_block(merge_block);
  fx.bcx.block_params(merge_block)[0]
}

fn apply_shuffle(fx: &mut FunctionCx<'_, '_>, raw: Value, lane_mask: [u8; 4]) -> Value {
  let mut bytes = [0u8; 16];
  for (j, &src_lane) in lane_mask.iter().enumerate() {
    let base = src_lane as usize * 4;
    for k in 0..4 {
      bytes[j * 4 + k] = (base + k) as u8;
    }
  }
  let imm = fx
    .bcx
    .func
    .dfg
    .immediates
    .push(ConstantData::from(bytes.as_slice()));
  let byte_flags = MemFlagsData::new().with_endianness(Endianness::Little);
  let raw_i8 = fx.bcx.ins().bitcast(types::I8X16, byte_flags, raw);
  let shuffled_i8 = fx.bcx.ins().shuffle(raw_i8, raw_i8, imm);
  fx.bcx.ins().bitcast(VEC_TYPE, byte_flags, shuffled_i8)
}

fn translate_rel_access_mirror_x(
  fx: &mut FunctionCx<'_, '_>,
  clip_idx: usize,
  rel_x: i32,
  target_y: Value,
  x_base: Value,
) -> Value {
  let src_type = fx.src_types[clip_idx];
  let lanes = SIMD_LANES;
  let rel_x_val = fx.bcx.ins().iconst(types::I64, i64::from(rel_x));

  let zero = fx.bcx.ins().f32const(0.0f32);
  let mut vec = fx.bcx.ins().splat(VEC_TYPE, zero);
  for lane in 0..lanes {
    let lane_off = fx.bcx.ins().iconst(types::I64, lane);
    let x_unshifted = fx.bcx.ins().iadd(x_base, lane_off);
    let x_shifted = fx.bcx.ins().iadd(x_unshifted, rel_x_val);
    let x_mirrored = codegen_boundary_mode(fx, x_shifted, fx.width, BoundaryMode::Mirror);
    let (src_ptr, pixel_offset) =
      codegen_pixel_offset(fx, clip_idx, x_mirrored, target_y, src_type);
    let scalar = load_scalar_as_f32(fx, src_ptr, pixel_offset, src_type);
    vec = fx.bcx.ins().insertlane(vec, scalar, lane as u8);
  }
  vec
}

pub(crate) fn translate_abs_access_simd(
  fx: &mut FunctionCx<'_, '_>,
  clip_idx: usize,
  x_vec: Value,
  y_vec: Value,
  boundary_mode: BoundaryMode,
) -> Value {
  let src_type = fx.src_types[clip_idx];
  let x_rounded = fx.bcx.ins().nearest(x_vec);
  let y_rounded = fx.bcx.ins().nearest(y_vec);
  let x_i32x4 = fx.bcx.ins().fcvt_to_sint_sat(types::I32X4, x_rounded);
  let y_i32x4 = fx.bcx.ins().fcvt_to_sint_sat(types::I32X4, y_rounded);

  let zero = fx.bcx.ins().f32const(0.0f32);
  let mut vec = fx.bcx.ins().splat(VEC_TYPE, zero);
  for lane in 0..SIMD_LANES {
    let x_lane_i32 = fx.bcx.ins().extractlane(x_i32x4, lane as u8);
    let y_lane_i32 = fx.bcx.ins().extractlane(y_i32x4, lane as u8);
    let x_lane = fx.bcx.ins().sextend(types::I64, x_lane_i32);
    let y_lane = fx.bcx.ins().sextend(types::I64, y_lane_i32);
    let clamped_x = codegen_boundary_mode(fx, x_lane, fx.width, boundary_mode);
    let clamped_y = codegen_boundary_mode(fx, y_lane, fx.height, boundary_mode);
    let (src_ptr, pixel_offset) =
      codegen_pixel_offset(fx, clip_idx, clamped_x, clamped_y, src_type);
    let scalar = load_scalar_as_f32(fx, src_ptr, pixel_offset, src_type);
    vec = fx.bcx.ins().insertlane(vec, scalar, lane as u8);
  }
  vec
}

/// Loads a single pixel of arbitrary component type at `src_ptr + offset` and
/// promotes it to an `F32` scalar.
fn load_scalar_as_f32(
  fx: &mut FunctionCx<'_, '_>,
  src_ptr: Pointer,
  offset: Value,
  src_type: ComponentType,
) -> Value {
  let ptr = src_ptr.offset_value(fx, offset);
  match src_type {
    ComponentType::F16 => {
      // Load the raw f16 bits as an integer and convert via bit twiddling.
      let halfbits = ptr.load(fx, types::I16, SRC_MEMFLAGS);
      let halfbits = fx.bcx.ins().uextend(types::I32, halfbits);
      half_to_float_scalar(fx, halfbits)
    }
    ComponentType::U8 | ComponentType::U16 => {
      let val = ptr.load(fx, src_type.into(), SRC_MEMFLAGS);
      fx.bcx.ins().fcvt_from_uint(types::F32, val)
    }
    ComponentType::F32 => ptr.load(fx, types::F32, SRC_MEMFLAGS),
  }
}

/// Computes the byte offset for a pixel at (x, y).
pub(crate) fn codegen_pixel_offset(
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
  let col_offset = fx.bcx.ins().imul_imm_s(x, src_type.bytes() as i64);
  let pixel_offset = fx.bcx.ins().iadd(row_offset, col_offset);

  (src_ptr, pixel_offset)
}

/// Resolves a clip name (e.g., "x", "y", "src0") to a clip index.
pub(crate) fn resolve_clip_name(
  clip: &str,
  src_types: &[ComponentType],
) -> Result<usize, CodegenError> {
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
