#[macro_use]
extern crate num_derive;

mod codegen;
mod errors;
mod lexer;
mod parser;
mod pixel;

use const_str::cstr;
use num_traits::FromPrimitive;
use serde::Serialize;
use std::{
  ffi::{CStr, c_void},
  ptr, slice,
};
use vapours::{frame::VapoursVideoFrame, generic::HoldsVideoFormat};
use vapoursynth4_rs::{
  ColorFamily, SampleType, VideoInfo,
  core::CoreRef,
  declare_plugin,
  frame::{Frame, FrameContext, VideoFrame},
  key,
  map::MapRef,
  node::{
    ActivationReason, Dependencies, Filter, FilterDependency, Node, RequestPattern, VideoNode,
  },
  utils::is_constant_video_format,
};

use crate::{
  codegen::{MainFunction, compiler::compile_jit},
  errors::CranexprError,
  pixel::Pixel,
};

#[derive(Clone, Copy, Debug, PartialEq)]
enum PlaneOp {
  Process,
  Copy,
  Undefined,
}

/// Boundary handling mode.
#[derive(Clone, Copy, Debug, FromPrimitive, Serialize)]
pub(crate) enum BoundaryMode {
  /// Clamped boundary.
  Clamp = 0,

  /// Mirrored boundary.
  Mirror = 1,
}

struct CranexprFilter {
  nodes: Vec<VideoNode>,
  vi: VideoInfo,
  planes: [PlaneOp; 3],
  bytecode: [Option<MainFunction>; 3],
}

impl Filter for CranexprFilter {
  type Error = CranexprError;
  type FrameType = VideoFrame;
  type FilterData = ();

  #[inline]
  fn create(
    input: MapRef<'_>,
    output: MapRef<'_>,
    _data: Option<Box<Self::FilterData>>,
    mut core: CoreRef<'_>,
  ) -> Result<(), Self::Error> {
    let Some(num_inputs) = input.num_elements(key!(c"clips")) else {
      return Err(CranexprError::NumberOfClips);
    };

    let nodes = (0..num_inputs)
      .map(|i| input.get_video_node(key!(c"clips"), i).unwrap())
      .collect::<Vec<_>>();
    let video_infos = nodes
      .iter()
      .map(|node| node.info().clone())
      .collect::<Vec<_>>();
    for vi in &video_infos {
      if !is_constant_video_format(vi) {
        return Err(CranexprError::VariableFormat);
      }
    }

    let mut vi = video_infos[0].clone();
    if let Ok(format) = input.get_int_saturated(key!(c"format"), 0) {
      let f = core.get_video_format_by_id(format as u32);
      if f.color_family != ColorFamily::Undefined {
        if vi.format.num_planes != f.num_planes {
          return Err(CranexprError::PlanesMismatch);
        }
        vi.format = core.query_video_format(
          f.color_family,
          f.sample_type,
          f.bits_per_sample,
          f.sub_sampling_w,
          f.sub_sampling_h,
        );
      }
    }

    let num_exprs = input.num_elements(key!(c"expr")).unwrap();
    if num_exprs > vi.format.num_planes {
      return Err(CranexprError::MoreExpressionsThanPlanes);
    }

    let mut expr = [""; 3];
    for i in 0..num_exprs {
      expr[i as usize] = input.get_utf8(key!(c"expr"), i).unwrap();
    }

    // Fill the rest of the exprs with the last specified one.
    for i in num_exprs..3 {
      expr[i as usize] = expr[num_exprs as usize - 1];
    }

    let Some(boundary_mode) =
      BoundaryMode::from_i64(input.get_int(key!(c"boundary"), 0).unwrap_or(0))
    else {
      return Err(CranexprError::UnrecognizedBoundaryMode);
    };

    let mut planes = [PlaneOp::Undefined; 3];
    let mut bytecode: [Option<MainFunction>; 3] = [None, None, None];

    for i in 0..vi.format.num_planes as usize {
      planes[i] = if expr[i].is_empty() {
        if vi.depth() == video_infos[0].depth() && vi.sample_type() == video_infos[0].sample_type()
        {
          PlaneOp::Copy
        } else {
          PlaneOp::Undefined
        }
      } else {
        PlaneOp::Process
      };

      if planes[i] != PlaneOp::Process {
        continue;
      }

      let dst_type = match vi.sample_type() {
        SampleType::Integer => match vi.format.bytes_per_sample {
          1 => Pixel::U8,
          2 => Pixel::U16,
          _ => unreachable!(),
        },
        SampleType::Float => Pixel::F32,
      };
      let src_types = video_infos
        .iter()
        .map(|vi| match vi.sample_type() {
          SampleType::Integer => match vi.format.bytes_per_sample {
            1 => Pixel::U8,
            2 => Pixel::U16,
            _ => unreachable!(),
          },
          SampleType::Float => Pixel::F32,
        })
        .collect::<Vec<_>>();

      bytecode[i] = Some(compile_jit(
        expr[i],
        dst_type,
        &src_types,
        Some(boundary_mode),
      )?);
    }

    let filter = Self {
      nodes: nodes.clone(),
      vi: vi.clone(),
      planes,
      bytecode,
    };

    let deps = nodes
      .iter()
      .map(|node| FilterDependency {
        source: node.as_ptr(),
        request_pattern: if filter.vi.num_frames <= node.info().num_frames {
          RequestPattern::StrictSpatial
        } else {
          RequestPattern::NoFrameReuse
        },
      })
      .collect::<Vec<FilterDependency>>();

    core.create_video_filter(
      output,
      cstr!("Expr"),
      &vi,
      Box::new(filter),
      Dependencies::new(&deps).unwrap(),
    );

    Ok(())
  }

  #[inline]
  fn get_frame(
    &self,
    n: i32,
    activation_reason: ActivationReason,
    _frame_data: *mut *mut c_void,
    mut ctx: FrameContext,
    core: CoreRef<'_>,
  ) -> Result<Option<VideoFrame>, Self::Error> {
    match activation_reason {
      ActivationReason::Initial => {
        for node in &self.nodes {
          ctx.request_frame_filter(n, node);
        }
      }
      ActivationReason::AllFramesReady => {
        let src = self
          .nodes
          .iter()
          .map(|node| node.get_frame_filter(n, &mut ctx))
          .collect::<Vec<_>>();

        let height = src[0].frame_height(0);
        let width = src[0].frame_width(0);
        let mut dst = core.new_video_frame2(
          &self.vi.format,
          width,
          height,
          &[
            if self.planes[0] == PlaneOp::Copy {
              src[0].as_ptr()
            } else {
              ptr::null()
            },
            if self.planes[1] == PlaneOp::Copy {
              src[0].as_ptr()
            } else {
              ptr::null()
            },
            if self.planes[2] == PlaneOp::Copy {
              src[0].as_ptr()
            } else {
              ptr::null()
            },
          ],
          &[0, 1, 2],
          Some(&src[0]),
        );

        for (plane_idx, (_plane_op, expr)) in self
          .planes
          .iter()
          .zip(self.bytecode.iter())
          .enumerate()
          .filter(|(_i, (plane_op, _expr))| **plane_op == PlaneOp::Process)
        {
          let width = dst.frame_width(plane_idx as i32);
          let height = dst.frame_height(plane_idx as i32);

          let expr = unsafe { expr.as_ref().unwrap_unchecked() };

          let src_slices = src.iter().map(|f| {
            let plane_idx = plane_idx as i32;
            let len = f.frame_height(plane_idx) as usize * f.stride(plane_idx) as usize;
            let ptr = f.as_slice::<u8>(plane_idx).as_ptr();
            unsafe { slice::from_raw_parts(ptr, len) }
          });

          match self.vi.sample_type() {
            SampleType::Integer => match self.vi.format.bytes_per_sample {
              1 => unsafe {
                expr.invoke(
                  dst.as_mut_slice::<u8>(plane_idx as i32),
                  src_slices.clone(),
                  width,
                  height,
                  n,
                );
              },
              2 => unsafe {
                expr.invoke(
                  dst.as_mut_slice::<u16>(plane_idx as i32),
                  src_slices.clone(),
                  width,
                  height,
                  n,
                );
              },
              _ => unreachable!(),
            },
            SampleType::Float => unsafe {
              expr.invoke(
                dst.as_mut_slice::<f32>(plane_idx as i32),
                src_slices.clone(),
                width,
                height,
                n,
              );
            },
          }
        }

        return Ok(Some(dst));
      }
      ActivationReason::Error => {}
    }

    Ok(None)
  }

  const NAME: &'static CStr = cstr!("Expr");
  const ARGS: &'static CStr = cstr!("clips:vnode[];expr:data[];format:int:opt;boundary:int:opt;");
  const RETURN_TYPE: &'static CStr = cstr!("clip:vnode;");
}

declare_plugin!(
  c"sgt.cranexpr",
  c"cranexpr",
  c"Cranelift Expr",
  (0, 1),
  VAPOURSYNTH_API_VERSION,
  0,
  (CranexprFilter, None)
);
