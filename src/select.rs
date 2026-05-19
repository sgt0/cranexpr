use std::{
  ffi::{CStr, c_void},
  ptr,
};

use const_str::cstr;
use vapoursynth4_rs::{
  VideoInfo,
  core::CoreRef,
  ffi::VSFrame,
  frame::{Frame, FrameContext, VideoFrame},
  key,
  map::{Key, MapRef},
  node::{
    ActivationReason, Dependencies, Filter, FilterDependency, Node, RequestPattern, VideoNode,
  },
  utils::{is_constant_video_format, is_same_video_info},
};

use cranexpr_codegen::{SelectFunction, compile_jit_select};
use cranexpr_transforms::{PixelAccessVisitor, PropVisitor, Visitor};

use crate::errors::CranexprError;

pub(crate) struct SelectFilter {
  clip_nodes: Vec<VideoNode>,
  prop_nodes: Vec<VideoNode>,
  vi: VideoInfo,
  bytecode: Vec<SelectFunction>,
  required_frame_props: Vec<Vec<(usize, String)>>,
  frame_prop_keys: Vec<Vec<Key>>,
}

struct FrameState {
  indices: Vec<i32>,
}

fn read_frame_prop(frame: &VideoFrame, key: &Key) -> f32 {
  if let Some(props) = frame.properties() {
    if let Ok(float_val) = props.get_float(key, 0) {
      return float_val as f32;
    }
    if let Ok(int_val) = props.get_int(key, 0) {
      #[allow(clippy::cast_precision_loss)]
      return int_val as f32;
    }
    if let Ok(bin_val) = props.get_binary(key, 0)
      && !bin_val.is_empty()
    {
      return f32::from(bin_val[0]);
    }
  }
  0.0
}

impl SelectFilter {
  fn compute_indices(&self, n: i32, prop_frames: &[VideoFrame]) -> Vec<i32> {
    let num_clips = self.clip_nodes.len() as i32;
    let num_planes = self.vi.format.num_planes as usize;
    let mut indices = Vec::with_capacity(num_planes);

    // akarin.Select uses the clip's dimensions rather than the plane's
    // dimensions like Expr does. Not sure if this is intentional or not.
    let width = self.vi.width;
    let height = self.vi.height;

    for plane in 0..num_planes {
      let required = &self.required_frame_props[plane];
      let keys = &self.frame_prop_keys[plane];

      let mut props = Vec::with_capacity(required.len());
      for (i, (clip_idx, _)) in required.iter().enumerate() {
        let frame = &prop_frames[*clip_idx];
        let key = &keys[i];
        props.push(read_frame_prop(frame, key));
      }

      let raw = unsafe { self.bytecode[plane].invoke(n, width, height, &props) };
      let rounded = raw.round();
      #[allow(clippy::cast_precision_loss)]
      let max_f = (num_clips - 1) as f32;
      let idx = if rounded.is_nan() || rounded <= 0.0 {
        0
      } else if rounded >= max_f {
        num_clips - 1
      } else {
        rounded as i32
      };
      indices.push(idx);
    }

    indices
  }
}

impl Filter for SelectFilter {
  type Error = CranexprError;
  type FrameType = VideoFrame;
  type FilterData = ();

  const NAME: &'static CStr = cstr!("Select");
  const ARGS: &'static CStr = cstr!("clip_src:vnode[];prop_src:vnode[];expr:data[];");
  const RETURN_TYPE: &'static CStr = cstr!("clip:vnode;");

  #[inline]
  fn create(
    input: MapRef<'_>,
    output: MapRef<'_>,
    _data: Option<Box<Self::FilterData>>,
    mut core: CoreRef<'_>,
  ) -> Result<(), Self::Error> {
    let Some(num_clip_src) = input.num_elements(key!(c"clip_src")) else {
      return Err(CranexprError::EmptyClipSrc);
    };
    if num_clip_src == 0 {
      return Err(CranexprError::EmptyClipSrc);
    }

    let clip_nodes = (0..num_clip_src)
      .map(|i| input.get_video_node(key!(c"clip_src"), i).unwrap())
      .collect::<Vec<_>>();

    let Some(num_prop_src) = input.num_elements(key!(c"prop_src")) else {
      return Err(CranexprError::EmptyPropSrc);
    };
    if num_prop_src == 0 {
      return Err(CranexprError::EmptyPropSrc);
    }

    let prop_nodes = (0..num_prop_src)
      .map(|i| input.get_video_node(key!(c"prop_src"), i).unwrap())
      .collect::<Vec<_>>();

    for node in &clip_nodes {
      if !is_constant_video_format(node.info()) {
        return Err(CranexprError::VariableFormat);
      }
    }
    for node in &prop_nodes {
      if !is_constant_video_format(node.info()) {
        return Err(CranexprError::VariableFormat);
      }
    }

    let vi = clip_nodes[0].info().clone();
    for node in clip_nodes.iter().skip(1) {
      let other = node.info();
      if !is_same_video_info(&vi, other) {
        return Err(CranexprError::ClipSrcMismatch);
      }
    }

    let num_exprs = input.num_elements(key!(c"expr")).unwrap_or(0);
    if num_exprs == 0 {
      return Err(CranexprError::NoExpression);
    }
    if num_exprs > vi.format.num_planes {
      return Err(CranexprError::MoreExpressionsThanPlanes);
    }

    let num_planes = vi.format.num_planes as usize;
    let mut expr = vec![""; num_planes];
    for i in 0..num_exprs {
      expr[i as usize] = input.get_utf8(key!(c"expr"), i).unwrap();
    }
    // Fill the rest of the exprs with the last specified one.
    for i in (num_exprs as usize)..num_planes {
      expr[i] = expr[num_exprs as usize - 1];
    }

    let mut bytecode = Vec::with_capacity(num_planes);
    let mut required_frame_props: Vec<Vec<(usize, String)>> = Vec::with_capacity(num_planes);
    let mut frame_prop_keys: Vec<Vec<Key>> = Vec::with_capacity(num_planes);

    for plane_expr in &expr {
      let ast = cranexpr_parser::parse_expr(plane_expr)?;

      let mut pa_visitor = PixelAccessVisitor::new();
      for node in &ast {
        pa_visitor.visit_expr(node);
      }
      if let Some(err) = pa_visitor.error {
        return Err(err.into());
      }

      let mut prop_visitor = PropVisitor::new(prop_nodes.len());
      for node in &ast {
        prop_visitor.visit_expr(node);
      }
      if let Some(err) = prop_visitor.error {
        return Err(err.into());
      }

      let required: Vec<(usize, String)> = prop_visitor.props.into_iter().collect();
      let mut keys = Vec::with_capacity(required.len());
      for (_, prop_name) in &required {
        keys.push(
          Key::new(prop_name.as_str())
            .map_err(|_| CranexprError::InvalidFramePropertyName(prop_name.clone()))?,
        );
      }

      bytecode.push(compile_jit_select(&ast, prop_nodes.len(), &required)?);
      required_frame_props.push(required);
      frame_prop_keys.push(keys);
    }

    let mut deps: Vec<FilterDependency> = Vec::with_capacity(clip_nodes.len() + prop_nodes.len());
    for node in &clip_nodes {
      deps.push(FilterDependency {
        source: node.as_ptr(),
        request_pattern: if vi.num_frames <= node.info().num_frames {
          RequestPattern::StrictSpatial
        } else {
          RequestPattern::General
        },
      });
    }
    for node in &prop_nodes {
      deps.push(FilterDependency {
        source: node.as_ptr(),
        request_pattern: if vi.num_frames <= node.info().num_frames {
          RequestPattern::StrictSpatial
        } else {
          RequestPattern::General
        },
      });
    }

    let filter = Self {
      clip_nodes,
      prop_nodes,
      vi: vi.clone(),
      bytecode,
      required_frame_props,
      frame_prop_keys,
    };

    core.create_video_filter(
      output,
      cstr!("Select"),
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
    frame_data: *mut *mut c_void,
    mut ctx: FrameContext,
    core: CoreRef<'_>,
  ) -> Result<Option<VideoFrame>, Self::Error> {
    match activation_reason {
      ActivationReason::Initial => {
        for node in &self.prop_nodes {
          ctx.request_frame_filter(n, node);
        }
        Ok(None)
      }
      ActivationReason::AllFramesReady => {
        let state_ptr = unsafe { *frame_data }.cast::<FrameState>();
        if state_ptr.is_null() {
          // First pass: evaluate expressions and request the selected clips.
          let prop_frames = self
            .prop_nodes
            .iter()
            .map(|node| node.get_frame_filter(n, &mut ctx))
            .collect::<Vec<_>>();

          let indices = self.compute_indices(n, &prop_frames);

          // Request each unique selected clip index once.
          let mut requested = vec![false; self.clip_nodes.len()];
          for &idx in &indices {
            let uidx = idx as usize;
            if !requested[uidx] {
              requested[uidx] = true;
              ctx.request_frame_filter(n, &self.clip_nodes[uidx]);
            }
          }

          let state = Box::new(FrameState { indices });
          unsafe {
            *frame_data = Box::into_raw(state).cast::<c_void>();
          }
          return Ok(None);
        }

        // Second pass: all requested clip frames are ready.
        let state = unsafe { Box::from_raw(state_ptr) };
        unsafe {
          *frame_data = ptr::null_mut();
        }
        let indices = state.indices;

        // Fetch one frame per plane.
        let num_planes = self.vi.format.num_planes as usize;
        let plane_frames: Vec<VideoFrame> = indices
          .iter()
          .take(num_planes)
          .map(|&idx| self.clip_nodes[idx as usize].get_frame_filter(n, &mut ctx))
          .collect();

        let plane_src_ptrs: Vec<*const VSFrame> = plane_frames
          .iter()
          .map(|f| f.as_ptr().cast_const())
          .collect();
        let plane_src_indices: Vec<i32> = (0..num_planes as i32).collect();

        let dst = core.new_video_frame2(
          &self.vi.format,
          self.vi.width,
          self.vi.height,
          &plane_src_ptrs,
          &plane_src_indices,
          Some(&plane_frames[0]),
        );

        Ok(Some(dst))
      }
      ActivationReason::Error => {
        let state_ptr = unsafe { *frame_data }.cast::<FrameState>();
        if !state_ptr.is_null() {
          // Drop the state.
          let _ = unsafe { Box::from_raw(state_ptr) };
          unsafe {
            *frame_data = ptr::null_mut();
          }
        }
        Ok(None)
      }
    }
  }
}
