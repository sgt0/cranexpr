use std::ffi::{CStr, c_void};

use const_str::cstr;
use vapoursynth4_rs::{
  VideoInfo,
  core::CoreRef,
  frame::{Frame, FrameContext, VideoFrame},
  key,
  map::{AppendMode, Key, MapRef, Value},
  node::{
    ActivationReason, Dependencies, Filter, FilterDependency, Node, RequestPattern, VideoNode,
  },
  utils::is_constant_video_format,
};

use cranexpr_codegen::{SelectFunction, compile_jit_select};
use cranexpr_transforms::{PixelAccessVisitor, PropVisitor, Visitor};

use crate::errors::CranexprError;

struct PropBinding {
  key: Key,
  bytecode: SelectFunction,
  required_frame_props: Vec<(usize, String)>,
  frame_prop_keys: Vec<Key>,
}

pub(crate) struct PropExprFilter {
  nodes: Vec<VideoNode>,
  vi: VideoInfo,
  bindings: Vec<PropBinding>,
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

impl Filter for PropExprFilter {
  type Error = CranexprError;
  type FrameType = VideoFrame;
  type FilterData = ();

  const NAME: &'static CStr = cstr!("PropExpr");
  const ARGS: &'static CStr = cstr!("clips:vnode[];any");
  const RETURN_TYPE: &'static CStr = cstr!("clip:vnode;");

  fn create(
    input: MapRef<'_>,
    output: MapRef<'_>,
    _data: Option<Box<Self::FilterData>>,
    mut core: CoreRef<'_>,
  ) -> Result<(), Self::Error> {
    let Some(num_clips) = input.num_elements(key!(c"clips")) else {
      return Err(CranexprError::NumberOfClips);
    };
    if num_clips == 0 {
      return Err(CranexprError::NumberOfClips);
    }

    let nodes = (0..num_clips)
      .map(|i| input.get_video_node(key!(c"clips"), i).unwrap())
      .collect::<Vec<_>>();

    for node in &nodes {
      if !is_constant_video_format(node.info()) {
        return Err(CranexprError::VariableFormat);
      }
    }

    let vi = nodes[0].info().clone();

    let num_keys = input.len();
    let mut bindings = Vec::new();

    for i in 0..num_keys {
      let map_key = input.get_key(i);

      if map_key == key!(c"clips") {
        continue;
      }

      let Ok(expr_str) = input.get_utf8(map_key, 0) else {
        return Err(CranexprError::PropExprValueNotString(map_key.to_string()));
      };

      let prop_key = Key::new(map_key.to_bytes())
        .map_err(|_| CranexprError::InvalidFramePropertyName(map_key.to_string()))?;

      let ast = cranexpr_parser::parse_expr(expr_str)?;

      let mut pa_visitor = PixelAccessVisitor::new();
      for node in &ast {
        pa_visitor.visit_expr(node);
      }
      if let Some(err) = pa_visitor.error {
        return Err(err.into());
      }

      let mut prop_visitor = PropVisitor::new(num_clips as usize);
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

      let bytecode = compile_jit_select(&ast, num_clips as usize, &required)?;

      bindings.push(PropBinding {
        key: prop_key,
        bytecode,
        required_frame_props: required,
        frame_prop_keys: keys,
      });
    }

    if bindings.is_empty() {
      return Err(CranexprError::NoExpression);
    }

    let deps: Vec<FilterDependency> = nodes
      .iter()
      .map(|node| FilterDependency {
        source: node.as_ptr(),
        request_pattern: if vi.num_frames <= node.info().num_frames {
          RequestPattern::StrictSpatial
        } else {
          RequestPattern::General
        },
      })
      .collect();

    let filter = Self {
      nodes,
      vi: vi.clone(),
      bindings,
    };

    core.create_video_filter(
      output,
      cstr!("PropExpr"),
      &vi,
      Box::new(filter),
      Dependencies::new(&deps).unwrap(),
    );

    Ok(())
  }

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
        let src_frames: Vec<VideoFrame> = self
          .nodes
          .iter()
          .map(|node| node.get_frame_filter(n, &mut ctx))
          .collect();

        let width = self.vi.width;
        let height = self.vi.height;

        // First compute all values.
        let mut values = Vec::with_capacity(self.bindings.len());
        for binding in &self.bindings {
          let mut props = Vec::with_capacity(binding.required_frame_props.len());
          for (i, (clip_idx, _)) in binding.required_frame_props.iter().enumerate() {
            let frame = &src_frames[*clip_idx];
            let key = &binding.frame_prop_keys[i];
            props.push(read_frame_prop(frame, key));
          }

          let val = unsafe { binding.bytecode.invoke(n, width, height, &props) };
          values.push(val);
        }

        // Then write them to the output frame.
        let mut dst = core.copy_frame(&src_frames[0]);
        if let Some(mut dst_props) = dst.properties_mut() {
          for (binding, &val) in self.bindings.iter().zip(values.iter()) {
            let int_val = val as i64;
            #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
            let roundtrips = val.is_finite() && val == int_val as f32;

            if roundtrips {
              dst_props
                .set(&binding.key, Value::Int(int_val), AppendMode::Replace)
                .ok();
            } else {
              dst_props
                .set(
                  &binding.key,
                  Value::Float(f64::from(val)),
                  AppendMode::Replace,
                )
                .ok();
            }
          }
        }

        return Ok(Some(dst));
      }
      ActivationReason::Error => {}
    }

    Ok(None)
  }
}
