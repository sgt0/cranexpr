//! Errors.

use std::ffi::{CStr, CString};
use std::string::String;

use miette::Diagnostic;
use thiserror::Error;

/// Errors from cranexpr.
#[derive(Debug, Diagnostic, Error)]
pub enum CranexprError {
  #[error(transparent)]
  Parse(#[from] cranexpr_parser::ParseError),

  #[error(transparent)]
  Transform(#[from] cranexpr_transforms::TransformError),

  #[error(transparent)]
  Codegen(#[from] cranexpr_codegen::errors::CodegenError),

  #[error("Clips in clip_src differ in format or dimensions.")]
  ClipSrcMismatch,

  #[error("clip_src must not be empty.")]
  EmptyClipSrc,

  #[error("prop_src must not be empty.")]
  EmptyPropSrc,

  #[error("At least one expression is required.")]
  NoExpression,

  #[error("Invalid frame property name '{0}'.")]
  InvalidFramePropertyName(String),

  #[error("More expressions given than there are planes.")]
  MoreExpressionsThanPlanes,

  #[error("Failed to get number of clips.")]
  NumberOfClips,

  #[error("Input and output formats have a different number of planes.")]
  PlanesMismatch,

  #[error("Unrecognized boundary mode.")]
  UnrecognizedBoundaryMode,

  #[error("Only clips with constant format and dimensions are allowed")]
  VariableFormat,
}

impl AsRef<CStr> for CranexprError {
  fn as_ref(&self) -> &CStr {
    let s = format!("cranexpr: {self}");
    let cs = CString::new(s).expect("error message should not contain NUL bytes");
    Box::leak(cs.into_boxed_c_str())
  }
}
