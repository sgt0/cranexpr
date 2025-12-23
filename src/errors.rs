//! Errors.

use std::ffi::{CStr, CString};
use std::string::String;

use miette::Diagnostic;
use thiserror::Error;

/// Errors from cranexpr.
#[derive(Debug, Diagnostic, Error)]
pub enum CranexprError {
  #[error("Compilation error: {0}")]
  CompilationError(String),

  #[error("Expression evaluates to nothing.")]
  ExpressionEvaluatesToNothing,

  #[error("Expression does not evaluate to a single value.")]
  ExpressionDoesNotEvaluateToSingleValue,

  #[error("Missing frame property name.")]
  MissingPropertyName,

  #[error("More expressions given than there are planes.")]
  MoreExpressionsThanPlanes,

  #[error("Failed to get number of clips.")]
  NumberOfClips,

  #[error("Input and output formats have a different number of planes.")]
  PlanesMismatch,

  #[error("Frame property name is not an identifier.")]
  PropertyNameNotAnIdentifier,

  #[error("Stack underflow.")]
  StackUnderflow,

  #[error("Undefined variable '{0}'.")]
  UndefinedVariable(String),

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

/// Alias for a `Result` that uses `CranexprError` as the error type.
pub type CranexprResult<T> = Result<T, CranexprError>;
