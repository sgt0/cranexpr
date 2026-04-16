//! Errors.

use std::string::String;

use miette::Diagnostic;
use thiserror::Error;

/// Errors from cranexpr.
#[derive(Debug, Diagnostic, Error)]
pub enum TransformError {
  #[error("Invalid clip identifier '{0}'.")]
  InvalidClipIdentifier(String),
}
