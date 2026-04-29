//! Errors.

use std::string::String;

use miette::Diagnostic;
use thiserror::Error;

/// Errors from cranexpr.
#[derive(Debug, Diagnostic, Error)]
pub enum TransformError {
  #[error("Invalid clip identifier '{0}'.")]
  InvalidClipIdentifier(String),

  #[error("Pixel access and per-pixel context are not allowed here: '{0}'.")]
  PixelAccessNotAllowed(String),
}
