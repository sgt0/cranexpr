use miette::Diagnostic;
use thiserror::Error;

/// Errors from code generation.
#[derive(Debug, Diagnostic, Error)]
pub enum CodegenError {
  #[error(transparent)]
  Parse(#[from] cranexpr_parser::ParseError),

  #[error("Compilation error: {0}")]
  CompilationError(String),

  #[error("Undefined variable '{0}'.")]
  UndefinedVariable(String),
}

/// Alias for a `Result` that uses `CodegenError` as the error type.
pub type CodegenResult<T> = Result<T, CodegenError>;
