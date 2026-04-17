use thiserror::Error;

/// Errors from the parser.
#[derive(Debug, Error)]
pub enum ParseError {
  #[error("Attempt to drop out of bounds.")]
  DropOutOfBounds,

  #[error("Attempt to dup out of bounds.")]
  DupOutOfBounds,

  #[error("Expression evaluates to nothing.")]
  ExpressionEvaluatesToNothing,

  #[error("Expression does not evaluate to a single value.")]
  ExpressionDoesNotEvaluateToSingleValue,

  #[error("Invalid literal: {0}")]
  InvalidLiteral(String),

  #[error("Missing frame property name.")]
  MissingFramePropertyName,

  #[error("Frame property name is not an identifier.")]
  PropertyNameNotAnIdentifier,

  #[error("Stack underflow.")]
  StackUnderflow,

  #[error("Attempt to swap out of bounds.")]
  SwapOutOfBounds,

  #[error("Reference to uninitialized variable: {0}@")]
  UninitializedVariable(String),

  #[error("Unrecognized token: {0}")]
  UnrecognizedToken(String),
}
