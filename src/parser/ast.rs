use serde::Serialize;
use strum_macros::{Display, EnumString};

/// Binary operator.
#[derive(Clone, Copy, Debug, Display, EnumString, PartialEq, Serialize)]
pub(crate) enum BinOp {
  /// The `+` operator (addition)
  #[strum(serialize = "+")]
  Add,
  /// The `-` operator (subtraction)
  #[strum(serialize = "-")]
  Sub,
  /// The `*` operator (multiplication)
  #[strum(serialize = "*")]
  Mul,
  /// The `/` operator (division)
  #[strum(serialize = "/")]
  Div,

  /// The `%` operator (modulo)
  #[strum(serialize = "%")]
  Rem,

  /// The `pow` operator (power)
  #[strum(serialize = "pow")]
  Pow,

  /// The `>` operator (greater than)
  #[strum(serialize = ">")]
  Gt,
  /// The `<` operator (less than)
  #[strum(serialize = "<")]
  Lt,

  /// The `max` operator (maximum)
  #[strum(serialize = "max")]
  Max,
  /// The `min` operator (minimum)
  #[strum(serialize = "min")]
  Min,
}

/// Unary operator.
#[derive(Clone, Copy, Debug, Display, EnumString, PartialEq, Serialize)]
pub(crate) enum UnOp {
  /// The `abs` operator (absolute value).
  #[strum(serialize = "abs")]
  Abs,
  /// The exponential function (`e^x`).
  #[strum(serialize = "exp")]
  Exp,
  /// The floor function.
  #[strum(serialize = "floor")]
  Floor,
  /// The round function.
  #[strum(serialize = "round")]
  Round,
  /// The natural logarithm.
  #[strum(serialize = "log")]
  Log,
  /// The `-` operator for negation.
  #[strum(serialize = "-")]
  Neg,
  /// The `not` operator for logical inversion
  #[strum(serialize = "not")]
  Not,
  /// The square root function.
  #[strum(serialize = "sqrt")]
  Sqrt,

  #[strum(serialize = "sin")]
  Sine,
  #[strum(serialize = "cos")]
  Cosine,
  #[strum(serialize = "tan")]
  Tangent,
}

/// The AST node for expressions.
#[derive(Clone, Debug, Serialize)]
pub(crate) enum Expr {
  /// A binary operation (e.g., `a b +`, `a b *`).
  Binary(BinOp, Box<Expr>, Box<Expr>),

  /// A unary operation (e.g., `x sin`, `x exp`).
  Unary(UnOp, Box<Expr>),

  /// A literal (e.g., `1e-06`).
  Lit(f32),

  /// An identifier.
  Ident(String),

  /// If/else, ternary.
  IfElse(Box<Expr>, Box<Expr>, Box<Expr>),

  /// Access of a frame property (e.g. `x.PlaneStatsAverage`).
  Prop(String, String),
}
