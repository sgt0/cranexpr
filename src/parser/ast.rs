use serde::Serialize;
use strum_macros::{Display, EnumString};

use crate::BoundaryMode;

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

  /// The `atan2` operator
  #[strum(serialize = "atan2")]
  Atan2,

  /// Bitwise AND
  #[strum(serialize = "bitand")]
  BitAnd,
  /// Bitwise OR
  #[strum(serialize = "bitor")]
  BitOr,
  /// Bitwise XOR
  #[strum(serialize = "bitxor")]
  BitXor,
}

/// Unary operator.
#[derive(Clone, Copy, Debug, Display, EnumString, PartialEq, Serialize)]
pub(crate) enum UnOp {
  /// The `abs` operator (absolute value).
  #[strum(serialize = "abs")]
  Abs,
  /// Bitwise NOT
  #[strum(serialize = "bitnot")]
  BitNot,
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
  /// The `not` operator for logical inversion.
  #[strum(serialize = "not")]
  Not,
  /// The `sgn` operator.
  #[strum(serialize = "sgn")]
  Sign,
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

/// Ternary operator.
#[derive(Clone, Copy, Debug, Display, EnumString, PartialEq, Serialize)]
pub(crate) enum TernaryOp {
  /// The `clip` or `clamp` operator (clamps a value to a range).
  #[strum(serialize = "clip")]
  Clip,
}

/// The AST node for expressions.
#[derive(Clone, Debug, Serialize)]
pub(crate) enum Expr {
  /// A binary operation (e.g., `a b +`, `a b *`).
  Binary(BinOp, Box<Self>, Box<Self>),

  /// A unary operation (e.g., `x sin`, `x exp`).
  Unary(UnOp, Box<Self>),

  /// A ternary operation (e.g., `x min_val max_val clip`).
  Ternary(TernaryOp, Box<Self>, Box<Self>, Box<Self>),

  /// A literal (e.g., `1e-06`).
  Lit(f32),

  /// An identifier.
  Ident(String),

  /// If/else, ternary.
  IfElse(Box<Self>, Box<Self>, Box<Self>),

  /// Access of a frame property (e.g. `x.PlaneStatsAverage`).
  Prop(String, String),

  /// Store a value into a variable.
  Store(String, Box<Self>),

  /// Load a value from a variable.
  Load(String),

  /// Relative pixel access (e.g. `x[-1, 0]:c`).
  RelAccess {
    /// Clip identifier.
    clip: String,

    /// Relative X offset.
    rel_x: i32,

    /// Relative Y offset.
    rel_y: i32,

    /// Optional boundary mode override.
    boundary_mode: Option<BoundaryMode>,
  },
}
