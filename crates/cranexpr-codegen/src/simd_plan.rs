use cranexpr_ast::{BinOp, Expr, UnOp};

use crate::component_type::ComponentType;

pub(crate) const SIMD_LANES: i64 = 4;

/// Marker struct for a SIMD-eligible expression.
pub(crate) struct SimdPlan {
  pub(crate) lanes: i64,
}

pub(crate) fn try_simd_plan(
  ast: &[Expr],
  dst_type: ComponentType,
  src_types: &[ComponentType],
) -> Option<SimdPlan> {
  if dst_type != ComponentType::F32 {
    return None;
  }
  if src_types.iter().any(|t| *t != ComponentType::F32) {
    return None;
  }
  if ast.is_empty() {
    return None;
  }

  for expr in ast {
    if !is_simd_eligible(expr) {
      return None;
    }
  }

  Some(SimdPlan { lanes: SIMD_LANES })
}

/// Returns `true` if every node of the given expression tree is supported by
/// the SIMD translator.
fn is_simd_eligible(expr: &Expr) -> bool {
  match expr {
    Expr::Lit(_) | Expr::Ident(_) | Expr::Prop(..) | Expr::Load(_) | Expr::RelAccess { .. } => true,

    Expr::Binary(op, lhs, rhs) => {
      is_binop_simd_eligible(*op) && is_simd_eligible(lhs) && is_simd_eligible(rhs)
    }

    Expr::Unary(op, x) => is_unop_simd_eligible(*op) && is_simd_eligible(x),

    Expr::Ternary(_, a, b, c) | Expr::IfElse(a, b, c) => {
      is_simd_eligible(a) && is_simd_eligible(b) && is_simd_eligible(c)
    }

    Expr::Store(_, inner) => is_simd_eligible(inner),

    Expr::AbsAccess { x, y, .. } => is_simd_eligible(x) && is_simd_eligible(y),
  }
}

const fn is_binop_simd_eligible(op: BinOp) -> bool {
  matches!(
    op,
    BinOp::Add
      | BinOp::Sub
      | BinOp::Mul
      | BinOp::Div
      | BinOp::Rem
      | BinOp::Gt
      | BinOp::Lt
      | BinOp::Eq
      | BinOp::Gte
      | BinOp::Lte
      | BinOp::Max
      | BinOp::Min
      | BinOp::And
      | BinOp::Or
      | BinOp::Xor
      | BinOp::BitAnd
      | BinOp::BitOr
      | BinOp::BitXor
      | BinOp::Pow
      | BinOp::Atan2
  )
}

const fn is_unop_simd_eligible(op: UnOp) -> bool {
  matches!(
    op,
    UnOp::Abs
      | UnOp::Neg
      | UnOp::Sqrt
      | UnOp::Not
      | UnOp::Sign
      | UnOp::BitNot
      | UnOp::Floor
      | UnOp::Round
      | UnOp::Trunc
      | UnOp::Sine
      | UnOp::Cosine
      | UnOp::Tangent
      | UnOp::Exp
      | UnOp::Log
  )
}
