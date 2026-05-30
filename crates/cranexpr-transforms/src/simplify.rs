//! Algebraic simplification of expression ASTs.

use cranexpr_ast::{Expr, UnOp};

use crate::visit::{MutVisitor, walk_expr_mut};

const fn are_inverses(a: UnOp, b: UnOp) -> bool {
  matches!((a, b), (UnOp::Exp, UnOp::Log) | (UnOp::Log, UnOp::Exp))
}

struct Simplify;

impl MutVisitor for Simplify {
  fn fold_expr(&mut self, expr: &Expr) -> Expr {
    let expr = walk_expr_mut(self, expr);
    if let Expr::Unary(outer_op, inner) = &expr
      && let Expr::Unary(inner_op, x) = inner.as_ref()
      && are_inverses(*outer_op, *inner_op)
    {
      return x.as_ref().clone();
    }
    expr
  }
}

#[must_use]
pub fn simplify(expr: &Expr) -> Expr {
  Simplify.fold_expr(expr)
}

#[cfg(test)]
mod tests {
  use std::sync::Arc;

  use super::*;
  use cranexpr_ast::BinOp;

  fn ident(name: &str) -> Expr {
    Expr::Ident(name.to_string())
  }

  fn unary(op: UnOp, inner: Expr) -> Expr {
    Expr::Unary(op, Arc::new(inner))
  }

  #[test]
  fn exp_then_log_cancels() {
    let expr = unary(UnOp::Log, unary(UnOp::Exp, ident("x")));
    let result = simplify(&expr);
    assert!(matches!(result, Expr::Ident(s) if s == "x"));
  }

  #[test]
  fn log_then_exp_cancels() {
    let expr = unary(UnOp::Exp, unary(UnOp::Log, ident("x")));
    let result = simplify(&expr);
    assert!(matches!(result, Expr::Ident(s) if s == "x"));
  }

  #[test]
  fn nested_double_inverse_cancels() {
    // exp log exp log -> identity
    let expr = unary(
      UnOp::Log,
      unary(UnOp::Exp, unary(UnOp::Log, unary(UnOp::Exp, ident("x")))),
    );
    let result = simplify(&expr);
    assert!(matches!(result, Expr::Ident(s) if s == "x"));
  }

  #[test]
  fn non_inverse_pair_unchanged() {
    let expr = unary(UnOp::Log, unary(UnOp::Sine, ident("x")));
    let result = simplify(&expr);
    assert!(matches!(result, Expr::Unary(UnOp::Log, _)));
  }

  #[test]
  fn cancellation_inside_binary() {
    let expr = Expr::Binary(
      BinOp::Add,
      Arc::new(unary(UnOp::Log, unary(UnOp::Exp, ident("x")))),
      Arc::new(ident("y")),
    );
    let result = simplify(&expr);
    if let Expr::Binary(BinOp::Add, lhs, _) = &result {
      assert!(matches!(lhs.as_ref(), Expr::Ident(s) if s == "x"));
    } else {
      panic!("expected Binary");
    }
  }

  #[test]
  fn preserves_surrounding_ops() {
    // x sqrt sin exp log -> x sqrt sin
    let expr = unary(
      UnOp::Log,
      unary(UnOp::Exp, unary(UnOp::Sine, unary(UnOp::Sqrt, ident("x")))),
    );
    let result = simplify(&expr);
    assert!(matches!(result, Expr::Unary(UnOp::Sine, _)));
    if let Expr::Unary(UnOp::Sine, inner) = &result {
      assert!(matches!(inner.as_ref(), Expr::Unary(UnOp::Sqrt, _)));
    }
  }
}
