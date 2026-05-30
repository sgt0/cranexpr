use std::sync::Arc;

use cranexpr_ast::{BinOp, Expr, TernaryOp, UnOp};

/// A trait for AST visitors. Visits all nodes in the AST recursively.
pub trait Visitor<'a>: Sized {
  fn visit_expr(&mut self, expr: &'a Expr) {
    walk_expr(self, expr);
  }
  fn visit_binary_op(&mut self, op: &'a BinOp) {
    walk_binary_op(self, op);
  }
  fn visit_ternary_op(&mut self, op: &'a TernaryOp) {
    walk_ternary_op(self, op);
  }
  fn visit_ident(&mut self, ident: &'a str) {
    walk_ident(self, ident);
  }
  fn visit_prop(&mut self, name: &'a str, prop: &'a str) {
    walk_prop(self, name, prop);
  }
  fn visit_unary_op(&mut self, op: &'a UnOp) {
    walk_unary_op(self, op);
  }
  fn visit_store(&mut self, name: &'a str) {
    walk_store(self, name);
  }
  fn visit_load(&mut self, name: &'a str) {
    walk_load(self, name);
  }
}

pub fn walk_expr<'a, V: Visitor<'a>>(visitor: &mut V, expr: &'a Expr) {
  match expr {
    Expr::Ident(ident) => {
      visitor.visit_ident(ident);
    }
    Expr::Binary(op, left, right) => {
      visitor.visit_expr(left);
      visitor.visit_binary_op(op);
      visitor.visit_expr(right);
    }
    Expr::Ternary(op, a, b, c) => {
      visitor.visit_ternary_op(op);
      visitor.visit_expr(a);
      visitor.visit_expr(b);
      visitor.visit_expr(c);
    }
    Expr::IfElse(condition, then_body, else_body) => {
      visitor.visit_expr(condition);
      visitor.visit_expr(then_body);
      visitor.visit_expr(else_body);
    }
    Expr::Prop(name, prop) => {
      visitor.visit_prop(name, prop);
    }
    Expr::Unary(op, operand) => {
      visitor.visit_unary_op(op);
      visitor.visit_expr(operand);
    }
    Expr::Store(name, expr) => {
      visitor.visit_expr(expr);
      visitor.visit_store(name);
    }
    Expr::Load(name) => {
      visitor.visit_load(name);
    }
    Expr::AbsAccess { x, y, .. } => {
      visitor.visit_expr(x);
      visitor.visit_expr(y);
    }
    Expr::Lit(_) | Expr::RelAccess { .. } => {}
  }
}

const fn walk_binary_op<'a, V: Visitor<'a>>(_visitor: &mut V, _op: &'a BinOp) {}
const fn walk_ternary_op<'a, V: Visitor<'a>>(_visitor: &mut V, _op: &'a TernaryOp) {}
const fn walk_ident<'a, V: Visitor<'a>>(_visitor: &mut V, _ident: &'a str) {}
const fn walk_unary_op<'a, V: Visitor<'a>>(_visitor: &mut V, _op: &'a UnOp) {}
const fn walk_prop<'a, V: Visitor<'a>>(_visitor: &mut V, _name: &'a str, _prop: &'a str) {}
const fn walk_store<'a, V: Visitor<'a>>(_visitor: &mut V, _name: &'a str) {}
const fn walk_load<'a, V: Visitor<'a>>(_visitor: &mut V, _name: &'a str) {}

/// A folding visitor that produces a new AST.
pub trait MutVisitor: Sized {
  fn fold_expr(&mut self, expr: &Expr) -> Expr {
    walk_expr_mut(self, expr)
  }
}

pub fn walk_expr_mut<V: MutVisitor>(visitor: &mut V, expr: &Expr) -> Expr {
  match expr {
    Expr::Unary(op, inner) => Expr::Unary(*op, Arc::new(visitor.fold_expr(inner))),
    Expr::Binary(op, lhs, rhs) => Expr::Binary(
      *op,
      Arc::new(visitor.fold_expr(lhs)),
      Arc::new(visitor.fold_expr(rhs)),
    ),
    Expr::Ternary(op, a, b, c) => Expr::Ternary(
      *op,
      Arc::new(visitor.fold_expr(a)),
      Arc::new(visitor.fold_expr(b)),
      Arc::new(visitor.fold_expr(c)),
    ),
    Expr::IfElse(cond, then_br, else_br) => Expr::IfElse(
      Arc::new(visitor.fold_expr(cond)),
      Arc::new(visitor.fold_expr(then_br)),
      Arc::new(visitor.fold_expr(else_br)),
    ),
    Expr::Store(name, val) => Expr::Store(name.clone(), Arc::new(visitor.fold_expr(val))),
    Expr::AbsAccess {
      clip,
      x,
      y,
      boundary_mode,
    } => Expr::AbsAccess {
      clip: clip.clone(),
      x: Arc::new(visitor.fold_expr(x)),
      y: Arc::new(visitor.fold_expr(y)),
      boundary_mode: *boundary_mode,
    },
    Expr::Lit(_) | Expr::Ident(_) | Expr::Prop(..) | Expr::Load(_) | Expr::RelAccess { .. } => {
      expr.clone()
    }
  }
}
