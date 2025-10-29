#![allow(dead_code)]

use crate::parser::ast::{BinOp, Expr, TernaryOp, UnOp};

/// A trait for AST visitors. Visits all nodes in the AST recursively.
pub(crate) trait Visitor<'a>: Sized {
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

fn walk_expr<'a, V: Visitor<'a>>(visitor: &mut V, expr: &'a Expr) {
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
    Expr::Lit(_) => {}
    Expr::Store(name, expr) => {
      visitor.visit_expr(expr);
      visitor.visit_store(name);
    }
    Expr::Load(name) => {
      visitor.visit_load(name);
    }
  }
}

const fn walk_binary_op<'a, V: Visitor<'a>>(_visitor: &mut V, _op: &'a BinOp) {}
const fn walk_ternary_op<'a, V: Visitor<'a>>(_visitor: &mut V, _op: &'a TernaryOp) {}
const fn walk_ident<'a, V: Visitor<'a>>(_visitor: &mut V, _ident: &'a str) {}
const fn walk_unary_op<'a, V: Visitor<'a>>(_visitor: &mut V, _op: &'a UnOp) {}
const fn walk_prop<'a, V: Visitor<'a>>(_visitor: &mut V, _name: &'a str, _prop: &'a str) {}
const fn walk_store<'a, V: Visitor<'a>>(_visitor: &mut V, _name: &'a str) {}
const fn walk_load<'a, V: Visitor<'a>>(_visitor: &mut V, _name: &'a str) {}
