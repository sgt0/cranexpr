#![allow(dead_code)]

use std::collections::HashSet;

use crate::parser::{ast::Expr, visit::Visitor};

#[derive(Default)]
pub(crate) struct Globals(HashSet<String>);

impl Globals {
  pub(crate) fn from_expr(expr: &Expr) -> Self {
    let mut collector = Self::default();
    collector.visit_expr(expr);
    collector
  }

  pub(crate) fn iter(&self) -> impl Iterator<Item = &String> {
    self.0.iter()
  }
}

impl Visitor<'_> for Globals {
  fn visit_ident(&mut self, ident: &str) {
    self.0.insert(ident.to_string());
  }

  fn visit_prop(&mut self, name: &str, prop: &str) {
    self.0.insert(format!("{name}.{prop}"));
  }
}

#[cfg(test)]
mod tests {
  use rstest::rstest;

  use crate::parser::parse_expr;

  use super::*;

  #[rstest]
  #[case(
    "x 42 1 1 1 -4 y.PlaneStatsMin y.PlaneStatsAverage 1e-6 - - / y y.PlaneStatsAverage - * exp + / - * z / + a 42 z / + /",
    &["a", "x", "y", "y.PlaneStatsAverage", "y.PlaneStatsMin", "z"]
  )]
  #[case(
    "x._Combed N 42 % 1 + y._Combed 42 0 ? + 0 ?",
    &["N", "x._Combed", "y._Combed"]
  )]
  fn test_ident_collector(#[case] input: &str, #[case] expected: &[&str]) {
    let expr = parse_expr(input).unwrap();

    let mut collector = Globals::default();
    collector.visit_expr(&expr);
    let mut actual = collector.0.iter().collect::<Vec<_>>();

    actual.sort_unstable();
    assert_eq!(actual, expected);
  }
}
