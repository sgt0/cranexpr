use cranexpr_ast::Expr;

use crate::errors::TransformError;
use crate::visit::{Visitor, walk_expr};

fn is_clip_identifier(name: &str) -> bool {
  matches!(name.as_bytes(), [b'a'..=b'z'])
    || name
      .strip_prefix("src")
      .is_some_and(|s| s.parse::<usize>().is_ok())
}

fn is_pixel_context_identifier(name: &str) -> bool {
  matches!(name, "X" | "Y")
}

/// Visitor that rejects any expression node that would require access to
/// pixel data or per-pixel context.
pub struct PixelAccessVisitor {
  pub error: Option<TransformError>,
}

impl PixelAccessVisitor {
  #[must_use]
  pub const fn new() -> Self {
    Self { error: None }
  }
}

impl Default for PixelAccessVisitor {
  fn default() -> Self {
    Self::new()
  }
}

impl<'a> Visitor<'a> for PixelAccessVisitor {
  fn visit_expr(&mut self, expr: &'a Expr) {
    if self.error.is_some() {
      return;
    }

    match expr {
      Expr::RelAccess { clip, .. } => {
        self.error = Some(TransformError::PixelAccessNotAllowed(format!(
          "{clip}[...]"
        )));
      }
      Expr::AbsAccess { clip, .. } => {
        self.error = Some(TransformError::PixelAccessNotAllowed(format!("{clip}[]")));
      }
      Expr::Ident(name) | Expr::Load(name) => {
        if is_clip_identifier(name) || is_pixel_context_identifier(name) {
          self.error = Some(TransformError::PixelAccessNotAllowed(name.clone()));
          return;
        }
        walk_expr(self, expr);
      }
      _ => walk_expr(self, expr),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_rejects_clip_shorthand() {
    let ast = Expr::Ident("x".to_string());
    let mut v = PixelAccessVisitor::new();
    v.visit_expr(&ast);
    assert!(v.error.is_some());

    let ast = Expr::Ident("src0".to_string());
    let mut v = PixelAccessVisitor::new();
    v.visit_expr(&ast);
    assert!(v.error.is_some());

    let ast = Expr::Ident("a".to_string());
    let mut v = PixelAccessVisitor::new();
    v.visit_expr(&ast);
    assert!(v.error.is_some());
  }

  #[test]
  fn test_rejects_pixel_context_identifiers() {
    for name in ["X", "Y"] {
      let ast = Expr::Ident(name.to_string());
      let mut v = PixelAccessVisitor::new();
      v.visit_expr(&ast);
      assert!(v.error.is_some(), "expected error for {name}");
    }
  }

  #[test]
  fn test_rejects_rel_access() {
    let ast = Expr::RelAccess {
      clip: "x".to_string(),
      rel_x: -1,
      rel_y: 0,
      boundary_mode: None,
    };
    let mut v = PixelAccessVisitor::new();
    v.visit_expr(&ast);
    assert!(v.error.is_some());
  }

  #[test]
  fn test_rejects_abs_access() {
    let ast = Expr::AbsAccess {
      clip: "x".to_string(),
      x: Box::new(Expr::Lit(0.0)),
      y: Box::new(Expr::Lit(0.0)),
      boundary_mode: None,
    };
    let mut v = PixelAccessVisitor::new();
    v.visit_expr(&ast);
    assert!(v.error.is_some());
  }

  #[test]
  fn test_allows_frame_prop_access() {
    let ast = Expr::Prop("x".to_string(), "PlaneStatsAverage".to_string());
    let mut v = PixelAccessVisitor::new();
    v.visit_expr(&ast);
    assert!(v.error.is_none());
  }

  #[test]
  fn test_allows_n_and_pi_and_literals() {
    for name in ["N", "pi", "width", "height"] {
      let ast = Expr::Ident(name.to_string());
      let mut v = PixelAccessVisitor::new();
      v.visit_expr(&ast);
      assert!(v.error.is_none(), "unexpected error for {name}");
    }

    let ast = Expr::Lit(1.5);
    let mut v = PixelAccessVisitor::new();
    v.visit_expr(&ast);
    assert!(v.error.is_none());
  }

  #[test]
  fn test_recurses_into_children() {
    let ast = Expr::Binary(
      cranexpr_ast::BinOp::Add,
      Box::new(Expr::Lit(1.0)),
      Box::new(Expr::Ident("x".to_string())),
    );
    let mut v = PixelAccessVisitor::new();
    v.visit_expr(&ast);
    assert!(v.error.is_some());
  }
}
