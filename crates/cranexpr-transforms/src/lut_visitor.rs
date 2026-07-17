use cranexpr_ast::Expr;

use crate::ident::is_clip_identifier;
use crate::visit::{Visitor, walk_expr};

/// Visitor that decides whether an expression can become a lookup table.
///
/// An expression qualifies when it is a pure function of the current pixel of
/// a single clip, so the whole expression can be precomputed per input value.
///
/// Eligible expressions may only read one clip's current pixel (`x`), use
/// literals, arithmetic, and user variables. Anything positional (`X`, `Y`,
/// relative/absolute pixel access), temporal (`N`), per-clip context
/// (`width`, `height`), frame properties, or a second clip disqualifies it.
pub struct LutVisitor<'a> {
  /// The single clip identifier referenced so far, if any.
  pub clip: Option<&'a str>,
  // Whether or not the expression can become a lookup table.
  pub eligible: bool,
}

impl LutVisitor<'_> {
  #[must_use]
  pub const fn new() -> Self {
    Self {
      clip: None,
      eligible: true,
    }
  }
}

impl Default for LutVisitor<'_> {
  fn default() -> Self {
    Self::new()
  }
}

impl<'a> Visitor<'a> for LutVisitor<'a> {
  fn visit_expr(&mut self, expr: &'a Expr) {
    if !self.eligible {
      return;
    }
    match expr {
      Expr::RelAccess { .. } | Expr::AbsAccess { .. } | Expr::Prop(..) => {
        self.eligible = false;
      }
      Expr::Ident(name) => {
        if is_clip_identifier(name) {
          match self.clip {
            None => self.clip = Some(name),
            Some(prev) if prev == name => {}
            Some(_) => self.eligible = false,
          }
        } else {
          // X, Y, N, width, height, etc.
          self.eligible = false;
        }
      }
      _ => walk_expr(self, expr),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn check(expr: &str) -> (bool, Option<String>) {
    let ast = cranexpr_parser::parse_expr(expr).unwrap();
    let mut v = LutVisitor::new();
    for node in &ast {
      v.visit_expr(node);
    }
    (v.eligible, v.clip.map(str::to_owned))
  }

  #[test]
  fn accepts_single_clip_pure_expr() {
    let (eligible, clip) = check("x 32768 / 0.86 pow 65535 *");
    assert!(eligible);
    assert_eq!(clip.as_deref(), Some("x"));
  }

  #[test]
  fn accepts_user_variables() {
    let (eligible, clip) = check("x 2 * v! v@ v@ *");
    assert!(eligible);
    assert_eq!(clip.as_deref(), Some("x"));
  }

  #[test]
  fn rejects_two_clips() {
    assert!(!check("x y +").0);
  }

  #[test]
  fn rejects_positional_and_temporal() {
    assert!(!check("x X +").0);
    assert!(!check("x Y +").0);
    assert!(!check("x N +").0);
    assert!(!check("x width +").0);
  }

  #[test]
  fn rejects_pixel_access() {
    assert!(!check("x[-1,0]").0);
    assert!(!check("0 0 x[]").0);
  }

  #[test]
  fn rejects_frame_props() {
    assert!(!check("x x.PlaneStatsAverage *").0);
  }
}
