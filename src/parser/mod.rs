pub(crate) mod ast;
pub(crate) mod visit;

use crate::{
  BoundaryMode,
  errors::{CranexprError, CranexprResult},
  lexer::{TokenKind, tokenize_with_text},
  parser::ast::{BinOp, Expr, TernaryOp, UnOp},
};

fn add_unary_op(stack: &mut Vec<Expr>, op: UnOp) -> CranexprResult<()> {
  let x = stack.pop().ok_or(CranexprError::StackUnderflow)?;
  stack.push(Expr::Unary(op, Box::new(x)));
  Ok(())
}

fn add_binary_op(stack: &mut Vec<Expr>, op: BinOp) -> CranexprResult<()> {
  let right = stack.pop().ok_or(CranexprError::StackUnderflow)?;
  let left = stack.pop().ok_or(CranexprError::StackUnderflow)?;
  stack.push(Expr::Binary(op, Box::new(left), Box::new(right)));
  Ok(())
}

fn add_ternary_op(stack: &mut Vec<Expr>, op: TernaryOp) -> CranexprResult<()> {
  let c = stack.pop().ok_or(CranexprError::StackUnderflow)?;
  let b = stack.pop().ok_or(CranexprError::StackUnderflow)?;
  let a = stack.pop().ok_or(CranexprError::StackUnderflow)?;
  stack.push(Expr::Ternary(op, Box::new(a), Box::new(b), Box::new(c)));
  Ok(())
}

/// Parses an expression into a sequence of ASTs.
pub(crate) fn parse_expr(expr: &str) -> CranexprResult<Vec<Expr>> {
  let mut stack = Vec::new();

  // We don't filter out whitespace tokens because they're significant when it
  // comes to differentiating between subtraction (`x - y`) and
  // negative literals (`-x`).
  let mut tokens = tokenize_with_text(expr).peekable();

  while let Some((kind, text)) = tokens.next() {
    match kind {
      TokenKind::Whitespace => { /* usually insignificant */ }
      TokenKind::Literal {
        kind: _,
        suffix_start: _,
      } => {
        stack.push(Expr::Lit(text.parse::<f32>().unwrap()));
      }
      TokenKind::Ident => {
        // An identifier followed by an open bracket is relative or absolute pixel access.
        if tokens
          .peek()
          .is_some_and(|(k, _)| *k == TokenKind::OpenBracket)
        {
          tokens.next(); // Consume `[`.

          // Check for empty brackets `[]` which indicates absolute access.
          if tokens
            .peek()
            .is_some_and(|(k, _)| *k == TokenKind::CloseBracket)
          {
            tokens.next(); // Consume `]`.

            // Parse optional boundary mode suffix.
            let boundary_mode = if tokens.peek().is_some_and(|(k, _)| *k == TokenKind::Colon) {
              tokens.next(); // Consume `:`.
              // Skip whitespace.
              while tokens
                .peek()
                .is_some_and(|(k, _)| *k == TokenKind::Whitespace)
              {
                tokens.next();
              }
              let (mode_kind, mode_text) = tokens.next().ok_or(CranexprError::StackUnderflow)?;
              if mode_kind != TokenKind::Ident {
                return Err(CranexprError::StackUnderflow);
              }
              match mode_text {
                "c" => Some(BoundaryMode::Clamp),
                "m" => Some(BoundaryMode::Mirror),
                _ => return Err(CranexprError::StackUnderflow),
              }
            } else {
              Some(BoundaryMode::Clamp) // Default to clamp if no suffix
            };

            let y = stack.pop().ok_or(CranexprError::StackUnderflow)?;
            let x = stack.pop().ok_or(CranexprError::StackUnderflow)?;

            stack.push(Expr::AbsAccess {
              clip: text.to_string(),
              x: Box::new(x),
              y: Box::new(y),
              boundary_mode,
            });
          } else {
            // Relative access parsing
            // Parse relX (must be an integer literal).
            let (rel_x_kind, rel_x_text) = tokens.next().ok_or(CranexprError::StackUnderflow)?;
            let rel_x = match rel_x_kind {
              TokenKind::Literal { .. } => rel_x_text
                .parse::<i32>()
                .map_err(|_| CranexprError::StackUnderflow)?,
              TokenKind::Minus => {
                // Negative integer literal.
                let (lit_kind, lit_text) = tokens.next().ok_or(CranexprError::StackUnderflow)?;
                if !matches!(lit_kind, TokenKind::Literal { .. }) {
                  return Err(CranexprError::StackUnderflow);
                }
                -(lit_text
                  .parse::<i32>()
                  .map_err(|_| CranexprError::StackUnderflow)?)
              }
              _ => return Err(CranexprError::StackUnderflow),
            };

            // Skip whitespace and optional comma.
            while tokens
              .peek()
              .is_some_and(|(k, _)| matches!(k, TokenKind::Whitespace | TokenKind::Comma))
            {
              tokens.next();
            }

            // Parse relY.
            let (rel_y_kind, rel_y_text) = tokens.next().ok_or(CranexprError::StackUnderflow)?;
            let rel_y = match rel_y_kind {
              TokenKind::Literal { .. } => rel_y_text
                .parse::<i32>()
                .map_err(|_| CranexprError::StackUnderflow)?,
              TokenKind::Minus => {
                let (lit_kind, lit_text) = tokens.next().ok_or(CranexprError::StackUnderflow)?;
                if !matches!(lit_kind, TokenKind::Literal { .. }) {
                  return Err(CranexprError::StackUnderflow);
                }
                -(lit_text
                  .parse::<i32>()
                  .map_err(|_| CranexprError::StackUnderflow)?)
              }
              _ => return Err(CranexprError::StackUnderflow),
            };

            // Expect close bracket, but skip whitespace first.
            while tokens
              .peek()
              .is_some_and(|(k, _)| *k == TokenKind::Whitespace)
            {
              tokens.next();
            }
            let (rbracket_kind, _) = tokens.next().ok_or(CranexprError::StackUnderflow)?;
            if rbracket_kind != TokenKind::CloseBracket {
              return Err(CranexprError::StackUnderflow);
            }

            // Parse optional boundary mode suffix.
            let boundary_override = if tokens.peek().is_some_and(|(k, _)| *k == TokenKind::Colon) {
              tokens.next(); // Consume `:`.
              // Skip whitespace.
              while tokens
                .peek()
                .is_some_and(|(k, _)| *k == TokenKind::Whitespace)
              {
                tokens.next();
              }
              let (mode_kind, mode_text) = tokens.next().ok_or(CranexprError::StackUnderflow)?;
              if mode_kind != TokenKind::Ident {
                return Err(CranexprError::StackUnderflow);
              }
              match mode_text {
                "c" => Some(BoundaryMode::Clamp),
                "m" => Some(BoundaryMode::Mirror),
                _ => return Err(CranexprError::StackUnderflow),
              }
            } else {
              None
            };

            stack.push(Expr::RelAccess {
              clip: text.to_string(),
              rel_x,
              rel_y,
              boundary_mode: boundary_override,
            });
          }
        }
        // An identifier followed by a dot is the start of frame property access.
        else if tokens.peek().is_some_and(|(k, _)| *k == TokenKind::Dot) {
          tokens.next(); // Consume dot.
          let (prop_kind, prop_text) = tokens.next().ok_or(CranexprError::MissingPropertyName)?;
          if prop_kind != TokenKind::Ident {
            return Err(CranexprError::PropertyNameNotAnIdentifier);
          }
          stack.push(Expr::Prop(text.to_string(), prop_text.to_string()));
        }
        // Drops the top N values from the stack. `drop` is equivalent to
        // `drop1`.
        else if let Some(steps) = text.strip_prefix("drop") {
          let steps = steps.parse::<usize>().unwrap_or(1);
          let new_len = stack
            .len()
            .checked_sub(steps)
            .expect("attempt to drop out of bounds");
          stack.truncate(new_len);
        }
        // Duplicates the topmost stack value.
        //
        // `dupN` allows a value N steps up in the stack to be
        // duplicated. The top value of the stack has index 0 meaning
        // that `dup` is equivalent to `dup0`.
        else if let Some(steps) = text.strip_prefix("dup") {
          let steps = steps.parse::<usize>().unwrap_or(0);
          let idx = stack
            .len()
            .checked_sub(steps + 1)
            .expect("attempt to dup out of bounds");
          let to_dupe = stack.get(idx).expect("attempt to dup out of bounds");
          stack.push(to_dupe.clone());
        }
        // `swapN` allows a value N steps up in the stack to be swapped.
        // The top value of the stack has index 0 meaning that `swap` is
        // equivalent to `swap1`. This is because `swapN` always swaps
        // with the topmost value at index 0.
        else if let Some(steps) = text.strip_prefix("swap") {
          let steps = steps.parse::<usize>().unwrap_or(1);
          let idx = stack
            .len()
            .checked_sub(steps + 1)
            .expect("attempt to swap out of bounds");
          let len = stack.len();
          stack.swap(idx, len - 1);
        }
        // `var!`: Pops the top value from the stack and stores it in a variable
        // named `var`.
        else if tokens.peek().is_some_and(|(k, _)| *k == TokenKind::Bang) {
          tokens.next(); // Consume `!`.
          let value = stack.pop().ok_or(CranexprError::StackUnderflow)?;
          stack.push(Expr::Store(text.to_string(), Box::new(value)));
        }
        // `var@`: Pushes the value of the variable `var` onto the stack.
        else if tokens.peek().is_some_and(|(k, _)| *k == TokenKind::At) {
          tokens.next(); // Consume `@`.
          stack.push(Expr::Load(text.to_string()));
        } else {
          match text {
            "abs" => add_unary_op(&mut stack, UnOp::Abs),
            "pow" => add_binary_op(&mut stack, BinOp::Pow),
            "exp" => add_unary_op(&mut stack, UnOp::Exp),
            "floor" => add_unary_op(&mut stack, UnOp::Floor),
            "round" => add_unary_op(&mut stack, UnOp::Round),
            "log" => add_unary_op(&mut stack, UnOp::Log),
            "max" => add_binary_op(&mut stack, BinOp::Max),
            "min" => add_binary_op(&mut stack, BinOp::Min),
            "not" => add_unary_op(&mut stack, UnOp::Not),
            "sgn" => add_unary_op(&mut stack, UnOp::Sign),
            "sqrt" => add_unary_op(&mut stack, UnOp::Sqrt),
            "sin" => add_unary_op(&mut stack, UnOp::Sine),
            "tan" => add_unary_op(&mut stack, UnOp::Tangent),
            "cos" => add_unary_op(&mut stack, UnOp::Cosine),
            "clip" | "clamp" => add_ternary_op(&mut stack, TernaryOp::Clip),
            "atan2" => add_binary_op(&mut stack, BinOp::Atan2),
            "bitand" => add_binary_op(&mut stack, BinOp::BitAnd),
            "bitor" => add_binary_op(&mut stack, BinOp::BitOr),
            "bitxor" => add_binary_op(&mut stack, BinOp::BitXor),
            "bitnot" => add_unary_op(&mut stack, UnOp::BitNot),
            _ => {
              stack.push(Expr::Ident(text.to_string()));
              Ok(())
            }
          }?;
        }
      }
      TokenKind::Plus => add_binary_op(&mut stack, BinOp::Add)?,
      TokenKind::Minus => {
        // First check if the next token is a literal.
        if tokens
          .peek()
          .is_some_and(|(token_kind, _)| matches!(token_kind, TokenKind::Literal { .. }))
        {
          let (_, text) = tokens.next().unwrap();
          stack.push(Expr::Lit(0.0 - text.parse::<f32>().unwrap()));
        } else {
          add_binary_op(&mut stack, BinOp::Sub)?;
        }
      }
      TokenKind::Star => add_binary_op(&mut stack, BinOp::Mul)?,
      TokenKind::Slash => add_binary_op(&mut stack, BinOp::Div)?,
      TokenKind::Gt => add_binary_op(&mut stack, BinOp::Gt)?,
      TokenKind::Lt => add_binary_op(&mut stack, BinOp::Lt)?,
      TokenKind::Eq => add_binary_op(&mut stack, BinOp::Eq)?,
      TokenKind::Percent => add_binary_op(&mut stack, BinOp::Rem)?,
      TokenKind::Question => {
        // Ternary follows the pattern `A B C ?`, which evaluates to `B` if
        // `A > 0`, and `C` otherwise.
        let no = stack.pop().ok_or(CranexprError::StackUnderflow)?;
        let yes = stack.pop().ok_or(CranexprError::StackUnderflow)?;
        let cond = stack.pop().ok_or(CranexprError::StackUnderflow)?;

        stack.push(Expr::IfElse(Box::new(cond), Box::new(yes), Box::new(no)));
      }
      _ => return Err(CranexprError::UnrecognizedToken(text.to_string())),
    }
  }

  if let Some((last_expr, preceding_exprs)) = stack.split_last() {
    // All expressions before the final one must be side effects.
    for expr in preceding_exprs {
      if !matches!(expr, Expr::Store(..)) {
        return Err(CranexprError::ExpressionDoesNotEvaluateToSingleValue);
      }
    }

    // Last expression must not be a side effect.
    if matches!(last_expr, Expr::Store(..)) {
      return Err(CranexprError::ExpressionEvaluatesToNothing);
    }
  } else {
    return Err(CranexprError::ExpressionEvaluatesToNothing);
  }

  Ok(stack)
}

#[cfg(test)]
mod tests {
  use insta::assert_yaml_snapshot;
  use rstest::rstest;

  use super::*;

  #[rstest]
  fn test_literals() {
    assert_yaml_snapshot!(parse_expr("1234").unwrap());
  }

  #[rstest]
  fn test_example() {
    assert_yaml_snapshot!(parse_expr("x y 64 * + z 256 * + 3 /").unwrap());
  }

  #[rstest]
  fn test_dup() {
    assert_yaml_snapshot!(parse_expr("x dup *").unwrap());
    assert_yaml_snapshot!(parse_expr("x x y - dup 1.0 * dup1 0.0 * ? -").unwrap());
  }

  #[rstest]
  fn test_swap() {
    assert_yaml_snapshot!(parse_expr("3.14 dup 0 > swap 0 < -").unwrap());
  }

  #[rstest]
  fn test_prop() {
    assert_yaml_snapshot!(parse_expr("x.PlaneStatsAverage").unwrap());
  }

  #[rstest]
  fn test_exponent() {
    assert_yaml_snapshot!(parse_expr("x 1e-06 + y.PlaneStatsAverage *").unwrap());
  }

  #[rstest]
  fn test_negative_literal() {
    assert_yaml_snapshot!(parse_expr("-4 -5 - 6 -").unwrap());
  }

  #[rstest]
  fn test_clip() {
    assert_yaml_snapshot!(parse_expr("x 16 235 clip").unwrap());
  }

  #[rstest]
  fn test_atan2() {
    assert_yaml_snapshot!(parse_expr("y x atan2").unwrap());
  }

  #[rstest]
  fn test_eq_op() {
    assert_yaml_snapshot!(parse_expr("x y =").unwrap());
  }

  #[rstest]
  fn test_variables() {
    assert_yaml_snapshot!(parse_expr("x 2 / my_var! my_var@ my_var@ *").unwrap());
  }
}
