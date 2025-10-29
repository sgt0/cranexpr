//! Lexer for exprs.

mod cursor;

use serde::Serialize;

use crate::lexer::cursor::Cursor;

/// Enum representing common lexeme types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) enum TokenKind {
  /// Any whitespace character sequence.
  Whitespace,

  /// An identifier or keyword.
  Ident,

  /// Literals, e.g. `12u8`, `1.0e-40`.
  ///
  /// See [`LiteralKind`] for more details.
  Literal {
    kind: LiteralKind,
    suffix_start: u32,
  },

  /// `.`
  Dot,
  /// `?`
  Question,
  /// `+`
  Plus,
  /// `*`
  Star,
  /// `/`
  Slash,
  /// `-`
  Minus,
  /// `<`
  Lt,
  /// `>`
  Gt,
  /// `%`
  Percent,
  /// `!`
  Bang,
  /// `@`
  At,

  /// Unknown token, not expected by the lexer.
  Unknown,

  /// End of input.
  Eof,
}

/// Parsed token.
/// It doesn't contain information about data that has been parsed,
/// only the type of the token and its size.
#[derive(Debug, Serialize)]
pub(crate) struct Token {
  pub kind: TokenKind,
  pub len: u32,
}

impl Token {
  const fn new(kind: TokenKind, len: u32) -> Self {
    Self { kind, len }
  }
}

/// Base of numeric literal encoding according to its prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub(crate) enum Base {
  /// Literal starts with "0b".
  Binary = 2,
  /// Literal starts with "0o".
  Octal = 8,
  /// Literal doesn't contain a prefix.
  Decimal = 10,
  /// Literal starts with "0x".
  Hexadecimal = 16,
}

/// Enum representing the literal types supported by the lexer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub(crate) enum LiteralKind {
  /// `12_u8`, `0o100`, `0b120i99`, `1f32`.
  Int { base: Base, empty_int: bool },
  /// `12.34f32`, `1e3`, but not `1f32`.
  Float { base: Base, empty_exponent: bool },
}

/// True if `c` is valid as a first character of an identifier.
const fn is_id_start(c: char) -> bool {
  c.is_ascii_alphabetic() || c == '_'
}

/// True if `c` is valid as a non-first character of an identifier.
const fn is_id_continue(c: char) -> bool {
  c.is_ascii_alphanumeric() || c == '_'
}

/// True if `c` is considered a whitespace character.
pub(crate) const fn is_whitespace(c: char) -> bool {
  matches!(
    c,
    // Usual ASCII suspects.
    '\u{0009}'   // \t
        | '\u{000A}' // \n
        | '\u{000B}' // vertical tab
        | '\u{000C}' // form feed
        | '\u{000D}' // \r
        | '\u{0020}' // space

        // NEXT LINE from latin1.
        | '\u{0085}'

        // Bidi markers.
        | '\u{200E}' // LEFT-TO-RIGHT MARK
        | '\u{200F}' // RIGHT-TO-LEFT MARK

        // Dedicated whitespace characters from Unicode.
        | '\u{2028}' // LINE SEPARATOR
        | '\u{2029}' // PARAGRAPH SEPARATOR
  )
}

impl Cursor<'_> {
  /// Parses a token from the input string.
  pub(crate) fn advance_token(&mut self) -> Token {
    let Some(first_char) = self.bump() else {
      return Token::new(TokenKind::Eof, 0);
    };

    let token_kind = match first_char {
      // Whitespace.
      c if is_whitespace(c) => self.whitespace(),

      // Identifier (this should be checked after other variant that can
      // start as identifier).
      c if is_id_start(c) => self.ident_or_unknown_prefix(),

      // Numeric literal.
      c @ '0'..='9' => {
        let literal_kind = self.number(c);
        let suffix_start = self.pos_within_token();
        self.eat_literal_suffix();
        TokenKind::Literal {
          kind: literal_kind,
          suffix_start,
        }
      }

      // One-symbol tokens.
      '.' => TokenKind::Dot,
      '?' => TokenKind::Question,
      '-' => TokenKind::Minus,
      '+' => TokenKind::Plus,
      '*' => TokenKind::Star,
      '/' => TokenKind::Slash,
      '<' => TokenKind::Lt,
      '>' => TokenKind::Gt,
      '%' => TokenKind::Percent,
      '!' => TokenKind::Bang,
      '@' => TokenKind::At,

      _ => TokenKind::Unknown,
    };

    let res = Token::new(token_kind, self.pos_within_token());
    self.reset_pos_within_token();
    res
  }

  // Eats the identifier. Note: succeeds on `_`, which isn't a valid
  // identifier.
  fn eat_identifier(&mut self) {
    if !is_id_start(self.first()) {
      return;
    }
    self.bump();

    self.eat_while(is_id_continue);
  }

  // Eats the suffix of the literal, e.g. "u8".
  fn eat_literal_suffix(&mut self) {
    self.eat_identifier();
  }

  fn eat_decimal_digits(&mut self) -> bool {
    let mut has_digits = false;
    loop {
      match self.first() {
        '_' => {
          self.bump();
        }
        '0'..='9' => {
          has_digits = true;
          self.bump();
        }
        _ => break,
      }
    }
    has_digits
  }

  fn eat_hexadecimal_digits(&mut self) -> bool {
    let mut has_digits = false;
    loop {
      match self.first() {
        '_' => {
          self.bump();
        }
        '0'..='9' | 'a'..='f' | 'A'..='F' => {
          has_digits = true;
          self.bump();
        }
        _ => break,
      }
    }
    has_digits
  }

  /// Eats the float exponent. Returns true if at least one digit was met,
  /// and returns false otherwise.
  fn eat_float_exponent(&mut self) -> bool {
    if self.first() == '-' || self.first() == '+' {
      self.bump();
    }
    self.eat_decimal_digits()
  }

  fn ident_or_unknown_prefix(&mut self) -> TokenKind {
    // Start is already eaten, eat the rest of identifier.
    self.eat_while(is_id_continue);
    TokenKind::Ident
  }

  fn number(&mut self, first_digit: char) -> LiteralKind {
    let mut base = Base::Decimal;
    if first_digit == '0' {
      // Attempt to parse encoding base.
      match self.first() {
        'b' => {
          base = Base::Binary;
          self.bump();
          if !self.eat_decimal_digits() {
            return LiteralKind::Int {
              base,
              empty_int: true,
            };
          }
        }
        'o' => {
          base = Base::Octal;
          self.bump();
          if !self.eat_decimal_digits() {
            return LiteralKind::Int {
              base,
              empty_int: true,
            };
          }
        }
        'x' => {
          base = Base::Hexadecimal;
          self.bump();
          if !self.eat_hexadecimal_digits() {
            return LiteralKind::Int {
              base,
              empty_int: true,
            };
          }
        }
        // Not a base prefix; consume additional digits.
        '0'..='9' | '_' => {
          self.eat_decimal_digits();
        }

        // Also not a base prefix; nothing more to do here.
        '.' | 'e' | 'E' => {}

        // Just a 0.
        _ => {
          return LiteralKind::Int {
            base,
            empty_int: false,
          };
        }
      }
    } else {
      // No base prefix, parse number in the usual way.
      self.eat_decimal_digits();
    }

    match self.first() {
      // Don't be greedy if this is actually an
      // integer literal followed by field/method access or a range pattern
      // (`0..2` and `12.foo()`)
      '.' if self.second() != '.' && !is_id_start(self.second()) => {
        // might have stuff after the ., and if it does, it needs to start
        // with a number
        self.bump();
        let mut empty_exponent = false;
        if self.first().is_ascii_digit() {
          self.eat_decimal_digits();
          match self.first() {
            'e' | 'E' => {
              self.bump();
              empty_exponent = !self.eat_float_exponent();
            }
            _ => (),
          }
        }
        LiteralKind::Float {
          base,
          empty_exponent,
        }
      }
      'e' | 'E' => {
        self.bump();
        let empty_exponent = !self.eat_float_exponent();
        LiteralKind::Float {
          base,
          empty_exponent,
        }
      }
      _ => LiteralKind::Int {
        base,
        empty_int: false,
      },
    }
  }

  fn whitespace(&mut self) -> TokenKind {
    self.eat_while(is_whitespace);
    TokenKind::Whitespace
  }
}

/// Creates an iterator that produces tokens from the input string.
pub(crate) fn tokenize(input: &str) -> impl Iterator<Item = Token> {
  let mut cursor = Cursor::new(input);
  std::iter::from_fn(move || {
    let token = cursor.advance_token();
    if token.kind == TokenKind::Eof {
      None
    } else {
      Some(token)
    }
  })
}

/// Tokenizes the input while keeping the text associated with each token.
pub(crate) fn tokenize_with_text(s: &str) -> impl Iterator<Item = (TokenKind, &str)> {
  let mut pos = 0;
  tokenize(s).map(move |t| {
    let end = pos + t.len;
    let range = pos as usize..end as usize;
    pos = end;
    (t.kind, s.get(range).unwrap_or_default())
  })
}

#[cfg(test)]
mod tests {
  use insta::assert_yaml_snapshot;
  use rstest::rstest;

  use super::*;

  fn lex(expr: &str) -> Vec<Token> {
    tokenize(expr).collect()
  }

  #[rstest]
  fn test_literals() {
    assert_yaml_snapshot!(lex("1234"));
    assert_yaml_snapshot!(lex("0b101"));
    assert_yaml_snapshot!(lex("0xABC"));
    assert_yaml_snapshot!(lex("1.0"));
    assert_yaml_snapshot!(lex("1.0e10"));
  }

  #[rstest]
  fn test_ternary() {
    assert_yaml_snapshot!(lex("a b c ?"));
  }

  #[rstest]
  fn test_ident() {
    assert_yaml_snapshot!(lex("f abs"));
  }

  #[rstest]
  fn test_binary_ops() {
    assert_yaml_snapshot!(lex("d e <"));
  }

  #[rstest]
  fn test_example() {
    assert_yaml_snapshot!(lex("x y 64 * + z 256 * + 3 /"));
  }

  #[rstest]
  fn test_dot() {
    assert_yaml_snapshot!(tokenize("x.PlaneStatsAverage").collect::<Vec<_>>());
  }

  #[rstest]
  fn test_underscore_prop_name() {
    assert_yaml_snapshot!(tokenize("x._Combed").collect::<Vec<_>>());
  }
}
