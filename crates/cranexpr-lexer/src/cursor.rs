use std::str::Chars;

const EOF_CHAR: char = '\0';

/// Peekable iterator over a char sequence.
///
/// Next characters can be peeked via the `first` method, and position can be
/// shifted forward via the `bump` method.
pub(crate) struct Cursor<'a> {
  len_remaining: usize,
  /// Iterator over chars. Slightly faster than a &str.
  chars: Chars<'a>,
}

impl<'a> Cursor<'a> {
  pub(crate) fn new(input: &'a str) -> Self {
    Cursor {
      len_remaining: input.len(),
      chars: input.chars(),
    }
  }

  /// Moves to the next character.
  pub(crate) fn bump(&mut self) -> Option<char> {
    let c = self.chars.next()?;
    Some(c)
  }

  /// Eats symbols while predicate returns true or until the end of file is reached.
  pub(crate) fn eat_while(&mut self, mut predicate: impl FnMut(char) -> bool) {
    while predicate(self.first()) && !self.is_eof() {
      self.bump();
    }
  }

  /// Peeks the next symbol from the input stream without consuming it.
  /// If requested position doesn't exist, `EOF_CHAR` is returned.
  /// However, getting `EOF_CHAR` doesn't always mean actual end of file,
  /// it should be checked with `is_eof` method.
  pub(crate) fn first(&self) -> char {
    self.chars.clone().next().unwrap_or(EOF_CHAR)
  }

  /// Checks if there is nothing more to consume.
  pub(crate) fn is_eof(&self) -> bool {
    self.chars.as_str().is_empty()
  }

  /// Returns amount of already consumed symbols.
  pub(crate) fn pos_within_token(&self) -> u32 {
    (self.len_remaining - self.chars.as_str().len()) as u32
  }

  /// Resets the number of bytes consumed to 0.
  pub(crate) fn reset_pos_within_token(&mut self) {
    self.len_remaining = self.chars.as_str().len();
  }

  /// Peeks the second symbol from the input stream without consuming it.
  pub(crate) fn second(&self) -> char {
    let mut iter = self.chars.clone();
    iter.next();
    iter.next().unwrap_or(EOF_CHAR)
  }
}
