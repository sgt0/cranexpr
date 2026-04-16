use std::collections::BTreeSet;

use crate::errors::TransformError;
use crate::visit::Visitor;

fn shorthand_to_clip_idx(name: &str) -> Option<usize> {
  if name.len() == 1 {
    let c = name.as_bytes()[0];
    if (b'x'..=b'z').contains(&c) {
      return Some((c - b'x') as usize);
    }
    if (b'a'..=b'w').contains(&c) {
      return Some((c - b'a' + 3) as usize);
    }
  } else if let Some(stripped) = name.strip_prefix("src")
    && let Ok(idx) = stripped.parse::<usize>()
  {
    return Some(idx);
  }
  None
}

pub struct PropVisitor {
  pub props: BTreeSet<(usize, String)>,
  pub error: Option<TransformError>,
  num_inputs: usize,
}

impl PropVisitor {
  #[must_use]
  pub const fn new(num_inputs: usize) -> Self {
    Self {
      props: BTreeSet::new(),
      num_inputs,
      error: None,
    }
  }
}

impl<'a> Visitor<'a> for PropVisitor {
  fn visit_prop(&mut self, name: &'a str, prop: &'a str) {
    if self.error.is_some() {
      return;
    }

    if let Some(idx) = shorthand_to_clip_idx(name)
      && idx < self.num_inputs
    {
      self.props.insert((idx, prop.to_string()));
    } else {
      self.error = Some(TransformError::InvalidClipIdentifier(name.to_string()));
    }
  }
}
