use std::collections::HashMap;
use std::fmt::{self, Write};

use cranelift::codegen::entity::SecondaryMap;
use cranelift::codegen::ir::entities::AnyEntity;
use cranelift::codegen::ir::{Block, Function, Inst, Value};
use cranelift::codegen::write::{FuncWriter, PlainWriter, write_block_header};

/// A [`FuncWriter`] implementation that decorates Cranelift IR with comments.
pub struct CommentWriter {
  /// Comments written at the very top of the function.
  pub global_comments: Vec<String>,

  /// Comments attached to Cranelift entities.
  pub entity_comments: HashMap<AnyEntity, String>,

  /// Comments appended immediately after a specific instruction.
  pub inst_post_comments: HashMap<Inst, String>,
}

impl CommentWriter {
  #[must_use]
  pub fn new() -> Self {
    Self {
      global_comments: Vec::new(),
      entity_comments: HashMap::new(),
      inst_post_comments: HashMap::new(),
    }
  }
}

impl Default for CommentWriter {
  fn default() -> Self {
    Self::new()
  }
}

impl FuncWriter for CommentWriter {
  fn write_preamble(&mut self, w: &mut dyn Write, func: &Function) -> Result<bool, fmt::Error> {
    for comment in &self.global_comments {
      writeln!(w, "    ; {comment}")?;
    }
    self.super_preamble(w, func)
  }

  fn write_entity_definition(
    &mut self,
    w: &mut dyn Write,
    func: &Function,
    entity: AnyEntity,
    value: &dyn fmt::Display,
  ) -> fmt::Result {
    self.super_entity_definition(w, func, entity, value)?;
    if let Some(comment) = self.entity_comments.get(&entity) {
      writeln!(w, "    ; {entity}: {comment}")?;
    }
    Ok(())
  }

  fn write_block_header(
    &mut self,
    w: &mut dyn Write,
    func: &Function,
    block: Block,
    indent: usize,
  ) -> fmt::Result {
    write_block_header(w, func, block, indent)
  }

  fn write_instruction(
    &mut self,
    w: &mut dyn Write,
    func: &Function,
    aliases: &SecondaryMap<Value, Vec<Value>>,
    inst: Inst,
    indent: usize,
  ) -> fmt::Result {
    PlainWriter.write_instruction(w, func, aliases, inst, indent)?;
    if let Some(comment) = self.inst_post_comments.get(&inst) {
      writeln!(w, "{1:0$}; ^ {comment}", indent, "")?;
    }
    Ok(())
  }
}
