use cranelift::{codegen::ir::immediates::Offset32, prelude::*};

use crate::codegen::compiler::FunctionCx;

#[derive(Copy, Clone, Debug)]
pub(crate) struct Pointer {
  base: Value,
  offset: Offset32,
}

impl Pointer {
  pub(crate) fn new(addr: Value) -> Self {
    Self {
      base: addr,
      offset: Offset32::new(0),
    }
  }

  #[allow(dead_code)]
  pub(crate) fn get_addr(self, fx: &mut FunctionCx<'_, '_>) -> Value {
    let offset: i64 = self.offset.into();
    if offset == 0 {
      self.base
    } else {
      fx.bcx.ins().iadd_imm(self.base, offset)
    }
  }

  pub(crate) fn offset(self, fx: &mut FunctionCx<'_, '_>, extra_offset: Offset32) -> Self {
    self.offset_i64(fx, extra_offset.into())
  }

  pub(crate) fn offset_i64(self, fx: &mut FunctionCx<'_, '_>, extra_offset: i64) -> Self {
    if let Some(new_offset) = self.offset.try_add_i64(extra_offset) {
      Self {
        base: self.base,
        offset: new_offset,
      }
    } else {
      let base_offset: i64 = self.offset.into();
      if let Some(new_offset) = base_offset.checked_add(extra_offset) {
        let base_addr = self.base;
        let addr = fx.bcx.ins().iadd_imm(base_addr, new_offset);
        Self {
          base: addr,
          offset: Offset32::new(0),
        }
      } else {
        panic!(
          "self.offset ({base_offset}) + extra_offset ({extra_offset}) not representable in i64",
        );
      }
    }
  }

  pub(crate) fn offset_value(self, fx: &mut FunctionCx<'_, '_>, extra_offset: Value) -> Self {
    Self {
      base: fx.bcx.ins().iadd(self.base, extra_offset),
      offset: self.offset,
    }
  }

  /// Load from memory at this pointer.
  pub(crate) fn load(self, fx: &mut FunctionCx<'_, '_>, ty: Type, flags: MemFlags) -> Value {
    fx.bcx.ins().load(ty, flags, self.base, self.offset)
  }

  /// Store `value` to memory at this pointer.
  pub(crate) fn store(self, fx: &mut FunctionCx<'_, '_>, value: Value, flags: MemFlags) {
    fx.bcx.ins().store(flags, value, self.base, self.offset);
  }
}
