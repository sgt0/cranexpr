use cranelift::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComponentType {
  U8,
  U16,
  F32,
}

impl ComponentType {
  #[must_use]
  pub const fn bytes(self) -> usize {
    match self {
      Self::U8 => 1,
      Self::U16 => 2,
      Self::F32 => 4,
    }
  }

  #[must_use]
  pub const fn peak_value(self) -> f32 {
    match self {
      Self::U8 => 255.0,
      Self::U16 => 65535.0,
      Self::F32 => 1.0,
    }
  }
}

impl From<ComponentType> for types::Type {
  fn from(value: ComponentType) -> Self {
    match value {
      ComponentType::U8 => types::I8,
      ComponentType::U16 => types::I16,
      ComponentType::F32 => types::F32,
    }
  }
}

impl From<u8> for ComponentType {
  fn from(_: u8) -> Self {
    Self::U8
  }
}

impl From<u16> for ComponentType {
  fn from(_: u16) -> Self {
    Self::U16
  }
}

impl From<f32> for ComponentType {
  fn from(_: f32) -> Self {
    Self::F32
  }
}
