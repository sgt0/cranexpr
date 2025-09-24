use cranelift::prelude::*;
use vapours::{
  enums::ColorRange,
  generic::HoldsVideoFormat,
  vs_enums::{GRAY8, GRAY16, GRAYS},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PixelType {
  U8,
  U16,
  F32,
}

impl PixelType {
  pub(crate) const fn bytes(self) -> usize {
    match self {
      Self::U8 => 1,
      Self::U16 => 2,
      Self::F32 => 4,
    }
  }

  pub(crate) fn peak_value(self) -> f32 {
    match self {
      Self::U8 => GRAY8.peak_value(None, Some(ColorRange::Full)),
      Self::U16 => GRAY16.peak_value(None, Some(ColorRange::Full)),
      Self::F32 => GRAYS.peak_value(None, Some(ColorRange::Full)),
    }
  }
}

impl From<PixelType> for types::Type {
  fn from(value: PixelType) -> Self {
    match value {
      PixelType::U8 => types::I8,
      PixelType::U16 => types::I16,
      PixelType::F32 => types::F32,
    }
  }
}
