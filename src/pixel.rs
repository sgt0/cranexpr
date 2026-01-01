use cranelift::prelude::*;
use vapours::{
  enums::ColorRange,
  generic::HoldsVideoFormat,
  vs_enums::{GRAY8, GRAY16, GRAYS},
};
use vapoursynth4_rs::SampleType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pixel {
  U8,
  U16,
  F32,
}

impl Pixel {
  /// Maps a video format to a [Pixel].
  pub(crate) fn from_video_format<T: HoldsVideoFormat>(format: &T) -> Self {
    match format.sample_type() {
      SampleType::Integer => match format.video_format().bytes_per_sample {
        1 => Self::U8,
        2 => Self::U16,
        _ => unreachable!(),
      },
      SampleType::Float => Self::F32,
    }
  }

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

impl From<Pixel> for types::Type {
  fn from(value: Pixel) -> Self {
    match value {
      Pixel::U8 => types::I8,
      Pixel::U16 => types::I16,
      Pixel::F32 => types::F32,
    }
  }
}

impl From<u8> for Pixel {
  fn from(_: u8) -> Self {
    Self::U8
  }
}

impl From<u16> for Pixel {
  fn from(_: u16) -> Self {
    Self::U16
  }
}

impl From<f32> for Pixel {
  fn from(_: f32) -> Self {
    Self::F32
  }
}
