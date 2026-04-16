pub(crate) use cranexpr_codegen::component_type::ComponentType;

use vapours::generic::HoldsVideoFormat;
use vapoursynth4_rs::SampleType;

pub(crate) trait FromVideoFormat {
  fn from_video_format<T: HoldsVideoFormat>(format: &T) -> Self;
}

impl FromVideoFormat for ComponentType {
  /// Maps a video format to a [`ComponentType`].
  fn from_video_format<T: HoldsVideoFormat>(format: &T) -> Self {
    match format.sample_type() {
      SampleType::Integer => match format.video_format().bytes_per_sample {
        1 => Self::U8,
        2 => Self::U16,
        _ => unreachable!(),
      },
      SampleType::Float => Self::F32,
    }
  }
}
