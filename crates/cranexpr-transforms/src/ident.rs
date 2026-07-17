/// Returns whether `name` refers to a clip.
pub(crate) fn is_clip_identifier(name: &str) -> bool {
  matches!(name.as_bytes(), [b'a'..=b'z'])
    || name
      .strip_prefix("src")
      .is_some_and(|s| s.parse::<usize>().is_ok())
}
