pub mod component_type;
pub mod errors;

mod compiler;
mod pointer;
mod translate;

pub use compiler::compile_jit;

type MainFunc =
  unsafe extern "C" fn(*mut u8, i64, *const *const u8, *const i64, i64, i64, i64, *const f32);

#[derive(Debug)]
pub struct MainFunction {
  ptr: *const u8,
}

impl MainFunction {
  pub(crate) const fn from_ptr(ptr: *const u8) -> Self {
    Self { ptr }
  }

  /// Invokes the compiled function.
  ///
  /// # Safety
  ///
  /// The caller must ensure that `dst`, `srcs`, and `frame_props` are valid
  /// and that the types match those used during compilation.
  #[inline]
  pub unsafe fn invoke<D>(
    &self,
    dst: &mut [D],
    dst_stride: i64,
    srcs: &[*const u8],
    src_strides: &[i64],
    width: i32,
    height: i32,
    n: i32,
    frame_props: &[f32],
  ) {
    debug_assert!(width > 0, "width must be greater than 0");
    debug_assert!(height > 0, "height must be greater than 0");

    let dst_ptr = dst.as_mut_ptr().cast::<u8>();

    let frame_props_ptr = if frame_props.is_empty() {
      std::ptr::null()
    } else {
      frame_props.as_ptr()
    };

    let func = unsafe { std::mem::transmute::<*const u8, MainFunc>(self.ptr) };
    unsafe {
      func(
        dst_ptr,
        dst_stride,
        srcs.as_ptr(),
        src_strides.as_ptr(),
        i64::from(width),
        i64::from(height),
        i64::from(n),
        frame_props_ptr,
      );
    };
  }
}
