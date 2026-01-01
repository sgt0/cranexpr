pub mod compiler;
pub mod globals;
pub mod pointer;
pub mod translate;

type MainFunc = unsafe extern "C" fn(*mut u8, i64, *const *const u8, i64, i64, i64, i64);

#[derive(Debug)]
pub(crate) struct MainFunction {
  ptr: *const u8,
}

impl MainFunction {
  pub(crate) const fn from_ptr(ptr: *const u8) -> Self {
    Self { ptr }
  }

  #[inline]
  pub(crate) unsafe fn invoke<D, S, I>(
    &self,
    dst: &mut [D],
    srcs: I,
    width: i32,
    height: i32,
    n: i32,
  ) where
    S: AsRef<[u8]>,
    I: IntoIterator<Item = S>,
    I::IntoIter: ExactSizeIterator,
  {
    debug_assert!(width > 0, "width must be greater than 0");
    debug_assert!(height > 0, "height must be greater than 0");

    let dst_ptr = dst.as_mut_ptr().cast::<u8>();
    let dst_len = dst.len() as i64;

    let srcs_iter = srcs.into_iter();
    let srcs_len = srcs_iter.len() as i64;
    let srcs_ptrs: Vec<*const u8> = srcs_iter.map(|s| s.as_ref().as_ptr()).collect();
    let srcs_ptr = srcs_ptrs.as_ptr();

    let func = unsafe { std::mem::transmute::<*const u8, MainFunc>(self.ptr) };
    unsafe {
      func(
        dst_ptr,
        dst_len,
        srcs_ptr.cast(),
        srcs_len,
        i64::from(width),
        i64::from(height),
        i64::from(n),
      );
    };
  }
}
