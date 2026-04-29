mod component_type;
mod errors;
mod expr;

use vapoursynth4_rs::declare_plugin;

use crate::expr::CranexprFilter;

declare_plugin!(
  c"sgt.cranexpr",
  c"cranexpr",
  c"Cranelift Expr",
  (0, 4),
  VAPOURSYNTH_API_VERSION,
  0,
  (CranexprFilter, None)
);
