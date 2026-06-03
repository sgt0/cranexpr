mod component_type;
mod errors;
mod expr;
mod prop_expr;
mod select;

use vapoursynth4_rs::declare_plugin;

use crate::expr::CranexprFilter;
use crate::prop_expr::PropExprFilter;
use crate::select::SelectFilter;

declare_plugin!(
  c"sgt.cranexpr",
  c"cranexpr",
  c"Cranelift Expr",
  (0, 8),
  VAPOURSYNTH_API_VERSION,
  0,
  (CranexprFilter, None),
  (PropExprFilter, None),
  (SelectFilter, None)
);
