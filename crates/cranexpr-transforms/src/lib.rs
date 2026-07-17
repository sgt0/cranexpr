//! AST traversal patterns, optimizations, and rewrites.

mod errors;
mod ident;
mod lut_visitor;
mod pixel_access_visitor;
mod prop_visitor;
mod simplify;
mod visit;

pub use errors::TransformError;
pub use lut_visitor::LutVisitor;
pub use pixel_access_visitor::PixelAccessVisitor;
pub use prop_visitor::PropVisitor;
pub use simplify::simplify;
pub use visit::{MutVisitor, Visitor, walk_expr, walk_expr_mut};
