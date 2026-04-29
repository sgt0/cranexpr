//! AST traversal patterns, optimizations, and rewrites.

mod errors;
mod pixel_access_visitor;
mod prop_visitor;
mod visit;

pub use errors::TransformError;
pub use pixel_access_visitor::PixelAccessVisitor;
pub use prop_visitor::PropVisitor;
pub use visit::{Visitor, walk_expr};
