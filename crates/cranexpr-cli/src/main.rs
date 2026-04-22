use std::process::ExitCode;

use clap::Parser;
use cranexpr_ast::BoundaryMode;
use cranexpr_codegen::compile_clif;
use cranexpr_codegen::component_type::ComponentType;
use cranexpr_transforms::{PropVisitor, Visitor};

/// Compile an expr and output Cranelift IR.
#[derive(Parser)]
#[command(version)]
struct Args {
  /// The expression to compile.
  expr: String,
}

fn main() -> ExitCode {
  let args = Args::parse();

  let ast = match cranexpr_parser::parse_expr(&args.expr) {
    Ok(ast) => ast,
    Err(err) => {
      eprintln!("{err}");
      return ExitCode::FAILURE;
    }
  };

  let mut visitor = PropVisitor::new(1);
  for node in &ast {
    visitor.visit_expr(node);
  }
  let required_props: Vec<(usize, String)> = visitor.props.into_iter().collect();

  match compile_clif(
    &ast,
    ComponentType::F32,
    &[ComponentType::F32],
    Some(BoundaryMode::Clamp),
    &required_props,
  ) {
    Ok(clif) => {
      print!("{clif}");
      ExitCode::SUCCESS
    }
    Err(err) => {
      eprintln!("{err}");
      ExitCode::FAILURE
    }
  }
}
