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

  /// Source component type.
  #[arg(long, default_value = "f32")]
  src_type: String,

  /// Destination component type.
  #[arg(long, default_value = "f32")]
  dst_type: String,
}

fn parse_component_type(s: &str) -> Option<ComponentType> {
  match s {
    "f32" => Some(ComponentType::F32),
    "u8" => Some(ComponentType::U8),
    "u16" => Some(ComponentType::U16),
    _ => None,
  }
}

fn main() -> ExitCode {
  let args = Args::parse();

  let Some(src_type) = parse_component_type(&args.src_type) else {
    eprintln!("unknown src component type: {}", args.src_type);
    return ExitCode::FAILURE;
  };
  let Some(dst_type) = parse_component_type(&args.dst_type) else {
    eprintln!("unknown dst component type: {}", args.dst_type);
    return ExitCode::FAILURE;
  };

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
    dst_type,
    &[src_type],
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
