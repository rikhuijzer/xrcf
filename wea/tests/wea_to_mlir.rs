extern crate wea;

use indoc::indoc;
use std::panic::Location;
use wea::WeaParserDispatch;
use wea::WeaTransformDispatch;
use xrcf::ir::ModuleOp;
use xrcf::shared::SharedExt;
use xrcf::tester::Tester;

type WeaTester = Tester<WeaParserDispatch, WeaTransformDispatch>;

fn flags() -> Vec<&'static str> {
    vec!["--convert-wea-to-mlir"]
}

#[test]
fn preprocess() {
    WeaTester::init_tracing();
    let src = indoc! {r#"
    pub fn plus(a: i32, b: i32) -> i32:
        a + b
    "#};
    let expected = indoc! {r#"
    pub fn plus(a: i32, b: i32) -> i32 {
        a + b
    }
    "#};
    let actual = WeaTester::preprocess(src);
    WeaTester::check_lines_exact(&actual, expected, Location::caller());
}

#[test]
fn test_plus() {
    WeaTester::init_tracing();
    let src = indoc! {r#"
    pub fn plus(a: i32, b: i32) -> i32:
        a + b
    "#};
    let parsed = indoc! {r#"
    module {
      pub fn plus(arg0: i32, arg1: i32) -> i32 {
        arg0 + arg1
      }
    }
    "#};
    let (_module, actual) = WeaTester::parse(src);
    WeaTester::check_lines_exact(&actual, parsed, Location::caller());

    let expected = indoc! {r#"
    module {
      func.func @plus(%arg0: i32, %arg1: i32) -> i32 {
          arith.addi %arg0, %arg1 : i32
          return %0 : i32
      }
    }
    "#};
    let (module, actual) = WeaTester::transform(flags(), src);
    WeaTester::verify(module.clone());
    let module = module.rd();
    assert!(module.as_any().is::<ModuleOp>());
    WeaTester::check_lines_contain(&actual, expected, Location::caller());
}
