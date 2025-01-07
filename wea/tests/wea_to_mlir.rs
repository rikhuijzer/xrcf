extern crate wea;

use indoc::indoc;
use std::panic::Location;
use wea::WeaParserDispatch;
use xrcf::ir::ModuleOp;
use xrcf::shared::SharedExt;
use xrcf::tester::Tester;

type WeaTester = Tester<WeaParserDispatch>;

fn flags() -> Vec<&'static str> {
    vec!["--convert-wea-to-mlir"]
}

// #[test]
fn test_plus() {
    WeaTester::init_tracing();
    let src = indoc! {r#"
    module:
        pub fn plus(a: i32, b: i32) -> i32:
            a + b
    "#};
    let preprocessed = indoc! {r#"
    module:
        pub fn plus(a: i32, b: i32) -> i32:
            a + b
    "#};
    WeaTester::preprocess(src);
    WeaTester::check_lines_exact(src, preprocessed, Location::caller());

    let expected = indoc! {r#"
    module {
        fn plus(%arg0: i32, %arg1: i32) -> i32 {
            %arg0 + %arg1
        }
    }
    "#};

    let (_module, actual) = WeaTester::parse(src);
    WeaTester::check_lines_contain(&actual, src, Location::caller());
    let (module, actual) = WeaTester::transform(flags(), src);
    WeaTester::verify(module.clone());
    let module = module.rd();
    assert!(module.as_any().is::<ModuleOp>());
    WeaTester::check_lines_contain(&actual, expected, Location::caller());
}
