extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::ir::ModuleOp;
use xrcf::shared::SharedExt;
use xrcf::tester::Tester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-wea-to-wat"]
}

#[test]
fn test_constant() {
    Tester::init_tracing();
    let src = indoc! {r#"
    module:
        pub fn plus(a: i32, b: i32) -> i32:
            return a + b
    "#};

    let expected = indoc! {r#"
    (module
    (type (func (param i32 i32) (result i32)))
    (func (export "plus") (type 0) (param $a i32) (param $b i32) (result i32)
        local.get $a
        local.get $b
        i32.add))
    "#};
    let caller = Location::caller();
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module.clone());
    let module = module.rd();
    assert!(module.as_any().is::<ModuleOp>());
    Tester::check_lines_contain(&actual, expected, caller);
}