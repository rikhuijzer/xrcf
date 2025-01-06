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
            a + b
    "#};

    let expected = indoc! {r#"
    (module
        (func (export "plus") (param $a i32) (param $b i32) (result i32)
            local.get $a
            local.get $b
            i32.add))
    "#};

    let (mut store, instance) = Tester::load_wat(expected).unwrap();
    let plus = instance
        .get_func(&mut store, "plus")
        .expect("func not found");
    let plus = plus
        .typed::<(i32, i32), i32>(&store)
        .expect("no typed func");
    let result = plus.call(&mut store, (1, 2)).expect("call failed");
    assert_eq!(result, 3);

    let (_module, actual) = Tester::parse(src);
    Tester::check_lines_contain(&actual, src, Location::caller());
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module.clone());
    let module = module.rd();
    assert!(module.as_any().is::<ModuleOp>());
    Tester::check_lines_contain(&actual, expected, Location::caller());
}
