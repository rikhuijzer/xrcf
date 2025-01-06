extern crate wea;

use indoc::indoc;
use std::panic::Location;
use wea::transform::WeaParserDispatch;
use xrcf::ir::ModuleOp;
use xrcf::shared::SharedExt;
use xrcf::tester::Tester;

type WeaTester = Tester<WeaParserDispatch>;

fn flags() -> Vec<&'static str> {
    vec!["--convert-wea-to-wat"]
}

#[test]
fn test_plus() {
    Tester::init_tracing();
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

    let (_module, actual) = WeaTester::parse(src);
    WeaTester::check_lines_contain(&actual, src, Location::caller());
    let (module, actual) = WeaTester::transform(flags(), src);
    WeaTester::verify(module.clone());
    let module = module.rd();
    assert!(module.as_any().is::<ModuleOp>());
    WeaTester::check_lines_contain(&actual, expected, Location::caller());
}
