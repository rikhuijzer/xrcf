use indoc::indoc;
use std::panic::Location;
use xrcf::shared::SharedExt;
use xrcf::targ3t::wat::ModuleOp;
use xrcf::tester::DefaultTester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-mlir-to-wat"]
}

#[test]
fn test_plus() {
    DefaultTester::init_tracing();
    let src = indoc! {r#"
    module {
        func.func @i32_plus(%arg0: i32, %arg1: i32) -> i32 {
            %0 = arith.addi %arg0, %arg1 : i32
            return %0 : i32
        }
    }
    "#};

    let expected = indoc! {r#"
    (module
        (func (export "i32_plus") (param $a i32) (param $b i32) (result i32)
            local.get $a
            local.get $b
            i32.add))
    "#};

    let (mut store, instance) = DefaultTester::load_wat(expected).unwrap();
    let plus = instance
        .get_func(&mut store, "i32_plus")
        .expect("func not found");
    let plus = plus
        .typed::<(i32, i32), i32>(&store)
        .expect("no typed func");
    let result = plus.call(&mut store, (1, 2)).expect("call failed");
    assert_eq!(result, 3);

    let (module, actual) = DefaultTester::transform(flags(), src);
    DefaultTester::verify(module.clone());
    assert!(module.rd().as_any().is::<ModuleOp>());
    DefaultTester::check_lines_exact(&actual, expected, Location::caller());
}
