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
        (func (export "i32_plus") (param $arg0 i32) (param $arg1 i32) (result i32)
            (i32.add (local.get $arg0) (local.get $arg1))
            return
        )
    )
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

#[test]
fn test_arith() {
    DefaultTester::init_tracing();
    let transformations = vec![
        ("arith.subi", "i32.sub"),
        ("arith.addi", "i32.add"),
        ("arith.divsi", "i32.div_s"),
    ];
    for (mlir, wat) in transformations {
        let src = &format!(
            indoc! {r#"
        module {{
            func.func @i32_arith(%arg0: i32, %arg1: i32) -> i32 {{
                %0 = {} %arg0, %arg1 : i32
                return %0 : i32
            }}
        }}
        "#},
            mlir
        );

        let expected = &format!(
            indoc! {r#"
        (module
            (func (export "i32_arith") (param $arg0 i32) (param $arg1 i32) (result i32)
                ({} (local.get $arg0) (local.get $arg1))
                return
            )
        )
        "#},
            wat
        );

        let (module, actual) = DefaultTester::transform(flags(), src);
        DefaultTester::verify(module.clone());
        DefaultTester::check_lines_exact(&actual, expected, Location::caller());
    }
}
