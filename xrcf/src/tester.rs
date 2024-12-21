use crate::convert::RewriteResult;
use crate::init_subscriber;
use crate::ir::GuardedBlock;
use crate::ir::GuardedOp;
use crate::ir::GuardedOperation;
use crate::ir::Op;
use crate::parser::DefaultParserDispatch;
use crate::parser::Parser;
use crate::shared::SharedExt;
use crate::transform;
use crate::DefaultTransformDispatch;
use crate::Passes;
use crate::TransformOptions;
use std::cmp::max;
use std::panic::Location;
use std::sync::Arc;
use std::sync::RwLock;
use tracing::info;

pub struct Tester;

impl Tester {
    /// Initialize the subscriber for the tests.
    ///
    /// Cannot pass options, since the tests run concurrently.
    pub fn init_tracing() {
        let level = tracing::Level::INFO;
        match init_subscriber(level) {
            Ok(_) => (),
            Err(_e) => (),
        }
    }
    fn point_to_missing_line(expected: &str, index: usize) -> String {
        let mut result = String::new();
        result.push_str("A line is missing from the output:\n");
        result.push_str("```");
        for (i, line) in expected.lines().enumerate() {
            if i == index {
                let msg = format!("{line}   <== missing");
                result.push_str(&format!("\n{msg}"));
            } else {
                result.push_str(&format!("\n{line}"));
            }
        }
        result.push_str("\n```");
        result
    }
    pub fn check_lines_exact(actual: &str, expected: &str, caller: &Location<'_>) {
        let actual = actual.trim();
        let expected = expected.trim();
        let l = max(actual.lines().count(), expected.lines().count());
        for i in 0..l {
            let actual_line = match actual.lines().nth(i) {
                None => {
                    panic!(
                        "Expected line {i} not found in output: called from {}",
                        caller
                    );
                }
                Some(actual_line) => actual_line,
            };
            let expected_line = match expected.lines().nth(i) {
                None => {
                    panic!(
                        "Expected line {i} not found in output: called from {}",
                        caller
                    );
                }
                Some(expected_line) => expected_line,
            };
            assert_eq!(actual_line, expected_line, "called from {}", caller);
        }
    }
    /// Check whether the expected lines are present in the actual output.
    ///
    /// The actual output may contain additional lines that are not in the expected output.
    pub fn check_lines_contain(actual: &str, expected: &str, caller: &Location<'_>) {
        let actual = actual.trim();
        let expected = expected.trim();
        let mut actual_index = 0;
        'outer: for i in 0..expected.lines().count() {
            let expected_line = expected.lines().nth(i).unwrap().trim();
            // If not skipping these, an empty line will match any line (which
            // can then cause the next expected line to be reported as missing).
            if expected_line.is_empty() {
                continue;
            }
            let start = actual_index;
            for j in start..actual.lines().count() {
                let actual_line = actual.lines().nth(j).unwrap();
                if actual_line.trim() == expected_line.trim() {
                    actual_index = j + 1;
                    continue 'outer;
                }
            }
            let msg = Self::point_to_missing_line(expected, i);
            panic!("{msg}\nwhen called from {caller}");
        }
    }
    fn print_heading(msg: &str, src: &str) {
        info!("{msg}:\n```\n{src}\n```\n");
    }
    pub fn parse(src: &str) -> (Arc<RwLock<dyn Op>>, String) {
        Self::print_heading("Before parse", src.trim());
        let module = Parser::<DefaultParserDispatch>::parse(&src).unwrap();
        let actual = format!("{}", module.rd());
        Self::print_heading("After parse", &actual);
        (module, actual)
    }
    pub fn transform(arguments: Vec<&str>, src: &str) -> (Arc<RwLock<dyn Op>>, String) {
        let src = src.trim();
        let module = Parser::<DefaultParserDispatch>::parse(src).unwrap();
        let msg = format!("Before (transform {arguments:?})");
        Self::print_heading(&msg, src);

        for arg in arguments.clone() {
            if arg.starts_with("convert-") {
                panic!("conversion passes should be prefixed with `--convert-`");
            }
        }
        let passes = Passes::from_convert_vec(arguments.clone());
        let options = TransformOptions::from_passes(passes);
        let result = transform::<DefaultTransformDispatch>(module.clone(), &options).unwrap();
        let new_root_op = match result {
            RewriteResult::Changed(changed_op) => changed_op.op,
            RewriteResult::Unchanged => {
                panic!("Expected changes");
            }
        };
        let actual = format!("{}", new_root_op.rd());
        let msg = format!("After (transform {arguments:?})");
        Self::print_heading(&msg, &actual);
        (new_root_op, actual)
    }
    fn verify_core(op: Arc<RwLock<dyn Op>>) {
        let op = op.rd();
        if !op.name().to_string().contains("module") {
            let parent = op.operation().parent();
            assert!(
                op.operation().parent().is_some(),
                "Parent was not set for:\n{}",
                op
            );
            let parent = parent.unwrap();
            let operation = op.operation();
            let operation = operation.rd();
            assert!(parent.index_of(&operation).is_some(),
            "Could not find the following op in parent. Is the parent field pointing to the wrong block?\n{}",
            op);
            for result in operation.results().vec().rd().iter() {
                let value = result.rd();
                assert!(value.typ().is_ok(), "type was not set for {value}");
            }
        }
    }
    /// Run some extra verification on the IR.
    ///
    /// This method can be used to run some extra verification on generated IR.
    /// MLIR runs verification inside the production code, but XRCF has more
    /// flexibility due to Rust's testing framework and thus XRCF aims to run
    /// verification only during testing. If something is wrong in production
    /// code, it means that test coverage was insufficient.
    ///
    /// Essentially, this verification aims to catch problems that are not
    /// visible in the textual representation. For example, whether an op is
    /// added to it's parent is visible or the op wouldn't be printed, but
    /// whether the op has also a pointer to the parent is not visible.
    pub fn verify(op: Arc<RwLock<dyn Op>>) {
        Self::verify_core(op.clone());
        let ops = op.ops();
        for op in ops {
            Self::verify(op);
        }
    }
}
