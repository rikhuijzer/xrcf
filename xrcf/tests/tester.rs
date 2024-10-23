#![allow(dead_code)]
extern crate xrcf;

use std::cmp::max;
use std::panic::Location;
use std::sync::Arc;
use std::sync::RwLock;
use tracing::info;
use xrcf::convert::RewriteResult;
use xrcf::init_subscriber;
use xrcf::ir::Op;
use xrcf::parser::DefaultParserDispatch;
use xrcf::parser::Parser;
use xrcf::transform;
use xrcf::DefaultTransformDispatch;

pub struct Test;

impl Test {
    /// Initialize the subscriber for the tests.
    ///
    /// Cannot pass options, since the tests run concurrently.
    pub fn init_subscriber() {
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
            let actual_line = actual.lines().nth(i);
            let actual_line = match actual_line {
                None => {
                    panic!(
                        "Expected line {i} not found in output: called from {}",
                        caller
                    );
                }
                Some(actual_line) => actual_line,
            };
            let expected_line = expected.lines().nth(i);
            let expected_line = match expected_line {
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
                if actual_line.contains(expected_line) {
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
        let src = src.trim();
        Self::print_heading("Before parse", src);
        let module = Parser::<DefaultParserDispatch>::parse(&src).unwrap();
        let read_module = module.clone();
        let read_module = read_module.try_read().unwrap();
        let actual = format!("{}", read_module);
        Self::print_heading("After parse", &actual);
        (module, actual)
    }
    pub fn transform(arguments: &str, src: &str) -> (Arc<RwLock<dyn Op>>, String) {
        let src = src.trim();
        let module = Parser::<DefaultParserDispatch>::parse(src).unwrap();
        let msg = format!("Before (transform {arguments})");
        Self::print_heading(&msg, src);

        let result = transform::<DefaultTransformDispatch>(module.clone(), arguments).unwrap();
        let new_root_op = match result {
            RewriteResult::Changed(changed_op) => changed_op.0,
            RewriteResult::Unchanged => {
                panic!("Expected changes");
            }
        };
        let actual = format!("{}", new_root_op.try_read().unwrap());
        let msg = format!("After (transform {arguments})");
        Self::print_heading(&msg, &actual);
        (new_root_op, actual)
    }
}
