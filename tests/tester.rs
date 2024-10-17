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
use xrcf::opt;
use xrcf::parser::BuiltinParse;
use xrcf::OptOptions;
use xrcf::Parser;

pub struct Test;

impl Test {
    pub fn init_subscriber(level: tracing::Level) {
        match init_subscriber(level) {
            Ok(_) => (),
            Err(_e) => (),
        }
    }
    fn point_missing_line(expected: &str, index: usize) -> String {
        let mut result = String::new();
        result.push_str("Line is missing from output:\n");
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
    /// Check whether the expected lines are present in the actual output.
    /// The expected lines are required to be present in the actual output,
    /// but the actual output may contain additional lines that are not in the expected output.
    fn check_lines(actual: &str, expected: &str, caller: &Location<'_>) {
        let mut actual_index = 0;
        'outer: for i in 0..expected.lines().count() {
            let expected_line = expected.lines().nth(i).unwrap();
            let start = actual_index;
            for j in start..actual.lines().count() {
                let actual_line = actual.lines().nth(j).unwrap();
                if actual_line == expected_line {
                    actual_index = j + 1;
                    continue 'outer;
                }
            }
            let msg = Self::point_missing_line(expected, i);
            panic!("{msg}\nwhen called from {caller}");
        }
    }
    fn print_heading(msg: &str, src: &str) {
        info!("{msg}:\n```\n{src}\n```\n");
    }
    pub fn parse(src: &str, expected: &str, caller: &Location<'_>) -> Arc<RwLock<dyn Op>> {
        Self::print_heading("Before parse", src);
        let module = Parser::<BuiltinParse>::parse(&src).unwrap();
        let read_module = module.clone();
        let read_module = read_module.try_read().unwrap();
        let actual = format!("{}", read_module);
        Self::print_heading("After parse", &actual);
        Self::check_lines(&actual, expected, caller);
        module
    }
    pub fn opt(
        flags: &str,
        src: &str,
        expected: &str,
        caller: &Location<'_>,
    ) -> Arc<RwLock<dyn Op>> {
        let options = OptOptions::from_str(flags).unwrap();
        let module = Parser::<BuiltinParse>::parse(src).unwrap();
        let msg = format!("Before (opt {flags})");
        Self::print_heading(&msg, src);

        let result = opt(module.clone(), options).unwrap();
        let new_root_op = match result {
            RewriteResult::Changed(changed_op) => changed_op.0,
            RewriteResult::Unchanged => {
                panic!("Expected changes");
            }
        };
        let actual = format!("{}", new_root_op.try_read().unwrap());
        let msg = format!("After (opt {flags})");
        Self::print_heading(&msg, &actual);
        Self::check_lines(&actual, expected, caller);
        new_root_op
    }
}
