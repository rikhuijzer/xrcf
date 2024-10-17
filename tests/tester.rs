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
    fn compare_lines(actual: &str, expected: &str, caller: &Location<'_>) {
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
    fn print_heading(msg: &str, src: &str) {
        info!("{msg}:\n```\n{src}\n```\n");
    }
    pub fn parse(src: &str, expected: &str, caller: &Location<'_>) -> Arc<RwLock<dyn Op>> {
        Self::print_heading("Before parse", src);
        let module = Parser::<BuiltinParse>::parse(src).unwrap();
        let read_module = module.clone();
        let read_module = read_module.try_read().unwrap();
        let actual = format!("{}", read_module);
        Self::print_heading("After parse", &actual);
        Self::compare_lines(&actual, expected, caller);
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
        Self::print_heading(&format!("Before (opt {flags})"), src);

        let result = opt(module.clone(), options).unwrap();
        let new_root_op = match result {
            RewriteResult::Changed(changed_op) => changed_op.0,
            RewriteResult::Unchanged => {
                panic!("Expected changes");
            }
        };
        let actual = format!("{}", new_root_op.try_read().unwrap());
        Self::print_heading(&format!("After (opt {flags})"), &actual);
        Self::compare_lines(&actual, expected, caller);
        new_root_op
    }
}
