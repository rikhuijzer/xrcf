// Dialects can define new operations, attributes, and types.
// Each dialect is given an unique namespace that is prefixed.
//
// Dialects can co-exist and can be produced and consumed by different passes.
#![allow(dead_code)]
struct Dialect {
    name: String,
    description: String,
}

fn main() {
    println!("Hello, world!");
}
