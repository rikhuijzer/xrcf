# RRCF Code Style Guide

Some notes on the code style used in this project.

## Verbosity vs. Complexity

In case of doubt, prefer verbosity over complexity.
For example, prefer some duplication over things like macros, declarative code, or DSLs.
The main aim is to keep things easy to understand; for both humans as well as tooling.

For example, in my experience, declarative code is a beautiful idea, but in practice it's often hard to learn and understand.

## `Arc<RwLock<T>>`

This project uses `Arc<RwLock<T>>` for pretty much everything.
The reason is that the intermediate representation (IR) is essentially a tree with nodes.
This tree is traversed in multiple directions (up, down, and sideways), so we need to store pointers all over the place.
Also, this project uses `Arc` instead of `Rc` because at some point the compiler will be multi-threaded.

To simplify the usage, there are some helpers functions available via `GuardedOp`, `GuardedBlock`, etc.

In some cases, this can simplify code such as
```rust
let x = x.read().unwrap();
let parent = x.parent();
```
to
```rust
let x = x.parent();
```
