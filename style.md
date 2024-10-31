# RRCF Code Style Guide

Some notes on the code style used in this project.

## Verbosity vs. Complexity

In case of doubt, prefer verbosity over complexity.
For example, prefer some duplication over things like macros, declarative code, or DSLs.
The main aim is to keep things easy to understand; for both humans as well as tooling.

## Declarative Code

In my experience, declarative code is a beautiful idea, but in practice it often is hard to learn and understand.
There is now even data to back this up: in benchmarks, LLMs score higher on imperative code than declarative code.

## `Arc<RwLock<T>>`

This project uses `Arc<RwLock<T>>` for pretty much everything.
The reason is that the intermediate representation is essentially a tree with nodes.
This tree is traversed in multiple directions (up, down, and sideways),
so we need to store pointers all over the place.
The project uses `Arc` instead of `Rc` because at some point the compiler will be multi-threaded.

To simplify the usage, there are some helpers functions available via `GuardedOp` etc.
This reduces
```rust
let x = x.read().unwrap();
let parent = x.parent();
```
to
```rust
let x = x.parent();
```
