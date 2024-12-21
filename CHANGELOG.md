# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2024-12-21

### Added

- Add `block.inline_region_before` and a few `IntoIterator` implementations ([#33](https://github.com/rikhuijzer/xrcf/pull/33)).

### Changed

- Switch from the std `RwLock` to `parking_lot::RwLock` ([#39](https://github.com/rikhuijzer/xrcf/pull/39)).
- Remove `Arc` helpers such as `GuardedOp` and `GuardedBlock` ([#38](https://github.com/rikhuijzer/xrcf/pull/38)).
- Switch all `Arc<RwLock<T>>` to `Shared<T>` and add `re` and `wr` methods for read and write access ([#36](https://github.com/rikhuijzer/xrcf/pull/36)).
- Automatically rename blocks to simplify the API ([#34](https://github.com/rikhuijzer/xrcf/pull/34)).
- Automatically rename variables to avoid collisions ([#30](https://github.com/rikhuijzer/xrcf/pull/30)).

## [0.5.0] - 2024-12-07

### Added

- Allow lowering of `scf.if` to `cf` ([#25](https://github.com/rikhuijzer/xrcf/pull/25)).
- Support `scf.if` without `yield` ([#28](https://github.com/rikhuijzer/xrcf/pull/28)).
- Add missing passes to the ArnoldC compiler ([#29](https://github.com/rikhuijzer/xrcf/pull/29)).
- Add `--print-ir-before-all` command line argument ([#31](https://github.com/rikhuijzer/xrcf/pull/31)).

### Changed

- Multiple changes to IR logic ([#25](https://github.com/rikhuijzer/xrcf/pull/25)).
- Make `Tester` more strict ([#27](https://github.com/rikhuijzer/xrcf/pull/27)).

## [0.4.0] - 2024-12-02

### Changed

- Changed many core methods to allow lowering of if-else from LLVM dialect to LLVMIR ([#23](https://github.com/rikhuijzer/xrcf/pull/23)).

[unreleased]: https://github.com/rikhuijzer/xrcf/compare/v0.6.0...main
[0.6.0]: https://github.com/rikhuijzer/xrcf/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/rikhuijzer/xrcf/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/rikhuijzer/xrcf/releases/tag/v0.4.0