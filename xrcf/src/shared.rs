use std::ops::Deref;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;

struct SharedGuard<T> {
    inner: Arc<RwLock<T>>,
}

impl<T> SharedGuard<T> {
    fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(RwLock::new(inner)),
        }
    }
    fn read(&self) -> RwLockReadGuard<T> {
        self.inner.try_read().unwrap()
    }
    fn write(&self) -> RwLockWriteGuard<T> {
        self.inner.try_write().unwrap()
    }
    fn try_read(&self) -> RwLockReadGuard<T> {
        self.inner.try_read().unwrap()
    }
    fn try_write(&self) -> RwLockWriteGuard<T> {
        self.inner.try_write().unwrap()
    }
}

pub struct ReadRef<'a, T> {
    guard: RwLockReadGuard<'a, T>,
}

impl<'a, T> Deref for ReadRef<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

/// A shared object that can be read and written to.
///
/// Compared to `Arc<RwLock<T>>`, this API slightly less verbose.
///
/// # Example
///
/// ```
/// use xrcf::ir::Region;
/// use xrcf::shared::Shared;
///
/// let region = Region::default();
/// let _ = region.add_empty_block();
/// let shared = Shared::new(region);
/// assert_eq!(shared.read().blocks().into_iter().len(), 1);
/// ```
/// And the same now using `Arc<RwLock<T>>`:
/// ```
/// use std::sync::{Arc, RwLock};
/// use xrcf::ir::Region;
///
/// let region = Region::default();
/// let _unset_block = region.add_empty_block();
/// let shared = Arc::new(RwLock::new(region));
/// assert_eq!(shared.try_read().unwrap().blocks().into_iter().len(), 1);
/// ```
pub struct Shared<T> {
    inner: SharedGuard<T>,
}

impl<T> Shared<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: SharedGuard::new(inner),
        }
    }
    pub fn read(&self) -> ReadRef<T> {
        ReadRef {
            guard: self.inner.read(),
        }
    }
    pub fn write(&self) -> RwLockWriteGuard<T> {
        self.inner.write()
    }
    pub fn try_read(&self) -> ReadRef<T> {
        ReadRef {
            guard: self.inner.try_read(),
        }
    }
    pub fn try_write(&self) -> RwLockWriteGuard<T> {
        self.inner.try_write()
    }
}
