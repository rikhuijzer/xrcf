use std::sync::Arc;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;

/// A convenience type alias for [Arc<RwLock<T>>].
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use std::sync::RwLock;
/// use xrcf::shared::Shared;
///
/// let lock = Shared::new(42.into());
/// assert_eq!(*lock.try_read().unwrap(), 42);
/// ```
/// can also be used together with [SharedExt].
pub type Shared<T> = Arc<RwLock<T>>;

/// A trait for creating new Arc<RwLock<T>> values
pub trait SharedExt<T> {
    fn new(value: T) -> Self;
}

impl<T> SharedExt<T> for Shared<T> {
    fn new(value: T) -> Self {
        Arc::new(RwLock::new(value))
    }
}

#[test]
fn test_shared() {
    let lock = Shared::new(42.into());
    assert_eq!(*lock.try_read().unwrap(), 42);
}

/// A convenience trait for the `RwLock` API.
///
/// This trait makes using [RwLock] less verbose. Is this trait pretty? No. But
/// in the end it's just a convenience trait that could easily be removed later.
/// It does add some cognitive load for the read, but compared to
/// `try_read().unwrap()` it saves a **lot** of typing.
///
/// # Example
///
/// With this trait:
/// ```
/// use std::sync::Arc;
/// use std::sync::RwLock;
/// use xrcf::shared::RwLockExt;
///
/// let lock = Arc::new(RwLock::new(42));
/// assert_eq!(*lock.rd(), 42);
/// ```
/// and with `Shared`:
/// ```
/// use std::sync::Arc;
/// use std::sync::RwLock;
/// use xrcf::shared::RwLockExt;
/// use xrcf::shared::Shared;
///
/// let lock = Shared::new(42.into());
/// assert_eq!(*lock.rd(), 42);
/// ```
/// Without this trait:
/// ```
/// use std::sync::Arc;
/// use std::sync::RwLock;
///
/// let lock = Arc::new(RwLock::new(42));
/// assert_eq!(*lock.try_read().unwrap(), 42);
/// ```
pub trait RwLockExt<T> {
    /// Convenience method for reading.
    fn rd(&self) -> RwLockReadGuard<T>;
    /// Convenience method for writing.
    fn wr(&self) -> RwLockWriteGuard<T>;
}

impl<T> RwLockExt<T> for Arc<RwLock<T>> {
    fn rd(&self) -> RwLockReadGuard<T> {
        self.try_read().unwrap()
    }
    fn wr(&self) -> RwLockWriteGuard<T> {
        self.try_write().unwrap()
    }
}
