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

/// A convenience trait around [RwLock].
///
/// This trait makes using [RwLock] less verbose. It has two benefits. One is
/// obviously that it saves a lot of typing. Another one is that one-liners are
/// particularly powerful in Rust due to when variables are freed, see
/// <https://xrcf.org/blog/iterators/> for more information.
///
/// Regarding the lost access to error handling (now it just always crashes),
/// that's for now the default behavior until the project starts to support
/// multithreading. Until that feature is introduced, any blocking should crash
/// since it will hang anyway.
///
/// # Example
///
/// With this trait:
/// ```
/// use std::sync::Arc;
/// use std::sync::RwLock;
/// use xrcf::shared::SharedExt;
///
/// let lock = Arc::new(RwLock::new(42));
/// assert_eq!(*lock.rd(), 42);
/// ```
/// and with `Shared`:
/// ```
/// use std::sync::Arc;
/// use std::sync::RwLock;
/// use xrcf::shared::Shared;
/// use xrcf::shared::SharedExt;
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
pub trait SharedExt<T: ?Sized> {
    /// Convenience method for reading.
    fn rd(&self) -> RwLockReadGuard<T>;
    /// Convenience method for writing.
    fn wr(&self) -> RwLockWriteGuard<T>;
}

impl<T: ?Sized> SharedExt<T> for Shared<T> {
    fn rd(&self) -> RwLockReadGuard<T> {
        self.try_read().unwrap()
    }
    fn wr(&self) -> RwLockWriteGuard<T> {
        self.try_write().unwrap()
    }
}

#[test]
/// Just another test that runs even if the docstrings would not.
fn test_shared() {
    let lock = Shared::new(42.into());
    assert_eq!(*lock.rd(), 42);
}
