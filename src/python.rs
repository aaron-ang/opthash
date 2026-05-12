#![cfg(feature = "python")]
#![allow(clippy::ptr_as_ptr, clippy::borrow_as_ptr, clippy::ref_as_ptr)]

use std::hash::{Hash, Hasher};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use pyo3::Borrowed;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyString, PyTuple, PyType};

use crate::funnel::MAX_FUNNEL_RESERVE_FRACTION;
use crate::{ElasticHashMap, ElasticOptions, FunnelHashMap, FunnelOptions};

fn build_elastic_options(
    capacity: usize,
    reserve_fraction: Option<f64>,
    probe_scale: Option<f64>,
) -> PyResult<ElasticOptions> {
    let mut opts = ElasticOptions::with_capacity(capacity);
    if let Some(rf) = reserve_fraction {
        if !(rf > 0.0 && rf < 1.0) {
            return Err(PyValueError::new_err(
                "reserve_fraction must be in the open interval (0, 1)",
            ));
        }
        opts = opts.reserve_fraction(rf);
    }
    if let Some(ps) = probe_scale {
        if ps <= 0.0 {
            return Err(PyValueError::new_err("probe_scale must be positive"));
        }
        opts = opts.probe_scale(ps);
    }
    Ok(opts)
}

fn build_funnel_options(
    capacity: usize,
    reserve_fraction: Option<f64>,
    primary_probe_limit: Option<usize>,
) -> PyResult<FunnelOptions> {
    let mut opts = FunnelOptions::with_capacity(capacity);
    if let Some(rf) = reserve_fraction {
        if !(rf > 0.0 && rf <= MAX_FUNNEL_RESERVE_FRACTION) {
            return Err(PyValueError::new_err(format!(
                "reserve_fraction must be in (0, {MAX_FUNNEL_RESERVE_FRACTION}]; \
                 FunnelHashMap caps the load factor at 1/8 by design"
            )));
        }
        opts = opts.reserve_fraction(rf);
    }
    if let Some(limit) = primary_probe_limit {
        if limit == 0 {
            return Err(PyValueError::new_err(
                "primary_probe_limit must be positive",
            ));
        }
        opts = opts.primary_probe_limit(limit);
    }
    Ok(opts)
}

/// Tag for the cached object's Python type, used to short-circuit `__eq__`
/// dispatch in `HashedAny::eq` for the str-vs-str common case.
///
/// Encoded as the low 3 bits of `HashedAny::tagged`. `CPython` objects are
/// at least 8-byte aligned (`PyObject` starts with `Py_ssize_t ob_refcnt`),
/// so the low 3 bits of a real `PyObject*` are always zero and free for us.
#[derive(Clone, Copy, PartialEq)]
#[repr(usize)]
enum HashKind {
    Other = 0,
    Str = 1,
}

/// Mask covering the bits where `HashKind` lives in `HashedAny::tagged`.
const KIND_MASK: usize = 0b111;

/// Owning hashable wrapper around a `Py<PyAny>` used as a map key.
///
/// Caches the hash (computed once via Python `__hash__`) so `Hash` becomes a
/// `write_isize` instead of a Python call. The `HashKind` tag is packed into
/// the low bits of the object pointer so `PartialEq` can take the str-bytes
/// fast path without re-detecting the type — and the struct fits in 16 bytes
/// rather than 24 (8B pointer + 8B hash, no separate kind byte + padding).
struct HashedAny {
    /// Object pointer with `HashKind` tag in the low 3 bits.
    ///
    /// Holds one owned reference to the underlying `PyObject` — `Drop` calls
    /// `Py_DECREF` on the masked pointer. The tagged value is non-null
    /// because a valid `PyObject*` is non-null and the tag bits are also
    /// permitted to be zero (`HashKind::Other = 0`).
    tagged: NonNull<ffi::PyObject>,
    /// Cached `__hash__` result.
    hash: isize,
}

// Safety: `HashedAny` owns a refcount on the underlying `PyObject`, mirroring
// `Py<PyAny>`. The pointer is only dereferenced under the GIL.
unsafe impl Send for HashedAny {}
unsafe impl Sync for HashedAny {}

// Compile-time check that the tag-packing trick achieves the 16-byte goal:
// {NonNull<PyObject> = 8B} + {isize = 8B} with no padding. The whole point
// of packing `HashKind` into the pointer's low bits is this size reduction
// (from 24B with a separate `kind` byte + 7B padding).
const _: () = assert!(std::mem::size_of::<HashedAny>() == 16);

impl HashedAny {
    /// Pack a raw `PyObject*` together with its `HashKind` into a tagged
    /// `NonNull`. The caller is responsible for the refcount being correct —
    /// this routine performs no `Py_INCREF` / `Py_DECREF`.
    ///
    /// # Safety
    /// `obj` must be non-null, and its low 3 bits must be zero (which `CPython`
    /// guarantees for any real `PyObject*`).
    #[inline]
    unsafe fn pack(obj: *mut ffi::PyObject, kind: HashKind) -> NonNull<ffi::PyObject> {
        debug_assert!(!obj.is_null());
        debug_assert_eq!(obj as usize & KIND_MASK, 0);
        // SAFETY: caller guarantees `obj` is non-null. ORing in the tag bits
        // can only set bits, so the result is also non-null.
        unsafe { NonNull::new_unchecked(((obj as usize) | (kind as usize)) as *mut ffi::PyObject) }
    }

    /// Detect the kind of `ob` for the str-bytes equality fast path.
    #[inline]
    fn detect_kind(ob: &Bound<'_, PyAny>) -> HashKind {
        // SAFETY: `Bound` always holds a valid `PyObject*`.
        unsafe {
            if ffi::Py_TYPE(ob.as_ptr()) == &raw mut ffi::PyUnicode_Type {
                HashKind::Str
            } else {
                HashKind::Other
            }
        }
    }

    /// Build by computing `__hash__` once and bumping the object's refcount.
    /// We call `Py_INCREF` directly on the raw pointer instead of going
    /// through `Bound::clone().unbind() + forget`, which would also produce
    /// one strong reference but with extra moves of a `Py<PyAny>` smart
    /// pointer the optimizer doesn't always elide.
    fn from_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let hash = ob.hash()?;
        let kind = Self::detect_kind(ob);
        let raw = ob.as_ptr();
        // SAFETY: `Bound` guarantees `raw` is a valid, non-null `PyObject*`
        // and the GIL is held (it's tied to `Bound`'s lifetime). We take
        // ownership of the new strong reference into our tagged slot.
        unsafe { ffi::Py_INCREF(raw) };
        // SAFETY: `raw` is a non-null PyObject pointer with zero low bits
        // (CPython aligns `PyObject` to at least 8 bytes).
        let tagged = unsafe { Self::pack(raw, kind) };
        Ok(Self { tagged, hash })
    }

    /// Refcount-bumping clone. Reuses the cached hash and kind so only the
    /// `Py_INCREF` is paid.
    fn clone_with_py(&self, _py: Python<'_>) -> Self {
        // SAFETY: we hold one strong reference to `obj_ptr()`, so it remains
        // valid for `Py_INCREF`. Calling under GIL (`_py`) is required.
        unsafe { ffi::Py_INCREF(self.obj_ptr()) };
        Self {
            tagged: self.tagged,
            hash: self.hash,
        }
    }

    /// Masked object pointer (tag bits stripped).
    #[inline]
    fn obj_ptr(&self) -> *mut ffi::PyObject {
        ((self.tagged.as_ptr() as usize) & !KIND_MASK) as *mut ffi::PyObject
    }

    /// Decoded `HashKind` tag.
    #[inline]
    fn kind(&self) -> HashKind {
        match (self.tagged.as_ptr() as usize) & KIND_MASK {
            1 => HashKind::Str,
            _ => HashKind::Other,
        }
    }

    /// Borrow the underlying object as a `Borrowed<PyAny>` without bumping
    /// refcount. Derefs to `&Bound<'py, PyAny>` for method calls.
    #[inline]
    fn obj_borrowed<'a, 'py>(&'a self, py: Python<'py>) -> Borrowed<'a, 'py, PyAny> {
        // SAFETY: `obj_ptr()` returns the live, masked PyObject pointer we
        // own a strong reference to. The borrow lifetime `'a` is tied to
        // `&self`, so the pointer can't outlive our refcount.
        unsafe { Borrowed::from_ptr(py, self.obj_ptr()) }
    }

    /// Return a fresh owned `Py<PyAny>` (bumps refcount).
    #[inline]
    fn obj_clone_ref(&self, py: Python<'_>) -> Py<PyAny> {
        // SAFETY: we hold a strong reference, so the borrowed pointer is
        // valid. `to_owned()` bumps the refcount, yielding a second valid
        // strong reference packaged as `Bound`, which `unbind()` converts
        // to a `Py` without further refcount changes.
        unsafe { Borrowed::from_ptr(py, self.obj_ptr()) }
            .to_owned()
            .unbind()
    }
}

impl Drop for HashedAny {
    fn drop(&mut self) {
        // CPython requires the GIL for refcount decrements (outside of the
        // free-threaded build, where this is still safe). PyO3 attaches a
        // GIL guard internally for `Py<T>::drop`; we do the same dance.
        Python::attach(|_py| {
            // SAFETY: we own one strong reference to the masked pointer.
            unsafe { ffi::Py_DECREF(self.obj_ptr()) };
        });
    }
}

/// Borrow-only key wrapper for hash-table lookups.
///
/// Built by `ptr::read`-ing the input `Py<PyAny>` into a `ManuallyDrop` so no
/// refcount bump occurs. The wrapped value is never dropped; the original
/// `Bound`'s lifetime keeps the underlying object alive. Use when you only
/// need to query the map (`get`, `contains_key`, `remove`) and won't keep the
/// key around afterward.
struct ProbeKey {
    inner: ManuallyDrop<HashedAny>,
}

impl ProbeKey {
    /// # Safety
    /// Caller must ensure `ob` outlives the returned `ProbeKey`. The probe
    /// holds a non-owning copy of the underlying `PyObject` pointer (with
    /// the kind tag packed into the low bits, same as a normal `HashedAny`).
    fn from_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let hash = ob.hash()?;
        let kind = HashedAny::detect_kind(ob);
        // SAFETY: `ob.as_ptr()` is a valid, properly-aligned `PyObject*`.
        let tagged = unsafe { HashedAny::pack(ob.as_ptr(), kind) };
        Ok(Self {
            inner: ManuallyDrop::new(HashedAny { tagged, hash }),
        })
    }

    /// Borrow as `&HashedAny` for use with map APIs that take a borrowed key.
    fn as_key(&self) -> &HashedAny {
        &self.inner
    }
}

impl Hash for HashedAny {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_isize(self.hash);
    }
}

impl PartialEq for HashedAny {
    /// Equality with three short-circuits before falling back to Python rich
    /// compare: hash mismatch, pointer identity, and a UTF-8 bytes compare
    /// when both sides are `str` (skips `PyObject_RichCompareBool` dispatch).
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            return false;
        }
        if self.obj_ptr() == other.obj_ptr() {
            return true;
        }
        Python::attach(|py| {
            // Direct UTF-8 compare bypasses PyObject_RichCompareBool dispatch.
            if self.kind() == HashKind::Str
                && other.kind() == HashKind::Str
                && let Ok(sa) = self.obj_borrowed(py).cast::<PyString>()
                && let Ok(sb) = other.obj_borrowed(py).cast::<PyString>()
                && let Ok(x) = sa.to_str()
                && let Ok(y) = sb.to_str()
            {
                return x == y;
            }
            self.obj_borrowed(py)
                .eq(other.obj_borrowed(py))
                .unwrap_or(false)
        })
    }
}

impl Eq for HashedAny {}

/// Emits one full Python-facing map surface (map class + iterators + views)
/// parameterized by backend type. Invoked once for `Elastic` and
/// once for `Funnel` so behavior changes land in both maps simultaneously.
/// `PyO3` can't express `#[pyclass]` over a generic, hence the macro.
macro_rules! define_map_classes {
    (
        py_map = $PyMap:ident,
        py_map_name = $py_map_name:literal,
        inner = $Inner:ident,
        build_options = $build_options:ident,
        extra_arg = $extra_arg:ident,
        extra_ty = $extra_ty:ty,
        key_iter = $KeyIter:ident,
        key_iter_name = $key_iter_name:literal,
        value_iter = $ValueIter:ident,
        value_iter_name = $value_iter_name:literal,
        item_iter = $ItemIter:ident,
        item_iter_name = $item_iter_name:literal,
        keys_view = $KeysView:ident,
        keys_view_name = $keys_view_name:literal,
        values_view = $ValuesView:ident,
        values_view_name = $values_view_name:literal,
        items_view = $ItemsView:ident,
        items_view_name = $items_view_name:literal,
    ) => {
        /// `PyO3` wrapper around the Rust hash map.
        #[pyclass(name = $py_map_name, module = "opthash")]
        struct $PyMap {
            /// Underlying Rust hash map.
            inner: $Inner<HashedAny, Py<PyAny>>,
            /// Mutation counter. Iterators snapshot this at construction and
            /// raise `RuntimeError` on next `__next__` if it changes.
            generation: u64,
        }

        impl $PyMap {
            /// Bump generation to invalidate any active iterator snapshots.
            /// Call after every mutating operation.
            #[inline]
            fn bump(&mut self) {
                self.generation = self.generation.wrapping_add(1);
            }
        }

        #[pymethods]
        impl $PyMap {
            #[new]
            #[pyo3(signature = (other = None, *, capacity = 0, **kwargs))]
            fn new(
                other: Option<&Bound<'_, PyAny>>,
                capacity: usize,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<Self> {
                let mut me = Self {
                    inner: $Inner::with_capacity(capacity),
                    generation: 0,
                };
                if other.is_some() || kwargs.is_some() {
                    me.update(other, kwargs)?;
                }
                Ok(me)
            }

            #[classmethod]
            #[pyo3(signature = (capacity = 0, reserve_fraction = None, $extra_arg = None))]
            fn with_options(
                _cls: &Bound<'_, PyType>,
                capacity: usize,
                reserve_fraction: Option<f64>,
                $extra_arg: Option<$extra_ty>,
            ) -> PyResult<Self> {
                let opts = $build_options(capacity, reserve_fraction, $extra_arg)?;
                Ok(Self {
                    inner: $Inner::with_options(opts),
                    generation: 0,
                })
            }

            #[classmethod]
            #[pyo3(signature = (iterable, value = None))]
            fn fromkeys(
                _cls: &Bound<'_, PyType>,
                iterable: &Bound<'_, PyAny>,
                value: Option<Py<PyAny>>,
                py: Python<'_>,
            ) -> PyResult<Self> {
                let cap = iterable.len().unwrap_or(0);
                let mut me = Self {
                    inner: $Inner::with_capacity(cap),
                    generation: 0,
                };
                let val = value.unwrap_or_else(|| py.None());
                for k in iterable.try_iter()? {
                    let k = k?;
                    let key = HashedAny::from_bound(&k)?;
                    me.inner.insert(key, val.clone_ref(py));
                }
                me.bump();
                Ok(me)
            }

            /// Support `Cls[K, V]` subscript syntax at runtime by returning a
            /// `types.GenericAlias` (same factory `CPython` uses for
            /// `dict[str, int]` etc). Required for parity with the typing
            /// stub that declares the class as `Generic[K, V]`.
            #[classmethod]
            fn __class_getitem__<'py>(
                cls: &Bound<'py, PyType>,
                item: &Bound<'py, PyAny>,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                py.import("types")?
                    .getattr("GenericAlias")?
                    .call1((cls, item))
            }

            fn __len__(&self) -> usize {
                self.inner.len()
            }

            #[getter]
            fn capacity(&self) -> usize {
                self.inner.capacity()
            }

            fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
                let probe = ProbeKey::from_bound(key)?;
                Ok(self.inner.contains_key(probe.as_key()))
            }

            fn __getitem__(&self, key: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
                let probe = ProbeKey::from_bound(key)?;
                match self.inner.get(probe.as_key()) {
                    Some(v) => Ok(v.clone_ref(py)),
                    None => Err(PyKeyError::new_err(key.clone().unbind())),
                }
            }

            fn __setitem__(
                &mut self,
                key: &Bound<'_, PyAny>,
                value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                let k = HashedAny::from_bound(key)?;
                self.inner.insert(k, value.clone().unbind());
                self.bump();
                Ok(())
            }

            fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
                let probe = ProbeKey::from_bound(key)?;
                match self.inner.remove(probe.as_key()) {
                    Some(_) => {
                        self.bump();
                        Ok(())
                    }
                    None => Err(PyKeyError::new_err(key.clone().unbind())),
                }
            }

            #[pyo3(signature = (key, default = None))]
            fn get(
                &self,
                key: &Bound<'_, PyAny>,
                default: Option<Py<PyAny>>,
                py: Python<'_>,
            ) -> PyResult<Py<PyAny>> {
                let probe = ProbeKey::from_bound(key)?;
                Ok(match self.inner.get(probe.as_key()) {
                    Some(v) => v.clone_ref(py),
                    None => default.unwrap_or_else(|| py.None()),
                })
            }

            fn clear(&mut self) {
                self.inner.clear();
                self.bump();
            }

            fn __repr__(&self) -> String {
                format!(
                    concat!($py_map_name, "(len={}, capacity={})"),
                    self.inner.len(),
                    self.inner.capacity()
                )
            }

            fn __iter__(slf: Bound<'_, Self>) -> $KeyIter {
                let py = slf.py();
                let m = slf.borrow();
                let snapshot = m.inner.iter().map(|(k, _)| k.obj_clone_ref(py)).collect();
                let expected_gen = m.generation;
                drop(m);
                $KeyIter {
                    map: slf.unbind(),
                    snapshot,
                    expected_gen,
                    pos: 0,
                }
            }

            fn keys(slf: Bound<'_, Self>) -> $KeysView {
                $KeysView { map: slf.unbind() }
            }

            fn values(slf: Bound<'_, Self>) -> $ValuesView {
                $ValuesView { map: slf.unbind() }
            }

            fn items(slf: Bound<'_, Self>) -> $ItemsView {
                $ItemsView { map: slf.unbind() }
            }

            /// Mirror of `dict.update`. Branches in priority order: same-type
            /// (downcast for direct inner-map access), `PyDict`, mapping with
            /// `keys()`, then iterable of `(k, v)` tuples. Each branch
            /// reserves up front when size is known. `bump()` only fires when
            /// at least one insert occurred so empty `update()` doesn't
            /// invalidate active iterators.
            #[pyo3(signature = (other = None, **kwargs))]
            fn update(
                &mut self,
                other: Option<&Bound<'_, PyAny>>,
                kwargs: Option<&Bound<'_, PyDict>>,
            ) -> PyResult<()> {
                let mut touched = false;
                if let Some(other) = other {
                    if let Ok(other_map) = other.cast::<Self>() {
                        let py = other.py();
                        let borrowed = other_map.borrow();
                        self.inner.reserve(borrowed.inner.len());
                        for (k, v) in &borrowed.inner {
                            self.inner.insert(k.clone_with_py(py), v.clone_ref(py));
                            touched = true;
                        }
                    } else if let Ok(dict) = other.cast::<PyDict>() {
                        self.inner.reserve(dict.len());
                        for (k, v) in dict.iter() {
                            let key = HashedAny::from_bound(&k)?;
                            self.inner.insert(key, v.unbind());
                            touched = true;
                        }
                    } else if other.hasattr("keys")? {
                        if let Ok(hint) = other.len() {
                            self.inner.reserve(hint);
                        }
                        let keys = other.call_method0("keys")?;
                        for k in keys.try_iter()? {
                            let k = k?;
                            let v = other.get_item(&k)?;
                            let key = HashedAny::from_bound(&k)?;
                            self.inner.insert(key, v.unbind());
                            touched = true;
                        }
                    } else {
                        if let Ok(hint) = other.len() {
                            self.inner.reserve(hint);
                        }
                        for item in other.try_iter()? {
                            let item = item?;
                            let len = item.len().map_err(|_| {
                                PyValueError::new_err("update sequence elements must be 2-tuples")
                            })?;
                            if len != 2 {
                                return Err(PyValueError::new_err(
                                    "update sequence elements must be 2-tuples",
                                ));
                            }
                            let k = item.get_item(0)?;
                            let v = item.get_item(1)?;
                            let key = HashedAny::from_bound(&k)?;
                            self.inner.insert(key, v.unbind());
                            touched = true;
                        }
                    }
                }
                if let Some(kwargs) = kwargs {
                    self.inner.reserve(kwargs.len());
                    for (k, v) in kwargs.iter() {
                        let key = HashedAny::from_bound(&k)?;
                        self.inner.insert(key, v.unbind());
                        touched = true;
                    }
                }
                if touched {
                    self.bump();
                }
                Ok(())
            }

            #[pyo3(signature = (key, default = None))]
            fn pop(
                &mut self,
                key: &Bound<'_, PyAny>,
                default: Option<Py<PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                let probe = ProbeKey::from_bound(key)?;
                match self.inner.remove(probe.as_key()) {
                    Some(v) => {
                        self.bump();
                        Ok(v)
                    }
                    None => default.ok_or_else(|| PyKeyError::new_err(key.clone().unbind())),
                }
            }

            fn popitem<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
                if self.inner.is_empty() {
                    return Err(PyKeyError::new_err("popitem(): map is empty"));
                }
                let probe = {
                    let (k, _) = self.inner.iter().next().expect("len > 0");
                    k.clone_with_py(py)
                };
                let key_obj = probe.obj_clone_ref(py);
                let value = self.inner.remove(&probe).expect("key from iter must exist");
                self.bump();
                PyTuple::new(py, [key_obj, value])
            }

            #[pyo3(signature = (key, default = None))]
            fn setdefault(
                &mut self,
                key: &Bound<'_, PyAny>,
                default: Option<Py<PyAny>>,
                py: Python<'_>,
            ) -> PyResult<Py<PyAny>> {
                let probe = ProbeKey::from_bound(key)?;
                if let Some(v) = self.inner.get(probe.as_key()) {
                    return Ok(v.clone_ref(py));
                }
                let k = HashedAny::from_bound(key)?;
                let value = default.unwrap_or_else(|| py.None());
                self.inner.insert(k, value.clone_ref(py));
                self.bump();
                Ok(value)
            }

            fn copy(&self, py: Python<'_>) -> Self {
                let mut new = $Inner::with_capacity(self.inner.len());
                for (k, v) in &self.inner {
                    new.insert(k.clone_with_py(py), v.clone_ref(py));
                }
                Self {
                    inner: new,
                    generation: 0,
                }
            }

            fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> bool {
                if let Ok(other_map) = other.cast::<Self>() {
                    let other_inner = &other_map.borrow().inner;
                    if self.inner.len() != other_inner.len() {
                        return false;
                    }
                    for (k, v) in &self.inner {
                        match other_inner.get(k) {
                            Some(ov) => {
                                if !v.bind(py).eq(ov.bind(py)).unwrap_or(false) {
                                    return false;
                                }
                            }
                            None => return false,
                        }
                    }
                    return true;
                }
                if let Ok(d) = other.cast::<PyDict>() {
                    if d.len() != self.inner.len() {
                        return false;
                    }
                    for (k, v) in &self.inner {
                        let key_b = k.obj_borrowed(py);
                        match d.get_item(key_b) {
                            Ok(Some(other_v)) => {
                                if !v.bind(py).eq(&other_v).unwrap_or(false) {
                                    return false;
                                }
                            }
                            _ => return false,
                        }
                    }
                    return true;
                }
                if !other.hasattr("keys").unwrap_or(false) {
                    return false;
                }
                let Ok(other_len) = other.len() else {
                    return false;
                };
                if other_len != self.inner.len() {
                    return false;
                }
                for (k, v) in &self.inner {
                    let key_b = k.obj_borrowed(py);
                    let Ok(other_v) = other.get_item(key_b) else {
                        return false;
                    };
                    if !v.bind(py).eq(&other_v).unwrap_or(false) {
                        return false;
                    }
                }
                true
            }

            fn __or__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
                let other_hint = other.len().unwrap_or(0);
                let cap = self.inner.len().saturating_add(other_hint);
                let mut new = Self {
                    inner: $Inner::with_capacity(cap),
                    generation: 0,
                };
                for (k, v) in &self.inner {
                    new.inner.insert(k.clone_with_py(py), v.clone_ref(py));
                }
                new.update(Some(other), None)?;
                Ok(new)
            }

            fn __ror__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Self> {
                let other_hint = other.len().unwrap_or(0);
                let cap = self.inner.len().saturating_add(other_hint);
                let mut new = Self {
                    inner: $Inner::with_capacity(cap),
                    generation: 0,
                };
                new.update(Some(other), None)?;
                for (k, v) in &self.inner {
                    new.inner.insert(k.clone_with_py(py), v.clone_ref(py));
                }
                new.bump();
                Ok(new)
            }

            fn __ior__(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
                self.update(Some(other), None)
            }
        }

        define_iter!($KeyIter, $key_iter_name, $PyMap);
        define_iter!($ValueIter, $value_iter_name, $PyMap);
        define_iter!($ItemIter, $item_iter_name, $PyMap);

        /// Live view over the map's keys (mirrors `dict.keys()`). Holds a
        /// `Py<map>` so each operation borrows current map state — no
        /// snapshotting at view construction.
        #[pyclass(name = $keys_view_name, module = "opthash")]
        struct $KeysView {
            map: Py<$PyMap>,
        }

        #[pymethods]
        impl $KeysView {
            fn __iter__(&self, py: Python<'_>) -> $KeyIter {
                let m = self.map.borrow(py);
                let snapshot = m.inner.iter().map(|(k, _)| k.obj_clone_ref(py)).collect();
                $KeyIter {
                    map: self.map.clone_ref(py),
                    snapshot,
                    expected_gen: m.generation,
                    pos: 0,
                }
            }
            fn __len__(&self, py: Python<'_>) -> usize {
                self.map.borrow(py).inner.len()
            }
            fn __contains__(&self, key: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<bool> {
                let m = self.map.borrow(py);
                let probe = ProbeKey::from_bound(key)?;
                Ok(m.inner.contains_key(probe.as_key()))
            }
            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                let m = self.map.borrow(py);
                let parts: PyResult<Vec<String>> = m
                    .inner
                    .iter()
                    .map(|(k, _)| Ok(k.obj_borrowed(py).repr()?.to_string()))
                    .collect();
                Ok(format!(
                    concat!($keys_view_name, "([{}])"),
                    parts?.join(", ")
                ))
            }
            fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> bool {
                let m = self.map.borrow(py);
                let Ok(other_len) = other.len() else {
                    return false;
                };
                if other_len != m.inner.len() {
                    return false;
                }
                for (k, _) in &m.inner {
                    if !other.contains(k.obj_borrowed(py)).unwrap_or(false) {
                        return false;
                    }
                }
                true
            }
            fn __and__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PySet>> {
                let result = PySet::empty(py)?;
                let m = self.map.borrow(py);
                for (k, _) in &m.inner {
                    let key_b = k.obj_borrowed(py);
                    if other.contains(key_b)? {
                        result.add(key_b)?;
                    }
                }
                Ok(result.unbind())
            }
            fn __or__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PySet>> {
                let result = PySet::empty(py)?;
                {
                    let m = self.map.borrow(py);
                    for (k, _) in &m.inner {
                        result.add(k.obj_borrowed(py))?;
                    }
                }
                for item in other.try_iter()? {
                    result.add(item?)?;
                }
                Ok(result.unbind())
            }
            fn __sub__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PySet>> {
                let result = PySet::empty(py)?;
                let m = self.map.borrow(py);
                for (k, _) in &m.inner {
                    let key_b = k.obj_borrowed(py);
                    if !other.contains(key_b)? {
                        result.add(key_b)?;
                    }
                }
                Ok(result.unbind())
            }
            fn __xor__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PySet>> {
                let result = PySet::empty(py)?;
                {
                    let m = self.map.borrow(py);
                    for (k, _) in &m.inner {
                        let key_b = k.obj_borrowed(py);
                        if !other.contains(key_b)? {
                            result.add(key_b)?;
                        }
                    }
                }
                let m = self.map.borrow(py);
                for item in other.try_iter()? {
                    let item = item?;
                    let probe = ProbeKey::from_bound(&item)?;
                    if !m.inner.contains_key(probe.as_key()) {
                        result.add(item)?;
                    }
                }
                Ok(result.unbind())
            }
        }

        /// Live view over the map's values (mirrors `dict.values()`).
        #[pyclass(name = $values_view_name, module = "opthash")]
        struct $ValuesView {
            map: Py<$PyMap>,
        }

        #[pymethods]
        impl $ValuesView {
            fn __iter__(&self, py: Python<'_>) -> $ValueIter {
                let m = self.map.borrow(py);
                let snapshot = m.inner.iter().map(|(_, v)| v.clone_ref(py)).collect();
                $ValueIter {
                    map: self.map.clone_ref(py),
                    snapshot,
                    expected_gen: m.generation,
                    pos: 0,
                }
            }
            fn __len__(&self, py: Python<'_>) -> usize {
                self.map.borrow(py).inner.len()
            }
            fn __contains__(&self, value: &Bound<'_, PyAny>, py: Python<'_>) -> bool {
                let m = self.map.borrow(py);
                for (_, v) in &m.inner {
                    if v.bind(py).eq(value).unwrap_or(false) {
                        return true;
                    }
                }
                false
            }
            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                let m = self.map.borrow(py);
                let parts: PyResult<Vec<String>> = m
                    .inner
                    .iter()
                    .map(|(_, v)| Ok(v.bind(py).repr()?.to_string()))
                    .collect();
                Ok(format!(
                    concat!($values_view_name, "([{}])"),
                    parts?.join(", ")
                ))
            }
        }

        /// Live view over the map's `(key, value)` pairs (mirrors
        /// `dict.items()`). Set operations build fresh `(k, v)` `PyTuple`s.
        #[pyclass(name = $items_view_name, module = "opthash")]
        struct $ItemsView {
            map: Py<$PyMap>,
        }

        #[pymethods]
        impl $ItemsView {
            fn __iter__(&self, py: Python<'_>) -> PyResult<$ItemIter> {
                let m = self.map.borrow(py);
                let snapshot: PyResult<Vec<Py<PyAny>>> = m
                    .inner
                    .iter()
                    .map(|(k, v)| {
                        let tup = PyTuple::new(py, [k.obj_clone_ref(py), v.clone_ref(py)])?;
                        Ok(tup.into_any().unbind())
                    })
                    .collect();
                Ok($ItemIter {
                    map: self.map.clone_ref(py),
                    snapshot: snapshot?,
                    expected_gen: m.generation,
                    pos: 0,
                })
            }
            fn __len__(&self, py: Python<'_>) -> usize {
                self.map.borrow(py).inner.len()
            }
            fn __contains__(&self, item: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<bool> {
                let Ok(tup) = item.cast::<PyTuple>() else {
                    return Ok(false);
                };
                if tup.len() != 2 {
                    return Ok(false);
                }
                let k = tup.get_item(0)?;
                let v = tup.get_item(1)?;
                let m = self.map.borrow(py);
                let probe = ProbeKey::from_bound(&k)?;
                match m.inner.get(probe.as_key()) {
                    Some(stored_v) => Ok(stored_v.bind(py).eq(&v).unwrap_or(false)),
                    None => Ok(false),
                }
            }
            fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
                let m = self.map.borrow(py);
                let parts: PyResult<Vec<String>> = m
                    .inner
                    .iter()
                    .map(|(k, v)| {
                        let kr = k.obj_borrowed(py).repr()?.to_string();
                        let vr = v.bind(py).repr()?.to_string();
                        Ok(format!("({kr}, {vr})"))
                    })
                    .collect();
                Ok(format!(
                    concat!($items_view_name, "([{}])"),
                    parts?.join(", ")
                ))
            }
            fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> bool {
                let m = self.map.borrow(py);
                let Ok(other_len) = other.len() else {
                    return false;
                };
                if other_len != m.inner.len() {
                    return false;
                }
                for (k, v) in &m.inner {
                    let Ok(tup) = PyTuple::new(py, [k.obj_clone_ref(py), v.clone_ref(py)]) else {
                        return false;
                    };
                    if !other.contains(&tup).unwrap_or(false) {
                        return false;
                    }
                }
                true
            }
            fn __and__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PySet>> {
                let result = PySet::empty(py)?;
                let m = self.map.borrow(py);
                for (k, v) in &m.inner {
                    let tup = PyTuple::new(py, [k.obj_clone_ref(py), v.clone_ref(py)])?;
                    if other.contains(&tup)? {
                        result.add(tup)?;
                    }
                }
                Ok(result.unbind())
            }
            fn __or__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PySet>> {
                let result = PySet::empty(py)?;
                {
                    let m = self.map.borrow(py);
                    for (k, v) in &m.inner {
                        let tup = PyTuple::new(py, [k.obj_clone_ref(py), v.clone_ref(py)])?;
                        result.add(tup)?;
                    }
                }
                for item in other.try_iter()? {
                    result.add(item?)?;
                }
                Ok(result.unbind())
            }
            fn __sub__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PySet>> {
                let result = PySet::empty(py)?;
                let m = self.map.borrow(py);
                for (k, v) in &m.inner {
                    let tup = PyTuple::new(py, [k.obj_clone_ref(py), v.clone_ref(py)])?;
                    if !other.contains(&tup)? {
                        result.add(tup)?;
                    }
                }
                Ok(result.unbind())
            }
        }
    };
}

/// Generates a single iterator pyclass.
///
/// `snapshot` holds the iter contents materialized eagerly at `__iter__`
/// time. Trades memory for borrow-checker simplicity (no self-referencing
/// borrow of the map). `expected_gen` is captured at iter construction;
/// each `__next__` checks the map's current `generation` and raises
/// `RuntimeError("dictionary changed size during iteration")` on mismatch.
macro_rules! define_iter {
    ($Iter:ident, $iter_name:literal, $PyMap:ident) => {
        #[pyclass(name = $iter_name, module = "opthash")]
        struct $Iter {
            /// Source map. Held to check generation per `__next__`.
            map: Py<$PyMap>,
            /// Eagerly materialized iter contents (keys / values / items).
            snapshot: Vec<Py<PyAny>>,
            /// Map's `generation` at iter construction.
            expected_gen: u64,
            /// Next index into `snapshot`.
            pos: usize,
        }

        #[pymethods]
        impl $Iter {
            fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }
            fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
                if self.map.borrow(py).generation != self.expected_gen {
                    return Err(PyRuntimeError::new_err(
                        "dictionary changed size during iteration",
                    ));
                }
                if self.pos >= self.snapshot.len() {
                    return Ok(None);
                }
                let v = self.snapshot[self.pos].clone_ref(py);
                self.pos += 1;
                Ok(Some(v))
            }
        }
    };
}

define_map_classes! {
    py_map = PyElasticHashMap,
    py_map_name = "ElasticHashMap",
    inner = ElasticHashMap,
    build_options = build_elastic_options,
    extra_arg = probe_scale,
    extra_ty = f64,
    key_iter = PyElasticKeyIter,
    key_iter_name = "_ElasticKeyIter",
    value_iter = PyElasticValueIter,
    value_iter_name = "_ElasticValueIter",
    item_iter = PyElasticItemIter,
    item_iter_name = "_ElasticItemIter",
    keys_view = PyElasticKeysView,
    keys_view_name = "elastic_keys",
    values_view = PyElasticValuesView,
    values_view_name = "elastic_values",
    items_view = PyElasticItemsView,
    items_view_name = "elastic_items",
}

define_map_classes! {
    py_map = PyFunnelHashMap,
    py_map_name = "FunnelHashMap",
    inner = FunnelHashMap,
    build_options = build_funnel_options,
    extra_arg = primary_probe_limit,
    extra_ty = usize,
    key_iter = PyFunnelKeyIter,
    key_iter_name = "_FunnelKeyIter",
    value_iter = PyFunnelValueIter,
    value_iter_name = "_FunnelValueIter",
    item_iter = PyFunnelItemIter,
    item_iter_name = "_FunnelItemIter",
    keys_view = PyFunnelKeysView,
    keys_view_name = "funnel_keys",
    values_view = PyFunnelValuesView,
    values_view_name = "funnel_values",
    items_view = PyFunnelItemsView,
    items_view_name = "funnel_items",
}

#[pymodule]
fn opthash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyElasticHashMap>()?;
    m.add_class::<PyFunnelHashMap>()?;
    m.add_class::<PyElasticKeysView>()?;
    m.add_class::<PyElasticValuesView>()?;
    m.add_class::<PyElasticItemsView>()?;
    m.add_class::<PyElasticKeyIter>()?;
    m.add_class::<PyElasticValueIter>()?;
    m.add_class::<PyElasticItemIter>()?;
    m.add_class::<PyFunnelKeysView>()?;
    m.add_class::<PyFunnelValuesView>()?;
    m.add_class::<PyFunnelItemsView>()?;
    m.add_class::<PyFunnelKeyIter>()?;
    m.add_class::<PyFunnelValueIter>()?;
    m.add_class::<PyFunnelItemIter>()?;
    Ok(())
}
