#![cfg(feature = "python")]
#![allow(clippy::ptr_as_ptr, clippy::borrow_as_ptr, clippy::ref_as_ptr)]

use std::hash::{Hash, Hasher};
use std::mem::ManuallyDrop;

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
        opts.reserve_fraction = rf;
    }
    if let Some(ps) = probe_scale {
        if ps <= 0.0 {
            return Err(PyValueError::new_err("probe_scale must be positive"));
        }
        opts.probe_scale = ps;
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
        opts.reserve_fraction = rf;
    }
    if let Some(limit) = primary_probe_limit {
        if limit == 0 {
            return Err(PyValueError::new_err(
                "primary_probe_limit must be positive",
            ));
        }
        opts.primary_probe_limit = Some(limit);
    }
    Ok(opts)
}

#[derive(Clone, Copy, PartialEq)]
#[repr(u8)]
enum HashKind {
    Str,
    Other,
}

struct HashedAny {
    obj: Py<PyAny>,
    hash: isize,
    kind: HashKind,
}

impl HashedAny {
    fn from_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let hash = ob.hash()?;
        let kind = unsafe {
            if ffi::Py_TYPE(ob.as_ptr()) == &raw mut ffi::PyUnicode_Type {
                HashKind::Str
            } else {
                HashKind::Other
            }
        };
        Ok(Self {
            obj: ob.clone().unbind(),
            hash,
            kind,
        })
    }

    fn clone_with_py(&self, py: Python<'_>) -> Self {
        Self {
            obj: self.obj.clone_ref(py),
            hash: self.hash,
            kind: self.kind,
        }
    }
}

struct ProbeKey {
    inner: ManuallyDrop<HashedAny>,
}

impl ProbeKey {
    fn from_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let hash = ob.hash()?;
        let kind = unsafe {
            if ffi::Py_TYPE(ob.as_ptr()) == &raw mut ffi::PyUnicode_Type {
                HashKind::Str
            } else {
                HashKind::Other
            }
        };
        let obj = unsafe { std::ptr::read(ob.as_unbound()) };
        Ok(Self {
            inner: ManuallyDrop::new(HashedAny { obj, hash, kind }),
        })
    }

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
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            return false;
        }
        if self.obj.as_ptr() == other.obj.as_ptr() {
            return true;
        }
        Python::attach(|py| {
            // Direct UTF-8 compare bypasses PyObject_RichCompareBool dispatch.
            if self.kind == HashKind::Str
                && other.kind == HashKind::Str
                && let Ok(sa) = self.obj.bind(py).cast::<PyString>()
                && let Ok(sb) = other.obj.bind(py).cast::<PyString>()
                && let Ok(x) = sa.to_str()
                && let Ok(y) = sb.to_str()
            {
                return x == y;
            }
            self.obj.bind(py).eq(other.obj.bind(py)).unwrap_or(false)
        })
    }
}

impl Eq for HashedAny {}

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
        #[pyclass(name = $py_map_name, module = "opthash")]
        struct $PyMap {
            inner: $Inner<HashedAny, Py<PyAny>>,
            generation: u64,
        }

        impl $PyMap {
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
                let snapshot = m.inner.iter().map(|(k, _)| k.obj.clone_ref(py)).collect();
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
                let key_obj = probe.obj.clone_ref(py);
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
                        let key_b = k.obj.bind(py);
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
                    let key_b = k.obj.bind(py);
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

        #[pyclass(name = $keys_view_name, module = "opthash")]
        struct $KeysView {
            map: Py<$PyMap>,
        }

        #[pymethods]
        impl $KeysView {
            fn __iter__(&self, py: Python<'_>) -> $KeyIter {
                let m = self.map.borrow(py);
                let snapshot = m.inner.iter().map(|(k, _)| k.obj.clone_ref(py)).collect();
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
                    .map(|(k, _)| Ok(k.obj.bind(py).repr()?.to_string()))
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
                    if !other.contains(k.obj.bind(py)).unwrap_or(false) {
                        return false;
                    }
                }
                true
            }
            fn __and__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PySet>> {
                let result = PySet::empty(py)?;
                let m = self.map.borrow(py);
                for (k, _) in &m.inner {
                    let key_b = k.obj.bind(py);
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
                        result.add(k.obj.bind(py))?;
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
                    let key_b = k.obj.bind(py);
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
                        let key_b = k.obj.bind(py);
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
                        let tup = PyTuple::new(py, [k.obj.clone_ref(py), v.clone_ref(py)])?;
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
                        let kr = k.obj.bind(py).repr()?.to_string();
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
                    let Ok(tup) = PyTuple::new(py, [k.obj.clone_ref(py), v.clone_ref(py)]) else {
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
                    let tup = PyTuple::new(py, [k.obj.clone_ref(py), v.clone_ref(py)])?;
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
                        let tup = PyTuple::new(py, [k.obj.clone_ref(py), v.clone_ref(py)])?;
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
                    let tup = PyTuple::new(py, [k.obj.clone_ref(py), v.clone_ref(py)])?;
                    if !other.contains(&tup)? {
                        result.add(tup)?;
                    }
                }
                Ok(result.unbind())
            }
        }
    };
}

macro_rules! define_iter {
    ($Iter:ident, $iter_name:literal, $PyMap:ident) => {
        #[pyclass(name = $iter_name, module = "opthash")]
        struct $Iter {
            map: Py<$PyMap>,
            snapshot: Vec<Py<PyAny>>,
            expected_gen: u64,
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
