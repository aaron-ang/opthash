#![cfg(feature = "python")]

use std::hash::{Hash, Hasher};

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet, PyString, PyTuple, PyType};

use crate::funnel::MAX_FUNNEL_RESERVE_FRACTION;
use crate::{ElasticHashMap, ElasticOptions, FunnelHashMap, FunnelOptions};

#[derive(Clone, Copy, PartialEq, Eq)]
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

fn validate_elastic_reserve_fraction(value: f64) -> PyResult<()> {
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "reserve_fraction must be in the open interval (0, 1)",
        ))
    }
}

fn validate_funnel_reserve_fraction(value: f64) -> PyResult<()> {
    if value > 0.0 && value <= MAX_FUNNEL_RESERVE_FRACTION {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "reserve_fraction must be in (0, {MAX_FUNNEL_RESERVE_FRACTION}]; \
             FunnelHashMap caps the load factor at 1/8 by design"
        )))
    }
}

#[pyclass(name = "ElasticOptions", module = "opthash")]
struct PyElasticOptions {
    inner: ElasticOptions,
}

#[pymethods]
impl PyElasticOptions {
    #[new]
    #[pyo3(signature = (capacity = 0, reserve_fraction = None, probe_scale = None))]
    fn new(
        capacity: usize,
        reserve_fraction: Option<f64>,
        probe_scale: Option<f64>,
    ) -> PyResult<Self> {
        let mut inner = ElasticOptions::with_capacity(capacity);
        if let Some(rf) = reserve_fraction {
            validate_elastic_reserve_fraction(rf)?;
            inner.reserve_fraction = rf;
        }
        if let Some(ps) = probe_scale {
            if ps <= 0.0 {
                return Err(PyValueError::new_err("probe_scale must be positive"));
            }
            inner.probe_scale = ps;
        }
        Ok(Self { inner })
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity
    }

    #[getter]
    fn reserve_fraction(&self) -> f64 {
        self.inner.reserve_fraction
    }

    #[getter]
    fn probe_scale(&self) -> f64 {
        self.inner.probe_scale
    }
}

#[pyclass(name = "FunnelOptions", module = "opthash")]
struct PyFunnelOptions {
    inner: FunnelOptions,
}

#[pymethods]
impl PyFunnelOptions {
    #[new]
    #[pyo3(signature = (capacity = 0, reserve_fraction = None, primary_probe_limit = None))]
    fn new(
        capacity: usize,
        reserve_fraction: Option<f64>,
        primary_probe_limit: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner = FunnelOptions::with_capacity(capacity);
        if let Some(rf) = reserve_fraction {
            validate_funnel_reserve_fraction(rf)?;
            inner.reserve_fraction = rf;
        }
        if let Some(limit) = primary_probe_limit {
            if limit == 0 {
                return Err(PyValueError::new_err(
                    "primary_probe_limit must be positive",
                ));
            }
            inner.primary_probe_limit = Some(limit);
        }
        Ok(Self { inner })
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity
    }

    #[getter]
    fn reserve_fraction(&self) -> f64 {
        self.inner.reserve_fraction
    }

    #[getter]
    fn primary_probe_limit(&self) -> Option<usize> {
        self.inner.primary_probe_limit
    }
}

#[pyclass(name = "ElasticHashMap", module = "opthash", unsendable)]
struct PyElasticHashMap {
    inner: ElasticHashMap<HashedAny, Py<PyAny>>,
    generation: u64,
}

impl PyElasticHashMap {
    #[inline]
    fn bump(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }
}

#[pymethods]
impl PyElasticHashMap {
    #[new]
    #[pyo3(signature = (other = None, *, capacity = 0))]
    fn new(other: Option<&Bound<'_, PyAny>>, capacity: usize) -> PyResult<Self> {
        let mut me = Self {
            inner: ElasticHashMap::with_capacity(capacity),
            generation: 0,
        };
        if let Some(other) = other {
            me.update(other)?;
        }
        Ok(me)
    }

    #[classmethod]
    fn with_options(_cls: &Bound<'_, PyType>, options: &PyElasticOptions) -> Self {
        Self {
            inner: ElasticHashMap::with_options(options.inner),
            generation: 0,
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let k = HashedAny::from_bound(key)?;
        Ok(self.inner.contains_key(&k))
    }

    fn __getitem__(&self, key: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let k = HashedAny::from_bound(key)?;
        match self.inner.get(&k) {
            Some(v) => Ok(v.clone_ref(py)),
            None => Err(PyKeyError::new_err(key.clone().unbind())),
        }
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let k = HashedAny::from_bound(key)?;
        self.inner.insert(k, value.clone().unbind());
        self.bump();
        Ok(())
    }

    fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let k = HashedAny::from_bound(key)?;
        match self.inner.remove(&k) {
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
        let k = HashedAny::from_bound(key)?;
        Ok(match self.inner.get(&k) {
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
            "ElasticHashMap(len={}, capacity={})",
            self.inner.len(),
            self.inner.capacity()
        )
    }

    fn __iter__(slf: Bound<'_, Self>) -> PyElasticKeyIter {
        let py = slf.py();
        let m = slf.borrow();
        let snapshot = m.inner.iter().map(|(k, _)| k.obj.clone_ref(py)).collect();
        let expected_gen = m.generation;
        drop(m);
        PyElasticKeyIter {
            map: slf.unbind(),
            snapshot,
            expected_gen,
            pos: 0,
        }
    }

    fn keys(slf: Bound<'_, Self>) -> PyElasticKeysView {
        PyElasticKeysView { map: slf.unbind() }
    }

    fn values(slf: Bound<'_, Self>) -> PyElasticValuesView {
        PyElasticValuesView { map: slf.unbind() }
    }

    fn items(slf: Bound<'_, Self>) -> PyElasticItemsView {
        PyElasticItemsView { map: slf.unbind() }
    }

    fn update(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(dict) = other.cast::<PyDict>() {
            for (k, v) in dict.iter() {
                let key = HashedAny::from_bound(&k)?;
                self.inner.insert(key, v.unbind());
            }
            self.bump();
            return Ok(());
        }
        if other.hasattr("keys")? {
            let keys = other.call_method0("keys")?;
            for k in keys.try_iter()? {
                let k = k?;
                let v = other.get_item(&k)?;
                let key = HashedAny::from_bound(&k)?;
                self.inner.insert(key, v.unbind());
            }
            self.bump();
            return Ok(());
        }
        for item in other.try_iter()? {
            let item = item?;
            let len = item
                .len()
                .map_err(|_| PyValueError::new_err("update sequence elements must be 2-tuples"))?;
            if len != 2 {
                return Err(PyValueError::new_err(
                    "update sequence elements must be 2-tuples",
                ));
            }
            let k = item.get_item(0)?;
            let v = item.get_item(1)?;
            let key = HashedAny::from_bound(&k)?;
            self.inner.insert(key, v.unbind());
        }
        self.bump();
        Ok(())
    }

    #[pyo3(signature = (key, default = None))]
    fn pop(&mut self, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let k = HashedAny::from_bound(key)?;
        match self.inner.remove(&k) {
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
        let k = HashedAny::from_bound(key)?;
        if let Some(v) = self.inner.get(&k) {
            return Ok(v.clone_ref(py));
        }
        let value = default.unwrap_or_else(|| py.None());
        self.inner.insert(k, value.clone_ref(py));
        self.bump();
        Ok(value)
    }

    fn copy(&self, py: Python<'_>) -> Self {
        let mut new = ElasticHashMap::with_capacity(self.inner.len());
        for (k, v) in &self.inner {
            new.insert(k.clone_with_py(py), v.clone_ref(py));
        }
        Self {
            inner: new,
            generation: 0,
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> bool {
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
        let mut new = self.copy(py);
        new.update(other)?;
        Ok(new)
    }
}

#[pyclass(name = "FunnelHashMap", module = "opthash", unsendable)]
struct PyFunnelHashMap {
    inner: FunnelHashMap<HashedAny, Py<PyAny>>,
    generation: u64,
}

impl PyFunnelHashMap {
    #[inline]
    fn bump(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }
}

#[pymethods]
impl PyFunnelHashMap {
    #[new]
    #[pyo3(signature = (other = None, *, capacity = 0))]
    fn new(other: Option<&Bound<'_, PyAny>>, capacity: usize) -> PyResult<Self> {
        let mut me = Self {
            inner: FunnelHashMap::with_capacity(capacity),
            generation: 0,
        };
        if let Some(other) = other {
            me.update(other)?;
        }
        Ok(me)
    }

    #[classmethod]
    fn with_options(_cls: &Bound<'_, PyType>, options: &PyFunnelOptions) -> Self {
        Self {
            inner: FunnelHashMap::with_options(options.inner),
            generation: 0,
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let k = HashedAny::from_bound(key)?;
        Ok(self.inner.contains_key(&k))
    }

    fn __getitem__(&self, key: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let k = HashedAny::from_bound(key)?;
        match self.inner.get(&k) {
            Some(v) => Ok(v.clone_ref(py)),
            None => Err(PyKeyError::new_err(key.clone().unbind())),
        }
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let k = HashedAny::from_bound(key)?;
        self.inner.insert(k, value.clone().unbind());
        self.bump();
        Ok(())
    }

    fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let k = HashedAny::from_bound(key)?;
        match self.inner.remove(&k) {
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
        let k = HashedAny::from_bound(key)?;
        Ok(match self.inner.get(&k) {
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
            "FunnelHashMap(len={}, capacity={})",
            self.inner.len(),
            self.inner.capacity()
        )
    }

    fn __iter__(slf: Bound<'_, Self>) -> PyFunnelKeyIter {
        let py = slf.py();
        let m = slf.borrow();
        let snapshot = m.inner.iter().map(|(k, _)| k.obj.clone_ref(py)).collect();
        let expected_gen = m.generation;
        drop(m);
        PyFunnelKeyIter {
            map: slf.unbind(),
            snapshot,
            expected_gen,
            pos: 0,
        }
    }

    fn keys(slf: Bound<'_, Self>) -> PyFunnelKeysView {
        PyFunnelKeysView { map: slf.unbind() }
    }

    fn values(slf: Bound<'_, Self>) -> PyFunnelValuesView {
        PyFunnelValuesView { map: slf.unbind() }
    }

    fn items(slf: Bound<'_, Self>) -> PyFunnelItemsView {
        PyFunnelItemsView { map: slf.unbind() }
    }

    fn update(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(dict) = other.cast::<PyDict>() {
            for (k, v) in dict.iter() {
                let key = HashedAny::from_bound(&k)?;
                self.inner.insert(key, v.unbind());
            }
            self.bump();
            return Ok(());
        }
        if other.hasattr("keys")? {
            let keys = other.call_method0("keys")?;
            for k in keys.try_iter()? {
                let k = k?;
                let v = other.get_item(&k)?;
                let key = HashedAny::from_bound(&k)?;
                self.inner.insert(key, v.unbind());
            }
            self.bump();
            return Ok(());
        }
        for item in other.try_iter()? {
            let item = item?;
            let len = item
                .len()
                .map_err(|_| PyValueError::new_err("update sequence elements must be 2-tuples"))?;
            if len != 2 {
                return Err(PyValueError::new_err(
                    "update sequence elements must be 2-tuples",
                ));
            }
            let k = item.get_item(0)?;
            let v = item.get_item(1)?;
            let key = HashedAny::from_bound(&k)?;
            self.inner.insert(key, v.unbind());
        }
        self.bump();
        Ok(())
    }

    #[pyo3(signature = (key, default = None))]
    fn pop(&mut self, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let k = HashedAny::from_bound(key)?;
        match self.inner.remove(&k) {
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
        let k = HashedAny::from_bound(key)?;
        if let Some(v) = self.inner.get(&k) {
            return Ok(v.clone_ref(py));
        }
        let value = default.unwrap_or_else(|| py.None());
        self.inner.insert(k, value.clone_ref(py));
        self.bump();
        Ok(value)
    }

    fn copy(&self, py: Python<'_>) -> Self {
        let mut new = FunnelHashMap::with_capacity(self.inner.len());
        for (k, v) in &self.inner {
            new.insert(k.clone_with_py(py), v.clone_ref(py));
        }
        Self {
            inner: new,
            generation: 0,
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> bool {
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
        let mut new = self.copy(py);
        new.update(other)?;
        Ok(new)
    }
}

// =============================================================================
// Elastic views + iterators
// =============================================================================

#[pyclass(name = "_ElasticKeyIter", module = "opthash", unsendable)]
struct PyElasticKeyIter {
    map: Py<PyElasticHashMap>,
    snapshot: Vec<Py<PyAny>>,
    expected_gen: u64,
    pos: usize,
}

#[pymethods]
impl PyElasticKeyIter {
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

#[pyclass(name = "_ElasticValueIter", module = "opthash", unsendable)]
struct PyElasticValueIter {
    map: Py<PyElasticHashMap>,
    snapshot: Vec<Py<PyAny>>,
    expected_gen: u64,
    pos: usize,
}

#[pymethods]
impl PyElasticValueIter {
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

#[pyclass(name = "_ElasticItemIter", module = "opthash", unsendable)]
struct PyElasticItemIter {
    map: Py<PyElasticHashMap>,
    snapshot: Vec<Py<PyAny>>,
    expected_gen: u64,
    pos: usize,
}

#[pymethods]
impl PyElasticItemIter {
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

#[pyclass(name = "elastic_keys", module = "opthash", unsendable)]
struct PyElasticKeysView {
    map: Py<PyElasticHashMap>,
}

#[pymethods]
impl PyElasticKeysView {
    fn __iter__(&self, py: Python<'_>) -> PyElasticKeyIter {
        let m = self.map.borrow(py);
        let snapshot = m.inner.iter().map(|(k, _)| k.obj.clone_ref(py)).collect();
        PyElasticKeyIter {
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
        let k = HashedAny::from_bound(key)?;
        Ok(m.inner.contains_key(&k))
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let m = self.map.borrow(py);
        let parts: PyResult<Vec<String>> = m
            .inner
            .iter()
            .map(|(k, _)| Ok(k.obj.bind(py).repr()?.to_string()))
            .collect();
        Ok(format!("elastic_keys([{}])", parts?.join(", ")))
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
            let h = HashedAny::from_bound(&item)?;
            if !m.inner.contains_key(&h) {
                result.add(item)?;
            }
        }
        Ok(result.unbind())
    }
}

#[pyclass(name = "elastic_values", module = "opthash", unsendable)]
struct PyElasticValuesView {
    map: Py<PyElasticHashMap>,
}

#[pymethods]
impl PyElasticValuesView {
    fn __iter__(&self, py: Python<'_>) -> PyElasticValueIter {
        let m = self.map.borrow(py);
        let snapshot = m.inner.iter().map(|(_, v)| v.clone_ref(py)).collect();
        PyElasticValueIter {
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
        Ok(format!("elastic_values([{}])", parts?.join(", ")))
    }
}

#[pyclass(name = "elastic_items", module = "opthash", unsendable)]
struct PyElasticItemsView {
    map: Py<PyElasticHashMap>,
}

#[pymethods]
impl PyElasticItemsView {
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyElasticItemIter> {
        let m = self.map.borrow(py);
        let snapshot: PyResult<Vec<Py<PyAny>>> = m
            .inner
            .iter()
            .map(|(k, v)| {
                let tup = PyTuple::new(py, [k.obj.clone_ref(py), v.clone_ref(py)])?;
                Ok(tup.into_any().unbind())
            })
            .collect();
        Ok(PyElasticItemIter {
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
        let h = HashedAny::from_bound(&k)?;
        match m.inner.get(&h) {
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
        Ok(format!("elastic_items([{}])", parts?.join(", ")))
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

// =============================================================================
// Funnel views + iterators (mirrors Elastic)
// =============================================================================

#[pyclass(name = "_FunnelKeyIter", module = "opthash", unsendable)]
struct PyFunnelKeyIter {
    map: Py<PyFunnelHashMap>,
    snapshot: Vec<Py<PyAny>>,
    expected_gen: u64,
    pos: usize,
}

#[pymethods]
impl PyFunnelKeyIter {
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

#[pyclass(name = "_FunnelValueIter", module = "opthash", unsendable)]
struct PyFunnelValueIter {
    map: Py<PyFunnelHashMap>,
    snapshot: Vec<Py<PyAny>>,
    expected_gen: u64,
    pos: usize,
}

#[pymethods]
impl PyFunnelValueIter {
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

#[pyclass(name = "_FunnelItemIter", module = "opthash", unsendable)]
struct PyFunnelItemIter {
    map: Py<PyFunnelHashMap>,
    snapshot: Vec<Py<PyAny>>,
    expected_gen: u64,
    pos: usize,
}

#[pymethods]
impl PyFunnelItemIter {
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

#[pyclass(name = "funnel_keys", module = "opthash", unsendable)]
struct PyFunnelKeysView {
    map: Py<PyFunnelHashMap>,
}

#[pymethods]
impl PyFunnelKeysView {
    fn __iter__(&self, py: Python<'_>) -> PyFunnelKeyIter {
        let m = self.map.borrow(py);
        let snapshot = m.inner.iter().map(|(k, _)| k.obj.clone_ref(py)).collect();
        PyFunnelKeyIter {
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
        let k = HashedAny::from_bound(key)?;
        Ok(m.inner.contains_key(&k))
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let m = self.map.borrow(py);
        let parts: PyResult<Vec<String>> = m
            .inner
            .iter()
            .map(|(k, _)| Ok(k.obj.bind(py).repr()?.to_string()))
            .collect();
        Ok(format!("funnel_keys([{}])", parts?.join(", ")))
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
            let h = HashedAny::from_bound(&item)?;
            if !m.inner.contains_key(&h) {
                result.add(item)?;
            }
        }
        Ok(result.unbind())
    }
}

#[pyclass(name = "funnel_values", module = "opthash", unsendable)]
struct PyFunnelValuesView {
    map: Py<PyFunnelHashMap>,
}

#[pymethods]
impl PyFunnelValuesView {
    fn __iter__(&self, py: Python<'_>) -> PyFunnelValueIter {
        let m = self.map.borrow(py);
        let snapshot = m.inner.iter().map(|(_, v)| v.clone_ref(py)).collect();
        PyFunnelValueIter {
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
        Ok(format!("funnel_values([{}])", parts?.join(", ")))
    }
}

#[pyclass(name = "funnel_items", module = "opthash", unsendable)]
struct PyFunnelItemsView {
    map: Py<PyFunnelHashMap>,
}

#[pymethods]
impl PyFunnelItemsView {
    fn __iter__(&self, py: Python<'_>) -> PyResult<PyFunnelItemIter> {
        let m = self.map.borrow(py);
        let snapshot: PyResult<Vec<Py<PyAny>>> = m
            .inner
            .iter()
            .map(|(k, v)| {
                let tup = PyTuple::new(py, [k.obj.clone_ref(py), v.clone_ref(py)])?;
                Ok(tup.into_any().unbind())
            })
            .collect();
        Ok(PyFunnelItemIter {
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
        let h = HashedAny::from_bound(&k)?;
        match m.inner.get(&h) {
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
        Ok(format!("funnel_items([{}])", parts?.join(", ")))
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

#[pymodule]
fn opthash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyElasticHashMap>()?;
    m.add_class::<PyFunnelHashMap>()?;
    m.add_class::<PyElasticOptions>()?;
    m.add_class::<PyFunnelOptions>()?;
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
