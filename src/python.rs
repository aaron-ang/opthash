#![cfg(feature = "python")]

use std::hash::{Hash, Hasher};

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use crate::funnel::MAX_FUNNEL_RESERVE_FRACTION;
use crate::{ElasticHashMap, ElasticOptions, FunnelHashMap, FunnelOptions};

struct HashedAny {
    obj: Py<PyAny>,
    hash: isize,
}

impl HashedAny {
    fn from_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let hash = ob.hash()?;
        Ok(Self {
            obj: ob.clone().unbind(),
            hash,
        })
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
        Python::attach(|py| self.obj.bind(py).eq(other.obj.bind(py)).unwrap_or(false))
    }
}

impl Eq for HashedAny {}

#[pyclass(unsendable)]
struct PyKeyIter {
    keys: Vec<Py<PyAny>>,
    pos: usize,
}

#[pymethods]
impl PyKeyIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> Option<Py<PyAny>> {
        let v = self.keys.get(self.pos)?.clone_ref(py);
        self.pos += 1;
        Some(v)
    }
}

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
}

#[pymethods]
impl PyElasticHashMap {
    #[new]
    #[pyo3(signature = (capacity = 0))]
    fn new(capacity: usize) -> Self {
        Self {
            inner: ElasticHashMap::with_capacity(capacity),
        }
    }

    #[classmethod]
    fn with_options(_cls: &Bound<'_, PyType>, options: &PyElasticOptions) -> Self {
        Self {
            inner: ElasticHashMap::with_options(options.inner),
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
        Ok(())
    }

    fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let k = HashedAny::from_bound(key)?;
        match self.inner.remove(&k) {
            Some(_) => Ok(()),
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
    }

    fn __repr__(&self) -> String {
        format!(
            "ElasticHashMap(len={}, capacity={})",
            self.inner.len(),
            self.inner.capacity()
        )
    }

    fn __iter__(&self, py: Python<'_>) -> PyKeyIter {
        let keys = self
            .inner
            .iter()
            .map(|(k, _)| k.obj.clone_ref(py))
            .collect();
        PyKeyIter { keys, pos: 0 }
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Py<PyAny>> = self
            .inner
            .iter()
            .map(|(k, _)| k.obj.clone_ref(py))
            .collect();
        PyList::new(py, items)
    }

    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Py<PyAny>> = self.inner.iter().map(|(_, v)| v.clone_ref(py)).collect();
        PyList::new(py, items)
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: PyResult<Vec<Bound<'py, PyTuple>>> = self
            .inner
            .iter()
            .map(|(k, v)| PyTuple::new(py, [k.obj.clone_ref(py), v.clone_ref(py)]))
            .collect();
        PyList::new(py, items?)
    }

    fn update(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(dict) = other.cast::<PyDict>() {
            for (k, v) in dict.iter() {
                let key = HashedAny::from_bound(&k)?;
                self.inner.insert(key, v.unbind());
            }
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
            return Ok(());
        }
        for item in other.try_iter()? {
            let item = item?;
            let pair = item
                .cast::<PyTuple>()
                .map_err(|_| PyValueError::new_err("update sequence elements must be 2-tuples"))?;
            if pair.len() != 2 {
                return Err(PyValueError::new_err(
                    "update sequence elements must be 2-tuples",
                ));
            }
            let k = pair.get_item(0)?;
            let v = pair.get_item(1)?;
            let key = HashedAny::from_bound(&k)?;
            self.inner.insert(key, v.unbind());
        }
        Ok(())
    }

    #[pyo3(signature = (key, default = None))]
    fn pop(&mut self, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let k = HashedAny::from_bound(key)?;
        match self.inner.remove(&k) {
            Some(v) => Ok(v),
            None => default.ok_or_else(|| PyKeyError::new_err(key.clone().unbind())),
        }
    }

    fn popitem<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        if self.inner.is_empty() {
            return Err(PyKeyError::new_err("popitem(): map is empty"));
        }
        let (key_obj, key_hash) = {
            let (k, _) = self.inner.iter().next().expect("len > 0");
            (k.obj.clone_ref(py), k.hash)
        };
        let probe = HashedAny {
            obj: key_obj.clone_ref(py),
            hash: key_hash,
        };
        let value = self.inner.remove(&probe).expect("key from iter must exist");
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
        Ok(value)
    }

    fn copy(&self, py: Python<'_>) -> Self {
        let mut new = ElasticHashMap::with_capacity(self.inner.len());
        for (k, v) in &self.inner {
            let key = HashedAny {
                obj: k.obj.clone_ref(py),
                hash: k.hash,
            };
            new.insert(key, v.clone_ref(py));
        }
        Self { inner: new }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> bool {
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
}

#[pymethods]
impl PyFunnelHashMap {
    #[new]
    #[pyo3(signature = (capacity = 0))]
    fn new(capacity: usize) -> Self {
        Self {
            inner: FunnelHashMap::with_capacity(capacity),
        }
    }

    #[classmethod]
    fn with_options(_cls: &Bound<'_, PyType>, options: &PyFunnelOptions) -> Self {
        Self {
            inner: FunnelHashMap::with_options(options.inner),
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
        Ok(())
    }

    fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let k = HashedAny::from_bound(key)?;
        match self.inner.remove(&k) {
            Some(_) => Ok(()),
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
    }

    fn __repr__(&self) -> String {
        format!(
            "FunnelHashMap(len={}, capacity={})",
            self.inner.len(),
            self.inner.capacity()
        )
    }

    fn __iter__(&self, py: Python<'_>) -> PyKeyIter {
        let keys = self
            .inner
            .iter()
            .map(|(k, _)| k.obj.clone_ref(py))
            .collect();
        PyKeyIter { keys, pos: 0 }
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Py<PyAny>> = self
            .inner
            .iter()
            .map(|(k, _)| k.obj.clone_ref(py))
            .collect();
        PyList::new(py, items)
    }

    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Py<PyAny>> = self.inner.iter().map(|(_, v)| v.clone_ref(py)).collect();
        PyList::new(py, items)
    }

    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: PyResult<Vec<Bound<'py, PyTuple>>> = self
            .inner
            .iter()
            .map(|(k, v)| PyTuple::new(py, [k.obj.clone_ref(py), v.clone_ref(py)]))
            .collect();
        PyList::new(py, items?)
    }

    fn update(&mut self, other: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(dict) = other.cast::<PyDict>() {
            for (k, v) in dict.iter() {
                let key = HashedAny::from_bound(&k)?;
                self.inner.insert(key, v.unbind());
            }
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
            return Ok(());
        }
        for item in other.try_iter()? {
            let item = item?;
            let pair = item
                .cast::<PyTuple>()
                .map_err(|_| PyValueError::new_err("update sequence elements must be 2-tuples"))?;
            if pair.len() != 2 {
                return Err(PyValueError::new_err(
                    "update sequence elements must be 2-tuples",
                ));
            }
            let k = pair.get_item(0)?;
            let v = pair.get_item(1)?;
            let key = HashedAny::from_bound(&k)?;
            self.inner.insert(key, v.unbind());
        }
        Ok(())
    }

    #[pyo3(signature = (key, default = None))]
    fn pop(&mut self, key: &Bound<'_, PyAny>, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let k = HashedAny::from_bound(key)?;
        match self.inner.remove(&k) {
            Some(v) => Ok(v),
            None => default.ok_or_else(|| PyKeyError::new_err(key.clone().unbind())),
        }
    }

    fn popitem<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        if self.inner.is_empty() {
            return Err(PyKeyError::new_err("popitem(): map is empty"));
        }
        let (key_obj, key_hash) = {
            let (k, _) = self.inner.iter().next().expect("len > 0");
            (k.obj.clone_ref(py), k.hash)
        };
        let probe = HashedAny {
            obj: key_obj.clone_ref(py),
            hash: key_hash,
        };
        let value = self.inner.remove(&probe).expect("key from iter must exist");
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
        Ok(value)
    }

    fn copy(&self, py: Python<'_>) -> Self {
        let mut new = FunnelHashMap::with_capacity(self.inner.len());
        for (k, v) in &self.inner {
            let key = HashedAny {
                obj: k.obj.clone_ref(py),
                hash: k.hash,
            };
            new.insert(key, v.clone_ref(py));
        }
        Self { inner: new }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> bool {
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

#[pymodule]
fn opthash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyElasticHashMap>()?;
    m.add_class::<PyFunnelHashMap>()?;
    m.add_class::<PyElasticOptions>()?;
    m.add_class::<PyFunnelOptions>()?;
    m.add_class::<PyKeyIter>()?;
    Ok(())
}
