#![cfg(feature = "python")]

use std::hash::{Hash, Hasher};

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

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

fn validate_reserve_fraction(value: f64) -> PyResult<()> {
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "reserve_fraction must be in the open interval (0, 1)",
        ))
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
            validate_reserve_fraction(rf)?;
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
            validate_reserve_fraction(rf)?;
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
}

#[pymodule]
fn opthash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyElasticHashMap>()?;
    m.add_class::<PyFunnelHashMap>()?;
    m.add_class::<PyElasticOptions>()?;
    m.add_class::<PyFunnelOptions>()?;
    Ok(())
}
