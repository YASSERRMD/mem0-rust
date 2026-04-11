#![cfg(feature = "python")]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::runtime::Runtime;

use crate::{AddOptions, Memory, MemoryConfig, SearchOptions};

fn to_py_err(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Python-facing Memory wrapper.
#[pyclass(unsendable)]
pub struct PyMemory {
    inner: Memory,
    runtime: Runtime,
}

#[pymethods]
impl PyMemory {
    #[new]
    #[pyo3(signature = (collection_name=None))]
    pub fn new(collection_name: Option<String>) -> PyResult<Self> {
        let runtime = Runtime::new().map_err(to_py_err)?;
        let mut config = MemoryConfig::default();

        if let Some(name) = collection_name {
            config.collection_name = name;
        }

        let inner = runtime.block_on(Memory::new(config)).map_err(to_py_err)?;

        Ok(Self { inner, runtime })
    }

    #[pyo3(signature = (text, user_id=None, agent_id=None, run_id=None, infer=false))]
    pub fn add(
        &self,
        text: String,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        infer: bool,
    ) -> PyResult<String> {
        let options = AddOptions {
            user_id,
            agent_id,
            run_id,
            infer,
            ..Default::default()
        };

        let result = self.runtime.block_on(self.inner.add(text, options)).map_err(to_py_err)?;
        serde_json::to_string(&result).map_err(to_py_err)
    }

    #[pyo3(signature = (query, user_id=None, agent_id=None, run_id=None, limit=10, threshold=0.0))]
    pub fn search(
        &self,
        query: String,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
        limit: usize,
        threshold: f32,
    ) -> PyResult<String> {
        let options = SearchOptions {
            user_id,
            agent_id,
            run_id,
            limit: Some(limit),
            threshold: Some(threshold),
            ..Default::default()
        };

        let result = self
            .runtime
            .block_on(self.inner.search(&query, options))
            .map_err(to_py_err)?;

        serde_json::to_string(&result).map_err(to_py_err)
    }
}

#[pyfunction]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn mem0_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemory>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
