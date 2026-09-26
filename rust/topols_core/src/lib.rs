//! TopoLS embedding core.
//!
//! A faithful port of `topols.routing` / `topols.embedding` / `topols.driver`:
//! the layer-by-layer MCTS embedding of a layered ZX diagram into a 3D grid.
//! Every decision is deterministic and follows the Python implementation
//! exactly (same iteration orders, same RNG, same work-unit budget), so a
//! compile returns the same result as the Python reference.

pub mod driver;
pub mod embedding;
pub mod geometry;
pub mod payload;
pub mod pyrandom;
pub mod routing;

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    /// Fallback blocks fetched from Python when the search needs them.
    struct PyBlocks(Py<PyAny>);
    impl crate::driver::FallbackSource for PyBlocks {
        fn block(&self, block: u16) -> crate::driver::FallbackBlock {
            Python::with_gil(|py| {
                let text: String = self.0.bind(py).call1((block,)).expect("fallback provider failed").extract().expect("fallback provider must return a JSON string");
                let v: serde_json::Value = serde_json::from_str(&text).expect("fallback block is not JSON");
                crate::payload::parse_block(&v)
            })
        }
    }

    /// Compile a layered circuit. `payload` is the JSON produced by
    /// `topols.engine.compile_payload`; `fallback` is called with a block
    /// index and returns `topols.engine.block_payload` as JSON when the
    /// gate-by-gate fallback needs that block. The result is the JSON that
    /// `topols.engine.run_rust` reads back.
    #[pyfunction]
    fn compile(py: Python<'_>, payload: &str, fallback: Py<PyAny>) -> PyResult<String> {
        let out = py.allow_threads(|| {
            let input = crate::payload::parse_input_with(payload, Some(Box::new(PyBlocks(fallback))));
            crate::payload::output_json(&crate::driver::operation(&input))
        });
        Ok(out)
    }

    #[pymodule]
    fn topols_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(compile, m)?)?;
        Ok(())
    }
}
