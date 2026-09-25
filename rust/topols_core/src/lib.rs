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

    /// Compile a layered circuit. `payload` is the JSON produced by
    /// `topols.engine.compile_payload`; the result is the JSON that
    /// `topols.engine.run_rust` reads back.
    #[pyfunction]
    fn compile(py: Python<'_>, payload: &str) -> PyResult<String> {
        let out = py.allow_threads(|| {
            let input = crate::payload::parse_input(payload);
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
