//! pyvolr Rust core.
//!
//! All PyO3-exposed functions operate on flat 1-D numpy arrays. The Python wrapper
//! is responsible for broadcasting and reshape; this layer assumes equal-length
//! contiguous f64 (and i8 for option-flag) inputs.

mod bsm;
mod greeks;
mod iv;
mod normal;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bsm::Flag;

/// Validate that all input slices share a length; otherwise return a Python
/// `ValueError`. This is defense-in-depth — the Python wrapper enforces this
/// upstream via numpy broadcasting.
fn check_len(lens: &[usize]) -> PyResult<usize> {
    let n = lens[0];
    for &l in &lens[1..] {
        if l != n {
            return Err(PyValueError::new_err(format!(
                "input arrays have mismatched lengths: {lens:?}"
            )));
        }
    }
    Ok(n)
}

macro_rules! define_price_or_greek {
    ($pyname:ident, $rustfn:path, with_flag) => {
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $pyname<'py>(
            py: Python<'py>,
            flag: PyReadonlyArray1<'py, i8>,
            s: PyReadonlyArray1<'py, f64>,
            k: PyReadonlyArray1<'py, f64>,
            t: PyReadonlyArray1<'py, f64>,
            r: PyReadonlyArray1<'py, f64>,
            q: PyReadonlyArray1<'py, f64>,
            sigma: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let flag = flag.as_slice()?;
            let s = s.as_slice()?;
            let k = k.as_slice()?;
            let t = t.as_slice()?;
            let r = r.as_slice()?;
            let q = q.as_slice()?;
            let sigma = sigma.as_slice()?;
            let n = check_len(&[
                flag.len(),
                s.len(),
                k.len(),
                t.len(),
                r.len(),
                q.len(),
                sigma.len(),
            ])?;
            let out: Vec<f64> = (0..n)
                .map(|i| {
                    $rustfn(
                        Flag::from_i8(flag[i]),
                        s[i],
                        k[i],
                        t[i],
                        r[i],
                        q[i],
                        sigma[i],
                    )
                })
                .collect();
            Ok(out.into_pyarray(py))
        }
    };
    ($pyname:ident, $rustfn:path, no_flag) => {
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $pyname<'py>(
            py: Python<'py>,
            s: PyReadonlyArray1<'py, f64>,
            k: PyReadonlyArray1<'py, f64>,
            t: PyReadonlyArray1<'py, f64>,
            r: PyReadonlyArray1<'py, f64>,
            q: PyReadonlyArray1<'py, f64>,
            sigma: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let s = s.as_slice()?;
            let k = k.as_slice()?;
            let t = t.as_slice()?;
            let r = r.as_slice()?;
            let q = q.as_slice()?;
            let sigma = sigma.as_slice()?;
            let n = check_len(&[s.len(), k.len(), t.len(), r.len(), q.len(), sigma.len()])?;
            let out: Vec<f64> = (0..n)
                .map(|i| $rustfn(s[i], k[i], t[i], r[i], q[i], sigma[i]))
                .collect();
            Ok(out.into_pyarray(py))
        }
    };
}

define_price_or_greek!(bsm_price, bsm::price, with_flag);
define_price_or_greek!(bsm_delta, greeks::delta, with_flag);
define_price_or_greek!(bsm_theta, greeks::theta, with_flag);
define_price_or_greek!(bsm_rho, greeks::rho, with_flag);
define_price_or_greek!(bsm_gamma, greeks::gamma, no_flag);
define_price_or_greek!(bsm_vega, greeks::vega, no_flag);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bsm_iv<'py>(
    py: Python<'py>,
    target_price: PyReadonlyArray1<'py, f64>,
    flag: PyReadonlyArray1<'py, i8>,
    s: PyReadonlyArray1<'py, f64>,
    k: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let target_price = target_price.as_slice()?;
    let flag = flag.as_slice()?;
    let s = s.as_slice()?;
    let k = k.as_slice()?;
    let t = t.as_slice()?;
    let r = r.as_slice()?;
    let q = q.as_slice()?;
    let n = check_len(&[
        target_price.len(),
        flag.len(),
        s.len(),
        k.len(),
        t.len(),
        r.len(),
        q.len(),
    ])?;
    let out: Vec<f64> = (0..n)
        .map(|i| {
            iv::solve(
                target_price[i],
                Flag::from_i8(flag[i]),
                s[i],
                k[i],
                t[i],
                r[i],
                q[i],
            )
        })
        .collect();
    Ok(out.into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__doc__",
        "pyvolr Rust core: BSM pricing, Greeks, implied volatility.",
    )?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(bsm_price, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_delta, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_vega, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_theta, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_rho, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_iv, m)?)?;
    Ok(())
}
