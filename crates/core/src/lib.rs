//! pyvolr Rust core.
//!
//! All PyO3-exposed functions operate on flat 1-D numpy arrays. The Python wrapper
//! is responsible for broadcasting and reshape; this layer assumes equal-length
//! contiguous f64 (and i8 for option-flag) inputs.

// Modules are pub so the pyvolr-fuzz workspace (fuzz/) can drive them
// directly. The PyO3 surface below is the only consumer in normal builds.
pub mod black76;
pub mod bsm;
pub mod greeks;
pub mod iv;
pub mod normal;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::bsm::Flag;

/// Above this batch size, the IV solver dispatches per-row work to rayon's
/// global thread pool (after releasing the GIL). Below it, the call stays
/// serial — rayon's per-call setup overhead (~45 µs on Apple M4 Pro) only
/// amortises against per-row IV cost (~280 ns) above this threshold, with
/// a comfortable margin for run-to-run noise. Bench data: F4 experiment
/// in `crates/core/benches/pricing.rs`.
///
/// Set `RAYON_NUM_THREADS=1` in the environment to disable parallelism
/// entirely (useful when pyvolr is called from inside a caller-managed
/// thread pool that already saturates the cores).
const PARALLEL_THRESHOLD: usize = 1024;

/// Greeks have a much lower per-row cost (~19 ns for `greeks::all` vs
/// ~280 ns for `iv::solve`), so the rayon overhead amortises later.
/// Break-even is around N=2,600; threshold set comfortably above to
/// ensure parallel always wins by a meaningful margin at the gate.
const GREEKS_PARALLEL_THRESHOLD: usize = 4096;

/// Return type of `bsm_greeks` / `black76_greeks`: `(delta, gamma, vega, theta, rho)`.
type GreeksTuple<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

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

// Black-76 has no `q` parameter (forward, not spot). Otherwise identical
// macro shape to the BSM bindings above.
macro_rules! define_black76 {
    ($pyname:ident, $rustfn:path, with_flag) => {
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        fn $pyname<'py>(
            py: Python<'py>,
            flag: PyReadonlyArray1<'py, i8>,
            f: PyReadonlyArray1<'py, f64>,
            k: PyReadonlyArray1<'py, f64>,
            t: PyReadonlyArray1<'py, f64>,
            r: PyReadonlyArray1<'py, f64>,
            sigma: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let flag = flag.as_slice()?;
            let f = f.as_slice()?;
            let k = k.as_slice()?;
            let t = t.as_slice()?;
            let r = r.as_slice()?;
            let sigma = sigma.as_slice()?;
            let n = check_len(&[flag.len(), f.len(), k.len(), t.len(), r.len(), sigma.len()])?;
            let out: Vec<f64> = (0..n)
                .map(|i| $rustfn(Flag::from_i8(flag[i]), f[i], k[i], t[i], r[i], sigma[i]))
                .collect();
            Ok(out.into_pyarray(py))
        }
    };
    ($pyname:ident, $rustfn:path, no_flag) => {
        #[pyfunction]
        fn $pyname<'py>(
            py: Python<'py>,
            f: PyReadonlyArray1<'py, f64>,
            k: PyReadonlyArray1<'py, f64>,
            t: PyReadonlyArray1<'py, f64>,
            r: PyReadonlyArray1<'py, f64>,
            sigma: PyReadonlyArray1<'py, f64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let f = f.as_slice()?;
            let k = k.as_slice()?;
            let t = t.as_slice()?;
            let r = r.as_slice()?;
            let sigma = sigma.as_slice()?;
            let n = check_len(&[f.len(), k.len(), t.len(), r.len(), sigma.len()])?;
            let out: Vec<f64> = (0..n)
                .map(|i| $rustfn(f[i], k[i], t[i], r[i], sigma[i]))
                .collect();
            Ok(out.into_pyarray(py))
        }
    };
}

define_black76!(black76_price, black76::price, with_flag);
define_black76!(black76_delta, black76::delta, with_flag);
define_black76!(black76_theta, black76::theta, with_flag);
define_black76!(black76_rho, black76::rho, with_flag);
define_black76!(black76_gamma, black76::gamma, no_flag);
define_black76!(black76_vega, black76::vega, no_flag);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn black76_iv<'py>(
    py: Python<'py>,
    target_price: PyReadonlyArray1<'py, f64>,
    flag: PyReadonlyArray1<'py, i8>,
    f: PyReadonlyArray1<'py, f64>,
    k: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let target_price = target_price.as_slice()?;
    let flag = flag.as_slice()?;
    let f = f.as_slice()?;
    let k = k.as_slice()?;
    let t = t.as_slice()?;
    let r = r.as_slice()?;
    let n = check_len(&[
        target_price.len(),
        flag.len(),
        f.len(),
        k.len(),
        t.len(),
        r.len(),
    ])?;
    // Black-76 IV: reuse the BSM solver with q = r (the Black-76 specialization).
    let solve_at = |i: usize| {
        iv::solve(
            target_price[i],
            Flag::from_i8(flag[i]),
            f[i],
            k[i],
            t[i],
            r[i],
            r[i],
        )
    };
    let out: Vec<f64> = if n >= PARALLEL_THRESHOLD {
        py.detach(|| (0..n).into_par_iter().map(solve_at).collect())
    } else {
        (0..n).map(solve_at).collect()
    };
    Ok(out.into_pyarray(py))
}

/// Compute all five BSM Greeks in a single pass.
///
/// Returns `(delta, gamma, vega, theta, rho)` as a 5-tuple of `PyArray1<f64>`.
/// Equivalent to calling `bsm_delta`, `bsm_gamma`, `bsm_vega`, `bsm_theta`,
/// and `bsm_rho` separately, but shares the `d1_d2`, discount-factor, `cdf`,
/// and `pdf` evaluations — ~3× faster on the bundled `bs.greeks(...)` path
/// (measured at N=10k on Apple M4 Pro: 273 µs combined vs 568 µs individual).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bsm_greeks<'py>(
    py: Python<'py>,
    flag: PyReadonlyArray1<'py, i8>,
    s: PyReadonlyArray1<'py, f64>,
    k: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
    q: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
) -> PyResult<GreeksTuple<'py>> {
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
    let work = |i: usize| {
        greeks::all(
            Flag::from_i8(flag[i]),
            s[i],
            k[i],
            t[i],
            r[i],
            q[i],
            sigma[i],
        )
    };
    let mut delta = Vec::with_capacity(n);
    let mut gamma = Vec::with_capacity(n);
    let mut vega = Vec::with_capacity(n);
    let mut theta = Vec::with_capacity(n);
    let mut rho = Vec::with_capacity(n);
    if n >= GREEKS_PARALLEL_THRESHOLD {
        // Parallel path: collect a 5-tuple per row in rayon, then unzip
        // serially into the five output Vecs. The temp tuple Vec costs
        // 40 bytes/row; unzip is N memcpy-equivalents, negligible.
        let tuples: Vec<(f64, f64, f64, f64, f64)> =
            py.detach(|| (0..n).into_par_iter().map(work).collect());
        for (d, g, v, th, rh) in tuples {
            delta.push(d);
            gamma.push(g);
            vega.push(v);
            theta.push(th);
            rho.push(rh);
        }
    } else {
        // Serial path: write directly into the output Vecs, no temp alloc.
        for i in 0..n {
            let (d, g, v, th, rh) = work(i);
            delta.push(d);
            gamma.push(g);
            vega.push(v);
            theta.push(th);
            rho.push(rh);
        }
    }
    Ok((
        delta.into_pyarray(py),
        gamma.into_pyarray(py),
        vega.into_pyarray(py),
        theta.into_pyarray(py),
        rho.into_pyarray(py),
    ))
}

/// Compute all five Black-76 Greeks in a single pass. See `bsm_greeks`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn black76_greeks<'py>(
    py: Python<'py>,
    flag: PyReadonlyArray1<'py, i8>,
    f: PyReadonlyArray1<'py, f64>,
    k: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
) -> PyResult<GreeksTuple<'py>> {
    let flag = flag.as_slice()?;
    let f = f.as_slice()?;
    let k = k.as_slice()?;
    let t = t.as_slice()?;
    let r = r.as_slice()?;
    let sigma = sigma.as_slice()?;
    let n = check_len(&[flag.len(), f.len(), k.len(), t.len(), r.len(), sigma.len()])?;
    let work = |i: usize| black76::all(Flag::from_i8(flag[i]), f[i], k[i], t[i], r[i], sigma[i]);
    let mut delta = Vec::with_capacity(n);
    let mut gamma = Vec::with_capacity(n);
    let mut vega = Vec::with_capacity(n);
    let mut theta = Vec::with_capacity(n);
    let mut rho = Vec::with_capacity(n);
    if n >= GREEKS_PARALLEL_THRESHOLD {
        let tuples: Vec<(f64, f64, f64, f64, f64)> =
            py.detach(|| (0..n).into_par_iter().map(work).collect());
        for (d, g, v, th, rh) in tuples {
            delta.push(d);
            gamma.push(g);
            vega.push(v);
            theta.push(th);
            rho.push(rh);
        }
    } else {
        for i in 0..n {
            let (d, g, v, th, rh) = work(i);
            delta.push(d);
            gamma.push(g);
            vega.push(v);
            theta.push(th);
            rho.push(rh);
        }
    }
    Ok((
        delta.into_pyarray(py),
        gamma.into_pyarray(py),
        vega.into_pyarray(py),
        theta.into_pyarray(py),
        rho.into_pyarray(py),
    ))
}

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
    let solve_at = |i: usize| {
        iv::solve(
            target_price[i],
            Flag::from_i8(flag[i]),
            s[i],
            k[i],
            t[i],
            r[i],
            q[i],
        )
    };
    let out: Vec<f64> = if n >= PARALLEL_THRESHOLD {
        py.detach(|| (0..n).into_par_iter().map(solve_at).collect())
    } else {
        (0..n).map(solve_at).collect()
    };
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
    m.add_function(wrap_pyfunction!(bsm_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(bsm_iv, m)?)?;
    m.add_function(wrap_pyfunction!(black76_price, m)?)?;
    m.add_function(wrap_pyfunction!(black76_delta, m)?)?;
    m.add_function(wrap_pyfunction!(black76_gamma, m)?)?;
    m.add_function(wrap_pyfunction!(black76_vega, m)?)?;
    m.add_function(wrap_pyfunction!(black76_theta, m)?)?;
    m.add_function(wrap_pyfunction!(black76_rho, m)?)?;
    m.add_function(wrap_pyfunction!(black76_greeks, m)?)?;
    m.add_function(wrap_pyfunction!(black76_iv, m)?)?;
    Ok(())
}
