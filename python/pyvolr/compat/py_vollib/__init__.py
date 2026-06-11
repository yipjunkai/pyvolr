"""Drop-in compatibility shim for the abandoned `py_vollib` library.

Replace your imports:

    # Before
    from py_vollib.black_scholes import black_scholes
    from py_vollib.black_scholes.greeks.analytical import delta
    from py_vollib.black_scholes.implied_volatility import implied_volatility

    # After
    from pyvolr.compat.py_vollib.black_scholes import black_scholes
    from pyvolr.compat.py_vollib.black_scholes.greeks.analytical import delta
    from pyvolr.compat.py_vollib.black_scholes.implied_volatility import implied_volatility

Signatures, return types, the `'c'`/`'p'` flag convention, py_vollib's unit
conventions (per-day theta, per-1%-vol vega, per-1%-rate rho), and its
argument-order quirks (`flag` last in `implied_volatility`, `r` before `t` in
`black`) are all preserved. Values match py_vollib to f64 precision on
well-posed inputs.

py_vollib's *one* intentional error contract — `implied_volatility` raising
`BelowIntrinsicException` / `AboveMaximumException` on a price with no real
implied vol — is mirrored in `pyvolr.compat.py_vollib.exceptions` (the
originals live in `py_lets_be_rational.exceptions`, unimportable on Python
3.12+; re-point your `except` import there).

Every other py_vollib "raise" is a leaked implementation artifact, not a
contract, so pyvolr deliberately diverges by being more graceful:

    input                       py_vollib                     pyvolr.compat
    --------------------------  ----------------------------  -------------------------
    IV price below intrinsic    BelowIntrinsicException        BelowIntrinsicException
    IV price above maximum      AboveMaximumException          AboveMaximumException
    price/Greeks at t = 0       ZeroDivisionError /            clean intrinsic / limit
                                ValueError (NaN -> int)
    price/Greeks at sigma = 0   may warn / leak NaN            clean (discounted) intrinsic
    K = 0                       ZeroDivisionError              clean value (S*e^(-q*t))
    invalid flag (e.g. 'x')     KeyError                       ValueError with a message
    'C' / 'P' (uppercase)       KeyError (rejected)            accepted (case-insensitive)

For new code, prefer the modern `pyvolr.bs` API: numpy arrays, broadcasting,
all five Greeks in one call, per-unit conventions. Its `implied_vol` returns
NaN on a bound violation (vectorised semantics) rather than raising.
"""
