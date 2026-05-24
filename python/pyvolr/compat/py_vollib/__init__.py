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

Function signatures, return types, and `'c'`/`'p'` flag conventions match the
originals exactly. For new code, prefer the modern `pyvolr.bs` API.
"""
