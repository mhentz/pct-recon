"""Test that the compiled dependencies are installed."""


def test_ccpi_proximal_solvers_import():
    from ccpi.filters.regularisers import FGP_TV, FGP_dTV

    assert callable(FGP_TV)
    assert callable(FGP_dTV)


def test_pct_tools_imports():
    from pct_tools.recon import gradient_descent, squared_residuals

    assert callable(gradient_descent)
    assert callable(squared_residuals)
