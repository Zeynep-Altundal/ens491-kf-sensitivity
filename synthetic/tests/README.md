# synthetic/tests

Pytest tests for the scenario generator: noise-free strapdown integration of the ideal IMU against
the ground truth (absolute 1 cm bound for `straight` and `turning`, second-order convergence for
`aggressive`), white-noise standard deviation against the configured noise density, seed
reproducibility, `.npz` round trip and independence of the error sources.
