# CommonFeatures.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ivanuricardo.github.io/CommonFeatures.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ivanuricardo.github.io/CommonFeatures.jl/dev/)
[![Build Status](https://github.com/ivanuricardo/CommonFeatures.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ivanuricardo/CommonFeatures.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ivanuricardo/CommonFeatures.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ivanuricardo/CommonFeatures.jl)

## Common Feature detection for Reduced Rank Time Series Models

Provides a suite of tensor methods for applied statistics and econometrics.
This includes reduced rank regressions as well as tensor regressions.
Additionally, the package provides tools for model selection and information criteria.
The bulk of the work is done for the purpose of detecting co-movements in multivariate time series models.

### Basics
- [x] Tensor Lags
- [x] Ridge Rank Select
- [x] Tucker Number of Parameters
- [x] Information Criteria
    - [x] BIC
    - [x] AIC

### Regressions
- [x] Tensor OLS
- [x] Tensor Autoregressive
- [x] Tucker Autoregressive

### Simulations
- [x] Simulate Matrix AR model
- [x] Simulate Tucker AR model

