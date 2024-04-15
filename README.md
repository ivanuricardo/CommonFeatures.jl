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

## Structure

The most basic time series model is the Vector Autoregressive (VAR) model.
Unfortunately, VAR models are still plagued by the curse of dimensionality.
In order to remedy this, Reduced Rank Regressions have been proposed in order to create a more parsimonious model.

Reduced Rank Regressions assume that the autoregressive coefficient of the VAR is of reduced rank.
In this way, we can either detect co-movements or impose a factor structure.

### SCCF Restriction

We start with the VAR model with a reduced rank structure
$$y_t = A \sum_{j=1}^p B_j y_{t-j} + u_t$$
Premultiplying the equation by a left null space matrix $\mathbf{A}_\perp$ then removes the serial correlation feature and we are left with the white noise model
$$A_\perp^\top y_t = A_\perp^\top u_t$$
This nullspace matrix gives us important information regarding the co-movements in the system.
We can motivate this with an example.
Suppose we have a data generating process where two series $y_{1,t}$ and $y_{2,t}$ co-move with one another, and is generated by a VAR with one lag and a coefficient matrix
$$\begin{bmatrix}
    0.2 & 0.1 \\
    0.4 & 0.2\end{bmatrix}$$


