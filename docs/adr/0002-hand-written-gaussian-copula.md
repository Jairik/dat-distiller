# Hand-written Gaussian copula instead of SDV

The statistical and hybrid Generation Modes use a Gaussian copula implemented directly on numpy/scipy, not SDV or `copulas`. Both of those libraries are under the Business Source License, which is not an open-source license, and the part we need is small (marginal fitting, plus a correlation matrix in normal space). Don't replace it with SDV unless the licensing position changes.
