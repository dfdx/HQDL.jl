High-Quality Deep Learning

Curated list of high-quality operators for machine learning. See [REPORT.md](REPORT.md) for the current state of the ecosystem.

You can also test a new function using the `@inspect` or `@analyze` macros:

```julia
import Pkg
Pkg.add("https://github.com/dfdx/HQDL.jl")

import HQDL
import NNlib

HQDL.@inspect NNlib.softmax(r(3,4))
```

> :warning: This is the very beginning of the project, so many false negative results are expected. On the other hand, even positive result in the report by itself doesn't guarantee the correctness of implementation and thus doesn't avoid necesserity of the usual unit tests.