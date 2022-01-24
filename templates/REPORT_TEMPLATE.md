Call specification:

* `r(...)` or `rand(...)` - random array of the specified size and tested precision
* `X`, `Y` - aliases to `r(3, 4)` and `r(4, 3)` respectively

Status meaning:

* :heavy_check_mark: - check passed
* :x: - there was an error during the check
* :grey_question: - status is unclear (e.g. there's no rrule for the op, but an AD system may still be able to handle it)

## Basic

{{ basic }}

## Activations

{{ activations }}