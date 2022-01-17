* :heavy_check_mark: - check passed
* :x: - there was an error during the check
* :grey_question: - status is unclear (e.g. there's no rrule for the op, but an AD system may still be able to handle it)


| call | invoke_ok | cpu_f64 | cpu_f32 | gpu_f64 | gpu_f32 | docs_ok |
| --- | --- | --- | --- | --- | --- | --- |
| +(r(3, 4), r(3, 4)) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| -(r(3, 4), r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| *(r(3, 4), r(4, 3)) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| /(r(3, 4), r(3, 4)) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| abs.(r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| acos.(r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| asin.(r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| atan.(r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| batched_mul(r(3, 4, 5), r(4, 3)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| cbrt.(r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| clamp.(r(3, 4), 0.2, 0.4) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| conv(r(10, 10, 3, 1), r(3, 3, 3, 6)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| exp.(r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| log.(r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| maximum(r(3, 4)) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| minimum(r(3, 4)) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| reduce(+, r(3, 4)) | :x: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| reshape(r(3, 4), 4, 3) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| reverse(r(5,)) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| sin.(r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| sqrt(r(3, 3)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| sqrt.(r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| tan.(r(3, 4)) | :heavy_check_mark: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :heavy_check_mark: |
| transpose(r(3, 4)) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| softmax(r(3, 4)) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
