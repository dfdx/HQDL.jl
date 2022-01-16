| call | cpu_f64 | cpu_f32 | cpu_f16 | gpu_f64 | gpu_f32 | gpu_f16 | docs_ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| *(r(3, 4), r(4, 3)) | :heavy_check_mark: | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: | true |
| batched_mul(r(3, 4, 5), r(4, 3)) | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | :grey_question: | true |
| softmax(r(3, 4)) | :heavy_check_mark: | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: | true |
