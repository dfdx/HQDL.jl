| ex | arrtyp | eltyp | fwd_time | bwd_time | has_docstring |
| --- | --- | --- | --- | --- | --- |
| rand(128, 512) * rand(512, 64) | Array | Float64 | 5.9204e-5 | 0.00015304 | true |
| rand(128, 512) * rand(512, 64) | Array | Float32 | 3.4523e-5 | 7.7706e-5 | true |
| rand(128, 512) * rand(512, 64) | Array | Float16 | 0.018241482 | 0.036794532 | true |
| rand(128, 512) * rand(512, 64) | CuArray | Float64 | 1.472e-5 | 2.5534e-5 | true |
| rand(128, 512) * rand(512, 64) | CuArray | Float32 | 1.4545e-5 | 2.1011e-5 | true |
| rand(128, 512) * rand(512, 64) | CuArray | Float16 | 1.1528e-5 | 2.1288e-5 | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | Array | Float64 | 0.007861237 | no rrule | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | Array | Float32 | 0.004074859 | no rrule | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | Array | Float16 | 2.377835135 | no rrule | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | CuArray | Float64 | 6.5958000000000004e-6 | no rrule | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | CuArray | Float32 | 7.011e-6 | no rrule | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | CuArray | Float16 | 0.00056649 | no rrule | true |
| softmax(rand(512, 64)) | Array | Float64 | 0.000242196 | 4.2221e-5 | true |
| softmax(rand(512, 64)) | Array | Float32 | 0.000206443 | 2.9008e-5 | true |
| softmax(rand(512, 64)) | Array | Float16 | 0.000471519 | 0.000272722 | true |
| softmax(rand(512, 64)) | CuArray | Float64 | failed | 3.162e-6 | true |
| softmax(rand(512, 64)) | CuArray | Float32 | 3.590625e-6 | 3.157222222222222e-6 | true |
| softmax(rand(512, 64)) | CuArray | Float16 | 3.747375e-6 | 3.165875e-6 | true |
