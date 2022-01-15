| ex | arrtyp | eltyp | fwd_time | bwd_time | has_docstring |
| --- | --- | --- | --- | --- | --- |
| rand(128, 512) * rand(512, 64) | Array | Float64 | 5.8354e-5 | 0.000140514 | true |
| rand(128, 512) * rand(512, 64) | Array | Float32 | 3.4248e-5 | 7.3642e-5 | true |
| rand(128, 512) * rand(512, 64) | Array | Float16 | 0.018110031 | 0.036814182 | true |
| rand(128, 512) * rand(512, 64) | CuArray | Float64 | 1.3727e-5 | 2.4505e-5 | true |
| rand(128, 512) * rand(512, 64) | CuArray | Float32 | 1.4793e-5 | 2.0952e-5 | true |
| rand(128, 512) * rand(512, 64) | CuArray | Float16 | 1.1301e-5 | 2.1346e-5 | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | Array | Float64 | 0.008057484 | no rrule | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | Array | Float32 | 0.004196887 | no rrule | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | Array | Float16 | 2.364161332 | no rrule | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | CuArray | Float64 | 6.5261999999999995e-6 | no rrule | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | CuArray | Float32 | 6.9742e-6 | no rrule | true |
| batched_mul(rand(128, 512, 64), rand(512, 128)) | CuArray | Float16 | 0.000568066 | no rrule | true |
| softmax(rand(512, 64)) | Array | Float64 | 0.000234354 | 4.0755e-5 | true |
| softmax(rand(512, 64)) | Array | Float32 | 0.000209038 | 2.8946e-5 | true |
| softmax(rand(512, 64)) | Array | Float16 | 0.00047147 | 0.000272731 | true |
| softmax(rand(512, 64)) | CuArray | Float64 | 3.6875e-6 | failed | true |
| softmax(rand(512, 64)) | CuArray | Float32 | 3.74975e-6 | failed | true |
| softmax(rand(512, 64)) | CuArray | Float16 | 3.726625e-6 | failed | true |
