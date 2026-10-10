| Setting | Value |
| --- | --- |
| python | 3.12.4 |
| pytorch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| numpy | 1.26.4 |
| matplotlib | 3.11.0 |
| sklearn | 1.4.2 |
| operating_system | Windows-11-10.0.26200-SP0 |
| device | cpu |
| cuda_available | False |
| compiled_cuda | None |
| gpu_name | None |
| epochs | 10 |
| batch_size | 128 |
| learning_rate | 0.001 |
| optimizer | Adam |
| loss | CrossEntropyLoss |
| seed | 42 |
| mc_samples | 30 |
| dropout_convolution | 0.25 |
| dropout_fc | 0.5 |
| preprocessing | ToTensor only; pixels in [0,1] |
| cpu_threads | 4 |
| dataset | MNIST |
| train_samples | 60000 |
| test_samples | 10000 |
| best_epoch | 10 |

Pixels remain in [0,1]. Noise is added before clipping to [0,1]. All six noise levels use all 10,000 test images and 30 MC passes. A fixed noise generator is independent of dropout sampling, and the same underlying noise draws are scaled across levels. Only one seed and one noise realization per image are evaluated.
