# Windows Environment
## cuda
```
nvcc benchmark_sgemm.cu -lcublas
a.exe 1024 1024 8192
```

## sycl
```
icx-cl -fsycl benchmark_sgemm_sycl.cpp -Qmkl OpenCL.lib
a.exe 1024 1024 8192
```
