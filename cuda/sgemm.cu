#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

void gemm_cublas(float* A, float* B, float* C, int M, int N, int K, int numRuns) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warm-up run
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalElapsedTime = 0.0f;

    for (int i = 0; i < numRuns; ++i) {
        // Record the start event
        cudaEventRecord(start, 0);

        // Perform matrix multiplication: C = alpha * A * B + beta * C
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

        // Record the stop event
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Calculate the elapsed time
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        totalElapsedTime += elapsedTime;
    }

    // Calculate average elapsed time
    float averageElapsedTime = totalElapsedTime / numRuns;

    // Calculate GFlops
    float gflops = (2.0f * M * N * K) / (averageElapsedTime * 1e6f);
    float tflops = (2.0f * M * N * K) / (averageElapsedTime * 1e9f);
    std::cout << gflops << "GF" <<  std::endl;
    std::cout << tflops << "TF" << std::endl;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(handle);

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    // int numRuns = std::atoi(argv[4]);
    int numRuns = 1000;

    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    // Initialize matrices A and B with some values
    // ...

    gemm_cublas(A, B, C, M, N, K, numRuns);

    // Use the result in matrix C
    // ...

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}