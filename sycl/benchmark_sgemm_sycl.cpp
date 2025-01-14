#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>

void gemm_cublas(float *A, float *B, float *C, int M, int N, int K,
                 int numRuns) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    d_A = (float *)sycl::malloc_device(sizeA, q_ct1);
    d_B = (float *)sycl::malloc_device(sizeB, q_ct1);
    d_C = (float *)sycl::malloc_device(sizeC, q_ct1);

    q_ct1.memcpy(d_A, A, sizeA);
    q_ct1.memcpy(d_B, B, sizeB).wait();

    dpct::blas::descriptor_ptr handle;
    handle = new dpct::blas::descriptor();

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warm-up run
    oneapi::mkl::blas::column_major::gemm(
        handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, N, M, K, alpha, d_B, N, d_A, K, beta,
        d_C, N);
    dev_ct1.queues_wait_and_throw();

    // Create CUDA events for timing
    dpct::event_ptr start, stop;
    start = new sycl::event();
    stop = new sycl::event();

    float totalElapsedTime = 0.0f;

    for (int i = 0; i < numRuns; ++i) {
        // Record the start event
        dpct::sync_barrier(start, &q_ct1);

        // Perform matrix multiplication: C = alpha * A * B + beta * C
        oneapi::mkl::blas::column_major::gemm(
            handle->get_queue(), oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans, N, M, K, alpha, d_B, N, d_A, K,
            beta, d_C, N);

        // Record the stop event
        dpct::sync_barrier(stop, &q_ct1);
        stop->wait_and_throw();

        // Calculate the elapsed time
        float elapsedTime;
        elapsedTime = (stop->get_profiling_info<
                           sycl::info::event_profiling::command_end>() -
                       start->get_profiling_info<
                           sycl::info::event_profiling::command_start>()) /
                      1000000.0f;
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
    dpct::destroy_event(start);
    dpct::destroy_event(stop);

    delete (handle);

    q_ct1.memcpy(C, d_C, sizeC).wait();

    dpct::dpct_free(d_A, q_ct1);
    dpct::dpct_free(d_B, q_ct1);
    dpct::dpct_free(d_C, q_ct1);
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
