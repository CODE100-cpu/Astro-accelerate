#include <chrono>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <fstream>
#include <math_constants.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <utility>
#include <vector>

//#include "aa_host_rfi.hpp"
//#include "aa_params.hpp"

// Be aware the below code only applied for nchans or nsamp smaller than max
// grid x dimension, which is 2147483647 for 3090 and number of channels less
// than grid y dimension which is 65535 for 3090

namespace astroaccelerate {

static void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

static void CHECK_CURAND(curandStatus_t err) {
  if (err != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "CURAND Error: %d\n", err);
    exit(EXIT_FAILURE);
  }
}

static __global__ void fill_ones(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = 1.0f;
}

static __global__ void set_int_array(int *x, int n, int value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = value;
}

static __global__ void Curand_init(curandStatePhilox4_32_10_t *state,
                                   unsigned long long seed, int n) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < n) {
    curand_init(seed, tid, 0, &state[tid]);
  }
}
static __global__ void transpose_rowmajor_kernel(const float *__restrict__ in,
                                                 float *__restrict__ out, int m,
                                                 int n) {
  __shared__ float tile[32][33];

  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;

  if (x < n && y < m)
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];

  __syncthreads();

  int xt = blockIdx.y * 32 + threadIdx.x;
  int yt = blockIdx.x * 32 + threadIdx.y;

  if (xt < m && yt < n)
    out[yt * m + xt] = tile[threadIdx.x][threadIdx.y];
}

static float *transpose(float *d_in, int m, int n) {
  float *d_out = nullptr;
  cudaMalloc(&d_out, sizeof(float) * (size_t)m * (size_t)n);

  dim3 block(32, 32);
  dim3 grid((n + 31) / 32, (m + 31) / 32);
  transpose_rowmajor_kernel<<<grid, block>>>(d_in, d_out, m, n);
  cudaDeviceSynchronize();

  cudaFree(d_in);
  return d_out;
}

// dot product of double and int array, used to mask out finished rows
static __global__ void dot(double *input1, int *input2, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    input1[tid] = input1[tid] * input2[tid];
  }
}

static __global__ void BufferCopy(float *d_out, const unsigned short *d_in,
                                  int n, int m) {
  int tid_X = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_Y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_X < m && tid_Y < n) {
    d_out[tid_Y * (size_t)m + tid_X] = (float)d_in[tid_Y + (size_t)n * tid_X];
  }
}

// Calculate the sum, square sum, and count of active elements in each row.
// Make sure at least one warp is full

static __global__ void LocalStatistics(float *d_stage, int n, int m, int *mask,
                                       double *d_mean, double *d_var,
                                       int *d_count, int *finish, int offset) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x; // the column index
  int i = blockIdx.y * blockDim.y + offset;        // the row index
  int wid = threadIdx.x / 32;                      // warp ID
  int lane = threadIdx.x % 32;                     // lane ID within the warp

  double sum = 0.0, sum2 = 0.0;
  int cnt = 0;
  bool active = tid < m && i < n && *(finish + i) == 0;

  if (active) {

    // Adjust pointers for the current row

    d_stage = d_stage + i * (size_t)m;
    mask = mask + i * (size_t)m;
    d_mean = d_mean + i;
    d_var = d_var + i;
    d_count = d_count + i;
    finish = finish + i;

    // 1) compute local sums and sums of squares

    if (mask[tid]) {

      float v = d_stage[tid];
      sum += v;
      sum2 += double(v) * double(v);
      cnt++;
    }
  }
  __syncthreads();
  // 2) warp-level reduction, remember to put shuffle outside of guard to avoid
  // undefined behavior

  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(-1, sum, offset);
    sum2 += __shfl_down_sync(-1, sum2, offset);
    cnt += __shfl_down_sync(-1, cnt, offset);
  }

  __syncthreads();

  __shared__ double warp_sum[32], warp_sum2[32];
  __shared__ int warp_cnt[32];
  int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

  if (active && lane == 0) {
    warp_sum[wid] = sum;
    warp_sum2[wid] = sum2;
    warp_cnt[wid] = cnt;
  }

  __syncthreads();

  int block_start = blockIdx.x * blockDim.x;
  int valid = m - block_start;
  valid = (valid < 0) ? 0 : (valid > blockDim.x ? blockDim.x : valid);
  int warps_active = (valid + 31) >> 5;

  double bsum = 0.0, bsum2 = 0.0;
  int bcnt = 0;

  // 3) block-level reduction in first warp
  if (active && wid == 0 && lane < warps_active) {
    bsum = warp_sum[lane];
    bsum2 = warp_sum2[lane];
    bcnt = warp_cnt[lane];
  }

  __syncthreads();

  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    bsum += __shfl_down_sync(-1, bsum, offset);
    bsum2 += __shfl_down_sync(-1, bsum2, offset);
    bcnt += __shfl_down_sync(-1, bcnt, offset);
  }

  __syncthreads();

  if (active && wid == 0 && lane == 0) {
    atomicAdd(d_mean, bsum);
    atomicAdd(d_var, bsum2);
    atomicAdd(d_count, bcnt);
  }
}

// Turn sums into the mean and standard deviation
static __global__ void Calc(double *d_mean, double *d_var, int *count, int n) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  bool active = tid < n;
  if (active && count[tid] != 0) {
    d_mean[tid] /= count[tid];
    d_var[tid] = d_var[tid] / count[tid] - (d_mean[tid] * d_mean[tid]);
    d_var[tid] = sqrt(d_var[tid]);
  }
}

// Check if counter is zero or standard deviation is 0 ->  break.
// If mean and var have converged and no above termination condtition ->
// continue execute for last time
static __global__ void Termination(double *d_mean, double *d_var,
                                   double *old_mean, double *old_var, int *mask,
                                   int *count, int *finish, int n, int round,
                                   int *unfinish) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  bool active = tid < n && finish[tid] == 0;

  if (active) {

    if (count[tid] == 0) {
      finish[tid] = 1;
      mask[tid] = 0;
      printf("\nCount zero, Sample %d %d %.16lf %.16lf", tid, round,
             d_mean[tid], d_var[tid]);
      active = false;
    }

    double mean = d_mean[tid];
    double stdv = d_var[tid];
    if (stdv * 1000000.0 < 0.1) {
      printf("\nVariance zero, Sample %d %d %lf %.16lf", tid, round, mean,
             stdv);
      mask[tid] = 0;
      finish[tid] = 1;
      active = false;
    }
    double oldm = old_mean[tid];
    double oldv = old_var[tid];
    if (fabs(mean - oldm) < 1e-3 && fabs(stdv - oldv) < 1e-4 && round > 1) {
      finish[tid] = 1;
    }
    if (finish[tid] == 0) {
      atomicOr(unfinish, 1);
    }
  }

  if (active) {
    old_mean[tid] = d_mean[tid];
    old_var[tid] = d_var[tid];
  }
}

// Check if data is outlier and set mask.
static __global__ void SigmaClip(float *d_stage, int n, int m, int *mask1,
                                 int *mask2, double *d_mean, double *d_var,
                                 int *finish, float sigma_cut, int round,
                                 int flag, int offset) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int i = blockIdx.y * blockDim.y + offset;

  bool active = tid < m && i < n && (finish[i] == 0);

  if (active) {
    double mean = d_mean[i];
    double stdv = d_var[i];

    double val = (d_stage[tid + i * (size_t)m] - mean) / stdv;
    if (flag || mask1[i])
      mask2[tid + i * (size_t)m] = (fabs(val) < sigma_cut);
  }
}

// Normalize regular values or replaces masked values with random values from
// the random array
static __global__ void LocalReplace(float *d_stage, int n, int m, float *random,
                                    double *mean, double *var, int *mask,
                                    unsigned long long seed,
                                    curandStatePhilox4_32_10_t *state,
                                    int offset) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int i = blockIdx.y * blockDim.y + offset;
  bool active = tid < m && i < n;

  if (active) {
    if (mask[i])
      d_stage[tid + i * (size_t)m] =
          (d_stage[tid + i * (size_t)m] - mean[i]) / var[i];
    else {
      int perm_one = (int)(curand_uniform(&state[tid]) * m);
      d_stage[tid + i * (size_t)m] = random[(tid + perm_one) % m];
    }
  }
}

// Mask out the mean and var of rows that are completely masked
static __global__ void Mask(double *mean, double *var, int *mask, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = tid < n;
  if (active && mask[tid] == 0) {
    mean[tid] = 0.0;
    var[tid] = 1.0;
  }
}
// Calculate the entire global mean and variance, count etc.
// This is basically same as Local_Statistics, except *d_stage is double type

static __global__ void GlobalStatistics(double *d_stage, int n, int *mask,
                                        double *d_mean, double *d_var,
                                        int *d_count) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int wid = threadIdx.x / 32;  // warp ID
  int lane = threadIdx.x % 32; // lane ID within the warp
  double sum = 0.0, sum2 = 0.0;
  int cnt = 0;
  bool active = tid < n;

  // 1) compute local sums and sums of squares

  if (active && mask[tid]) {
    double v = d_stage[tid];
    sum += v;
    sum2 += double(v) * double(v);
    cnt++;
  }
  __syncthreads();

  // 2) warp-level reduction
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(-1, sum, offset);
    sum2 += __shfl_down_sync(-1, sum2, offset);
    cnt += __shfl_down_sync(-1, cnt, offset);
  }
  __syncthreads();

  // Shared memory to store per-warp results
  __shared__ double warp_sum[32], warp_sum2[32];
  __shared__ int warp_cnt[32];
  int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

  if (active && lane == 0) {
    warp_sum[wid] = sum;
    warp_sum2[wid] = sum2;
    warp_cnt[wid] = cnt;
  }
  __syncthreads();

  // 3) block-level reduction in first warp

  int block_start = blockIdx.x * blockDim.x;
  int valid = n - block_start;
  valid = (valid < 0) ? 0 : (valid > blockDim.x ? blockDim.x : valid);
  int warps_active = (valid + 31) >> 5;

  double bsum = 0.0, bsum2 = 0.0;
  int bcnt = 0;

  if (active && wid == 0 && lane < warps_active) {
    bsum = warp_sum[lane];
    bsum2 = warp_sum2[lane];
    bcnt = warp_cnt[lane];
  }
  __syncthreads();

  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    bsum += __shfl_down_sync(-1, bsum, offset);
    bsum2 += __shfl_down_sync(-1, bsum2, offset);
    bcnt += __shfl_down_sync(-1, bcnt, offset);
  }
  __syncthreads();

  if (active && wid == 0 && lane == 0) {
    atomicAdd(d_mean, bsum);
    atomicAdd(d_var, bsum2);
    atomicAdd(d_count, bcnt);
  }
}

// Mask outlier channels and check for convergence
static __global__ void
GlobalConverge(double *mean, double *var, double *old_mean_of_mean,
               double *old_var_of_mean, double *old_mean_of_var,
               double *old_var_of_var, double *mean_of_mean,
               double *var_of_mean, double *mean_of_var, double *var_of_var,
               int *mask, int n, float sigma_cut, int *finish) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  bool active = tid < n;

  if (active) {

    if (fabs(mean[tid] - *mean_of_mean) / *var_of_mean > sigma_cut ||
        fabs(var[tid] - *mean_of_var) / *var_of_var > sigma_cut)
      mask[tid] = 0;

    if (tid == 0) {
      if (fabs(*mean_of_mean - *old_mean_of_mean) < 0.001 &&
          fabs(*var_of_mean - *old_var_of_mean) < 0.001 &&
          fabs(*mean_of_var - *old_mean_of_var) < 0.001 &&
          fabs(*var_of_var - *old_var_of_var) < 0.001) {
        *finish = 1;
      }
      *old_mean_of_mean = *mean_of_mean;
      *old_var_of_mean = *var_of_mean;
      *old_mean_of_var = *mean_of_var;
      *old_var_of_var = *var_of_var;
    }
  }
}

// COompute Clipping COonstants  using the below 2 functions
static __global__ void Clipping1(double *clipping_constant, int *mask, int n) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < n)
    atomicAdd(clipping_constant, mask[tid]);
}

static __global__ void Clipping2(int n, double *clipping_constant) {
  *clipping_constant = (n - *clipping_constant) / n;
  *clipping_constant = sqrt(-2.0 * log(*clipping_constant * 2.506628275));
}

// Simliar to LocalReplace, but here we replace entire rows if the mean and var
// are both outliers with respect to entire channel.
// Also, remember to initialize
// curand states before calling both this function and LocalReplace

static __global__ void
GlobalReplace(float *d_stage, double *clipping_constant, double *mean,
              double *var, double *mean_of_mean, double *var_of_mean,
              double *mean_of_var, double *var_of_var, int *mask, int n, int m,
              float *random, unsigned long long seed,
              curandStatePhilox4_32_10_t *state, int offset) {
  int tid_X = threadIdx.x + blockDim.x * blockIdx.x;
  int tid_Y = threadIdx.y + blockDim.y * blockIdx.y + offset;
  bool active = tid_X < m && tid_Y < n;

  if (active) {
    double val = (mean[tid_Y] - *mean_of_mean) / *var_of_mean;
    double val2 = (var[tid_Y] - *mean_of_var) / *var_of_var;
    if (fabs(val) > *clipping_constant && fabs(val2) > *clipping_constant) {
      int perm_one = (int)((curand_uniform(&state[tid_Y]) * m));
      d_stage[tid_Y * (size_t)m + tid_X] = random[(tid_X + perm_one) % m];
    }
  }
}

// Finally, rescale all data
static __global__ void Scale(float *d_stage, int n, int m, float mean_rescale,
                             float var_rescale) {
  int tid_X = threadIdx.x + blockDim.x * blockIdx.x;
  int tid_Y = threadIdx.y + blockDim.y * blockIdx.y;
  if (tid_X < n && tid_Y < m) {
    d_stage[tid_X * (size_t)m + tid_Y] =
        (d_stage[tid_X * (size_t)m + tid_Y] * var_rescale) + mean_rescale;
  }
}

// Mian RFI function for local RFI mitigation

static void RFILocal(float *d_stage, int n, int m, double *d_mean,
                     double *d_var, int *d_mask1, int *d_mask2, float sigma_cut,
                     float *d_random, curandStatePhilox4_32_10_t *state,
                     int flag, int blocks_x, int threads_x, int blocks_y,
                     int threads_y, cublasHandle_t cublas_handle,
                     int grid_y_Max, unsigned long long seed) {
  int *finish, *unfinish, *count, *mask2;
  double *mean, *var, *old_mean, *old_var;

  checkCudaError(cudaMallocManaged((void **)&finish, n * sizeof(int)));
  checkCudaError(cudaMallocManaged((void **)&unfinish, sizeof(int)));
  checkCudaError(cudaMalloc((void **)&old_mean, n * sizeof(double)));
  checkCudaError(cudaMalloc((void **)&old_var, n * sizeof(double)));
  checkCudaError(cudaMalloc((void **)&mean, n * sizeof(double)));
  checkCudaError(cudaMalloc((void **)&var, n * sizeof(double)));
  checkCudaError(cudaMalloc((void **)&count, n * sizeof(int)));
  checkCudaError(cudaMalloc((void **)&mask2, n * m * sizeof(int)));

  checkCudaError(cudaMemset(finish, 0, n * sizeof(int)));
  checkCudaError(cudaMemset(old_mean, 0, n * sizeof(double)));
  checkCudaError(cudaMemset(old_var, 0, n * sizeof(double)));
  checkCudaError(cudaMemset(mean, 0, n * sizeof(double)));
  checkCudaError(cudaMemset(var, 0, n * sizeof(double)));
  checkCudaError(cudaMemset(count, 0, n * sizeof(int)));
  checkCudaError(cudaMemset(mask2, 0, n * m * sizeof(int)));
  *unfinish = 1;

  printf("\nblocks = %d, threads = %d\n", blocks_x, threads_x);
  set_int_array<<<blocks_x * n, threads_x>>>(mask2, n * m, 1);
  checkCudaError(cudaDeviceSynchronize());
  dim3 blockDim(threads_x, 1);

  int loop = n > grid_y_Max ? (n + grid_y_Max - 1) / grid_y_Max : 1;
  int grid_1 = n > grid_y_Max ? grid_y_Max : n;

  dim3 gridDim(blocks_x, grid_1);

  int round = 1;
  while (*unfinish == 1) {

    *unfinish = 0;
    dot<<<blocks_y, threads_y /*, 0, stream[i]*/>>>(mean, finish, n);

    dot<<<blocks_y, threads_y /*, 0, stream[i]*/>>>(var, finish, n);

    checkCudaError(cudaMemset(count, 0, n * sizeof(int)));

    for (int i = 0; i < loop; ++i) {
      LocalStatistics<<<gridDim, blockDim>>>(d_stage, n, m, mask2, mean, var,
                                             count, finish, i * grid_y_Max);
    }

    Calc<<<blocks_y, threads_y>>>(mean, var, count, n);
    Termination<<<blocks_y, threads_y>>>(mean, var, old_mean, old_var, d_mask1,
                                         count, finish, n, round, unfinish);

    for (int i = 0; i < loop; ++i) {
      SigmaClip<<<gridDim, blockDim>>>(d_stage, n, m, d_mask1, mask2, mean, var,
                                       finish, sigma_cut, round, flag,
                                       i * grid_y_Max);
    }

    round++;
    checkCudaError(cudaDeviceSynchronize());
    // printf("\nRound %d, unfinish = %d", round, unfinish);
  }

  Curand_init<<<blocks_x, threads_x>>>(state, seed, m);

  for (int i = 0; i < loop; ++i) {
    LocalReplace<<<gridDim, blockDim /*, 0, stream[i]*/>>>(
        d_stage, n, m, d_random, mean, var, d_mask1, seed, state,
        grid_y_Max * i);
  }
  Mask<<<blocks_x, threads_x>>>(mean, var, d_mask1, n);

  checkCudaError(
      cudaMemcpy(d_mean, mean, n * sizeof(double), cudaMemcpyDeviceToDevice));
  checkCudaError(
      cudaMemcpy(d_var, var, n * sizeof(double), cudaMemcpyDeviceToDevice));
  checkCudaError(cudaMemcpy(d_mask2, mask2 + (n - 1) * m, m * sizeof(int),
                            cudaMemcpyDeviceToDevice));

  checkCudaError(cudaDeviceSynchronize());
  checkCudaError(cudaFree(finish));
  checkCudaError(cudaFree(old_mean));
  checkCudaError(cudaFree(old_var));
  checkCudaError(cudaFree(mean));
  checkCudaError(cudaFree(var));
  checkCudaError(cudaFree(count));
  checkCudaError(cudaFree(mask2));
  checkCudaError(cudaDeviceSynchronize());
}

// Main RFI function for global RFI mitigation
static std::vector<double>
RFIGlobal(float *d_stage, int n, int m, float sigma_cut, double *mean,
          double *var, int *mask, float *random_one,
          curandStatePhilox4_32_10_t *state, int block_x, int thread_x,
          int block_y, int thread_y, unsigned long long seed, int grid_y_Max) {
  double *mean_rescale, *var_rescale, *clipping_constant;
  checkCudaError(cudaMalloc(&clipping_constant, sizeof(double)));
  checkCudaError(cudaMemset(clipping_constant, 0, sizeof(double)));
  checkCudaError(cudaMalloc(&mean_rescale, sizeof(double)));
  checkCudaError(cudaMalloc(&var_rescale, sizeof(double)));
  checkCudaError(cudaMemset(mean_rescale, 0, sizeof(double)));
  checkCudaError(cudaMemset(var_rescale, 0, sizeof(double)));

  // Find the mean and SD of the mean and SD...
  int *finish, *counter;
  checkCudaError(cudaMalloc(&counter, sizeof(int)));
  checkCudaError(cudaMemset(counter, 0, sizeof(int)));
  checkCudaError(cudaMallocManaged(&finish, sizeof(int)));
  checkCudaError(cudaMemset(finish, 0, sizeof(int)));

  double *mean_of_mean, *var_of_mean, *mean_of_var, *var_of_var;
  checkCudaError(cudaMalloc(&mean_of_mean, sizeof(double)));
  checkCudaError(cudaMalloc(&var_of_mean, sizeof(double)));
  checkCudaError(cudaMalloc(&mean_of_var, sizeof(double)));
  checkCudaError(cudaMalloc(&var_of_var, sizeof(double)));

  double *old_mean_of_mean, *old_var_of_mean, *old_mean_of_var, *old_var_of_var;
  checkCudaError(cudaMalloc(&old_mean_of_mean, sizeof(double)));
  checkCudaError(cudaMalloc(&old_var_of_mean, sizeof(double)));
  checkCudaError(cudaMalloc(&old_mean_of_var, sizeof(double)));
  checkCudaError(cudaMalloc(&old_var_of_var, sizeof(double)));
  checkCudaError(cudaMemset(old_mean_of_mean, 0, sizeof(double)));
  checkCudaError(cudaMemset(old_var_of_mean, 0, sizeof(double)));
  checkCudaError(cudaMemset(old_mean_of_var, 0, sizeof(double)));
  checkCudaError(cudaMemset(old_var_of_var, 0, sizeof(double)));

  set_int_array<<<block_x, thread_x>>>(mask, n, 1);
  Curand_init<<<block_x, thread_x>>>(state, seed, n);

  while (*finish == 0) {
    checkCudaError(cudaMemset(mean_of_mean, 0, sizeof(double)));
    checkCudaError(cudaMemset(var_of_mean, 0, sizeof(double)));
    checkCudaError(cudaMemset(mean_of_var, 0, sizeof(double)));
    checkCudaError(cudaMemset(var_of_var, 0, sizeof(double)));
    checkCudaError(cudaMemset(counter, 0, sizeof(int)));

    GlobalStatistics<<<block_x, thread_x>>>(mean, n, mask, mean_of_mean,
                                            var_of_mean, counter);
    Calc<<<1, 1>>>(mean_of_mean, var_of_mean, counter, 1);
    checkCudaError(cudaMemset(counter, 0, sizeof(int)));
    GlobalStatistics<<<block_x, thread_x>>>(var, n, mask, mean_of_var,
                                            var_of_var, counter);
    Calc<<<1, 1>>>(mean_of_var, var_of_var, counter, 1);

    GlobalConverge<<<block_x, thread_x>>>(
        mean, var, old_mean_of_mean, old_var_of_mean, old_mean_of_var,
        old_var_of_var, mean_of_mean, var_of_mean, mean_of_var, var_of_var,
        mask, n, sigma_cut, finish);
    checkCudaError(cudaDeviceSynchronize());
  }

  double h_mean_of_mean, h_var_of_mean, h_mean_of_var, h_var_of_var;
  checkCudaError(cudaMemcpy(&h_mean_of_mean, mean_of_mean, sizeof(double),
                            cudaMemcpyDeviceToHost));
  checkCudaError(cudaMemcpy(&h_var_of_mean, var_of_mean, sizeof(double),
                            cudaMemcpyDeviceToHost));
  checkCudaError(cudaMemcpy(&h_mean_of_var, mean_of_var, sizeof(double),
                            cudaMemcpyDeviceToHost));
  checkCudaError(cudaMemcpy(&h_var_of_var, var_of_var, sizeof(double),
                            cudaMemcpyDeviceToHost));
  checkCudaError(cudaDeviceSynchronize());

  std::vector<double> stats = {h_mean_of_mean, h_var_of_mean, h_mean_of_var,
                               h_var_of_var};

  printf("mean_of_mean = %lf\n", h_mean_of_mean);
  printf("var_of_mean  = %lf\n", h_var_of_mean);
  printf("mean_of_var  = %lf\n", h_mean_of_var);
  printf("var_of_var   = %lf\n", h_var_of_var);

  checkCudaError(cudaMemcpy(mean_rescale, mean_of_mean, sizeof(double),
                            cudaMemcpyDeviceToDevice));
  checkCudaError(cudaMemcpy(var_rescale, mean_of_var, sizeof(double),
                            cudaMemcpyDeviceToDevice));

  Clipping1<<<block_x, thread_x>>>(clipping_constant, mask, n);
  Clipping2<<<1, 1>>>(n, clipping_constant);
  dim3 block(thread_y, 1);

  int loop = n > grid_y_Max ? (n + grid_y_Max - 1) / grid_y_Max : 1;
  int grid_1 = n > grid_y_Max ? grid_y_Max : n;
  dim3 grid(block_y, grid_1);

  for (int i = 0; i < loop; ++i) {
    GlobalReplace<<<grid, block>>>(d_stage, clipping_constant, mean, var,
                                   mean_of_mean, var_of_mean, mean_of_var,
                                   var_of_var, mask, n, m, random_one, seed,
                                   state, i * grid_y_Max);
  }

  checkCudaError(cudaFree(mean_of_mean));
  checkCudaError(cudaFree(var_of_mean));
  checkCudaError(cudaFree(mean_of_var));
  checkCudaError(cudaFree(var_of_var));
  checkCudaError(cudaFree(old_mean_of_mean));
  checkCudaError(cudaFree(old_var_of_mean));
  checkCudaError(cudaFree(old_mean_of_var));
  checkCudaError(cudaFree(old_var_of_var));
  checkCudaError(cudaFree(mean_rescale));
  checkCudaError(cudaFree(var_rescale));
  checkCudaError(cudaFree(counter));
  checkCudaError(cudaFree(finish));
  checkCudaError(cudaFree(clipping_constant));
  return stats;
}

// The main RFI function called by external code

void rfi(int nsamp, int nchans, std::vector<unsigned short> &input_buffer) {

  int dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  printf("Using device %d: %s\n", dev, prop.name);
  int GridMAX_X = prop.maxGridSize[0];
  int GridMAX_Y = prop.maxGridSize[1];
  int GridMAX_Z = prop.maxGridSize[2];
  printf("Max grid size: %d x %d x %d\n", prop.maxGridSize[0],
         prop.maxGridSize[1], prop.maxGridSize[2]);

  int file_reducer = 1;
  float sigma_cut = 2.0f;
  size_t N = (size_t)nsamp * (size_t)nchans;

  int thread_chan = 1024;
  if (nsamp >= 1024)
    thread_chan = 1024;
  else {
    for (thread_chan = 1; thread_chan <= nsamp; thread_chan *= 2) {
    }
    thread_chan /= 2;
  }

  int block_chan =
      (nsamp + thread_chan - 1) / thread_chan; // parameters for covering nsamp

  int thread_spectra = 1024;
  if (nchans >= 1024)
    thread_spectra = 1024;
  else {
    for (thread_spectra = 1; thread_spectra <= nchans; thread_spectra *= 2) {
    }
    thread_spectra /= 2;
  }
  int block_spectra = (nchans + thread_spectra - 1) /
                      thread_spectra; // parameters for covering nchans

  float *stage = (float *)malloc(N * sizeof(float)), *dev_stage;
  unsigned short *dev_input_buffer;

  printf("Input data: %d samples, %d channels\n", nsamp, nchans);
  checkCudaError(cudaMalloc((void **)&dev_stage, N * sizeof(float)));
  checkCudaError(
      cudaMalloc((void **)&dev_input_buffer, N * sizeof(unsigned short)));

  checkCudaError(cudaMemcpy(dev_input_buffer, input_buffer.data(),
                            N * sizeof(unsigned short),
                            cudaMemcpyHostToDevice));

  printf("Copying data to device...\n");

  BufferCopy<<<dim3(block_chan, nchans), dim3(thread_chan, 1)>>>(
      dev_stage, dev_input_buffer, nchans, nsamp);
  checkCudaError(cudaDeviceSynchronize());

  printf("\nRFI Reduction: %d samples, %d channels", nsamp, nchans);

  // ~~~ RFI Correct ~~~ //
  float orig_mean = 0, orig_var = 0;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

  float *d_ones;
  cudaMalloc(&d_ones, N * sizeof(float));

  fill_ones<<<block_chan * nchans, thread_chan>>>(d_ones, N);

  cublasSdot(handle, N, dev_stage, 1, d_ones, 1, &orig_mean);

  cublasSdot(handle, N, dev_stage, 1, dev_stage, 1, &orig_var);

  orig_mean /= N;
  orig_var = orig_var / N - orig_mean * orig_mean;
  orig_var = sqrt(orig_var);

  // Random Vectors
  float *d_random_spectra_one, *d_random_spectra_two, *d_random_chan_one,
      *d_random_chan_two;

  checkCudaError(
      cudaMalloc((void **)&d_random_spectra_one, nchans * sizeof(float)));
  checkCudaError(
      cudaMalloc((void **)&d_random_spectra_two, nchans * sizeof(float)));
  checkCudaError(
      cudaMalloc((void **)&d_random_chan_one, nsamp * sizeof(float)));
  checkCudaError(
      cudaMalloc((void **)&d_random_chan_two, nsamp * sizeof(float)));

  curandGenerator_t gen;
  CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  curandStatePhilox4_32_10_t *d_states;
  cudaMalloc(&d_states, max(nsamp, nchans) * sizeof(*d_states));

  unsigned long long seed = (unsigned long long)12345;
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));

  CHECK_CURAND(curandGenerateNormal(gen, d_random_spectra_one, nchans, 0, 1));
  CHECK_CURAND(curandGenerateNormal(gen, d_random_spectra_two, nchans, 0, 1));
  CHECK_CURAND(curandGenerateNormal(gen, d_random_chan_one, nsamp, 0, 1));
  CHECK_CURAND(curandGenerateNormal(gen, d_random_chan_two, nsamp, 0, 1));

  // Allocate working arrays

  int *d_chan_mask;
  checkCudaError(cudaMalloc((void **)&d_chan_mask, nchans * sizeof(int)));
  set_int_array<<<block_chan, thread_chan>>>(d_chan_mask, nchans, 1);

  int *d_spectra_mask;
  checkCudaError(cudaMalloc((void **)&d_spectra_mask, nsamp * sizeof(int)));
  set_int_array<<<block_spectra, thread_spectra>>>(d_spectra_mask, nsamp, 1);

  double *d_chan_mean;
  checkCudaError(cudaMalloc((void **)&d_chan_mean, nchans * sizeof(double)));
  checkCudaError(cudaMemset(d_chan_mean, 0, nchans * sizeof(double)));

  double *d_chan_var;
  checkCudaError(cudaMalloc((void **)&d_chan_var, nchans * sizeof(double)));
  checkCudaError(cudaMemset(d_chan_var, 0, nchans * sizeof(double)));

  double *d_spectra_mean;
  checkCudaError(cudaMalloc((void **)&d_spectra_mean, nsamp * sizeof(double)));
  checkCudaError(cudaMemset(d_spectra_mean, 0, nsamp * sizeof(double)));

  double *d_spectra_var;
  checkCudaError(cudaMalloc((void **)&d_spectra_var, nsamp * sizeof(double)));
  checkCudaError(cudaMemset(d_spectra_var, 0, nsamp * sizeof(double)));

  // Find the BLN and try to flatten the input data per channel & spectra
  // (remove non-stationary component).

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  auto t0 = std::chrono::steady_clock::now();

  RFILocal(dev_stage, nchans, nsamp, d_chan_mean, d_chan_var, d_chan_mask,
           d_spectra_mask, sigma_cut, d_random_chan_one, d_states, 1,
           block_chan, thread_chan, block_spectra, thread_spectra,
           cublas_handle, GridMAX_Y, seed);

  auto t1 = std::chrono::steady_clock::now();
  auto gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::ofstream time_record_gpu("time_gpu.txt");
  time_record_gpu << "Time for per channel sigma clip: " << gpu_ms << "ms\n";

  /* double h_chan_mean[nchans], h_chan_var[nchans];
   checkCudaError(cudaMemcpy(h_chan_mean, d_chan_mean, nchans * sizeof(double),
                             cudaMemcpyDeviceToHost));
   checkCudaError(cudaMemcpy(h_chan_var, d_chan_var, nchans * sizeof(double),
                             cudaMemcpyDeviceToHost));
   std::ofstream mean_file("chan_mean_gpu.txt");
   std::ofstream var_file("chan_var_gpu.txt");
   for (int c = 0; c < nchans; c++) {
     mean_file << h_chan_mean[c] << "\n";
     var_file << h_chan_var[c] << "\n";
   }
   mean_file.close();
   var_file.close();

   float *h_stage = (float *)malloc(N * sizeof(float));
   checkCudaError(cudaMemcpy(h_stage, dev_stage, N * sizeof(float),
                             cudaMemcpyDeviceToHost));

   std::ofstream stage_file("stage_gpu.txt");
   for (int c = 0; c < nchans; c++) {
     for (int t = 0; t < (nsamp); t++) {
       stage_file << (h_stage[c * (size_t)nsamp + t]) << " ";
     }
     stage_file << "\n";
   }
   stage_file.close();*/

  t0 = std::chrono::steady_clock::now();

  dev_stage = transpose(dev_stage, nchans, nsamp);

  RFILocal(dev_stage, nsamp, nchans, d_spectra_mean, d_spectra_var,
           d_spectra_mask, d_chan_mask, sigma_cut, d_random_spectra_one,
           d_states, 0, block_spectra, thread_spectra, block_chan, thread_chan,
           cublas_handle, GridMAX_Y, seed);

  t1 = std::chrono::steady_clock::now();
  gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  time_record_gpu << "Time for per spectra sigma clip: " << gpu_ms << " ms\n";

  std::vector<double> h_spectra_mean(nsamp), h_spectra_var(nsamp);
  checkCudaError(cudaMemcpy(h_spectra_mean.data(), d_spectra_mean,
                            nsamp * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaError(cudaMemcpy(h_spectra_var.data(), d_spectra_var,
                            nsamp * sizeof(double), cudaMemcpyDeviceToHost));

  /*std::ofstream spectra_mean_file("spectra_mean_gpu.txt");
  std::ofstream spectra_var_file("spectra_var_gpu.txt");
  for (int c = 0; c < nsamp; c++) {
    spectra_mean_file << h_spectra_mean[c] << "\n";
    spectra_var_file << h_spectra_var[c] << "\n";
  }
  spectra_mean_file.close();
  spectra_var_file.close();*/

  t0 = std::chrono::steady_clock::now();

  dev_stage = transpose(dev_stage, nsamp, nchans);

  // Find the BLN and try to flatten the input data per channel & spectra for
  // global data (remove non-stationary component).

  std::vector<double> holder =
      RFIGlobal(dev_stage, nchans, nsamp, sigma_cut, d_chan_mean, d_chan_var,
                d_chan_mask, d_random_chan_two, d_states, block_spectra,
                thread_spectra, block_chan, thread_chan, seed, GridMAX_Y);

  t1 = std::chrono::steady_clock::now();
  gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  time_record_gpu << "Time for global channel sigma clip: " << gpu_ms
                  << " ms\n";

  double mean_rescale = holder[0], var_rescale = holder[2];

  t0 = std::chrono::steady_clock::now();

  RFIGlobal(dev_stage, nsamp, nchans, sigma_cut, d_spectra_mean, d_spectra_var,
            d_spectra_mask, d_random_spectra_two, d_states, block_chan,
            thread_chan, block_spectra, thread_spectra, seed, GridMAX_Y);

  t1 = std::chrono::steady_clock::now();
  gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  time_record_gpu << "Time for global spectra sigma clip: " << gpu_ms
                  << " ms\n";

  time_record_gpu.close();

  dim3 block(thread_chan, min(1024 / thread_chan, thread_spectra));
  dim3 grid(block_chan, block_spectra * thread_spectra / block.y);
  Scale<<<grid, block>>>(dev_stage, nchans, nsamp, mean_rescale, var_rescale);

  checkCudaError(cudaDeviceSynchronize());
  checkCudaError(
      cudaMemcpy(stage, dev_stage, N * sizeof(float), cudaMemcpyDeviceToHost));

  for (int c = 0; c < nchans; c++) {
    for (int t = 0; t < (nsamp); t++) {
      input_buffer[c + (size_t)nchans * t] =
          (unsigned char)(stage[c * (size_t)nsamp + t]);
    }
  }

  FILE *fp_mask = fopen("gpu_masked_chans.txt", "w+");
  for (int c = 0; c < nchans; c++) {
    for (int t = 0; t < (nsamp) / file_reducer; t++) {
      fprintf(fp_mask, "%d ", (unsigned char)((stage[c * (size_t)nsamp + t])));
    }

    fprintf(fp_mask, "\n");
  }
  fclose(fp_mask);

  printf("\n%lf %lf\n", mean_rescale / orig_mean, var_rescale / orig_var);

  free(stage);
  checkCudaError(cudaFree(d_chan_mask));
  checkCudaError(cudaFree(d_spectra_mask));
  checkCudaError(cudaFree(d_chan_mean));
  checkCudaError(cudaFree(d_chan_var));
  checkCudaError(cudaFree(d_spectra_mean));
  checkCudaError(cudaFree(d_spectra_var));
  checkCudaError(cudaFree(d_random_chan_one));
  checkCudaError(cudaFree(d_random_chan_two));
  checkCudaError(cudaFree(d_random_spectra_one));
  checkCudaError(cudaFree(d_random_spectra_two));
  checkCudaError(cudaFree(dev_stage));
  checkCudaError(cudaFree(d_states));
  cublasDestroy(handle);
  checkCudaError(cudaFree(d_ones));
  curandDestroyGenerator(gen);
  cublasDestroy(cublas_handle);
  checkCudaError(cudaFree(dev_input_buffer));

  cudaDeviceReset();
}
} // namespace astroaccelerate

using namespace astroaccelerate;
int main() {
  std::vector<unsigned short> input_buffer;
  std::ifstream infile("input.txt");
  int nsamp, nchans;
  infile >> nsamp >> nchans;
  unsigned short value;
  while (infile >> value) {
    input_buffer.push_back(value);
  }
  infile.close();
  rfi(nsamp, nchans, input_buffer);
  return 0;
}