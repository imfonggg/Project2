#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <sys/time.h>
#include <cuda_runtime.h>
 
 
#define BOX_SIZE	23000

#define FLOAT_MARGIN	0.1f

#define DEFAULT_SMEM_BYTES	(48 * 1024)
#define COPIES_SMEM_BYTES	(16 * 1024)
#define MAX_HIST_COPIES		16

#define SHOW_EXTRA_TIMING	1
 

typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;
 
typedef struct hist_entry{
	long long d_cnt;
} bucket;
 
bucket * histogram;
long long	PDH_acnt;
int num_buckets;
double   PDH_res;
atom * atom_list;
 
struct timezone Idunno;
struct timeval startTime, endTime;
 
#define CUDA_CHECK(call)                                                      \
	do {                                                                      \
		cudaError_t err_ = (call);                                            \
		if (err_ != cudaSuccess) {                                            \
			fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
			        cudaGetErrorString(err_));                                \
			exit(EXIT_FAILURE);                                               \
		}                                                                     \
	} while (0)
 
 
double p2p_distance(int ind1, int ind2) {
 
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
 
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}
 
 
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
 
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		}
	}
	return 0;
}
 
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld s\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}
 
 
void output_histogram(bucket *b){
	int i;
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0)
			printf("\n%02d: ", i);
		printf("%15lld ", b[i].d_cnt);
		total_cnt += b[i].d_cnt;
		if(i == num_buckets - 1)
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}
 
bool compare_histograms(bucket *cpu_hist, bucket *gpu_hist) {
    bool match = true;
    printf("\nDifference between CPU and GPU histograms:\n");
    for (int i = 0; i < num_buckets; i++) {
        long long diff = cpu_hist[i].d_cnt - gpu_hist[i].d_cnt;
        if (diff != 0) match = false;
        if (i % 5 == 0) printf("\n%02d: ", i);
        printf("%15lld ", diff);
        if (i != num_buckets - 1) printf("| ");
    }
    printf("\n");
    return match;
}
 
 
/* Exact bucket of a pair, computed in double precision with exactly the
   same operations and rounding as the CPU baseline. */
__device__ __forceinline__ int exact_bucket(double x1, double y1, double z1,
                                            double x2, double y2, double z2,
                                            double w)
{
    double dx = __dsub_rn(x1, x2);
    double dy = __dsub_rn(y1, y2);
    double dz = __dsub_rn(z1, z2);
    double s = __dadd_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)), __dmul_rn(dz, dz));
    return (int)__ddiv_rn(__dsqrt_rn(s), w);
}
 
__device__ __forceinline__ int pair_bucket(float xi, float yi, float zi,
                                           float xj, float yj, float zj,
                                           int i, int j,
                                           const double * __restrict__ dX,
                                           const double * __restrict__ dY,
                                           const double * __restrict__ dZ,
                                           double w, float inv_w)
{
    float ax = xi - xj;
    float ay = yi - yj;
    float az = zi - zj;
    float d = sqrtf(ax * ax + ay * ay + az * az);
    int lo = (int)((d - FLOAT_MARGIN) * inv_w);
    int hi = (int)((d + FLOAT_MARGIN) * inv_w);
    if (lo == hi)
        return lo;
    return exact_bucket(dX[i], dY[i], dZ[i], dX[j], dY[j], dZ[j], w);
}
 
__device__ __forceinline__ void add_count(int h, unsigned int *my_hist,
                                          int win_lo, int win_size,
                                          unsigned long long *g_hist, int nb)
{
    if (h >= nb) h = nb - 1;   /* guard: 1.732 < sqrt(3) in num_buckets */
    int k = h - win_lo;
    if ((unsigned int)k < (unsigned int)win_size)
        atomicAdd(&my_hist[k], 1u);
    else
        atomicAdd(&g_hist[h], 1ULL);
}
 
__global__ void __launch_bounds__(1024) sdh_tiled_kernel(const float * __restrict__ X,
                                 const float * __restrict__ Y,
                                 const float * __restrict__ Z,
                                 const double * __restrict__ dX,
                                 const double * __restrict__ dY,
                                 const double * __restrict__ dZ,
                                 int N, int M, double w, float inv_w,
                                 int nb, int win_lo, int win_size, int n_copies,
                                 int d_begin, int d_end, int do_intra,
                                 unsigned long long *g_hist)
{
    extern __shared__ float smem[];
    const int B = blockDim.x;
    const int t = threadIdx.x;
    float *sx = smem;
    float *sy = sx + B;
    float *sz = sy + B;
    unsigned int *s_hist = (unsigned int *)(sz + B);
 
    for (int k = t; k < win_size * n_copies; k += B)
        s_hist[k] = 0;
    unsigned int *my_hist = s_hist + ((t >> 5) % n_copies) * win_size;
 
    const int bi = blockIdx.x;
    const int i = bi * B + t;
    const bool valid = i < N;
    float xi = 0.0f, yi = 0.0f, zi = 0.0f;
    if (valid) {
        xi = X[i];
        yi = Y[i];
        zi = Z[i];
    }
 
    if (do_intra) {
        const int cnt = min(B, N - bi * B);
        if (t < cnt) {
            sx[t] = xi;
            sy[t] = yi;
            sz[t] = zi;
        }
        __syncthreads();
        if (t < cnt) {
            const int half = (cnt - 1) / 2;
            for (int k = 1; k <= half; k++) {
                int jt = t + k;
                if (jt >= cnt) jt -= cnt;
                int h = pair_bucket(xi, yi, zi, sx[jt], sy[jt], sz[jt],
                                    i, bi * B + jt, dX, dY, dZ, w, inv_w);
                add_count(h, my_hist, win_lo, win_size, g_hist, nb);
            }
            if ((cnt & 1) == 0 && t < cnt / 2) {
                int jt = t + cnt / 2;
                int h = pair_bucket(xi, yi, zi, sx[jt], sy[jt], sz[jt],
                                    i, bi * B + jt, dX, dY, dZ, w, inv_w);
                add_count(h, my_hist, win_lo, win_size, g_hist, nb);
            }
        }
    }
 
    for (int d = d_begin; d < d_end; d++) {
        int bj = bi + d;
        if (bj >= M) bj -= M;
        if (2 * d == M && bi > bj)
            continue;
 
        const int j0 = bj * B;
        const int cnt = min(B, N - j0);
        __syncthreads();
        if (t < cnt) {
            sx[t] = X[j0 + t];
            sy[t] = Y[j0 + t];
            sz[t] = Z[j0 + t];
        }
        __syncthreads();
 
        if (valid) {
            #pragma unroll 4
            for (int k = 0; k < cnt; k++) {
                int h = pair_bucket(xi, yi, zi, sx[k], sy[k], sz[k],
                                    i, j0 + k, dX, dY, dZ, w, inv_w);
                add_count(h, my_hist, win_lo, win_size, g_hist, nb);
            }
        }
    }
 
    __syncthreads();
    for (int k = t; k < win_size; k += B) {
        unsigned long long sum = 0;
        for (int c = 0; c < n_copies; c++)
            sum += s_hist[c * win_size + k];
        if (sum != 0)
            atomicAdd(&g_hist[win_lo + k], sum);
    }
}
 
 
static void usage(const char *prog)
{
    fprintf(stderr, "Usage: %s {#of_samples} {bucket_width} {block_size}\n", prog);
    fprintf(stderr, "  #of_samples  : positive integer, number of points\n");
    fprintf(stderr, "  bucket_width : positive number, width of each histogram bucket\n");
    fprintf(stderr, "  block_size   : positive integer, CUDA threads per block\n");
}
 
static bool parse_long(const char *s, long long *out)
{
    char *end;
    errno = 0;
    long long v = strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0')
        return false;
    *out = v;
    return true;
}
 
static bool parse_double(const char *s, double *out)
{
    char *end;
    errno = 0;
    double v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(v))
        return false;
    *out = v;
    return true;
}
 
 
int main(int argc, char **argv)
{
	int i;
 
	if (argc != 4) {
		fprintf(stderr, "Error: expected 3 arguments, got %d.\n", argc - 1);
		usage(argv[0]);
		return EXIT_FAILURE;
	}
 
	long long n_arg, bs_arg;
	double w_arg;
	if (!parse_long(argv[1], &n_arg) || n_arg <= 0 || n_arg > INT_MAX) {
		fprintf(stderr, "Error: #of_samples must be a positive integer (got \"%s\").\n", argv[1]);
		usage(argv[0]);
		return EXIT_FAILURE;
	}
	if (!parse_double(argv[2], &w_arg) || w_arg <= 0.0) {
		fprintf(stderr, "Error: bucket_width must be a positive number (got \"%s\").\n", argv[2]);
		usage(argv[0]);
		return EXIT_FAILURE;
	}
	if (!parse_long(argv[3], &bs_arg) || bs_arg <= 0) {
		fprintf(stderr, "Error: block_size must be a positive integer (got \"%s\").\n", argv[3]);
		usage(argv[0]);
		return EXIT_FAILURE;
	}
 
	int dev;
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDevice(&dev));
	CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
	if (bs_arg > prop.maxThreadsPerBlock) {
		fprintf(stderr, "Error: block_size must be at most %d on this GPU (got %lld).\n",
		        prop.maxThreadsPerBlock, bs_arg);
		return EXIT_FAILURE;
	}
 
	double nb_d = floor(BOX_SIZE * 1.732 / w_arg) + 1;
	if (nb_d > 200000000.0) {
		fprintf(stderr, "Error: bucket_width %s is too small (%.0f buckets).\n", argv[2], nb_d);
		return EXIT_FAILURE;
	}
 
	PDH_acnt = n_arg;
	PDH_res	 = w_arg;
	int block_size = (int)bs_arg;
 
	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)calloc(num_buckets, sizeof(bucket));
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);
	if (histogram == NULL || atom_list == NULL) {
		fprintf(stderr, "Error: out of host memory.\n");
		return EXIT_FAILURE;
	}
 
	srand(1);
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
 
#if SHOW_EXTRA_TIMING
	gettimeofday(&startTime, &Idunno);
#endif
 
	PDH_baseline();
 
#if SHOW_EXTRA_TIMING
	double cpu_time = report_running_time();
#endif
 
	printf("\nCPU histogram:\n");
	output_histogram(histogram);
 
	int N = (int)PDH_acnt;
	float  *hX = (float *)malloc(sizeof(float) * N);
	float  *hY = (float *)malloc(sizeof(float) * N);
	float  *hZ = (float *)malloc(sizeof(float) * N);
	double *hdX = (double *)malloc(sizeof(double) * N);
	double *hdY = (double *)malloc(sizeof(double) * N);
	double *hdZ = (double *)malloc(sizeof(double) * N);
	bucket *gpu_histogram = (bucket *)calloc(num_buckets, sizeof(bucket));
	if (!hX || !hY || !hZ || !hdX || !hdY || !hdZ || !gpu_histogram) {
		fprintf(stderr, "Error: out of host memory.\n");
		return EXIT_FAILURE;
	}
	for (i = 0; i < N; i++) {
		hdX[i] = atom_list[i].x_pos;
		hdY[i] = atom_list[i].y_pos;
		hdZ[i] = atom_list[i].z_pos;
		hX[i] = (float)hdX[i];
		hY[i] = (float)hdY[i];
		hZ[i] = (float)hdZ[i];
	}
 
	float *dX, *dY, *dZ;
	double *ddX, *ddY, *ddZ;
	unsigned long long *d_hist;
	CUDA_CHECK(cudaMalloc((void **)&dX, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((void **)&dY, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((void **)&dZ, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((void **)&ddX, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((void **)&ddY, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((void **)&ddZ, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((void **)&d_hist, sizeof(unsigned long long) * num_buckets));
	CUDA_CHECK(cudaMemcpy(dX, hX, sizeof(float) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dY, hY, sizeof(float) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dZ, hZ, sizeof(float) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(ddX, hdX, sizeof(double) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(ddY, hdY, sizeof(double) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(ddZ, hdZ, sizeof(double) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemset(d_hist, 0, sizeof(unsigned long long) * num_buckets));
 
	int smem_optin = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
	if (smem_optin < (int)prop.sharedMemPerBlock)
		smem_optin = (int)prop.sharedMemPerBlock;
 
	size_t tile_bytes = 3 * sizeof(float) * block_size;
	size_t hist_bytes = sizeof(unsigned int) * (size_t)num_buckets;
	int win_lo = 0, win_size, n_copies = 1;
	size_t smem_bytes;
 
	if (tile_bytes + hist_bytes <= DEFAULT_SMEM_BYTES) {
		int warps = (block_size + 31) / 32;
		n_copies = (int)(COPIES_SMEM_BYTES / hist_bytes);
		if (n_copies > warps) n_copies = warps;
		if (n_copies > MAX_HIST_COPIES) n_copies = MAX_HIST_COPIES;
		if (n_copies < 1) n_copies = 1;
		win_size = num_buckets;
	} else {
		if (tile_bytes + sizeof(unsigned int) > (size_t)smem_optin) {
			fprintf(stderr, "Error: block_size %d needs more shared memory than this GPU has.\n",
			        block_size);
			return EXIT_FAILURE;
		}
		size_t max_win = ((size_t)smem_optin - tile_bytes) / sizeof(unsigned int);
		win_size = (int)((size_t)num_buckets < max_win ? (size_t)num_buckets : max_win);
		int centre = (int)(0.66 * BOX_SIZE / PDH_res);
		win_lo = centre - win_size / 2;
		if (win_lo + win_size > num_buckets) win_lo = num_buckets - win_size;
		if (win_lo < 0) win_lo = 0;
	}
	smem_bytes = tile_bytes + sizeof(unsigned int) * (size_t)win_size * n_copies;
	if (smem_bytes > DEFAULT_SMEM_BYTES)
		CUDA_CHECK(cudaFuncSetAttribute(sdh_tiled_kernel,
		           cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
 
	int M = (N + block_size - 1) / block_size;
	int D = M / 2;
	long long pairs_per_offset = (long long)block_size * block_size;
	long long max_offsets = (4294967295LL - pairs_per_offset) / pairs_per_offset;
	if (max_offsets < 1) max_offsets = 1;
	double w = PDH_res;
	float inv_w = (float)(1.0 / PDH_res);
 
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
 
	CUDA_CHECK(cudaEventRecord(start, 0));
	int d_begin = 1;
	int first = 1;
	do {
		int d_end = (int)((long long)d_begin + max_offsets < (long long)D + 1
		                  ? d_begin + max_offsets : D + 1);
		sdh_tiled_kernel<<<M, block_size, smem_bytes>>>(dX, dY, dZ, ddX, ddY, ddZ,
		        N, M, w, inv_w, num_buckets, win_lo, win_size, n_copies,
		        d_begin, d_end, first, d_hist);
		first = 0;
		d_begin = d_end;
	} while (d_begin <= D);
	CUDA_CHECK(cudaEventRecord(stop, 0));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaGetLastError());
 
	float elapsed_ms = 0;
	CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
 
	CUDA_CHECK(cudaMemcpy(gpu_histogram, d_hist, sizeof(bucket) * num_buckets,
	                      cudaMemcpyDeviceToHost));
 
	printf("\nGPU histogram:\n");
	output_histogram(gpu_histogram);
 
	compare_histograms(histogram, gpu_histogram);
 
	printf("\n******** Total Kernel Execution Time = %.4f sec *******\n", elapsed_ms / 1000.0);
 
#if SHOW_EXTRA_TIMING
	printf("\nSpeedup over CPU: %.2fx\n", (cpu_time * 1000) / elapsed_ms);
#endif
 
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	cudaFree(dX); cudaFree(dY); cudaFree(dZ);
	cudaFree(ddX); cudaFree(ddY); cudaFree(ddZ);
	cudaFree(d_hist);
	free(hX); free(hY); free(hZ);
	free(hdX); free(hdY); free(hdZ);
	free(gpu_histogram);
	free(histogram);
	free(atom_list);
 
	return 0;
}
 