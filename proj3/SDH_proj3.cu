/* ==================================================================
	Programmers: Yicheng Tu (tuy@usf.edu), Hanieh Toutouni (hanieht@usf.edu)
	Project 3: SDH of 3D data with an octree, starter code
	To compile and run in the GAIVI machines:
	    /apps/GPU_course/runScript.sh SDH_proj3.cu <N> <w> [block_size] [run_brute]
	run_brute 0 (optional scalability runs only) skips the brute-force
	reference; the traversals are then NOT CHECKED, never PASS.
	Implement: PDH_dft_kernel (Part A) and PDH_bft_gpu with its kernels
	(Part B).  Keep the output and RESULT lines.
   ==================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>


#define BOX_SIZE	23000 /* length of each side of the 3D data box           */
#define NO_VALUE	-1    /* octree: no child or parent                       */
#define TREE_DEGREE	8     /* children per internal octree node                */
#define LEAF_TARGET	32    /* target average points per leaf for depth selection (single leaves may hold more) */
#define START_LEVEL	3     /* initial node-pair level, capped at the leaf level; each level up
                             multiplies the number of starting pairs by about 64                */
#define MAX_START_PAIRS	67108864LL  /* refuse larger starting-pair lists */
#define MAX_BUCKETS	1000000     /* refuse larger histograms (w too small) */

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
	fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
	exit(1); } } while (0)

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	unsigned long long d_cnt;   /* the count might be huge; unsigned so that atomicAdd works on the GPU */
} bucket;


long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */
int block_size = 256;	/* threads per CUDA block (Project 2)     */
float gpu_time_ms = 0.0f;	/* kernel time of the last GPU method, CUDA events */
int max_grid_x;			/* largest gridDim.x of this GPU                */


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(bucket * h){
	int i; 
	unsigned long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15llu ", h[i].d_cnt);
		total_cnt += h[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%llu \n", total_cnt);
		else printf("| ");
	}
}

/* Compare a GPU histogram with the reference one and print the difference
   histogram (Project 1); returns true when they match exactly */
bool compare_histograms(const char * ref_name, const char * name, bucket * ref, bucket * test) {
	bool match = true;
	printf("\nDifference between the %s and %s histograms:\n", ref_name, name);
	for (int i = 0; i < num_buckets; i++) {
		long long diff = (long long)ref[i].d_cnt - (long long)test[i].d_cnt;
		if (diff != 0) match = false;
		if (i % 5 == 0) printf("\n%02d: ", i);
		printf("%15lld ", diff);
		if (i != num_buckets - 1) printf("| ");
	}
	printf("\n");
	return match;
}


/* ==================================================================
	The octree.  Node 0 is the root (the whole box).  Nodes are numbered
	level by level; the 8 children of node p are children[p] .. children[p]+7
	(= 8p+1 .. 8p+8); a leaf has children[n] == NO_VALUE.  Every node of the
	full tree exists, even if empty; all leaves are on the last level.
	The points are stored in leaf order in atom_sorted: the points of node n
	are atom_sorted[first_point[n]] .. atom_sorted[first_point[n]+a_cnt[n]-1].
	Data layout: the tree is a structure of arrays (one array per field,
	copied to the GPU as such); the points are an array of structures as in
	Projects 1 and 2.  Use the provided data structures for DFT and BFT.
   ================================================================== */

typedef struct {
	int total_nodes;
	int total_levels;     /* root is level 0, leaves are level total_levels-1 */
	int leaf_first;       /* index of the first leaf                          */
	int * level_first;    /* first node of each level (host only)             */
	double * x_min; double * x_max;     /* bounding box of each node        */
	double * y_min; double * y_max;
	double * z_min; double * z_max;
	int * children;       /* first child, NO_VALUE for a leaf                 */
	int * parent;         /* NO_VALUE for the root                            */
	long long * a_cnt;    /* number of points in the node                     */
	int * first_point;    /* position of the node's first point in atom_sorted */
} octree;

octree tree;              /* host copy                                        */
octree d_tree;            /* GPU copy (the arrays live in device memory)      */
atom * atom_sorted;       /* the points in leaf order, host                   */
atom * d_atom_list;       /* the points in the original order, device         */
atom * d_atom_sorted;     /* the points in leaf order, device                 */


/*
	the leaf that contains point (x, y, z)
*/
int octree_locate(const octree * t, double x, double y, double z) {
	int curr = 0;
	while(t->children[curr] != NO_VALUE) {
		double half = (t->x_max[curr] - t->x_min[curr]) / 2.0;
		int c = 0;
		if(!(x < t->x_min[curr] + half)) c |= 1;
		if(!(y > t->y_min[curr] + half)) c |= 2;
		if(  z < t->z_min[curr] + half ) c |= 4;
		curr = t->children[curr] + c;
	}
	return curr;
}

/*
	build the octree over atom_list; the depth is the smallest one with at
	most leaf_target points per leaf on average
*/
void octree_build(octree * t, double leaf_target) {
	int i, c, L, levels = 1, p;
	long long total = 1, pow8 = 1;

	while((double)PDH_acnt / pow(8.0, levels) > leaf_target) levels++;
	/* the full tree has (8^(levels+1)-1)/7 nodes of 68 bytes; refuse what int
	   indices or memory cannot hold (more than 8 levels needs over 1 GB)   */
	if(levels > 8) {
		fprintf(stderr, "octree with %d levels has too many nodes: raise LEAF_TARGET.\n", levels + 1);
		exit(1);
	}
	t->total_levels = levels + 1;
	t->level_first = (int *)malloc(sizeof(int) * (t->total_levels + 1));
	t->level_first[0] = 0;
	for(L = 1; L <= t->total_levels; L++) {
		pow8 *= TREE_DEGREE;
		total += pow8;
		t->level_first[L] = (int)(total - pow8);
	}
	t->total_nodes = (int)(total - pow8);

	t->x_min = (double *)malloc(sizeof(double) * t->total_nodes);
	t->x_max = (double *)malloc(sizeof(double) * t->total_nodes);
	t->y_min = (double *)malloc(sizeof(double) * t->total_nodes);
	t->y_max = (double *)malloc(sizeof(double) * t->total_nodes);
	t->z_min = (double *)malloc(sizeof(double) * t->total_nodes);
	t->z_max = (double *)malloc(sizeof(double) * t->total_nodes);
	t->children = (int *)malloc(sizeof(int) * t->total_nodes);
	t->parent   = (int *)malloc(sizeof(int) * t->total_nodes);
	t->a_cnt    = (long long *)calloc(t->total_nodes, sizeof(long long));
	t->first_point = (int *)malloc(sizeof(int) * t->total_nodes);
	if(!t->level_first || !t->x_min || !t->x_max || !t->y_min || !t->y_max || !t->z_min || !t->z_max
	   || !t->children || !t->parent || !t->a_cnt || !t->first_point) {
		fprintf(stderr, "cannot allocate an octree of %d nodes on the host: raise LEAF_TARGET.\n", t->total_nodes);
		exit(1);
	}

	/* the root covers the whole data box */
	t->x_min[0] = t->y_min[0] = t->z_min[0] = 0.0;
	t->x_max[0] = t->y_max[0] = t->z_max[0] = BOX_SIZE;
	t->parent[0] = NO_VALUE;

	/* the children of every node on levels 0 .. levels-1 */
	for(p = 0; p < t->level_first[levels]; p++) {
		double x_low = t->x_min[p], x_high = t->x_max[p];
		double y_low = t->y_min[p], y_high = t->y_max[p];
		double z_low = t->z_min[p], z_high = t->z_max[p];
		double half = (x_high - x_low) / 2.0;
		int first_child = TREE_DEGREE * p + 1;
		t->children[p] = first_child;
		for(c = 0; c < TREE_DEGREE; c++) {
			int n = first_child + c;
			if(c & 1) { t->x_min[n] = x_low + half; t->x_max[n] = x_high; }
			else      { t->x_min[n] = x_low;        t->x_max[n] = x_low + half; }
			if(c & 2) { t->y_min[n] = y_low;        t->y_max[n] = y_low + half; }
			else      { t->y_min[n] = y_low + half; t->y_max[n] = y_high; }
			if(c & 4) { t->z_min[n] = z_low;        t->z_max[n] = z_low + half; }
			else      { t->z_min[n] = z_low + half; t->z_max[n] = z_high; }
			t->parent[n] = p;
		}
	}
	t->leaf_first = t->level_first[levels];
	for(p = t->leaf_first; p < t->total_nodes; p++)
		t->children[p] = NO_VALUE;

	/* count the points in every leaf and all its ancestors */
	int * leaf_of = (int *)malloc(sizeof(int) * PDH_acnt);
	if(leaf_of == NULL) { fprintf(stderr, "cannot allocate the leaf index of %lld points.\n", PDH_acnt); exit(1); }
	for(i = 0; i < PDH_acnt; i++) {
		int n = octree_locate(t, atom_list[i].x_pos, atom_list[i].y_pos, atom_list[i].z_pos);
		leaf_of[i] = n;
		while(n != NO_VALUE) {
			t->a_cnt[n]++;
			n = t->parent[n];
		}
	}

	/* give every node a range of atom_sorted and copy the points into it */
	int pos = 0;
	for(p = t->leaf_first; p < t->total_nodes; p++) {
		t->first_point[p] = pos;
		pos += (int)t->a_cnt[p];
	}
	for(p = t->leaf_first - 1; p >= 0; p--)
		t->first_point[p] = t->first_point[t->children[p]];
	int * fill = (int *)calloc(t->total_nodes, sizeof(int));
	atom_sorted = (atom *)malloc(sizeof(atom) * PDH_acnt);
	if(fill == NULL || atom_sorted == NULL) { fprintf(stderr, "cannot allocate the leaf-ordered copy of %lld points.\n", PDH_acnt); exit(1); }
	for(i = 0; i < PDH_acnt; i++) {
		int n = leaf_of[i];
		atom_sorted[t->first_point[n] + fill[n]++] = atom_list[i];
	}
	free(fill);
	free(leaf_of);
}

void octree_print_stats(const octree * t) {
	int p, nonempty = 0, leaves = t->total_nodes - t->leaf_first;
	long long fullest = 0;
	for(p = t->leaf_first; p < t->total_nodes; p++) {
		if(t->a_cnt[p] > 0) nonempty++;
		if(t->a_cnt[p] > fullest) fullest = t->a_cnt[p];
	}
	printf("Octree: %d levels, %d nodes, %d leaves (%d non-empty), leaf side %.1f, "
	       "%.2f points per leaf on average, fullest leaf %lld\n",
	       t->total_levels, t->total_nodes, leaves, nonempty,
	       t->x_max[t->total_nodes - 1] - t->x_min[t->total_nodes - 1],
	       (double)PDH_acnt / leaves, fullest);
}

void octree_free(octree * t) {
	free(t->level_first); free(t->x_min); free(t->x_max); free(t->y_min);
	free(t->y_max); free(t->z_min); free(t->z_max); free(t->children);
	free(t->parent); free(t->a_cnt); free(t->first_point);
	free(atom_sorted);
}

/* copy the tree and the leaf-ordered points to the GPU */
void octree_to_device(const octree * h, octree * d) {
	size_t n = h->total_nodes;
	*d = *h;
	d->level_first = NULL;
	CUDA_CHECK(cudaMalloc((void **)&d->x_min, sizeof(double) * n));
	CUDA_CHECK(cudaMalloc((void **)&d->x_max, sizeof(double) * n));
	CUDA_CHECK(cudaMalloc((void **)&d->y_min, sizeof(double) * n));
	CUDA_CHECK(cudaMalloc((void **)&d->y_max, sizeof(double) * n));
	CUDA_CHECK(cudaMalloc((void **)&d->z_min, sizeof(double) * n));
	CUDA_CHECK(cudaMalloc((void **)&d->z_max, sizeof(double) * n));
	CUDA_CHECK(cudaMalloc((void **)&d->children, sizeof(int) * n));
	CUDA_CHECK(cudaMalloc((void **)&d->parent, sizeof(int) * n));
	CUDA_CHECK(cudaMalloc((void **)&d->a_cnt, sizeof(long long) * n));
	CUDA_CHECK(cudaMalloc((void **)&d->first_point, sizeof(int) * n));
	CUDA_CHECK(cudaMemcpy(d->x_min, h->x_min, sizeof(double) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d->x_max, h->x_max, sizeof(double) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d->y_min, h->y_min, sizeof(double) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d->y_max, h->y_max, sizeof(double) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d->z_min, h->z_min, sizeof(double) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d->z_max, h->z_max, sizeof(double) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d->children, h->children, sizeof(int) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d->parent, h->parent, sizeof(int) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d->a_cnt, h->a_cnt, sizeof(long long) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d->first_point, h->first_point, sizeof(int) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc((void **)&d_atom_sorted, sizeof(atom) * PDH_acnt));
	CUDA_CHECK(cudaMemcpy(d_atom_sorted, atom_sorted, sizeof(atom) * PDH_acnt, cudaMemcpyHostToDevice));
}

void octree_free_device(octree * d) {
	cudaFree(d->x_min); cudaFree(d->x_max); cudaFree(d->y_min); cudaFree(d->y_max);
	cudaFree(d->z_min); cudaFree(d->z_max); cudaFree(d->children); cudaFree(d->parent);
	cudaFree(d->a_cnt); cudaFree(d->first_point); cudaFree(d_atom_sorted);
}

/*
	all pairs (a, b), a <= b, of non-empty nodes on one tree level, including
	the pairs (a, a); returns the number of pairs, *pairs is malloc'ed
*/
long long octree_level_pairs(const octree * t, int level, int2 ** pairs) {
	int first = t->level_first[level], last = t->level_first[level + 1];
	int * nodes = (int *)malloc(sizeof(int) * (last - first));
	int i, j, m = 0;
	if(nodes == NULL) {
		fprintf(stderr, "cannot allocate the node list of level %d.\n", level);
		exit(1);
	}
	for(i = first; i < last; i++)
		if(t->a_cnt[i] > 0) nodes[m++] = i;
	long long k = 0, n_pairs = (long long)m * (m + 1) / 2;
	if(n_pairs > MAX_START_PAIRS) {
		fprintf(stderr, "level %d has %d non-empty nodes, %lld node pairs: lower START_LEVEL.\n",
		        level, m, n_pairs);
		exit(1);
	}
	*pairs = (int2 *)malloc(sizeof(int2) * n_pairs);
	if(*pairs == NULL) {
		fprintf(stderr, "cannot allocate %lld node pairs on the host.\n", n_pairs);
		exit(1);
	}
	for(i = 0; i < m; i++)
		for(j = i; j < m; j++)
			(*pairs)[k++] = make_int2(nodes[i], nodes[j]);
	free(nodes);
	return n_pairs;
}


/* ==================================================================
	Distance bounds of two boxes (provided) and Part A, the depth-first
	traversal.  Every pair of points (p in box 1, q in box 2) has a distance
	between min_distance and max_distance; if both fall into the same
	histogram bucket, all the pairs of the two nodes belong to that bucket.
   ================================================================== */

/* smallest possible distance between a point of box 1 and a point of box 2
   (0 if the boxes touch or overlap): the length of the per-axis gap vector */
__host__ __device__ double min_distance(
	double x1min, double x1max, double y1min, double y1max, double z1min, double z1max,
	double x2min, double x2max, double y2min, double y2max, double z2min, double z2max)
{
	double dx = 0.0, dy = 0.0, dz = 0.0;
	if(x1max < x2min) dx = x2min - x1max; else if(x2max < x1min) dx = x1min - x2max;
	if(y1max < y2min) dy = y2min - y1max; else if(y2max < y1min) dy = y1min - y2max;
	if(z1max < z2min) dz = z2min - z1max; else if(z2max < z1min) dz = z1min - z2max;
	return sqrt(dx * dx + dy * dy + dz * dz);
}

/* largest possible distance between a point of box 1 and a point of box 2:
   the distance of the two farthest corners (also valid for a box with itself) */
__host__ __device__ double max_distance(
	double x1min, double x1max, double y1min, double y1max, double z1min, double z1max,
	double x2min, double x2max, double y2min, double y2max, double z2min, double z2max)
{
	double dx = fabs(x1min - x2max); if(fabs(x1max - x2min) > dx) dx = fabs(x1max - x2min);
	double dy = fabs(y1min - y2max); if(fabs(y1max - y2min) > dy) dy = fabs(y1max - y2min);
	double dz = fabs(z1min - z2max); if(fabs(z1max - z2min) > dz) dz = fabs(z1max - z2min);
	return sqrt(dx * dx + dy * dy + dz * dz);
}

/*
	CPU dual-tree traversal (DM-SDH), a single-thread reference on the host; its
	call in main() is commented out.  Both nodes of a pair are always on the same
	level, and all leaves are on the last level, so the two nodes are leaves together.
*/
void dualtree_pair_cpu(const octree * t, int a, int b, double w, bucket * h) {
	long long ca = t->a_cnt[a], cb = t->a_cnt[b];
	if(ca == 0 || cb == 0) return;

	double dmin = min_distance(t->x_min[a], t->x_max[a], t->y_min[a], t->y_max[a], t->z_min[a], t->z_max[a],
	                           t->x_min[b], t->x_max[b], t->y_min[b], t->y_max[b], t->z_min[b], t->z_max[b]);
	double dmax = max_distance(t->x_min[a], t->x_max[a], t->y_min[a], t->y_max[a], t->z_min[a], t->z_max[a],
	                           t->x_min[b], t->x_max[b], t->y_min[b], t->y_max[b], t->z_min[b], t->z_max[b]);
	int bin_min = (int)(dmin / w), bin_max = (int)(dmax / w);
	if(bin_min == bin_max) {                 /* resolved: count the whole node pair */
		h[bin_min].d_cnt += (a == b) ? ca * (ca - 1) / 2 : ca * cb;
		return;
	}

	int i, j;
	if(t->children[a] == NO_VALUE) {         /* two leaves: compute every distance */
		const atom * pa = atom_sorted + t->first_point[a];
		const atom * pb = atom_sorted + t->first_point[b];
		for(i = 0; i < ca; i++) {
			for(j = (a == b) ? i + 1 : 0; j < cb; j++) {
				double dx = pa[i].x_pos - pb[j].x_pos;
				double dy = pa[i].y_pos - pb[j].y_pos;
				double dz = pa[i].z_pos - pb[j].z_pos;
				h[(int)(sqrt(dx * dx + dy * dy + dz * dz) / w)].d_cnt++;
			}
		}
		return;
	}

	for(i = 0; i < TREE_DEGREE; i++)         /* otherwise apply the same to the child pairs */
		for(j = (a == b) ? i : 0; j < TREE_DEGREE; j++)
			dualtree_pair_cpu(t, t->children[a] + i, t->children[b] + j, w, h);
}

/* starts at the first level whose node diagonal fits in one bucket (or at the
   leaf level), where a node pair can first be resolved; adds the counts to
   out[], which must start at zero                                          */
void PDH_dualtree_cpu(bucket * out) {
	int level = (int)ceil(log2(sqrt(3.0) * BOX_SIZE / PDH_res));
	int a, b;
	if(level < 0) level = 0;
	if(level > tree.total_levels - 1) level = tree.total_levels - 1;
	for(a = tree.level_first[level]; a < tree.level_first[level + 1]; a++)
		for(b = a; b < tree.level_first[level + 1]; b++)
			dualtree_pair_cpu(&tree, a, b, PDH_res, out);
}

/* ==================================================================
	Helpers shared by the GPU traversals (Parts A and B)
   ================================================================== */

/* where a block keeps its histogram, chosen from the number of buckets */
#define HIST_THREAD	0	/* a private 32-bit copy per thread in shared memory (no atomic
				   per distance) plus a 64-bit block copy for the bulk adds     */
#define HIST_BLOCK	1	/* one 64-bit copy per block in shared memory                   */
#define HIST_GLOBAL	2	/* atomics straight into the global histogram (huge histograms) */
#define PRIV_SMEM_MAX	(32 * 1024)	/* shared bytes allowed for HIST_THREAD (keeps occupancy) */
#define SMEM_MAX	(48 * 1024)	/* default limit of dynamic shared memory per block        */
#define DIST_MARGIN	0.1f	/* a float distance is within 0.06 of the exact one (measured: 0.006) */
#define DFT_STACK	16	/* stack entries per thread; a traversal is at most 9 levels deep   */
#define BFT_QUEUE_CAP	(1LL << 24)	/* node pairs per BFT frontier buffer (128 MB)           */

/* shared memory bytes of a block histogram (rounded up to 16 bytes) */
__host__ __device__ inline size_t hist_bytes(int mode, int nb, int threads) {
	if(mode == HIST_GLOBAL) return 0;
	size_t s = (size_t)nb * sizeof(unsigned long long);
	if(mode == HIST_THREAD) s += (size_t)nb * sizeof(unsigned int) * threads;
	return (s + 15) & ~(size_t)15;
}

/* the histogram mode of a kernel that also needs extra bytes of shared memory */
__host__ __device__ inline int hist_mode(int nb, int threads, size_t extra) {
	if(hist_bytes(HIST_THREAD, nb, threads) + extra <= PRIV_SMEM_MAX) return HIST_THREAD;
	if(hist_bytes(HIST_BLOCK, nb, threads) + extra <= SMEM_MAX) return HIST_BLOCK;
	return HIST_GLOBAL;
}

/* one thread's view of its block's histogram */
typedef struct {
	unsigned long long * blk;   /* 64-bit block copy in shared memory (HIST_THREAD, HIST_BLOCK)  */
	unsigned int * mine;        /* this thread's 32-bit counters, stride blockDim.x (HIST_THREAD) */
	bucket * g;                 /* the global histogram                                         */
	unsigned long long pending; /* upper bound of the counts now in mine[]                      */
} hist_view;

/* all threads of the block call it: clear the shared copies */
template<int MODE> __device__ void hist_begin(hist_view & h, unsigned long long * smem, int nb, bucket * g) {
	h.g = g;
	h.blk = smem;
	h.mine = (unsigned int *)(smem + nb) + threadIdx.x;
	h.pending = 0;
	if(MODE == HIST_GLOBAL) return;
	for(int k = threadIdx.x; k < nb; k += blockDim.x) smem[k] = 0;
	if(MODE == HIST_THREAD)
		for(int k = 0; k < nb; k++) h.mine[k * blockDim.x] = 0;
	__syncthreads();
}

/* one point pair in bucket b */
template<int MODE> __device__ __forceinline__ void hist_point(hist_view & h, int b) {
	if(MODE == HIST_THREAD) h.mine[b * blockDim.x]++;
	else if(MODE == HIST_BLOCK) atomicAdd(&h.blk[b], 1ULL);
	else atomicAdd(&h.g[b].d_cnt, 1ULL);
}

/* c point pairs of a resolved node pair, all in bucket b */
template<int MODE> __device__ __forceinline__ void hist_bulk(hist_view & h, int b, unsigned long long c) {
	if(MODE == HIST_GLOBAL) atomicAdd(&h.g[b].d_cnt, c);
	else atomicAdd(&h.blk[b], c);
}

/* move this thread's 32-bit counters into the block copy */
template<int MODE> __device__ void hist_flush_mine(hist_view & h, int nb) {
	if(MODE != HIST_THREAD) return;
	for(int k = 0; k < nb; k++) {
		unsigned int v = h.mine[k * blockDim.x];
		if(v) { atomicAdd(&h.blk[k], (unsigned long long)v); h.mine[k * blockDim.x] = 0; }
	}
	h.pending = 0;
}

/* call before adding at most n point pairs: keeps the 32-bit counters from overflowing */
template<int MODE> __device__ __forceinline__ void hist_reserve(hist_view & h, int nb, unsigned long long n) {
	if(MODE != HIST_THREAD) return;
	if(h.pending + n > 0xFFFFFFFFULL) hist_flush_mine<MODE>(h, nb);
	h.pending += n;
}

/* all threads of the block call it: add the block's counts to the global histogram */
template<int MODE> __device__ void hist_end(hist_view & h, int nb) {
	if(MODE == HIST_GLOBAL) return;
	hist_flush_mine<MODE>(h, nb);
	__syncthreads();
	for(int k = threadIdx.x; k < nb; k += blockDim.x)
		if(h.blk[k]) atomicAdd(&h.g[k].d_cnt, h.blk[k]);
}

/* bucket of a point pair from its single-precision coordinate differences, or -1
   when the distance is too close to a bucket boundary to be sure (the caller then
   uses exact_bucket).  With coordinates below 23,000 the float distance is within
   0.006 of the exact one (largest error over 2*10^8 random pairs; the worst-case
   bound with the float scaling by 1/w is 0.06), so a distance farther than
   DIST_MARGIN from every bucket boundary gets the same bucket as in double.    */
__device__ __forceinline__ int fast_bucket(float dx, float dy, float dz, float inv_w, int nb) {
	float d = sqrtf(dx * dx + dy * dy + dz * dz);
	int lo = (int)((d - DIST_MARGIN) * inv_w), hi = (int)((d + DIST_MARGIN) * inv_w);
	return (lo == hi && hi < nb) ? hi : -1;
}

/* bucket of a point pair in double precision: the same expression as in
   PDH_brute_kernel, so the result is bit for bit that of the reference      */
__device__ __forceinline__ int exact_bucket(double dx, double dy, double dz, double w) {
	return (int)(sqrt(dx * dx + dy * dy + dz * dz) / w);
}

/* the dual-tree rule for node pair (a, b) of one level: returns false when the
   pair is finished (no point pairs, or resolved and counted in bulk), true when
   it is unresolved; *ca_out and *cb_out get the point counts                  */
template<int MODE> __device__ __forceinline__ bool node_pair_open(const octree & t, int a, int b,
	double w, int nb, hist_view & h, long long * ca_out, long long * cb_out)
{
	long long ca = __ldg(&t.a_cnt[a]), cb = __ldg(&t.a_cnt[b]);
	*ca_out = ca; *cb_out = cb;
	if(ca == 0 || cb == 0 || (a == b && ca == 1)) return false;    /* no point pairs */
	double ax0 = __ldg(&t.x_min[a]), ax1 = __ldg(&t.x_max[a]);
	double ay0 = __ldg(&t.y_min[a]), ay1 = __ldg(&t.y_max[a]);
	double az0 = __ldg(&t.z_min[a]), az1 = __ldg(&t.z_max[a]);
	double bx0 = __ldg(&t.x_min[b]), bx1 = __ldg(&t.x_max[b]);
	double by0 = __ldg(&t.y_min[b]), by1 = __ldg(&t.y_max[b]);
	double bz0 = __ldg(&t.z_min[b]), bz1 = __ldg(&t.z_max[b]);
	double dmin = min_distance(ax0, ax1, ay0, ay1, az0, az1, bx0, bx1, by0, by1, bz0, bz1);
	double dmax = max_distance(ax0, ax1, ay0, ay1, az0, az1, bx0, bx1, by0, by1, bz0, bz1);
	int bin_min = (int)(dmin / w), bin_max = (int)(dmax / w);
	if(bin_min == bin_max && bin_max < nb) {          /* resolved: count the whole node pair */
		hist_bulk<MODE>(h, bin_min, (unsigned long long)((a == b) ? ca * (ca - 1) / 2 : ca * cb));
		return false;
	}
	return true;
}

/*
 * Part A (40 points): GPU depth-first traversal.
 * Implement the DFT algorithm described in the project description.
 * t = the octree in device memory (see the octree section), pts = the points
 * in leaf order (atom_sorted on the GPU), pairs[p] = (a, b) node indices with
 * a <= b, where (a, a) stands for the pairs of points inside a, w = bucket
 * width, nb = number of buckets, hist = the histogram in global memory
 * (update d_cnt with atomicAdd as in Projects 1 and 2).  The grid may hold
 * fewer threads than pairs.
 *
 * Implementation: each thread takes starting pairs in a grid-stride loop and
 * walks the node-pair tree below each one depth first, with an explicit stack.
 * A stack entry (a, b, k) is an open (unresolved, internal) pair; k is the next
 * of its 64 child pairs to visit (36 when a == b, children i <= j only).
 * Visiting a pair applies the dual-tree rule: a resolved pair is added in bulk,
 * an unresolved leaf pair is binned point by point (float distances with an
 * exact double fallback near bucket boundaries), and an unresolved internal
 * pair is pushed, so its first child is visited next: one branch is followed
 * to the bottom before its siblings.  Counts go to a block histogram in
 * shared memory, added to the global one once per block.
 */
template<int MODE> __device__ void dft_body(const octree & t, const atom * pts, const int2 * pairs,
	long long n_pairs, double w, int nb, bucket * hist, unsigned long long * smem)
{
	hist_view h;
	hist_begin<MODE>(h, smem, nb, hist);
	const float inv_w = (float)(1.0 / w);
	int st_a[DFT_STACK], st_b[DFT_STACK], st_k[DFT_STACK];
	long long stride = (long long)gridDim.x * blockDim.x;

	for(long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x; p < n_pairs; p += stride) {
		int2 sp = pairs[p];
		int a = sp.x, b = sp.y, top = 0;
		while(true) {
			/* visit (a, b) */
			long long ca, cb;
			if(node_pair_open<MODE>(t, a, b, w, nb, h, &ca, &cb)) {
				if(a < t.leaf_first) {                 /* internal: go deeper */
					st_a[top] = a; st_b[top] = b; st_k[top] = 0; top++;
				} else {                               /* two leaves: every distance */
					const atom * A = pts + __ldg(&t.first_point[a]);
					const atom * B = pts + __ldg(&t.first_point[b]);
					for(int i = 0; i < ca; i++) {
						double ax = __ldg(&A[i].x_pos), ay = __ldg(&A[i].y_pos), az = __ldg(&A[i].z_pos);
						float fx = (float)ax, fy = (float)ay, fz = (float)az;
						int j = (a == b) ? i + 1 : 0;
						hist_reserve<MODE>(h, nb, (unsigned long long)(cb - j));
						for(; j < cb; j++) {
							double bx = __ldg(&B[j].x_pos), by = __ldg(&B[j].y_pos), bz = __ldg(&B[j].z_pos);
							int bk = fast_bucket(fx - (float)bx, fy - (float)by, fz - (float)bz, inv_w, nb);
							if(bk < 0) bk = exact_bucket(ax - bx, ay - by, az - bz, w);
							hist_point<MODE>(h, bk);
						}
					}
				}
			}
			/* the next pair: the next unvisited child of the deepest open pair */
			while(top > 0 && st_k[top - 1] >= TREE_DEGREE * TREE_DEGREE) top--;   /* backtrack */
			if(top == 0) break;                        /* this starting pair is done */
			int pa = st_a[top - 1], pb = st_b[top - 1], k = st_k[top - 1];
			int nk = k + 1;                            /* (a, a): children i <= j only */
			if(pa == pb && (nk & 7) == 0) nk += nk >> 3;
			st_k[top - 1] = nk;
			a = __ldg(&t.children[pa]) + (k >> 3);
			b = __ldg(&t.children[pb]) + (k & 7);
		}
	}
	hist_end<MODE>(h, nb);
}

__global__ void PDH_dft_kernel(octree t, const atom * pts, const int2 * pairs, long long n_pairs,
                               double w, int nb, bucket * hist)
{
	extern __shared__ unsigned long long dft_smem[];
	/* the same rule as the host, which sizes the shared memory with it */
	switch(hist_mode(nb, blockDim.x, 0)) {
	case HIST_THREAD: dft_body<HIST_THREAD>(t, pts, pairs, n_pairs, w, nb, hist, dft_smem); break;
	case HIST_BLOCK:  dft_body<HIST_BLOCK>(t, pts, pairs, n_pairs, w, nb, hist, dft_smem); break;
	default:          dft_body<HIST_GLOBAL>(t, pts, pairs, n_pairs, w, nb, hist, dft_smem); break;
	}
}

/* ==================================================================
	GPU brute force (Project 2): the reference histogram, do not modify
   ================================================================== */

__global__ void PDH_brute_kernel(const atom * pts, long long n, double w, int nb,
                                 bucket * g_hist, int use_shared)
{
	extern __shared__ bucket sh_hist[];
	bucket * h = g_hist;
	int k;
	if(use_shared) {
		for(k = threadIdx.x; k < nb; k += blockDim.x) sh_hist[k].d_cnt = 0;
		__syncthreads();
		h = sh_hist;
	}
	long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x, j;
	if(i < n) {
		double x = pts[i].x_pos, y = pts[i].y_pos, z = pts[i].z_pos;
		for(j = i + 1; j < n; j++) {
			double dx = x - pts[j].x_pos, dy = y - pts[j].y_pos, dz = z - pts[j].z_pos;
			atomicAdd(&h[(int)(sqrt(dx * dx + dy * dy + dz * dz) / w)].d_cnt, 1ULL);
		}
	}
	if(use_shared) {
		__syncthreads();
		for(k = threadIdx.x; k < nb; k += blockDim.x)
			if(h[k].d_cnt) atomicAdd(&g_hist[k].d_cnt, h[k].d_cnt);
	}
}


/* ==================================================================
	host side of the GPU methods
   ================================================================== */

bucket * d_hist;                  /* histogram in device memory */

/* GPU kernel timing with CUDA events (Project 2): the interval covers the
   kernel(s) only, not memory allocation or host-device transfers         */
void gpu_timer_start(cudaEvent_t * start, cudaEvent_t * stop) {
	CUDA_CHECK(cudaEventCreate(start));
	CUDA_CHECK(cudaEventCreate(stop));
	CUDA_CHECK(cudaEventRecord(*start));
}

float gpu_timer_stop(cudaEvent_t start, cudaEvent_t stop) {
	float ms;
	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	return ms;
}

void report_kernel_time(const char * label) {
	printf("******** Total Kernel Execution Time (%s) = %.4f sec *******\n", label, gpu_time_ms / 1000.0);
}

void gpu_hist_reset() { CUDA_CHECK(cudaMemset(d_hist, 0, sizeof(bucket) * num_buckets)); }

void gpu_hist_fetch(bucket * out) {
	CUDA_CHECK(cudaMemcpy(out, d_hist, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost));
}

void PDH_brute_gpu(bucket * out) {
	size_t sh = (size_t)num_buckets * sizeof(bucket);
	int use_shared = (sh <= 48 * 1024);
	gpu_hist_reset();
	long long grid = (PDH_acnt + block_size - 1) / block_size;   /* one thread per point */
	if(grid > max_grid_x) {
		fprintf(stderr, "N too large for the brute-force launch on this GPU.\n");
		exit(1);
	}
	cudaEvent_t start, stop;
	gpu_timer_start(&start, &stop);
	PDH_brute_kernel<<<(int)grid, block_size, use_shared ? sh : 0>>>(
		d_atom_list, PDH_acnt, PDH_res, num_buckets, d_hist, use_shared);
	CUDA_CHECK(cudaGetLastError());
	gpu_time_ms = gpu_timer_stop(start, stop);
	gpu_hist_fetch(out);
}

/* Part A: one thread per node pair of start_level; change the launch if
   needed (grid, dynamic shared memory), but the kernel must process all
   n_pairs whatever the launch configuration is                           */
void PDH_dft_gpu(int start_level, bucket * out) {
	int2 * pairs, * d_pairs;
	long long n_pairs = octree_level_pairs(&tree, start_level, &pairs);
	CUDA_CHECK(cudaMalloc((void **)&d_pairs, sizeof(int2) * n_pairs));
	CUDA_CHECK(cudaMemcpy(d_pairs, pairs, sizeof(int2) * n_pairs, cudaMemcpyHostToDevice));
	gpu_hist_reset();
	long long grid = (n_pairs + block_size - 1) / block_size;
	if(grid > max_grid_x) grid = max_grid_x;   /* the kernel must then loop over the pairs */
	/* the block histogram in shared memory, sized by the rule the kernel uses */
	size_t smem = hist_bytes(hist_mode(num_buckets, block_size, 0), num_buckets, block_size);
	cudaEvent_t start, stop;
	gpu_timer_start(&start, &stop);
	PDH_dft_kernel<<<(int)grid, block_size, smem>>>(
		d_tree, d_atom_sorted, d_pairs, n_pairs, PDH_res, num_buckets, d_hist);
	CUDA_CHECK(cudaGetLastError());
	gpu_time_ms = gpu_timer_stop(start, stop);
	gpu_hist_fetch(out);
	cudaFree(d_pairs);
	free(pairs);
}

/* ==================================================================
	Part B: GPU breadth-first traversal

	The frontier is a list of unresolved node pairs per tree level, q[0] for
	the starting level .. q[levels-1] for the leaf level, in GPU memory.  Every
	pair in a list is known to be unresolved.  The host only launches kernels
	and reads the list lengths:

	  1. bft_level_kernel (filter): applies the dual-tree rule to the starting
	     pairs; resolved pairs are counted, unresolved ones go to q[0].
	  2. bft_level_kernel (expand), level by level: one thread per child pair
	     (64 per parent, 36 for (a, a)).  A resolved child is counted in bulk,
	     an unresolved one is appended to the next level's list (one atomic per
	     warp: ballot + popc + shuffle).
	  3. bft_leaf_kernel: the leaf-level list, one warp per leaf pair.  The warp
	     stages 32 points of node b at a time in shared memory as floats; each
	     lane holds a point of node a and bins its distances to them (small
	     leaves use several lane groups, so most lanes stay busy).

	Each list holds at most BFT_QUEUE_CAP pairs (fewer if GPU memory is low).
	When the next level could overflow, the host stops expanding, processes
	the next level first (the same procedure, recursively), then continues:
	with enough memory this is purely level by level, otherwise the traversal
	works on level-by-level chunks and never exceeds the memory budget.
   ================================================================== */

/* the dual-tree rule on the starting pairs (expand == 0) or on the child pairs
   of the pairs in[] (expand == 1); unresolved pairs are appended to out[] at
   *out_cnt.  Any grid size works: a block-stride loop covers all the work.   */
template<int MODE> __global__ void bft_level_kernel(octree t, const int2 * in, long long n_in, int expand,
	int2 * out, unsigned long long * out_cnt, double w, int nb, bucket * hist)
{
	extern __shared__ unsigned long long bft_smem[];
	hist_view h;
	hist_begin<MODE>(h, bft_smem, nb, hist);
	int lane = threadIdx.x & 31;
	int warp_lanes = (int)blockDim.x - (int)(threadIdx.x & ~31u);   /* lanes in this warp */
	unsigned int wmask = (warp_lanes >= 32) ? 0xFFFFFFFFu : ((1u << warp_lanes) - 1);
	long long total = expand ? n_in * TREE_DEGREE * TREE_DEGREE : n_in;
	long long stride = (long long)gridDim.x * blockDim.x;

	/* all threads of a block run the same iterations, so the warp votes see every lane */
	for(long long base = (long long)blockIdx.x * blockDim.x; base < total; base += stride) {
		long long idx = base + threadIdx.x, ca, cb;
		int a = 0, b = 0;
		bool open = false;
		if(idx < total) {
			if(expand) {
				int2 pr = in[idx >> 6];
				int i = (int)(idx >> 3) & 7, j = (int)idx & 7;
				if(pr.x != pr.y || i <= j) {          /* (a, a): children i <= j only */
					a = __ldg(&t.children[pr.x]) + i;
					b = __ldg(&t.children[pr.y]) + j;
					open = node_pair_open<MODE>(t, a, b, w, nb, h, &ca, &cb);
				}
			} else {
				int2 pr = in[idx];
				a = pr.x; b = pr.y;
				open = node_pair_open<MODE>(t, a, b, w, nb, h, &ca, &cb);
			}
		}
		/* append the warp's unresolved pairs with a single atomic */
		unsigned int votes = __ballot_sync(wmask, open);
		if(votes) {
			int leader = __ffs(votes) - 1;
			unsigned long long pos = 0;
			if(lane == leader) pos = atomicAdd(out_cnt, (unsigned long long)__popc(votes));
			pos = __shfl_sync(wmask, pos, leader);
			if(open) out[pos + __popc(votes & ((1u << lane) - 1))] = make_int2(a, b);
		}
	}
	hist_end<MODE>(h, nb);
}

/* shared memory of bft_leaf_kernel besides the histogram: 32 staged points per warp */
size_t bft_stage_bytes(int threads) {
	return (size_t)((threads + 31) / 32) * 32 * sizeof(float4);
}

/* every point distance of the unresolved leaf pairs lp[0 .. n-1], one warp per
   pair (warp-stride loop, so any grid size works)                            */
template<int MODE> __global__ void bft_leaf_kernel(octree t, const atom * pts, const int2 * lp, long long n,
	double w, int nb, bucket * hist)
{
	extern __shared__ unsigned long long bft_smem[];
	hist_view h;
	hist_begin<MODE>(h, bft_smem, nb, hist);
	const float inv_w = (float)(1.0 / w);
	int lane = threadIdx.x & 31;
	int W = (int)blockDim.x - (int)(threadIdx.x & ~31u);     /* lanes in this warp */
	if(W > 32) W = 32;
	unsigned int wmask = (W == 32) ? 0xFFFFFFFFu : ((1u << W) - 1);
	float4 * stage = (float4 *)((char *)bft_smem + hist_bytes(MODE, nb, blockDim.x)) + (threadIdx.x & ~31u);
	long long warps_per_block = (blockDim.x + 31) / 32;
	long long n_warps = gridDim.x * warps_per_block;

	for(long long p = blockIdx.x * warps_per_block + (threadIdx.x >> 5); p < n; p += n_warps) {
		int2 pr = lp[p];
		int a = pr.x, b = pr.y;
		int na = (int)__ldg(&t.a_cnt[a]), nbp = (int)__ldg(&t.a_cnt[b]);
		const atom * A = pts + __ldg(&t.first_point[a]);
		const atom * B = pts + __ldg(&t.first_point[b]);
		bool self = (a == b);
		/* R lanes hold R points of a; the W / R lane groups split the points of b */
		int R = (na < W) ? na : W, G = W / R;
		int grp = lane / R, r = lane % R;
		for(int i0 = 0; i0 < na; i0 += R) {
			int i = i0 + r;
			bool row = (grp < G) && (i < na);
			double ax = 0.0, ay = 0.0, az = 0.0;
			if(row) { ax = A[i].x_pos; ay = A[i].y_pos; az = A[i].z_pos; }
			float fx = (float)ax, fy = (float)ay, fz = (float)az;
			for(int j0 = self ? i0 + 1 : 0; j0 < nbp; j0 += W) {
				int jn = (nbp - j0 < W) ? nbp - j0 : W;
				__syncwarp(wmask);                     /* the previous chunk is used up */
				if(lane < jn) {
					const atom * q = B + j0 + lane;
					stage[lane] = make_float4((float)q->x_pos, (float)q->y_pos, (float)q->z_pos, 0.0f);
				}
				__syncwarp(wmask);
				if(row) {
					hist_reserve<MODE>(h, nb, (unsigned long long)((jn + G - 1) / G));
					for(int jj = grp; jj < jn; jj += G) {
						int j = j0 + jj;
						if(self && j <= i) continue;       /* (a, a): pairs i < j only */
						float4 q = stage[jj];
						int bk = fast_bucket(fx - q.x, fy - q.y, fz - q.z, inv_w, nb);
						if(bk < 0) bk = exact_bucket(ax - B[j].x_pos, ay - B[j].y_pos, az - B[j].z_pos, w);
						hist_point<MODE>(h, bk);
					}
				}
			}
		}
	}
	hist_end<MODE>(h, nb);
}

/* host-side state of the BFT frontier */
static int bft_levels;                    /* lists: starting level .. leaf level      */
static long long bft_cap;                 /* capacity of each list, in node pairs      */
static int2 ** bft_q;                     /* bft_q[i]: list of level start_level + i   */
static unsigned long long * bft_d_cnt;    /* list lengths on the GPU                   */
static unsigned long long * bft_cnt;      /* host copies of the list lengths           */
static int bft_mode_level, bft_mode_leaf; /* histogram modes of the two kernels        */
static size_t bft_smem_level, bft_smem_leaf;
static long long bft_grid_level, bft_grid_leaf;   /* resident blocks of each kernel    */

/* blocks of a kernel that fit on the GPU at once (its grid-stride loops cover the rest) */
static long long resident_blocks(const void * kernel, size_t smem) {
	int dev, sms, per_sm = 0;
	CUDA_CHECK(cudaGetDevice(&dev));
	CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
	CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, block_size, smem));
	if(per_sm < 1) per_sm = 1;
	return (long long)sms * per_sm;
}

/* run bft_level_kernel on n pairs of in[], appending to list li_out; reads the new length */
static void bft_launch_level(const int2 * in, long long n, int expand, int li_out) {
	long long work = expand ? n * TREE_DEGREE * TREE_DEGREE : n;
	long long grid = (work + block_size - 1) / block_size;
	if(grid > bft_grid_level) grid = bft_grid_level;
	switch(bft_mode_level) {
	case HIST_THREAD: bft_level_kernel<HIST_THREAD><<<(int)grid, block_size, bft_smem_level>>>(
		d_tree, in, n, expand, bft_q[li_out], bft_d_cnt + li_out, PDH_res, num_buckets, d_hist); break;
	case HIST_BLOCK: bft_level_kernel<HIST_BLOCK><<<(int)grid, block_size, bft_smem_level>>>(
		d_tree, in, n, expand, bft_q[li_out], bft_d_cnt + li_out, PDH_res, num_buckets, d_hist); break;
	default: bft_level_kernel<HIST_GLOBAL><<<(int)grid, block_size, bft_smem_level>>>(
		d_tree, in, n, expand, bft_q[li_out], bft_d_cnt + li_out, PDH_res, num_buckets, d_hist); break;
	}
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaMemcpy(&bft_cnt[li_out], bft_d_cnt + li_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
}

/* run bft_leaf_kernel on the n leaf pairs of lp[] */
static void bft_launch_leaf(const int2 * lp, long long n) {
	long long warps_per_block = (block_size + 31) / 32;
	long long grid = (n + warps_per_block - 1) / warps_per_block;
	if(grid > bft_grid_leaf) grid = bft_grid_leaf;
	switch(bft_mode_leaf) {
	case HIST_THREAD: bft_leaf_kernel<HIST_THREAD><<<(int)grid, block_size, bft_smem_leaf>>>(
		d_tree, d_atom_sorted, lp, n, PDH_res, num_buckets, d_hist); break;
	case HIST_BLOCK: bft_leaf_kernel<HIST_BLOCK><<<(int)grid, block_size, bft_smem_leaf>>>(
		d_tree, d_atom_sorted, lp, n, PDH_res, num_buckets, d_hist); break;
	default: bft_leaf_kernel<HIST_GLOBAL><<<(int)grid, block_size, bft_smem_leaf>>>(
		d_tree, d_atom_sorted, lp, n, PDH_res, num_buckets, d_hist); break;
	}
	CUDA_CHECK(cudaGetLastError());
}

static void bft_clear(int li) {
	bft_cnt[li] = 0;
	CUDA_CHECK(cudaMemset(bft_d_cnt + li, 0, sizeof(unsigned long long)));
}

/* process every pair in list li and everything below it, then empty list li */
static void bft_drain(int li) {
	if(li == bft_levels - 1) {                         /* leaf level */
		if(bft_cnt[li] > 0) bft_launch_leaf(bft_q[li], (long long)bft_cnt[li]);
		bft_clear(li);
		return;
	}
	long long n = (long long)bft_cnt[li], off = 0;
	long long min_chunk = bft_cap / (TREE_DEGREE * TREE_DEGREE) / 4;
	if(min_chunk < 1) min_chunk = 1;
	while(off < n) {
		/* parents whose 64 children surely fit in the next list */
		long long room = (bft_cap - (long long)bft_cnt[li + 1]) / (TREE_DEGREE * TREE_DEGREE);
		if(room < min_chunk && room < n - off) { bft_drain(li + 1); continue; }   /* next level first */
		long long c = (n - off < room) ? n - off : room;
		bft_launch_level(bft_q[li] + off, c, 1, li + 1);
		off += c;
	}
	if(bft_cnt[li + 1] > 0) bft_drain(li + 1);
	bft_clear(li);
}

/*
 * Part B (30 points): GPU breadth-first traversal.
 * Implement BFT as described in the project description.
 * You may add your own CUDA kernels and helper functions (define them
 * above this function; d_tree, d_atom_sorted, d_hist, block_size and
 * octree_level_pairs are available as in PDH_dft_gpu).
 * Interface: leave the histogram in out[] (gpu_hist_reset / gpu_hist_fetch)
 * and the time of all your traversal kernels, measured with CUDA events
 * (gpu_timer_start / gpu_timer_stop), in gpu_time_ms; check your launches
 * with CUDA_CHECK(cudaGetLastError()).
 */
void PDH_bft_gpu(int start_level, bucket * out)
{
	int i;
	int2 * pairs, * d_pairs;
	long long n_pairs = octree_level_pairs(&tree, start_level, &pairs);
	CUDA_CHECK(cudaMalloc((void **)&d_pairs, sizeof(int2) * n_pairs));
	CUDA_CHECK(cudaMemcpy(d_pairs, pairs, sizeof(int2) * n_pairs, cudaMemcpyHostToDevice));

	/* kernel configurations */
	bft_mode_level = hist_mode(num_buckets, block_size, 0);
	bft_smem_level = hist_bytes(bft_mode_level, num_buckets, block_size);
	bft_mode_leaf = hist_mode(num_buckets, block_size, bft_stage_bytes(block_size));
	bft_smem_leaf = hist_bytes(bft_mode_leaf, num_buckets, block_size) + bft_stage_bytes(block_size);
	switch(bft_mode_level) {
	case HIST_THREAD: bft_grid_level = resident_blocks((const void *)bft_level_kernel<HIST_THREAD>, bft_smem_level); break;
	case HIST_BLOCK:  bft_grid_level = resident_blocks((const void *)bft_level_kernel<HIST_BLOCK>, bft_smem_level); break;
	default:          bft_grid_level = resident_blocks((const void *)bft_level_kernel<HIST_GLOBAL>, bft_smem_level); break;
	}
	switch(bft_mode_leaf) {
	case HIST_THREAD: bft_grid_leaf = resident_blocks((const void *)bft_leaf_kernel<HIST_THREAD>, bft_smem_leaf); break;
	case HIST_BLOCK:  bft_grid_leaf = resident_blocks((const void *)bft_leaf_kernel<HIST_BLOCK>, bft_smem_leaf); break;
	default:          bft_grid_leaf = resident_blocks((const void *)bft_leaf_kernel<HIST_GLOBAL>, bft_smem_leaf); break;
	}
	if(bft_grid_level > max_grid_x) bft_grid_level = max_grid_x;
	if(bft_grid_leaf > max_grid_x) bft_grid_leaf = max_grid_x;

	/* the frontier lists: at most half of the free GPU memory */
	bft_levels = tree.total_levels - start_level;
	size_t free_mem, total_mem;
	CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
	bft_cap = BFT_QUEUE_CAP;
	if((long long)(free_mem / 2 / bft_levels / sizeof(int2)) < bft_cap)
		bft_cap = (long long)(free_mem / 2 / bft_levels / sizeof(int2));
	if(bft_cap < TREE_DEGREE * TREE_DEGREE) {
		fprintf(stderr, "not enough GPU memory for the BFT frontier.\n");
		exit(1);
	}
	bft_q = (int2 **)malloc(sizeof(int2 *) * bft_levels);
	bft_cnt = (unsigned long long *)calloc(bft_levels, sizeof(unsigned long long));
	if(bft_q == NULL || bft_cnt == NULL) { fprintf(stderr, "cannot allocate the BFT lists.\n"); exit(1); }
	for(i = 0; i < bft_levels; i++)
		CUDA_CHECK(cudaMalloc((void **)&bft_q[i], sizeof(int2) * bft_cap));
	CUDA_CHECK(cudaMalloc((void **)&bft_d_cnt, sizeof(unsigned long long) * bft_levels));
	CUDA_CHECK(cudaMemset(bft_d_cnt, 0, sizeof(unsigned long long) * bft_levels));
	gpu_hist_reset();

	/* the timed part: every traversal kernel (and the few list-length reads between them) */
	cudaEvent_t start, stop;
	gpu_timer_start(&start, &stop);
	long long off = 0;
	while(off < n_pairs) {                              /* starting pairs -> list 0 */
		long long room = bft_cap - (long long)bft_cnt[0];
		if(room == 0) { bft_drain(0); continue; }
		long long c = (n_pairs - off < room) ? n_pairs - off : room;
		bft_launch_level(d_pairs + off, c, 0, 0);
		off += c;
	}
	bft_drain(0);                                       /* then level by level */
	gpu_time_ms = gpu_timer_stop(start, stop);
	gpu_hist_fetch(out);

	for(i = 0; i < bft_levels; i++) cudaFree(bft_q[i]);
	cudaFree(bft_d_cnt);
	free(bft_q); free(bft_cnt);
	cudaFree(d_pairs);
	free(pairs);
}


int main(int argc, char **argv)
{
	int i, start_level, run_brute = 1;

	if(argc < 3 || argc > 5) {
		fprintf(stderr, "usage: %s <N> <w> [block_size] [run_brute]\n", argv[0]);
		return 1;
	}
	PDH_acnt = strtoll(argv[1], NULL, 10);
	PDH_res	 = atof(argv[2]);
	if(argc > 3) block_size = atoi(argv[3]);
	if(argc > 4) run_brute = atoi(argv[4]);
	if(PDH_acnt < 2 || PDH_acnt > 2147483647LL   /* point indices are int */
	   || !isfinite(PDH_res) || PDH_res <= 0.0 || block_size < 1 || block_size > 1024
	   || (run_brute != 0 && run_brute != 1)) {
		fprintf(stderr, "Invalid N (2 .. 2147483647), w, block_size, or run_brute.\n");
		return 1;
	}
	{
		int dev; cudaDeviceProp prop;
		CUDA_CHECK(cudaGetDevice(&dev));
		CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
		if(block_size > prop.maxThreadsPerBlock) {
			fprintf(stderr, "block_size exceeds the %d threads per block of this GPU.\n", prop.maxThreadsPerBlock);
			return 1;
		}
		max_grid_x = prop.maxGridSize[0];
	}

	/* histogram size as in Project 1 (with the exact box diagonal); a tiny w would ask for an absurd histogram */
	if(BOX_SIZE * sqrt(3.0) / PDH_res + 1 > (double)MAX_BUCKETS) {
		fprintf(stderr, "w is too small: more than %d buckets.\n", MAX_BUCKETS);
		return 1;
	}
	num_buckets = (int)(BOX_SIZE * sqrt(3.0) / PDH_res) + 1;
	bucket * h_brute = (bucket *)calloc(num_buckets, sizeof(bucket));
	bucket * h_dft   = (bucket *)calloc(num_buckets, sizeof(bucket));
	bucket * h_bft   = (bucket *)calloc(num_buckets, sizeof(bucket));

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);
	if(h_brute == NULL || h_dft == NULL || h_bft == NULL || atom_list == NULL) {
		fprintf(stderr, "cannot allocate the histograms or the %lld points on the host.\n", PDH_acnt);
		return 1;
	}

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* octree construction (while you develop, octree_print_stats(&tree) shows its
	   shape; the starting level used is the smaller of START_LEVEL and the leaf level) */
	octree_build(&tree, LEAF_TARGET);
	start_level = START_LEVEL;                 /* a shallow tree lowers the level actually used */
	if(start_level > tree.total_levels - 1) start_level = tree.total_levels - 1;

	/* copy the points and the tree to the GPU */
	CUDA_CHECK(cudaMalloc((void **)&d_atom_list, sizeof(atom) * PDH_acnt));
	CUDA_CHECK(cudaMemcpy(d_atom_list, atom_list, sizeof(atom) * PDH_acnt, cudaMemcpyHostToDevice));
	octree_to_device(&tree, &d_tree);
	CUDA_CHECK(cudaMalloc((void **)&d_hist, sizeof(bucket) * num_buckets));

	/* output as in Project 2: histogram, difference, kernel time per method */

	/* GPU brute force (Project 2): the reference histogram (skipped with run_brute 0) */
	if(run_brute) {
		printf("==== GPU brute force (reference) ====\n");
		PDH_brute_gpu(h_brute);
		output_histogram(h_brute);
		report_kernel_time("GPU brute force");
	}

	/* CPU dual-tree traversal (runs on the host, no GPU kernel); uncomment this block to run it
	bucket * h_cpu = (bucket *)calloc(num_buckets, sizeof(bucket));
	struct timeval t0, t1;
	printf("\n==== CPU dual-tree ====\n");
	gettimeofday(&t0, NULL);
	PDH_dualtree_cpu(h_cpu);
	gettimeofday(&t1, NULL);
	output_histogram(h_cpu);
	if(run_brute) printf("RESULT CPU %s\n", compare_histograms("reference", "CPU dual-tree", h_brute, h_cpu) ? "PASS" : "FAIL");
	printf("******** Total Execution Time (CPU dual-tree) = %.4f sec *******\n",
	       (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / 1000000.0);
	free(h_cpu);
	*/

	/* Part A: GPU depth-first traversal */
	printf("\n==== GPU DFT ====\n");
	PDH_dft_gpu(start_level, h_dft);
	output_histogram(h_dft);
	bool ok_dft = run_brute && compare_histograms("reference", "GPU DFT", h_brute, h_dft);
	report_kernel_time("GPU DFT");

	/* Part B: GPU breadth-first traversal */
	printf("\n==== GPU BFT ====\n");
	PDH_bft_gpu(start_level, h_bft);
	output_histogram(h_bft);
	bool ok_bft = run_brute && compare_histograms("reference", "GPU BFT", h_brute, h_bft);
	report_kernel_time("GPU BFT");

	/* one line per traversal for the grading script */
	bool same = false;
	if(run_brute) {
		printf("\nRESULT DFT %s\n", ok_dft ? "PASS" : "FAIL");
		printf("RESULT BFT %s\n", ok_bft ? "PASS" : "FAIL");
	} else {
		/* no reference: the two traversals are only checked against each other,
		   which catches a disagreement but is not a proof of correctness      */
		same = compare_histograms("GPU DFT", "GPU BFT", h_dft, h_bft);
		printf("\nRESULT DFT NOT CHECKED\n");
		printf("RESULT BFT %s\n", same ? "MATCHES DFT (not checked)" : "DIFFERS FROM DFT");
	}

	octree_free_device(&d_tree);
	cudaFree(d_atom_list);
	cudaFree(d_hist);
	octree_free(&tree);
	free(atom_list); free(h_brute); free(h_dft); free(h_bft);
	return run_brute ? ((ok_dft && ok_bft) ? 0 : 1) : (same ? 0 : 1);
}
