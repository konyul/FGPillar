//
// Created by sgs on 2021/7/13.
//

#include "atomics.cuh"
#include "pillar_ops_gpu.h"


#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


__global__ void createPillarIndicesStackKernel(int N, int B, int H, int W, float bev_size,
											   const float *xyz, const int *xyz_batch_cnt, bool *pillar_mask) {
	// xyz_batch_cnt: (N1+N2...)    xyz: (N1+N2..., 3) [x y z ...]
    // pillar_mask: [B, H, W]
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    xyz += tid * 3;

    int bid = 0, pts_cnt = xyz_batch_cnt[0];
    for (int b = 1; b < B; ++b){
    	if (tid < pts_cnt) break;
    	pts_cnt += xyz_batch_cnt[b];
    	bid = b;
    }

    pillar_mask += bid * H * W;

    float cx = xyz[0] / bev_size;
    float cy = xyz[1] / bev_size;
    int xid = int(cx);
    int yid = int(cy);

    if (xid < 0 || xid >= W || yid < 0 || yid >= H) return;
    pillar_mask[yid * W + xid] = 1;
}

__global__ void createVoxelIndicesStackKernel(int N, int B, int H, int W, int X, float bev_size_x, float bev_size_z,
											   const float *xyz, const int *xyz_batch_cnt, bool *pillar_mask) {
	// xyz_batch_cnt: (N1+N2...)    xyz: (N1+N2..., 3) [x y z ...]
    // voxel_mask: [B, H, W, X]
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    xyz += tid * 3;

    int bid = 0, pts_cnt = xyz_batch_cnt[0];
    for (int b = 1; b < B; ++b){
    	if (tid < pts_cnt) break;
    	pts_cnt += xyz_batch_cnt[b];
    	bid = b;
    }

    pillar_mask += bid * H * W * X;


    float cx = xyz[0] / bev_size_x;
    float cy = xyz[1] / bev_size_x;
    float cz = xyz[2] / bev_size_z;
    int xid = int(cx);
    int yid = int(cy);
    int zid = int(cz);

    if (xid < 0 || xid >= W || yid < 0 || yid >= H || zid < 0 || zid >= X ) return;
    pillar_mask[yid * W * X + xid * X + zid] = 1;
}

void create_pillar_indices_stack_kernel_launcher(int N, int B, int H, int W, float bev_size,
												 const float *xyz, const int *xyz_batch_cnt, bool *pillar_mask) {
    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    createPillarIndicesStackKernel<<<blocks, threads>>>(N, B, H, W, bev_size, xyz, xyz_batch_cnt, pillar_mask);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void create_voxel_indices_stack_kernel_launcher(int N, int B, int H, int W, int X, float bev_size_x, float bev_size_z,
												 const float *xyz, const int *xyz_batch_cnt, bool *pillar_mask) {
    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    createVoxelIndicesStackKernel<<<blocks, threads>>>(N, B, H, W, X, bev_size_x, bev_size_z, xyz, xyz_batch_cnt, pillar_mask);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void createPillarIndicesKernel(int B, int H, int W, const int *bevIndices, int *pillarIndices){
	// pillarIndices: (L, 3)  [bId, yId, xId]   bevIndices: (B*H*W)

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B*H*W) return;

	bevIndices += tid;
	if (bevIndices[0] < 0) return;

	int bId = tid / (H * W);
	int xId = tid % W;
	int yId = (tid / W) % H;

	const int index = bevIndices[0] * 3;
	pillarIndices[index + 0] = bId;
	pillarIndices[index + 1] = yId;
	pillarIndices[index + 2] = xId;
}

__global__ void createVoxelIndicesKernel(int B, int H, int W, int X, const int *bevIndices, int *voxelIndices){
	// pillarIndices: (L, 3)  [bId, yId, xId]   bevIndices: (B*H*W)

	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= B*H*W*X) return;

	bevIndices += tid;
	if (bevIndices[0] < 0) return;
    // Y -> H, X -> W
	int bId = tid / (H * W * X);
	int yId = (tid % (H * W * X)) / (W * X);
	int xId = (tid % (W * X)) / (X);
    int zId = tid % X;

	const int index = bevIndices[0] * 4;
	voxelIndices[index + 0] = bId;
	voxelIndices[index + 1] = yId;
	voxelIndices[index + 2] = xId;
    voxelIndices[index + 3] = zId;
}

void create_pillar_indices_kernel_launcher(int B, int H, int W, const int *bevIndices, int *pillarIndices){
	cudaError_t err;
	dim3 blocks(DIVUP(B * H * W, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	createPillarIndicesKernel<<<blocks, threads>>>(B, H, W, bevIndices, pillarIndices);

	// cudaDeviceSynchronize();  // for using printf in kernel function
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

void create_voxel_indices_kernel_launcher(int B, int H, int W, int X, const int *bevIndices, int *pillarIndices){
	cudaError_t err;
	dim3 blocks(DIVUP(B * H * W * X, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
	dim3 threads(THREADS_PER_BLOCK);

	createVoxelIndicesKernel<<<blocks, threads>>>(B, H, W, X, bevIndices, pillarIndices);

	// cudaDeviceSynchronize();  // for using printf in kernel function
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
		exit(-1);
	}
}

__global__ void createPillarIndicePairsStackKernel(int N, int B, int H, int W, float bev_size,
													  const float *xyz, const int *xyz_batch_cnt,
                                                      const int *pillar_bev_indices, int *indice_pairs) {
    // pointBatchCnt: (N1+N2...)    points: (N1+N2..., C) [x y z ...]    pillarBevIndices: [B, H, W]
    // indicePairs: (N1+N2..., 1)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    xyz += tid * 3;
    indice_pairs += tid;

    int bid = 0, pts_cnt = xyz_batch_cnt[0];
    for (int b = 1; b < B; ++b){
    	if (tid < pts_cnt) break;
    	pts_cnt += xyz_batch_cnt[b];
    	bid = b;
    }

    const int prefix = bid * H * W;
    pillar_bev_indices += prefix;

    const float cx = xyz[0] / bev_size;
    const float cy = xyz[1] / bev_size;
    int xid = int(cx);
    int yid = int(cy);

    if (xid < 0 || xid >= W || yid < 0 || yid >= H) return;
    indice_pairs[0] = pillar_bev_indices[yid * W + xid];
}

__global__ void createVoxelIndicePairsStackKernel(int N, int B, int H, int W, int X, float bev_size_x, float bev_size_z,
													  const float *xyz, const int *xyz_batch_cnt,
                                                      const int *pillar_bev_indices, int *indice_pairs) {
    // pointBatchCnt: (N1+N2...)    points: (N1+N2..., C) [x y z ...]    pillarBevIndices: [B, H, W]
    // indicePairs: (N1+N2..., 1)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    xyz += tid * 3;
    indice_pairs += tid;

    int bid = 0, pts_cnt = xyz_batch_cnt[0];
    for (int b = 1; b < B; ++b){
    	if (tid < pts_cnt) break;
    	pts_cnt += xyz_batch_cnt[b];
    	bid = b;
    }

    const int prefix = bid * H * W * X;
    pillar_bev_indices += prefix;

    const float cx = xyz[0] / bev_size_x;
    const float cy = xyz[1] / bev_size_x;
    const float cz = xyz[2] / bev_size_z;
    int xid = int(cx);
    int yid = int(cy);
    int zid = int(cz);

    if (xid < 0 || xid >= W || yid < 0 || yid >= H || zid < 0 || zid >= X ) return;
    indice_pairs[0] = pillar_bev_indices[yid * W * X + xid * X + zid];
}
void create_pillar_indice_pairs_stack_kernel_launcher(int N, int B, int H, int W, float bev_size,
													  const float *xyz, const int *xyz_batch_cnt,
													  const int *pillar_bev_indices, int *indice_pairs) {
    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    createPillarIndicePairsStackKernel<<<blocks, threads>>>(N, B, H, W, bev_size, xyz, xyz_batch_cnt,
															   pillar_bev_indices, indice_pairs);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

void create_voxel_indice_pairs_stack_kernel_launcher(int N, int B, int H, int W, int X, float bev_size_x, float bev_size_z,
													  const float *xyz, const int *xyz_batch_cnt,
													  const int *pillar_bev_indices, int *indice_pairs) {
    cudaError_t err;
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    createVoxelIndicePairsStackKernel<<<blocks, threads>>>(N, B, H, W, X, bev_size_x, bev_size_z, xyz, xyz_batch_cnt,
															   pillar_bev_indices, indice_pairs);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}