#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ 
void ChamferDistanceKernel(
	int b,
	int n,
	const float* xyz,
	int m,
	const float* xyz2,
	float* result,
	int* result_i)
{
	const int batch=512;
	__shared__ float buf[batch*4];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*4;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*4+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*4+0];
				float y1=xyz[(i*n+j)*4+1];
				float z1=xyz[(i*n+j)*4+2];
				float g1=xyz[(i*n+j)*4+3];
				int best_i=0;
				float best=0;
				int end_ka=end_k-(end_k&3);
				if (end_ka==batch){
					for (int k=0;k<batch;k+=4){
						{
							float x2=buf[k*4+0]-x1;
							float y2=buf[k*4+1]-y1;
							float z2=buf[k*4+2]-z1;
							float g2=buf[k*4+3]-g1;
							float d=x2*x2+y2*y2+z2*z2+g2*g2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*4+4]-x1;
							float y2=buf[k*4+5]-y1;
							float z2=buf[k*4+6]-z1;
							float g2=buf[k*4+7]-g1;
							float d=x2*x2+y2*y2+z2*z2+g2*g2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*4+8]-x1;
							float y2=buf[k*4+9]-y1;
							float z2=buf[k*4+10]-z1;
							float g2=buf[k*4+11]-g1;
							float d=x2*x2+y2*y2+z2*z2+g2*g2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*4+12]-x1;
							float y2=buf[k*4+13]-y1;
							float z2=buf[k*4+14]-z1;
							float g2=buf[k*4+15]-g1;
							float d=x2*x2+y2*y2+z2*z2+g2*g2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}else{
					for (int k=0;k<end_ka;k+=4){
						{
							float x2=buf[k*4+0]-x1;
							float y2=buf[k*4+1]-y1;
							float z2=buf[k*4+2]-z1;
							float g2=buf[k*4+3]-g1;
							float d=x2*x2+y2*y2+z2*z2+g2*g2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*4+4]-x1;
							float y2=buf[k*4+5]-y1;
							float z2=buf[k*4+6]-z1;
							float g2=buf[k*4+7]-g1;
							float d=x2*x2+y2*y2+z2*z2+g2*g2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*4+8]-x1;
							float y2=buf[k*4+9]-y1;
							float z2=buf[k*4+10]-z1;
							float g2=buf[k*4+11]-g1;
							float d=x2*x2+y2*y2+z2*z2+g2*g2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*4+12]-x1;
							float y2=buf[k*4+13]-y1;
							float z2=buf[k*4+14]-z1;
							float g2=buf[k*4+15]-g1;
							float d=x2*x2+y2*y2+z2*z2+g2*g2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}
				for (int k=end_ka;k<end_k;k++){
					float x2=buf[k*4+0]-x1;
					float y2=buf[k*4+1]-y1;
					float z2=buf[k*4+2]-z1;
					float g2=buf[k*4+3]-g1;
					float d=x2*x2+y2*y2+z2*z2+g2*g2;
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}

void ChamferDistanceKernelLauncher(
    const int b, const int n,
    const float* xyz,
    const int m,
    const float* xyz2,
    float* result,
    int* result_i,
    float* result2,
    int* result2_i)
{
	ChamferDistanceKernel<<<dim3(32,16,1),512>>>(b, n, xyz, m, xyz2, result, result_i);
	ChamferDistanceKernel<<<dim3(32,16,1),512>>>(b, m, xyz2, n, xyz, result2, result2_i);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("error in chamfer distance updateOutput: %s\n", cudaGetErrorString(err));
}


__global__ 
void ChamferDistanceGradKernel(
	int b, int n,
	const float* xyz1,
	int m,
	const float* xyz2,
	const float* grad_dist1,
	const int* idx1,
	float* grad_xyz1,
	float* grad_xyz2)
{
	for (int i = blockIdx.x; i<b; i += gridDim.x) {
		for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x*gridDim.y) {
			float x1=xyz1[(i*n+j)*4+0];
			float y1=xyz1[(i*n+j)*4+1];
			float z1=xyz1[(i*n+j)*4+2];
			int j2=idx1[i*n+j];
			float x2=xyz2[(i*m+j2)*4+0];
			float y2=xyz2[(i*m+j2)*4+1];
			float z2=xyz2[(i*m+j2)*4+2];
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(&(grad_xyz1[(i*n+j)*4+0]),g*(x1-x2));
			atomicAdd(&(grad_xyz1[(i*n+j)*4+1]),g*(y1-y2));
			atomicAdd(&(grad_xyz1[(i*n+j)*4+2]),g*(z1-z2));
			atomicAdd(&(grad_xyz2[(i*m+j2)*4+0]),-(g*(x1-x2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*4+1]),-(g*(y1-y2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*4+2]),-(g*(z1-z2)));
		}
	}
}

void ChamferDistanceGradKernelLauncher(
    const int b, const int n,
    const float* xyz1,
    const int m,
    const float* xyz2,
    const float* grad_dist1,
    const int* idx1,
    const float* grad_dist2,
    const int* idx2,
    float* grad_xyz1,
    float* grad_xyz2)
{
	cudaMemset(grad_xyz1, 0, b*n*4*4);
	cudaMemset(grad_xyz2, 0, b*m*4*4);
	ChamferDistanceGradKernel<<<dim3(1,16,1), 256>>>(b, n, xyz1, m, xyz2, grad_dist1, idx1, grad_xyz1, grad_xyz2);
	ChamferDistanceGradKernel<<<dim3(1,16,1), 256>>>(b, m, xyz2, n, xyz1, grad_dist2, idx2, grad_xyz2, grad_xyz1);

	cudaError_t err = cudaGetLastError();
  	if (err != cudaSuccess)
	    printf("error in chamfer distance get grad: %s\n", cudaGetErrorString(err));
}
