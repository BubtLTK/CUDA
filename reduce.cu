#include<stdlib.h>
#include<stdio.h>
#include<time.h>
//全局内存
__global__ void global_reduce(float *d_in,float *d_out){
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int idxn = threadIdx.x;
	for(int s = blockDim.x/2;s>0;s>>=1){
		if(idxn<s){
			d_in[idx] += d_in[idx+s];
		}
		__syncthreads();//同步
	}
	if(idxn == 0){
		d_out[blockIdx.x] = d_in[idx];
	}
}
//共享内存
__global__ void shared_reduce(float *d_in,float *d_out){
	extern __shared__ float s_in[];
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int idxn = threadIdx.x;
	s_in[idxn] = d_in[idx];
	__syncthreads();
	for(int s = blockDim.x/2;s>0;s>>=1){
		if(idxn<s){
			s_in[idxn] += s_in[idxn+s];
		}
		__syncthreads();//同步
	}
	if(idxn == 0){
		d_out[blockIdx.x] = s_in[0];
	}
}

void init(float *h_in,const int size){
	srand((unsigned int)time(NULL));
	for(int i=0;i<size;i++)
		h_in[i] =(float)(rand()%101)/100.0f;
}

int main(){
	int size = 1024;
	float *h_in;
	float h_out = 0;
	h_in = (float *)malloc(size*size*sizeof(float));
	init(h_in,size*size);//初始化
	time_t t_start = clock();
	for(int i=0;i<size*size;i++){
		h_out += h_in[i];
	}
	time_t t_end = clock();
	printf("CPU sum:%f\n",h_out);
	printf("CPU time:%fms\n",difftime(t_end,t_start));
	float *d_in;
	float *d_out;
	float *d_out_mid;
	dim3 block(size);
	dim3 thread(size);
	cudaMalloc((float **)&d_in,size*size*sizeof(float));
	cudaMalloc((float **)&d_out_mid,size*sizeof(float));
	cudaMalloc((float **)&d_out,sizeof(float));
	cudaMemcpy(d_in,h_in,size*size*sizeof(float),cudaMemcpyHostToDevice);
	t_start = clock();
	global_reduce<<<block,thread>>>(d_in,d_out_mid);
	global_reduce<<<1,thread>>>(d_out_mid,d_out);
	t_end = clock();
	cudaMemcpy(&h_out,d_out,sizeof(float),cudaMemcpyDeviceToHost);
	printf("GPU(global) sum:%f\n",h_out);
	printf("GPU(global) time:%fms\n",difftime(t_end,t_start));

	cudaMemcpy(d_in,h_in,size*size*sizeof(float),cudaMemcpyHostToDevice);
	t_start = clock();
	shared_reduce<<<block,thread,size*sizeof(float)>>>(d_in,d_out_mid);
	shared_reduce<<<1,thread,size*sizeof(float)>>>(d_out_mid,d_out);
	t_end = clock();
	cudaMemcpy(&h_out,d_out,sizeof(float),cudaMemcpyDeviceToHost);
	printf("GPU(shared) sum:%f\n",h_out);
	printf("GPU(shared) time:%fms\n",difftime(t_end,t_start));

	free(h_in);
	cudaFree(d_in);
	cudaFree(d_out_mid);
	cudaFree(d_out);
	cudaDeviceReset();//重置当前资源
	return 0;
}
