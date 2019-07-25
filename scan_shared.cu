#include<stdlib.h>
#include<stdio.h>
#include<time.h>
__global__ void scan_shared(float *d_in,float *d_out,const int size){
	extern __shared__ float s_in[];
	int idx = threadIdx.x;
	s_in[idx] = d_in[idx];
	__syncthreads();
	float out;
	for(int step=1;step<size;step*=2){
		if(idx-step>=0){
			out = s_in[idx]+s_in[idx-step];
		}
		__syncthreads();
		if(idx-step>=0){
			s_in[idx] = out;
		}
		__syncthreads();
	}
	d_out[idx] = s_in[idx];
}

void init(float *p,const int size){
	for(int i=0;i<size;i++)
		p[i] = i;
}

int main(){
	int size = 1024;
	float *h_in,*h_out;
	float *d_in,*d_out;
	h_in = (float *)malloc(size*sizeof(float));
	h_out = (float *)malloc(size*sizeof(float));
	init(h_in,size);
	printf("array:");
	for(int i=0;i<size;i++)
		printf("%f ",h_in[i]);
	printf("\n");
	cudaMalloc((float **)&d_in,size*sizeof(float));
	cudaMalloc((float **)&d_out,size*sizeof(float));
	cudaMemcpy(d_in,h_in,size*sizeof(float),cudaMemcpyHostToDevice);
	time_t t_start = clock();
	scan_shared<<<1,size,size*sizeof(float)>>>(d_in,d_out,size);
	time_t t_end = clock();
	cudaMemcpy(h_out,d_out,size*sizeof(float),cudaMemcpyDeviceToHost);
	printf("time:%fms\n",difftime(t_end,t_start));
	printf("result:");
	for(int i=0;i<size;i++)
		printf("%f ",h_out[i]);
	printf("\n");
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
