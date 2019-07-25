#include<stdlib.h>
#include<stdio.h>
#include<time.h>

__global__ void scan(float *d_in,float *d_out,const int size){
	int idx = threadIdx.x;
	d_out[idx] = d_in[idx];
	__syncthreads();
	float out;
	for(int step=1;step<size;step*=2){
		if(idx-step>=0){
			out = d_out[idx]+d_out[idx-step];
			/*
			__syncthreads();
			d_out[idx] = out;
			__syncthreads();
			*/
		}
		
		__syncthreads();
		if(idx-step>0){
			d_out[idx] = out;
		}
		__syncthreads();

	}
}

void init(float *h_in,const int size){
	for(int i=0;i<size;i++)
		h_in[i] = i;
}

int main(){
	int size = 1024;
	float *h_in,*h_out;
	float *d_in,*d_out;
	h_in = (float *)malloc(size*sizeof(float));
	h_out = (float *)malloc(size*sizeof(float));
	init(h_in,size);
	printf("array:");
	for(int i=0;i<size;i++){
		printf("%f ",h_in[i]);
	}
	printf("\n");
	cudaMalloc((float **)&d_in,size*sizeof(float));
	cudaMalloc((float **)&d_out,size*sizeof(float));
	cudaMemcpy(d_in,h_in,size*sizeof(float),cudaMemcpyHostToDevice);
	time_t t_start = clock();
	scan<<<1,size>>>(d_in,d_out,size);
	time_t t_end = clock();
	cudaMemcpy(h_out,d_out,size*sizeof(float),cudaMemcpyDeviceToHost);
	printf("time:%fms\n",difftime(t_end,t_start));
	printf("result:");
	for(int i=0;i<size;i++){
		printf("%f ",h_out[i]);
	}
	printf("\n");
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}

