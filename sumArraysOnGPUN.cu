#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<unistd.h>

__global__ void sumArraysOnGPUN(float *A,float *B,float *C,const int N){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if(idx<N)
		C[idx] = A[idx] + B[idx];
	printf(" %f + %f = %f   On GPU:block %d thread %d\n",A[idx],B[idx],C[idx],blockIdx.x,threadIdx.x);
}
void initialData(float *ip,const int size){
	time_t t;
	srand((unsigned int)time(&t));
	for(int i=0;i<size;i++){
		ip[i] = (float)(rand()%100)/1.0f;
	}
}
void print(float *array,const int size){
	for(int i=0;i<size;i++){
		printf(" %f",array[i]);
	}
	printf("\n");
}

int main(){
	int n;
	scanf("%d",&n);
	int nBytes = n*sizeof(float);
	float *h_A,*h_B,*h_C;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	h_C = (float *)malloc(nBytes);
	initialData(h_A,n);
	sleep(1);
	initialData(h_B,n);
	print(h_A,n);
	print(h_B,n);
	float *d_A,*d_B,*d_C;
	cudaMalloc((float **)&d_A,nBytes);
	cudaMalloc((float **)&d_B,nBytes);
	cudaMalloc((float **)&d_C,nBytes);
	cudaMemcpy(d_A,h_A,nBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,nBytes,cudaMemcpyHostToDevice);
	dim3 block(1);
	dim3 thread(n);
	sumArraysOnGPUN<<<block,thread>>>(d_A,d_B,d_C,n);
	cudaMemcpy(h_C,d_C,nBytes,cudaMemcpyDeviceToHost);
	print(h_C,n);
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}
