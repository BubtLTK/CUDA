#include<stdlib.h>
#include<stdio.h>
#include<time.h>
using namespace std;
__global__ void mul(int *d_in1,int *d_in2,int *d_out){
	int idx = threadIdx.x;
	d_out[idx] = d_in1[idx]*d_in2[idx];
}
__global__ void reduce_section(int *d_in,int &d_out,const int start,const int end){
	int idx = threadIdx.x;
	extern __shared__ int s_out[];
	s_out[idx] = d_in[start+idx];
	__syncthreads();
	int out;
	for(int step=1;step<end-start;step*=2){
		if(idx-step>=0){
			out = s_out[idx]+s_out[idx-1];
		}
		__syncthreads();
		if(idx-step>=0)
			s_out[idx] = out;
		__syncthreads();
	}
	if(idx == end-start-1)
		d_out = s_out[idx];
}
int main(){
	const int size = 6;
	int value[size] = {1,2,3,4,5,6};
	int cols[size] = {0,2,1,0,1,0};
	int rows[5] = {0,2,3,5,size};//最后一个元素记录非零元素个数
	int mul_val[3] = {1,2,3};
	int mul_valn[size];//非零元素相乘的对应元素
	printf("左矩阵:\n");
	int flag = 0;
	for(int i=0;i<4;i++){
		for(int i=0;i<3;i++){
			if(i == cols[flag])
				printf("%d ",value[flag++]);
			else
				printf("0 ");
		}
		printf("\n");
	}
	printf("\n右矩阵:\n");
	for(int i=0;i<3;i++){
		printf("%d\n",mul_val[i]);
	}
	printf("\n");
	for(int i=0;i<size;i++){
		mul_valn[i] = mul_val[cols[i]];
	}
	int *h_in1 = value;
	int *h_in2 = mul_valn;
	int *h_out;
	int *d_in1;
	int *d_in2;
	int *d_out_mid;
	int *d_out;
	h_out = (int *)malloc(4*sizeof(int));
	cudaMalloc((int **)&d_in1,size*sizeof(int));
	cudaMalloc((int **)&d_in2,size*sizeof(int));
	cudaMalloc((int **)&d_out,4*sizeof(int));
	cudaMalloc((int **)&d_out_mid,size*sizeof(int));
	cudaMemcpy(d_in1,h_in1,size*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_in2,h_in2,size*sizeof(int),cudaMemcpyHostToDevice);
	dim3 thread(size);
	mul<<<1,thread>>>(d_in1,d_in2,d_out_mid);
	for(int i=1;i<5;i++){
		int sizenew = rows[i]-rows[i-1];
		dim3 threadnew(sizenew);
		reduce_section<<<1,threadnew,sizenew>>>(d_out_mid,d_out[i-1],rows[i-1],rows[i]);
	}
	cudaMemcpy(h_out,d_out,4*sizeof(int),cudaMemcpyDeviceToHost);
	printf("结果:\n");
	for(int i=0;i<4;i++){
		printf("%d\n",h_out[i]);
	}
	printf("\n");
	free(h_out);
	cudaFree(d_in1);
	cudaFree(d_in2);
	cudaFree(d_out_mid);
	cudaFree(d_out);
	return 0;
}
