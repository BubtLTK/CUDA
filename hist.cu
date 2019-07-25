#include<stdlib.h>
#include<stdio.h>
#include<time.h>

void init_in(int *h_in,const int size){
	srand((unsigned int)time(NULL));
	for(int i=0;i<size;i++)
		h_in[i] = rand()%size;
}

void init_out(int *h_out,const int size){
	for(int i=0;i<size;i++)
		h_out[i] = 0;
}

void hist_normal(int *h_in,int *h_out,const int size,const int bin_size){
	for(int i=0;i<size;i++){
		int no = h_in[i]/bin_size;
		h_out[no]++;
	}
}

__global__ void hist_atomic(int *d_in,int *d_out,const int bin_size){
	int idx = threadIdx.x+blockIdx.x*blockDim.x;
	int no = d_in[idx]/bin_size;
	atomicAdd(&d_out[no],1);
}

__global__ void hist_local(int *d_in,int *d_out,const int bin_size,const int thread_size){
	int idx = threadIdx.x;
	int l_out[32] = {0};
	for(int i=idx*thread_size;i<idx*thread_size+thread_size;i++){
		int no = d_in[i]/bin_size;
		l_out[no]++;
	}
	for(int i=0;i<32;i++)
		atomicAdd(&d_out[i],l_out[i]);
}

void show_hist(int *h_out,const int bin_num){
	for(int i=0;i<bin_num;i++)
		printf("%d : %d\n",i+1,h_out[i]);
	printf("\n");
}

int main(){
	int size = 262144;
	int bin_num = 32;
	int bin_size = size/bin_num;
	int *h_in,*h_out;
	int *d_in,*d_out;
	h_in = (int *)malloc(size*sizeof(int));
	h_out = (int *)malloc(bin_num*sizeof(int));
	init_in(h_in,size);
	init_out(h_out,bin_num);
	time_t t_start = clock();
	hist_normal(h_in,h_out,size,bin_size);
	time_t t_end = clock();
	printf("hist_normal_time: %fms\n",difftime(t_end,t_start));
	printf("hist_normal_result: \n");
	show_hist(h_out,bin_num);

	cudaMalloc((int **)&d_in,size*sizeof(int));
	cudaMalloc((int **)&d_out,bin_num*sizeof(int));
	cudaMemcpy(d_in,h_in,size*sizeof(int),cudaMemcpyHostToDevice);
	dim3 block1(256);
	dim3 thread1(1024);
	t_start = clock();
	hist_atomic<<<block1,thread1>>>(d_in,d_out,bin_size);
	t_end = clock();
	cudaMemcpy(h_out,d_out,bin_num*sizeof(int),cudaMemcpyDeviceToHost);
	printf("hist_atomic_time: %fms\n",difftime(t_end,t_start));
	printf("hist_atomic_result: \n");
	show_hist(h_out,bin_num);

	int n = 64;
	dim3 thread2(n);
	init_out(h_out,bin_num);
	cudaMemcpy(d_out,h_out,bin_num*sizeof(int),cudaMemcpyHostToDevice);
	t_start = clock();
	hist_local<<<1,thread2>>>(d_in,d_out,bin_size,size/n);
	t_end =clock();
	cudaMemcpy(h_out,d_out,bin_num*sizeof(int),cudaMemcpyDeviceToHost);
	printf("hist_local_time: %fms\n",difftime(t_end,t_start));
	printf("hist_local_result: \n");
	show_hist(h_out,bin_num);

	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);

	cudaDeviceReset();
	return 0;
}


