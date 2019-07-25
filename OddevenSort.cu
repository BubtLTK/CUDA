#include<stdlib.h>
#include<stdio.h>
#include<time.h>
__global__ void oddevenSort(int *d_in,int size,int oe_flag,int &d_ch_flag){
	int idx = threadIdx.x+blockIdx.x*blockDim.x;
	int p = 2*idx+oe_flag;
	if(p+1<size){
		if(d_in[p]>d_in[p+1]){
			int temp = d_in[p];
			d_in[p] = d_in[p+1];
			d_in[p+1] = temp;
			d_ch_flag = 1;
		}
	}
}
void init(int *p,const int size){
	srand((unsigned int)time(NULL));
	for(int i=0;i<size;i++){
		p[i] = rand()%size;
	}
}
void show(int *p,const int size){
	for(int i=0;i<size;i++){
		printf("%d ",p[i]);
	}
	printf("\n");
}
void bubbleSort(int *p,const int size){
	for(int i=0;i<size-1;i++){
		for(int j=0;j<size-i-1;j++){
			if(p[j]>p[j+1]){
				int temp = p[j];
				p[j] = p[j+1];
				p[j+1] = temp;
			}
		}
	}
}
int main(){
	int size = 10*1024;
	int *h_in;
	int *h_out;
	h_in = (int *)malloc(size*sizeof(int));
	h_out = (int *)malloc(size*sizeof(int));
	init(h_in,size);
	//show(h_in,size);
	//printf("\n");
	int *d_in;
	int *d_ch_flag;
	cudaMalloc((int **)&d_in,size*sizeof(int));
	cudaMalloc((int **)&d_ch_flag,sizeof(int));
	cudaMemcpy(d_in,h_in,size*sizeof(int),cudaMemcpyHostToDevice);
	int oe_flag = 0;//判断当前进行排序的类型
	int ch_flag = 1;//判断数组是否发生改变
	dim3 block(10);
	dim3 thread(1024/2);
	time_t t_start = clock();
	while(ch_flag||oe_flag){//偶排序和奇排序必须成对出现
		ch_flag = 0;
		cudaMemcpy(d_ch_flag,&ch_flag,sizeof(int),cudaMemcpyHostToDevice);
		oddevenSort<<<block,thread>>>(d_in,size,oe_flag,d_ch_flag[0]);
		cudaMemcpy(&ch_flag,d_ch_flag,sizeof(int),cudaMemcpyDeviceToHost);
		oe_flag = 1^oe_flag;
	}
	time_t t_end = clock();
	cudaMemcpy(h_out,d_in,size*sizeof(int),cudaMemcpyDeviceToHost);
	//show(h_out,size);
	printf("GPU time:%fms\n",difftime(t_end,t_start));
	t_start = clock();
	bubbleSort(h_in,size);
	t_end = clock();
	printf("CPU time:%fms\n",difftime(t_end,t_start));
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_ch_flag);
	return 0;
}
