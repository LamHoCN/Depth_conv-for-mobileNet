__global__ void depth_conv_small(const float * __restrict__ bottom_data,float *top_data, const float *__restrict__ weights,
                                int channels,int kernel_single_size,int spatial_dim_in,int spatial_dim_out,
                                int spatial_dim_add_padding,int padding,int stride)
{   

    int kernel_size = kernel_single_size*kernel_single_size;
    extern __shared__ float bottom_data_shared[];
    __shared__ float weights_shared[9];

    const int warpid   = threadIdx.x / 32; 
    const int warp_num = blockDim.x  / 32;  
    const int laneid   = threadIdx.x % 32; 
    const int offset   = blockIdx.x * spatial_dim_in * spatial_dim_in; 

    for(int i = threadIdx.x;i<spatial_dim_add_padding*spatial_dim_add_padding;i+=128){
        bottom_data_shared[i] = 0.f;
    }
    __syncthreads();

    for( int i = warpid; i < spatial_dim_in; i += warp_num )
    {
        if( laneid < spatial_dim_in ){
            bottom_data_shared[spatial_dim_add_padding + padding + spatial_dim_add_padding*i + laneid] = __ldg(bottom_data+offset + spatial_dim_in*i + laneid);
        }
    }

    int weights_index = (blockIdx.x%channels)*kernel_size;
    if(threadIdx.x<kernel_size) weights_shared[threadIdx.x] = __ldg(weights+weights_index+threadIdx.x);

    __syncthreads();

    int top_index = blockIdx.x*spatial_dim_out*spatial_dim_out;
    float sum = 0;
    for(int i = warpid;i < spatial_dim_out;i += warp_num)
    {
        int index = laneid*stride;
        if( index <= spatial_dim_add_padding-kernel_single_size){
            sum  = bottom_data_shared[stride*padding*i*spatial_dim_add_padding+index]        * weights_shared[0];
            sum += bottom_data_shared[stride*padding*i*spatial_dim_add_padding+index+1]      * weights_shared[1];
            sum += bottom_data_shared[stride*padding*i*spatial_dim_add_padding+index+2]      * weights_shared[2];

            sum += bottom_data_shared[stride*padding*i*spatial_dim_add_padding+index+spatial_dim_add_padding]     * weights_shared[3];
            sum += bottom_data_shared[stride*padding*i*spatial_dim_add_padding+index+spatial_dim_add_padding+1]   * weights_shared[4];
            sum += bottom_data_shared[stride*padding*i*spatial_dim_add_padding+index+spatial_dim_add_padding+2]   * weights_shared[5];

            sum += bottom_data_shared[stride*padding*i*spatial_dim_add_padding+index+spatial_dim_add_padding*2]   * weights_shared[6];
            sum += bottom_data_shared[stride*padding*i*spatial_dim_add_padding+index+spatial_dim_add_padding*2+1] * weights_shared[7];
            sum += bottom_data_shared[stride*padding*i*spatial_dim_add_padding+index+spatial_dim_add_padding*2+2] * weights_shared[8];
            
            top_data[top_index + spatial_dim_out*i + laneid] = sum;

        }
    }
}

__global__ void depth_conv_big(const float *__restrict__ bottom_data_gpu, float *top_data_gpu, const float *__restrict__ weights_gpu,
                                int channels,int kernel_single_size,
                                int spatial_dim_in,int spatial_dim_out,int spatial_dim_add_padding,
                                int padding,int stride)

{
    extern __shared__ float bottom_data_shared[]; 
    __shared__ float weights_shared[9]; 

    const int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    for(int i = tidx; i<spatial_dim_add_padding*kernel_single_size; i += blockDim.x * blockDim.y) {
        bottom_data_shared[i] = 0.f;
    } 
    __syncthreads();

    int tid = (blockIdx.x/spatial_dim_out)*spatial_dim_in*spatial_dim_in;
    int height_index = (blockIdx.x % spatial_dim_out) * stride + threadIdx.y - padding; //-1

    if((unsigned int)height_index < spatial_dim_in) {
        for(int w = threadIdx.x; w < spatial_dim_in; w += blockDim.x)
        {
            bottom_data_shared[threadIdx.y * spatial_dim_add_padding + w + padding] = __ldg(bottom_data_gpu + tid + height_index * spatial_dim_in + w);
        }
    }
    
    if( threadIdx.y == 0 && threadIdx.x < 9 )
    {
        int threadblock_index_per_batch = blockIdx.x % (channels * spatial_dim_out);
        weights_shared[threadIdx.x] = __ldg(weights_gpu + ( threadblock_index_per_batch / spatial_dim_out ) * 9 + threadIdx.x);
    }
    __syncthreads();

    float sum = 0.f;
    for(int i = threadIdx.x * stride; i <= spatial_dim_add_padding-kernel_single_size; i += stride * blockDim.x) {
        sum  = bottom_data_shared[threadIdx.y * spatial_dim_add_padding + i]     * weights_shared[threadIdx.y * 3];
        sum += bottom_data_shared[threadIdx.y * spatial_dim_add_padding + i + 1] * weights_shared[threadIdx.y * 3 + 1];
        sum += bottom_data_shared[threadIdx.y * spatial_dim_add_padding + i + 2] * weights_shared[threadIdx.y * 3 + 2];


        atomicAdd(top_data_gpu + (blockIdx.x/spatial_dim_out)*spatial_dim_out*spatial_dim_out + (blockIdx.x%spatial_dim_out)*spatial_dim_out + i / stride,sum);
    }
}



/*               
     depth_conv_big<<<mobilenet_channels*out_w,dim3(32, 3),(w+2*pad)*kernelsize*sizeof(float)>>>
                (bottom_data,top_data,weights_gpu,mobilenet_channels,size,w,out_w,(w+2*pad),pad,stride);
*/

/*
    depth_conv_small<<<mobilenet_channels,128,(w+2*pad)*(w+2*pad)*sizeof(float)>>>
                (bottom_data,top_data,weights_gpu,mobilenet_channels,size,w,out_w,(w+2*pad),pad,stride);

*/
