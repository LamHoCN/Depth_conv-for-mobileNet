# Depth_conv-for-mobileNet
Depth_conv for MobileNet 3x3

       if the feature_map_size >32 (warpSize) use depth_conv_big()
              feature_map_size <32            use depth_conv_small()
       
       
## Depth_conv use group:

       layer        filters            size        input                  output            time 
       depth_conv_1     32             3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  32   gpu_time:0.542720 ms 
       depth_conv_2     64             3 x 3 / 2   208 x 208 x  64   ->   104 x 104 x  64   gpu_time:0.364544 ms
       depth_conv_3     128            3 x 3 / 1   104 x 104 x 128   ->   104 x 104 x 128   gpu_time:0.727040 ms
       depth_conv_4     128            3 x 3 / 2   104 x 104 x 128   ->    52 x  52 x 128   gpu_time:0.386048 ms
       depth_conv_5     256            3 x 3 / 1    52 x  52 x 256   ->    52 x  52 x 256   gpu_time:0.770048 ms
       depth_conv_6     256            3 x 3 / 2    52 x  52 x 256   ->    26 x  26 x 256   gpu_time:0.677888 ms
       depth_conv_7_1   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:1.367040 ms
       depth_conv_7_2   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:1.360896 ms
       depth_conv_7_3   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:1.358848 ms
       depth_conv_7_4   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:1.420288 ms
       depth_conv_7_5   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:1.361920 ms
       depth_conv_8     512            3 x 3 / 2    26 x  26 x 512   ->    13 x  13 x 512   gpu_time:1.297408 ms
       depth_conv_9    1024            3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024   gpu_time:2.599936 ms
## Total time: 12.867584ms
 
## CUDA implementation on caffe by liuhao 
## https://github.com/yonghenglh6/DepthwiseConvolution

       layer        filters            size        input                  output            time
       depth_conv_1     32             3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  32   gpu_time:0.248832 ms
       depth_conv_2     64             3 x 3 / 2   208 x 208 x  64   ->   104 x 104 x  64   gpu_time:0.148480 ms
       depth_conv_3     128            3 x 3 / 1   104 x 104 x 128   ->   104 x 104 x 128   gpu_time:0.245760 ms
       depth_conv_4     128            3 x 3 / 2   104 x 104 x 128   ->    52 x  52 x 128   gpu_time:0.060416 ms
       depth_conv_5     256            3 x 3 / 1    52 x  52 x 256   ->    52 x  52 x 256   gpu_time:0.096256 ms
       depth_conv_6     256            3 x 3 / 2    52 x  52 x 256   ->    26 x  26 x 256   gpu_time:0.031744 ms
       depth_conv_7_1   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:0.044032 ms
       depth_conv_7_2   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:0.045056 ms
       depth_conv_7_3   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:0.045056 ms
       depth_conv_7_4   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:0.045056 ms
       depth_conv_7_5   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:0.045056 ms
       depth_conv_8     512            3 x 3 / 2    26 x  26 x 512   ->    13 x  13 x 512   gpu_time:0.019456 ms
       depth_conv_9    1024            3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024   gpu_time:0.025600 ms
## Total time: 1.1008512 ms


 
 
## My CUDA implementation: 

       layer        filters            size        input                  output            time
       depth_conv_1     32             3 x 3 / 1   208 x 208 x  32   ->   208 x 208 x  32   gpu_time:0.169536 ms
       depth_conv_2     64             3 x 3 / 2   208 x 208 x  64   ->   104 x 104 x  64   gpu_time:0.122880 ms
       depth_conv_3     128            3 x 3 / 1   104 x 104 x 128   ->   104 x 104 x 128   gpu_time:0.182272 ms
       depth_conv_4     128            3 x 3 / 2   104 x 104 x 128   ->    52 x  52 x 128   gpu_time:0.072704 ms
       depth_conv_5     256            3 x 3 / 1    52 x  52 x 256   ->    52 x  52 x 256   gpu_time:0.099328 ms
       depth_conv_6     256            3 x 3 / 2    52 x  52 x 256   ->    26 x  26 x 256   gpu_time:0.040960 ms
       depth_conv_7_1   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:0.017408 ms
       depth_conv_7_2   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:0.017408 ms
       depth_conv_7_3   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:0.017952 ms
       depth_conv_7_4   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:0.018432 ms
       depth_conv_7_5   512            3 x 3 / 1    26 x  26 x 512   ->    26 x  26 x 512   gpu_time:0.018432 ms
       depth_conv_8     512            3 x 3 / 2    26 x  26 x 512   ->    13 x  13 x 512   gpu_time:0.012288 ms
       depth_conv_9    1024            3 x 3 / 1    13 x  13 x1024   ->    13 x  13 x1024   gpu_time:0.014336 ms
## Total time: 0.803936 ms








 

