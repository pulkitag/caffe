#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/imchannel2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

__device__ int my_modulus(int a, int b){
	if (a < 0){
		int q = (-a/b) + 1;
		return (a + b*q) % b;
	}else
		return a % b;	
}

template <typename Dtype>
__global__ void imchannel2col_gpu_kernel(const int n, const Dtype* data_im,
    const int imHeight, const int imWidth,
	  const int chHeight, const int chWidth,
	  const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
	
		//Total number of patches. 
		int totalPatches = imWidth * imHeight * width_col * height_col; 
		if (index > totalPatches)
			return;

		int numPatchPerImPosition = width_col*height_col; // Number of patches per position.
		int imPosition            = index / numPatchPerImPosition; //division is alwars rounded off.
		int imcol_index = imPosition % imWidth;
		int imrow_index = imPosition / imWidth; 
    int patch_index = index % numPatchPerImPosition;
		int w_out       = patch_index % width_col;
    int h_out       = patch_index / width_col;

    int h_in       = h_out * stride_h - pad_h;
    int w_in       = w_out * stride_w - pad_w;
    
		//Save data pointer
		Dtype* data_col_ptr = data_col;
    
		//Wrong format. 
		//data_col_ptr += imPosition * numPatchPerImPosition + h_out * width_col + w_out;
		data_col_ptr += (h_out * width_col + w_out) * imWidth * imHeight + 
										imrow_index * imWidth + imcol_index;   

		//Image data pointer
		const Dtype* data_im_ptr = data_im;
    data_im_ptr += imrow_index * imWidth + imcol_index;
	
		//Begin Copy
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
				//Introducing topography
				h = my_modulus(h, chHeight);
				w = my_modulus(w, chWidth);
				//
        *data_col_ptr = (h >= 0 && w >= 0 && h < chHeight && w < chWidth) ?
            data_im_ptr[(h * chWidth + w) * imWidth * imHeight] : 0;
        data_col_ptr += totalPatches;
      }
    }
  }
}

template <typename Dtype>
void imchannel2col_gpu(const Dtype* data_im, const int channels,
    const int imHeight, const int imWidth, 
    const int chHeight, const int chWidth, 
		const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  // We are going to launch processes/kernels, for  each position in the image and for all patches 
	// at that location.
  // Each Process is responsible for generating its image patch as a column.
  int height_col = (chHeight + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col  = (chWidth + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = imHeight * imWidth * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  imchannel2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, imHeight, imWidth, chHeight, chWidth,  kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void imchannel2col_gpu<float>(const float* data_im, const int channels,
    const int imHeight, const int imWidth,
    const int chHeight, const int chWidth,
	  const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
	  const int stride_h, const int stride_w,
    float* data_col);
template void imchannel2col_gpu<double>(const double* data_im, const int channels,
    const int imHeight, const int imWidth, 
    const int chHeight, const int chWidth, 
		const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
	  const int stride_h, const int stride_w,
    double* data_col);

template <typename Dtype>
__global__ void colchannel2im_gpu_kernel(const int n, const Dtype* data_col,
    const int imHeight, const int imWidth, 
    const int chHeight, const int chWidth, 
		const int channels,
    const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  
	CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
	
		//Get the location of bottom layer element. 
		int chn = index / (imWidth * imHeight);
    int col = index % imWidth;
    int row = (index / imWidth) % imHeight;
  
		//Now I need to compute the index of all the elements in data_col
		// which used the data_im(chn,row,col). 
		// data_col is K * N, where K is kernel_sz and N is number of patches.

		int ch_w = chn % chWidth + pad_w;
		int ch_h = chn / chWidth + pad_h; 
		int col_row_length = imHeight * imWidth * height_col * width_col; 
	  
		/*
		// compute the start and end of the output
    int w_col_start = (ch_w < patch_w) ? 0 : (ch_w - patch_w) / stride_w + 1;
    int w_col_end   = min(ch_w / stride_w + 1, width_col);
    int h_col_start = (ch_h < patch_h) ? 0 : (ch_h - patch_h) / stride_h + 1;
    int h_col_end   = min(ch_h / stride_h + 1, height_col);
	 */

	 //For wrapping into toroid. 
    int w_col_start = (ch_w - patch_w) / stride_w + 1;
    int w_col_end   = ch_w / stride_w + 1;
    int h_col_start = (ch_h - patch_h) / stride_h + 1;
    int h_col_end   = ch_h / stride_h + 1;
	  
		//int q;
		for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
				int offset = 0;
				//Get position within the patch.
				int h_start = h_col * stride_h;
				int w_start = w_col * stride_w;
				int h       = ch_h - h_start;
				int w       = ch_w - w_start;
				//Toroidal wrapping
				h           = my_modulus(h, patch_h);
				w           = my_modulus(w, patch_w);
				// 
				int pos     = h * patch_w + w;	
				offset     += pos * col_row_length;

				//Toroidal wrapping
				int h_col_wrap, w_col_wrap;
				h_col_wrap = my_modulus(h_col, height_col);
				w_col_wrap = my_modulus(w_col, width_col);
				//
	
				//Get the patch num
				int patch_num  = h_col_wrap * width_col + w_col_wrap;	
				offset        += patch_num * imHeight * imWidth;
				
				//offset due to image location. 
				offset  +=  row * imWidth + col; 
        val += data_col[offset];
    		//printf("offset: %d", offset);
		  }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void colchannel2im_gpu(const Dtype* data_col, const int channels,
    const int imHeight, const int imWidth, 
    const int chHeight, const int chWidth, 
		const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im) {
  int height_col = (chHeight + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col  = (chWidth + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * imHeight * imWidth;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  colchannel2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, imHeight, imWidth, chHeight, chWidth, 
			channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void colchannel2im_gpu<float>(const float* data_col, const int channels,
    const int imHeight, const int imWidth, 
    const int chHeight, const int chWidth, 
		const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void colchannel2im_gpu<double>(const double* data_col, const int channels,
    const int imHeight, const int imWidth, 
    const int chHeight, const int chWidth, 
		const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);

}  // namespace caffe
