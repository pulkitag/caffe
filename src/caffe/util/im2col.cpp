#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
	/*The way this works is as following:
	Suppose we want to extract patches on a N x N grid. 
	Then for each grid location - store the first element
	so that the first N^2 elements in data_col are the first element 
	of the patches. Then the N^2 elements of second patch element and so on.	
	*/
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);


template <typename Dtype>
void im2filtercol_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
  /*The way this works is as following:
	Suppose we want to extract patches on a N x N grid. 
	Then we extract the entire patch at each location and store.
	We store N x N such patches consequently. 	
	*/
	//The channels, height and width of the data_col
	int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col  = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	int filter_sz  = channels * kernel_h * kernel_w;
	for (int hcol = 0; hcol < height_col; ++hcol) {
	for (int wcol = 0; wcol < width_col; ++wcol) {
		int chcol = hcol * width_col + wcol;
		for (int c_im = 0; c_im < channels; ++c_im){
		for (int h = 0; h < kernel_h; ++h) {
		for (int w = 0; w < kernel_w; ++w) {
			int h_pad = hcol * stride_h - pad_h + h;
			int w_pad = wcol * stride_w - pad_w + w;
			if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
				data_col[chcol * filter_sz + (c_im * kernel_h + h) * kernel_w + w] = 
					data_im[(c_im * height + h_pad) * width + w_pad];
			else
				data_col[chcol * filter_sz + (c_im * kernel_h + h) * kernel_w + w] = 0;
		}
		}
		}
	}
	}
}

// Explicit instantiation
template void im2filtercol_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col);
template void im2filtercol_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col);

template <typename Dtype>
void filtercol2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im) {
	caffe_set(height * width * channels, Dtype(0), data_im);
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
	int filter_sz  = channels * patch_h * patch_w;
  for (int hcol = 0; hcol < height_col; ++hcol) {
	for (int wcol = 0; wcol < width_col; ++wcol) {
		int chcol = hcol * width_col + wcol;
		for (int c_im = 0; c_im < channels; ++c_im){
		for (int h = 0; h < patch_h; ++h) {
		for (int w = 0; w < patch_w; ++w) {
			int h_pad = hcol * stride_h - pad_h + h;
			int w_pad = wcol * stride_w - pad_w + w;
			if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
				data_im[(c_im * height + h_pad) * width + w_pad] += 
					data_col[chcol * filter_sz + (c_im * patch_h + h) * patch_w + w];
		}
		}
		}
	}
	}
}

// Explicit instantiation
template void filtercol2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void filtercol2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);



}  // namespace caffe
