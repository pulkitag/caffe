#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/imchannel2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void imchannel2col_cpu(const Dtype* data_im, const int channels,
    const int imHeight, const int imWidth, 
    const int chHeight, const int chWidth, 
		const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col) {
		//The input image provided as a column is converted into image patches
		// outputted as columns. 

  int height_col = (chHeight + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col  = (chWidth + 2 * pad_w - kernel_w) / stride_w + 1;
  int kernel_sz  =  kernel_h * kernel_w;
	
	for (int c = 0; c < kernel_sz; ++c) {
		//Arrange the elements into column, 1 element per patch at a time.
		//Offsets from the top left corner.
		int w_offset = c % kernel_w;
		int h_offset = c / kernel_w;
		for (int i = 0; i < imHeight; ++i){
			for (int j = 0; j < imWidth; ++j){
				//Goto all image positions indexed as (i,j)
			  //std::cout << "ImPosition: " << i << "," << j << "\n";	
				for (int h = 0; h < height_col; ++h) {
					for (int w = 0; w < width_col; ++w) {
						//For a given position in the patch, find values in all patches. 
						int h_pad = h * stride_h - pad_h + h_offset;
						int w_pad = w * stride_w - pad_w + w_offset;
						int c_im  = h_pad*height_col + w_pad;
						if (h_pad >= 0 && h_pad < chHeight && w_pad >= 0 && w_pad < chWidth){
							*data_col = 	data_im[(c_im * imHeight + i) * imWidth + j];
							//data_col[(c * height_col + h) * width_col + w] =
							//	data_im[(c_im * imHeight + i) * imWidth + j];
							//std::cout << data_im[(c_im * imHeight + i) * imWidth + j] << "\t";
						}
						else{
							//data_col[(c * height_col + h) * width_col + w] = 0;
							*data_col = 0;
						}
						data_col = data_col + 1;
					}
				}
			}
			//At each image position, thses many elements are generated.
			//data_col += height_col*width_col*kernel_sz;
			//std::cout << "\n";
		}
	}
}

// Explicit instantiation
template void imchannel2col_cpu<float>(const float* data_im, const int channels,
    const int imHeight, const int imWidth,
    const int chHeight, const int chWidth,
		const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, 
		const int stride_h, const int stride_w, float* data_col);
template void imchannel2col_cpu<double>(const double* data_im, const int channels,
    const int imHeight, const int imWidth,
    const int chHeight, const int chWidth,
	  const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
	  const int stride_h, const int stride_w, double* data_col);

template <typename Dtype>
void colchannel2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width,
  	const int patch_h, const int patch_w,
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
template void colchannel2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im);
template void colchannel2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im);

}  // namespace caffe
