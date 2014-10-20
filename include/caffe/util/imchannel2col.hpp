#ifndef _CAFFE_UTIL_IMCHANNEL2COL_HPP_
#define _CAFFE_UTIL_IMCHANNEL2COL_HPP_

namespace caffe {

template <typename Dtype>
void imchannel2col_cpu(const Dtype* data_im, const int channels,
    const int imHeight, const int imWidth, 
    const int chHeight, const int chWidth, 
		const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col);

template <typename Dtype>
void colchannel2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);


template <typename Dtype>
void imchannel2col_gpu(const Dtype* data_im, const int channels,
    const int imHeight, const int imWidth, 
    const int chHeight, const int chWidth, 
		const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col);

template <typename Dtype>
void colchannel2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im);

}  // namespace caffe

#endif  // CAFFE_UTIL_IMCHANNEL2COL_HPP_
