#ifndef CAFFE_FILTER_KERNEL_HPP_
#define CAFFE_FILTER_KERNEL_HPP_
namespace caffe{

template<typename Dtype>
void caffe_gaussian_kernel(Dtype* gk, const Dtype sd, const int sz); 

template<typename Dtype>
void caffe_gaussian_kernel_1d(Dtype* gk, const Dtype sd, const int sz); 
} //namespace caffe
#endif
