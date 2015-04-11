#include "caffe/common.hpp"
#include "caffe/util/filter_kernel.hpp"
#include <cmath>

namespace caffe {
template<typename Dtype>
void caffe_gaussian_kernel(Dtype* gk, const Dtype sd, const int sz){
	Dtype coeff_sum = 0;

	int cx = sz/2;
	int cy = sz/2;
	for (int r = 0; r < sz; ++r){
		for (int c = 0; c < sz; ++c){
			Dtype dist = (c - cx) * (c - cx) + (r - cy) * (r - cy);
			gk[r * sz + c]      = exp(- (dist / (2 * sd * sd)));
			coeff_sum  += gk[r * sz + c];
		}
	}

	CHECK_GT(coeff_sum,0)<<"Something wrong in forming kernel";
	//Normalize
	for (int r = 0; r < sz; ++r){
		for (int c = 0; c < sz; ++c){
			gk[r * sz + c] = (gk[r * sz + c]) / coeff_sum;
		}
	}
}

//Explicit instantiation
template void caffe_gaussian_kernel<float>(float* gk, const float sd, const int sz);
template void caffe_gaussian_kernel<double>(double* gk, const double sd, const int sz);
 
template<typename Dtype>
void caffe_gaussian_kernel_1d(Dtype* gk, const Dtype sd, const int sz){
	Dtype coeff_sum = 0;

	int center = sz/2;
	for (int r = 0; r < sz; ++r){
		Dtype dist = (r - center) * (r - center);
		gk[r]      = exp(- (dist / (2 * sd * sd)));
		coeff_sum  += gk[r];
	}

	CHECK_GT(coeff_sum,0)<<"Something wrong in forming kernel";
	//Normalize
	for (int r = 0; r < sz; ++r){
			gk[r] = (gk[r]) / coeff_sum;
	}
}

//Explicit instantiation
template void caffe_gaussian_kernel_1d<float>(float* gk, const float sd, const int sz);
template void caffe_gaussian_kernel_1d<double>(double* gk, const double sd, const int sz);
 

} //namespace caffe 
