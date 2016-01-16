#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"


namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
		Dtype scal = 1.0; 
   for (int n = 0; n < this->num_; ++n) {
			caffe_copy(imSz_, bottom_data + n * imSz_, this->blobs_[0]->mutable_gpu_data());
			switch (op_) { 
			case NormalizeParameter_NormalizeOp_DEMEAN:
				caffe_gpu_zero_mean(imSz_, this->blobs_[0]->mutable_gpu_data());
				break;
			case NormalizeParameter_NormalizeOp_SDSCALE:
				caffe_gpu_zero_mean(imSz_, this->blobs_[0]->mutable_gpu_data());
				caffe_gpu_dot<Dtype>(imSz_, this->blobs_[0]->gpu_data(), 
											this->blobs_[0]->gpu_data(), &scal); 
				caffe_scal(imSz_, Dtype(1.0 / scal), this->blobs_[0]->mutable_gpu_data());  
				break;
			default:
				LOG(FATAL) << "Unknown elementwise operation.";
			}
			caffe_copy(imSz_, this->blobs_[0]->gpu_data(), top_data + n * imSz_); 
    }
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			const Dtype* top_diff    = top[i]->gpu_diff();
			Dtype* bottom_diff       = bottom[i]->mutable_gpu_diff();
			Dtype scal = 1.0;	
			for (int n=0; n < num_; n++){
				switch (op_) { 
				case NormalizeParameter_NormalizeOp_DEMEAN:
					caffe_copy(imSz_, top_diff + n * imSz_, bottom_diff + n * imSz_);
					break;
				case NormalizeParameter_NormalizeOp_SDSCALE:
					caffe_copy(imSz_, bottom_data + n * imSz_, this->blobs_[0]->mutable_gpu_data());
					caffe_copy(imSz_, top_diff + n * imSz_, this->blobs_[0]->mutable_gpu_diff());
					//Find the Scaling Factor
					caffe_gpu_zero_mean(imSz_, this->blobs_[0]->mutable_gpu_data());
					caffe_gpu_dot<Dtype>(imSz_, this->blobs_[0]->gpu_data(),
												this-> blobs_[0]->gpu_data(), &scal); 
					//Apply the scaling to the gradients
					caffe_scal(imSz_, Dtype(1.0 / scal), this->blobs_[0]->mutable_gpu_diff());  
					caffe_copy(imSz_, this->blobs_[0]->gpu_diff() , bottom_diff + n * imSz_);
					break;
				default:
					LOG(FATAL) << "Unknown elementwise operation.";
				}
			}
    }
	}
}


INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);
}
