#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {


template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	//Initializw the Convolution Layer
	NormalizeParameter norm_param = this->layer_param_.normalize_param(); 

  num_      = bottom[0]->num();
	channels_ = bottom[0]->channels(); 
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();
	for (int i=1; i < bottom.size(); i++){
		int num2      = bottom[i]->num();
		int channels2 = bottom[i]->channels(); 
		int height2   = bottom[i]->height();
		int width2    = bottom[i]->width();
		CHECK_EQ(num_, num2) << "Number of examples mismatch";
		CHECK_EQ(channels_, channels2) << "Number of examples mismatch";
		CHECK_EQ(height_, height2) << "Height-MisMatch";
		CHECK_EQ(width_, width2)   << "Width-MisMatch";
	}

	this->param_propagate_down_.resize(1, true);
	
	imSz_ = channels_ * height_ * width_;
	this->blobs_.resize(1);
	this->blobs_[0].reset(new Blob<Dtype>(
			1, channels_, height_, width_));
}


template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
	//Set the shape of output blobs
	for (int i=0; i<top.size(); i++){
		top[i]->Reshape(num_, channels_, height_, width_);
	}
}


template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
   	Dtype scal;
		for (int n = 0; n < this->num_; ++n) {
			caffe_copy(imSz_, bottom_data + n * imSz_, this->blobs_[0]->mutable_cpu_data());
			switch (op_) { 
			case NormalizeParameter_NormalizeOp_DEMEAN:
				caffe_cpu_zero_mean(imSz_, this->blobs_[0]->mutable_cpu_data());
				break;
			case NormalizeParameter_NormalizeOp_SDSCALE:
				caffe_cpu_zero_mean(imSz_, this->blobs_[0]->mutable_cpu_data());
				scal = caffe_cpu_dot<Dtype>(imSz_, this->blobs_[0]->cpu_data(), 
											this->blobs_[0]->cpu_data()); 
				caffe_scal(imSz_, Dtype(1.0 / scal), this->blobs_[0]->mutable_cpu_data());  
				break;
			default:
				LOG(FATAL) << "Unknown elementwise operation.";
			}
			caffe_copy(imSz_, this->blobs_[0]->cpu_data(), top_data + n * imSz_); 
    }
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			const Dtype* top_diff    = top[i]->cpu_diff();
			Dtype* bottom_diff       = bottom[i]->mutable_cpu_diff();
			Dtype scal;
			for (int n=0; n <num_; n++){
				switch (op_) { 
				case NormalizeParameter_NormalizeOp_DEMEAN:
					caffe_copy(imSz_, top_diff + n * imSz_, bottom_diff + n * imSz_);
					break;
				case NormalizeParameter_NormalizeOp_SDSCALE:
					caffe_copy(imSz_, bottom_data + n * imSz_, this->blobs_[0]->mutable_cpu_data());
					caffe_copy(imSz_, top_diff + n * imSz_, this->blobs_[0]->mutable_cpu_diff());
					//Find the Scaling Factor
					caffe_cpu_zero_mean(imSz_, this->blobs_[0]->mutable_cpu_data());
					scal = caffe_cpu_dot<Dtype>(imSz_, this->blobs_[0]->cpu_data(), 
												this->blobs_[0]->cpu_data()); 
					//Apply the scaling to the gradients
					caffe_scal(imSz_, Dtype(1.0 / scal), this->blobs_[0]->mutable_cpu_diff());  
					caffe_copy(imSz_, this->blobs_[0]->cpu_diff() , bottom_diff + n * imSz_);
					break;
				default:
					LOG(FATAL) << "Unknown elementwise operation.";
				}
			}
    }
	}
}

#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
