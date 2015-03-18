#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void GlobalEltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "Global Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD)) << 
			"Global Eltwise Layer only takes SUM or PROD";
      
  op_ = this->layer_param_.eltwise_param().operation();

  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }

	//Assert number of top blobs is same as number of bottom blobs
	CHECK(bottom.size() == top.size()) << 
		 "For global Eltwise layer number of bottom and top blobs should be same";
}

template <typename Dtype>
void GlobalEltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	for (int i = 1; i < bottom.size(); ++i) {
    top[i]->Reshape(bottom[i]->num(), bottom[i]->channels(), bottom[i]->height(), bottom[i]->width());
  }
}

template <typename Dtype>
void GlobalEltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_cpu_scale(count, coeffs_[i], bottom[i]->cpu_data(), top[i]->mutable_cpu_data());
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();
			caffe_set(count, Dtype(0), top_data);
			for (int idx = 0; idx < count; idx++)
    		top_data[idx] = bottom_data[idx] + coeffs_[i];	
		}
    break;

   default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void GlobalEltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			const Dtype* top_diff = top[i]->cpu_diff();
		  switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
				caffe_cpu_scale(count, Dtype(1.0) / coeffs_[i], top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
				caffe_copy(count, top_diff, bottom_diff);
				break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GlobalEltwiseLayer);
#endif

INSTANTIATE_CLASS(GlobalEltwiseLayer);
REGISTER_LAYER_CLASS(GlobalEltwise);

}  // namespace caffe
