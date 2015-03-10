#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void GlobalEltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    for (int i = 0; i < bottom.size(); ++i) {
			Dtype* top_data = top[i]->mutable_gpu_data();
			caffe_gpu_set(count, Dtype(0.), top_data);
      caffe_gpu_scale(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data          = top[i]->mutable_gpu_data();
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
void GlobalEltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff    = bottom[i]->mutable_gpu_diff();
			const Dtype* top_diff = top[i]->gpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
				caffe_gpu_scale(count, Dtype(1.0) / coeffs_[i], top_diff, bottom_diff);
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

INSTANTIATE_LAYER_GPU_FUNCS(GlobalEltwiseLayer);

}  // namespace caffe
