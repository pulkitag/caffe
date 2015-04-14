#include <vector>
#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void CropDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //The thread joining will be taked care by GenericWindowData Layer
	// Check that thread has already ended. 
  CHECK(!this->is_started());
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  //New prefetch thread will be started by GenericWindowData layer.
}

INSTANTIATE_LAYER_GPU_FORWARD(CropDataLayer);

}  // namespace caffe
