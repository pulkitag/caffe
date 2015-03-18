#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BlockCorrelationLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
	//Check that there are two bottom blobs
	CHECK_EQ(bottom.size(), 2, "BLOCK_CORRELATION_LOSS_LAYER expects two bottom blobs");
	
	block_size_ = this->layer_param_.blockcorrelationloss_param().block_size();
	nb_         = bottom[0]->num() / block_size_;
	ch_         = bottom[0]->channels();
	CHECK_EQ(nb_ * block_size_, bottom[0]->num());

	//Setup the block data for the two input streams
	block_data_.clear();
	block_data_.push_back(Blob<Dtype>::Blob(nb_, ch_, 1, 1));
	block_data_.push_back(Blob<Dtype>::Blob(nb_, ch_, 1, 1));
}

template <typename Dtype>
void BlockCorrelationLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "BLOCK_CORRELATION_LOSS_Layer inputs must have the same count.";
	CHECK_EQ(bottom[0]->height(), 1, "Only Height of 1 is supported by BLOCK_CORRELATION_LAYER");
	CHECK_EQ(bottom[0]->width(),  1, "Only Width  of 1 is supported by BLOCK_CORRELATION_LAYER");

	//Shape the block Datas
	block_data_[0]->Reshape(nb_, ch_, 1, 1);
	block_data_[1]->Reshape(nb_, ch_, 1, 1);
}

template <typename Dtype>
void BlockCorrelationLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num   = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* ipData0 = bottom[0]->cpu_data();
  const Dtype* ipData1 = bottom[1]->cpu_data();
	Dtype*       bData0  = block_data_[0]->mutable_cpu_data();
	Dtype*       bData1  = block_data_[1]->mutable_cpu_data();
	int st, en, count;
	for (int b = 0; b < nb_; ++b){
		//Iterate over the blocks
		st = b * block_size_;
		en = st + block_size_;
		count = 0;
		for int(i = st; i < en; i++){
			//Copy each block
			for (int c = 0; c < ch_; ++c){
				bData0[count] = ipData0[i * ch_ + c];
				bData1[count] = ipData1[i * ch_ + c];
				count        += 1;	
			}
		}
		//Get the correlation
		caffe_mul(nb_ * ch_, bData0, bData1, opData);
	}
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void BlockCorrelationLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(BlockCorrelationLossLayer);
#endif

INSTANTIATE_CLASS(BlockCorrelationLossLayer);
REGISTER_LAYER_CLASS(BlockCorrelationLoss);

}  // namespace caffe
