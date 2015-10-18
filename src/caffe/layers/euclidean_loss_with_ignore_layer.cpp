#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossWithIgnoreLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top){
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		EuclideanLossParameter euclid_loss = this->layer_param_.euclideanloss_param();
		is_normalize_     = euclid_loss.is_normalized();
		nc_               = euclid_loss.normalize_choice(); 
		CHECK_GE(nc_, 0) << "normalize_choice can be 0 or 1";
		CHECK_LE(nc_, 1) << "normalize_choice can be 0 or 1"; 
		
		/*
		LOG(INFO) << "Setup Euclidean " << "Normalize: " << is_normalize_ << " nc_: " << nc_;
		if (is_normalize_)
			LOG(INFO) << "normalization is ON";
		else
			LOG(INFO) << "normalization is OFF";
		*/
}


template <typename Dtype>
void EuclideanLossWithIgnoreLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
	CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
				<< "Batch Size should be the same";
	CHECK_EQ(bottom[0]->shape(1) + 1, bottom[1]->shape(1))
				<< "The label sizes donot match";
	int extraCount;
	if (bottom[1]->num_axes() > 2)
		extraCount = bottom[1]->count(2, bottom[1]->num_axes());
	else
		extraCount = 1;
	extraCount = extraCount * bottom[0]->shape(0);
  CHECK_EQ(bottom[0]->count() + extraCount, bottom[1]->count())
      << "Inputs must have the same dimension.";

	//Check that number of axes are the same
	CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes())
				<<"Number of axes should match";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossWithIgnoreLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count  = bottom[0]->count();
	int N      = bottom[0]->shape(0);

	//Number of elements in the batch
	int bCount  = bottom[0]->count(1, bottom[0]->num_axes());
	int lbCount = bottom[1]->count(1, bottom[1]->num_axes());
	lCount_    = 0;
	Dtype loss = 0.0;
	Dtype Z    = 0.0; //The normalization factor
	const Dtype* labels = bottom[1]->cpu_data();
	const Dtype* preds  = bottom[0]->cpu_data();
	Dtype* diff         = diff_.mutable_cpu_data();
	const Dtype* diffC  = diff_.cpu_data();
	for (int i=0; i<N; i++){
		Dtype bLoss = 0;
		if (labels[bCount] == Dtype(1)){
			//Example needs to be considered
			caffe_sub(bCount, preds, labels, diff);
			bLoss =  caffe_cpu_dot(bCount, diffC, diffC);
			if (nc_==0){
				Z    = caffe_cpu_dot(bCount, preds, preds);
			}else {
				Z    = caffe_cpu_dot(bCount, labels, labels);
			}
			lCount_ += 1;
		} 
		preds   += bCount;
		labels  += lbCount;
		diff    += bCount;
		diffC   += bCount; 
		if (is_normalize_){
			if (Z > 0){
				bLoss = bLoss / Z;
			}
		}
		loss = loss + bLoss;
	}
	if (lCount_ > 0){
		loss = loss / lCount_ / Dtype(2);
	}
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossWithIgnoreLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int count  = bottom[0]->count();
	//Number of elements in the batch
	int bCount  = bottom[0]->count(1, bottom[0]->num_axes());
	int lbCount = bottom[1]->count(1, bottom[1]->num_axes());
	int N      = bottom[0]->shape(0);
	if (lCount_ == Dtype(0)){
		LOG(INFO) << "EuclideanLossWithIgnore was Silent for this batch";
		return;
	}
	//Compute the gradients
	for (int i = 0; i < 2; ++i) {
		const Dtype sign = (i == 0) ? 1 : -1;
		const Dtype alpha = sign * top[0]->cpu_diff()[0] / lCount_;
		Dtype Z;
		const Dtype* botZData = bottom[nc_]->cpu_data();
		Dtype* botDiff       = bottom[i]->mutable_cpu_diff(); 
		const Dtype* labels  = bottom[1]->cpu_data();       
		Dtype* diff          = diff_.mutable_cpu_data();
		const Dtype* diffC   = diff_.cpu_data();
		if (propagate_down[i]) {
			for (int n=0; n < N; ++n){
				if (labels[bCount] == Dtype(1)){ 
					if (is_normalize_){
						Z = caffe_cpu_dot(bCount, botZData, botZData);
						if (Z>0){
							caffe_scal(count, Z, diff);
						}
					}

				caffe_cpu_axpby(
						bCount,              // count
						alpha,               // alpha
						diffC,               // a
						Dtype(0),                           // beta
						botDiff);  // b
				}
				labels += lbCount;
				diff   += bCount;
				diffC  += bCount;
				if (nc_==0){
					botZData += bCount;
				}else{
					botZData += lbCount;
				}
				if (i==0){
					botDiff  += bCount;
				}else {
					botDiff  += lbCount; 
				} 
			}
		}
	}
}
#ifdef CPU_ONLY
STUB_GPU(EuclideanLossWithIgnoreLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossWithIgnoreLayer);
REGISTER_LAYER_CLASS(EuclideanLossWithIgnore);

}  // namespace caffe
