#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/euclidean_loss_with_ignore_layer.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossWithIgnoreLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count  = bottom[0]->count();
	int N      = bottom[0]->shape(0);

	//Number of elements in the batch
	int bCount = bottom[0]->count(1, bottom[0]->num_axes());
	int lbCount = bottom[1]->count(1, bottom[1]->num_axes());
	lCount_    = 0;
	Dtype loss = 0.0;
	Dtype Z    = 0.0;  //The normalization factor
	const Dtype* labels    = bottom[1]->gpu_data();
	const Dtype* labelsCpu = bottom[1]->cpu_data();
	const Dtype* preds     = bottom[0]->gpu_data();
	Dtype* diff            = diff_.mutable_gpu_data();
	const Dtype* diffC     = diff_.gpu_data();
	for (int i=0; i<N; i++){
		Dtype bLoss = 0;
		if (labelsCpu[bCount] == Dtype(1)){
			//Example needs to be considered
			caffe_gpu_sub(bCount, preds, labels, diff);
			caffe_gpu_dot(bCount, diffC, diffC, &bLoss);
			if (nc_==0){
				caffe_gpu_dot(bCount, preds, preds, &Z);
			}else {
				caffe_gpu_dot(bCount, labels, labels, &Z);
			}
			lCount_ += 1;
		} 
		//LOG(INFO) << "bLoss: " << bLoss;
		//LOG(INFO) << "CHECK-3";
		preds   += bCount;
		labels  += lbCount;
		labelsCpu  += lbCount;
		diff    += bCount;
		diffC   += bCount; 
		if (is_normalize_){
			if (Z > 0){
				bLoss = bLoss / Z;
			}
		}
		loss = loss + bLoss;
	}
	//LOG(INFO) << "CHECK-4";
	if (lCount_ > 0){
		loss = loss / lCount_ / Dtype(2);
	}
	//LOG(INFO) << "CHECK-5";
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossWithIgnoreLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int count  = bottom[0]->count();
	//Number of elements in the batch
	int bCount = bottom[0]->count(1, bottom[0]->num_axes());
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
		const Dtype* botZData = bottom[nc_]->gpu_data();
		Dtype* botDiff        = bottom[i]->mutable_gpu_diff(); 
		const Dtype* labels     = bottom[1]->gpu_data();       
		const Dtype* labelsCpu  = bottom[1]->cpu_data();       
		Dtype* diff          = diff_.mutable_gpu_data();
		const Dtype* diffC   = diff_.gpu_data();
		if (propagate_down[i]) {
			for (int n=0; n < N; ++n){
				if (labelsCpu[bCount] == Dtype(1)){ 
					if (is_normalize_){
						caffe_gpu_dot(bCount, botZData, botZData, &Z);
						if (Z>0){
							caffe_gpu_scale(count, Z, diffC, diff);
						}
					}
					caffe_gpu_axpby(
							bCount,              // count
							alpha,               // alpha
							diffC,               // a
							Dtype(0),                           // beta
							botDiff);  // b
				}
				labels += lbCount;
				labelsCpu += lbCount;
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

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossWithIgnoreLayer);

}  // namespace caffe
