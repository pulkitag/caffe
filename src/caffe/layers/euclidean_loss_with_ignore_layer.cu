#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossWithIgnoreLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count  = bottom[0]->count();
	int N      = bottom[0]->shape(0);

	//Number of elements in the batch
	int bCount = bottom[0]->count(1, bottom[0]->num_axes());
	lCount_    = 0;
	Dtype loss = 0.0;
	Dtype Z;  //The normalization factor
	const Dtype* labels = bottom[1]->gpu_data();
	const Dtype* labelsCpu = bottom[1]->cpu_data();
	const Dtype* preds  = bottom[0]->gpu_data();
	Dtype* diff         = diff_.mutable_gpu_data();
	const Dtype* diffC  = diff_.gpu_data();
	//LOG(INFO) << "CHECK-2";
	//LOG(INFO) << "N: " << N <<", count: " << count <<", bCount: " << bCount;
	//LOG(INFO) << bottom[1]->shape(0) << ", " << bottom[1]->shape(1);
	for (int i=0; i<N; i++){
		Dtype bLoss;
		//LOG(INFO) << "CHECK-2-0";
		if (labelsCpu[bCount] == Dtype(1)){
			//Example needs to be considered
			//LOG(INFO) << "CHECK-2-1";
			caffe_gpu_sub(bCount, preds, labels, diff);
			//LOG(INFO) << "CHECK-2-2";
			caffe_gpu_dot(bCount, diffC, diffC, &bLoss);
			if (nc_==0){
				caffe_gpu_dot(bCount, preds, preds, &Z);
			}else {
				caffe_gpu_dot(bCount, labels, labels, &Z);
			}
		} 
		//LOG(INFO) << "CHECK-3";
		preds   += bCount;
		labels  += (bCount + 1);
		diff    += bCount;
		diffC   += bCount; 
		lCount_ += 1;
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
	int N      = bottom[0]->shape(0);
	if (lCount_ == Dtype(0)){
		return;
	}
	//Compute the gradients
	for (int i = 0; i < 2; ++i) {
		const Dtype sign = (i == 0) ? 1 : -1;
		const Dtype alpha = sign * top[0]->cpu_diff()[0] / lCount_;
		Dtype Z;
		const Dtype* botData = bottom[nc_]->gpu_data();
		const Dtype* labels     = bottom[1]->gpu_data();       
		const Dtype* labelsCpu  = bottom[1]->cpu_data();       
		Dtype* diff          = diff_.mutable_gpu_data();
		const Dtype* diffC   = diff_.gpu_data();
		Dtype* botDiff       = bottom[nc_]->mutable_gpu_data(); 
		if (propagate_down[i]) {
			for (int n=0; n < N; ++n){
				if (labelsCpu[bCount] == Dtype(1)) 
					if (is_normalize_){
						caffe_gpu_dot(bCount, botData, botData, &Z);
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
			labels += (bCount + 1);
			diff   += bCount;
		  diffC  += bCount;
			if (i==0){
				botData += bCount;
				botDiff += bCount;
			}else {
				botData += (bCount+1);
				botDiff += (bCount+1); 
			} 
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossWithIgnoreLayer);

}  // namespace caffe
