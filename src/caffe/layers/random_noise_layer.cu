#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
void RandomNoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
	Dtype mu    = this->layer_param_.random_noise_param().mu();
  Dtype sigma = this->layer_param_.random_noise_param().sigma();
	bool adaptive_sigma   = this->layer_param_.random_noise_param().adaptive_sigma();
	Dtype adaptive_factor = this->layer_param_.random_noise_param().adaptive_factor();
	Dtype* noise = noise_.mutable_gpu_data();
	int nCount = noise_.count();
	LOG(INFO) << "nCount " << nCount << 
							 " tCount" << top[0]->count() << 
							 " count" << count;
	//Add the noise to the inputs. 
	caffe_gpu_set(count, Dtype(0.0), top_data);  
  for (int i = 0; i < count; ++i) {
		//LOG(INFO) << i;
		if (adaptive_sigma){
			sigma = adaptive_factor * bottom_data[i];
			mu    = 0; 
		}
		//When generating random numbers using cuda - it should be a multiple of 2
		//Otherwise error. 
		caffe_gpu_rng_gaussian(2, mu, sigma, noise);
		caffe_gpu_add(1, noise, bottom_data + i, top_data + i); 
		//LOG(INFO) << "Step Done";
  }
 	//LOG(INFO) << "done";
	// NOLINT_NEXT_LINE(whitespace/operators)
  //RandomNoiseForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
  //    count, bottom_data, top_data, mu, sigma, adaptive_sigma,
	//		adaptive_factor);
  // CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void RandomNoiseBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index];
  }
}

template <typename Dtype>
void RandomNoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    RandomNoiseBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RandomNoiseLayer);


}  // namespace caffe
