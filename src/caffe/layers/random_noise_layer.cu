#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void RandomNoiseForward(const int n, const Dtype sigma,
						 const bool adaptive_sigma, const Dtype adaptive_factor, 
						 const Dtype mu, const Dtype avgVal,
						 const Dtype* bottom_data, Dtype* top_data, const Dtype* noise) {
  CUDA_KERNEL_LOOP(index, n) {
		Dtype noise_tmp = noise[index];
		if (adaptive_sigma){
			if (bottom_data[index] < 0)
				noise_tmp = -noise_tmp * adaptive_factor * bottom_data[index];
			else{
				if (bottom_data[index] > 0)
					noise_tmp = noise_tmp * adaptive_factor * bottom_data[index];
				else
					noise_tmp = noise_tmp * adaptive_factor * avgVal;
			}
		}
		top_data[index] = bottom_data[index] + noise_tmp; 
  }
}

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

	//Get the noise
	Dtype* noise = noise_.mutable_gpu_data();
	caffe_gpu_rng_gaussian(count, mu, Dtype(1.0), noise); 

	//Compute the average activation of the layer. 	
	Dtype avgVal = 0;
	caffe_gpu_asum(count, bottom_data, &avgVal);
	avgVal = avgVal / count;

	//Initialize the top_data. 
	caffe_gpu_set(count, Dtype(0.0), top_data);  

	RandomNoiseForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, sigma, adaptive_sigma, adaptive_factor, mu, avgVal,
			bottom_data, top_data, noise);		  
	 CUDA_POST_KERNEL_CHECK;


	/*
  for (int i = 0; i < count; ++i) {
		//LOG(INFO) << i;
		if (adaptive_sigma){
			LOG(INFO) << "sigma_init " << sigma; 
			caffe_gpu_scale(1, adaptive_factor, bottom_data + i, &sigma);
			if (sigma < 0)
				sigma = -sigma;
			if (sigma == 0)
				sigma = adaptive_factor * avgVal;
			mu    = 0;
			LOG(INFO) << "Here"; 
		}
		//When generating random numbers using cuda - it should be a multiple of 2
		//Otherwise error.
		LOG(INFO) << sigma << ", " <<  avgVal << ", " << adaptive_factor; 
		caffe_gpu_rng_gaussian(2, mu, sigma, noise);
		caffe_gpu_add(1, noise, bottom_data + i, top_data + i); 
		//LOG(INFO) << "Step Done";
  }
	*/
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
