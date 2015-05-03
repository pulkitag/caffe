#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
void RandomNoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
}

template <typename Dtype>
void RandomNoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	//Initializing data arrays. 
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
	//Noise Parameters
  Dtype mu    = this->layer_param_.random_noise_param().mu();
  Dtype sigma = this->layer_param_.random_noise_param().sigma();
	bool adaptive_sigma   = this->layer_param_.random_noise_param().adaptive_sigma();
	Dtype adaptive_factor = this->layer_param_.random_noise_param().adaptive_factor();
	Dtype noise = 0;	
	//Add the noise to the inputs.  
  for (int i = 0; i < count; ++i) {
		if (adaptive_sigma){
			sigma = adaptive_factor * bottom_data[i];
			mu    = 0; 
		}
		caffe::caffe_rng_gaussian(1, mu, sigma, &noise); 
    top_data[i] = bottom_data[i] + noise;
  }
}

template <typename Dtype>
void RandomNoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		caffe_copy(top[0]->count(), top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(RandomNoiseLayer);
#endif

INSTANTIATE_CLASS(RandomNoiseLayer);

}  // namespace caffe
