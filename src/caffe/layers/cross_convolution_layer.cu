#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {

template <typename Dtype>
void CrossConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	//Construct the filter from one sample location at a time.
	vector<shared_ptr<Blob<Dtype> > > conv_layer_blobs = conv_layer_->blobs();
	const Dtype* bottom_data_0 = bottom[0]->gpu_data();
	const Dtype* bottom_data_1 = bottom[1]->gpu_data();
	Dtype* ipData       = conv_bottom_vec_[0]->mutable_gpu_data();
	Dtype* opData       = top[0]->mutable_gpu_data();
	for (int n=0; n < num_in_; n++){
		//Load the weights from the bottom[0]
		crossconv_getweights_gpu(bottom_data_0 + bottom[0]->offset(n), conv_layer_blobs[0]->mutable_gpu_data());

		//crossconv_getweights_gpu(bottom_data_0 + bottom[0]->offset(n), this->blobs_[0]->mutable_gpu_data());
		//Copy the data of bottom[1]
		caffe_copy(size_in_, bottom_data_1 + bottom[1]->offset(n), ipData);  
		//Perform the convolution
		conv_layer_->Forward(conv_bottom_vec_, conv_top_vec_);
		//Copy the result back into the top blob 
		caffe_copy(size_out_, conv_top_vec_[0]->gpu_data(), opData + top[0]->offset(n)); 
	}
}

template <typename Dtype>
void CrossConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
		//This is for weights
		//Construct the filter from one sample location at a time.
		const Dtype* bottom_data_0 = bottom[0]->gpu_data();
		const Dtype* bottom_data_1 = bottom[1]->gpu_data();
		const Dtype* top_diff      = top[0]->gpu_diff();
		Dtype* bottom_diff_0 = bottom[0]->mutable_gpu_diff();
		Dtype* bottom_diff_1 = bottom[1]->mutable_gpu_diff();
		Dtype* ipData       = conv_bottom_vec_[0]->mutable_gpu_data();
		Dtype* col_buff     = col_buffer_.mutable_gpu_diff();

		vector<shared_ptr<Blob<Dtype> > > conv_layer_blobs = conv_layer_->blobs();

		for (int n=0; n < num_in_; n++){
			//Load the weights from the bottom[0]
			crossconv_getweights_gpu(bottom_data_0, conv_layer_blobs[0]->mutable_gpu_data());
			//Copy the data of bottom[1]
			caffe_copy(size_in_, bottom_data_1, ipData);
			//Copy the diff from the top
 			caffe_copy(size_out_, top_diff, conv_top_vec_[0]->mutable_gpu_diff()); 
			//Perform the backward pass
			conv_layer_->Backward(conv_top_vec_, propagate_down, conv_bottom_vec_);
			//Copy the gradients wrt to weights, i.e. bottom[0] into a buffer 
			caffe_copy(conv_layer_blobs[0]->count(), conv_layer_blobs[0]->mutable_gpu_diff(), col_buff);
			//Transform the buffer into diff
			crossconv_col2im_gpu(col_buff, bottom_diff_0);
			if (propagate_down[1]){
				//Copy the gradients for bottom[1]
				caffe_copy(size_in_, conv_bottom_vec_[0]->mutable_gpu_diff(), bottom_diff_1);
			}
		  bottom_data_0 += bottom[0]->offset(1);
			bottom_data_1 += bottom[1]->offset(1);
			bottom_diff_0 += bottom[0]->offset(1);
			bottom_diff_1 += bottom[1]->offset(1);
			top_diff      += top[0]->offset(1);

		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(CrossConvolutionLayer);
}
