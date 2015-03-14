#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void CrossConvolutionLayer<Dtype>::compute_conv_num_samples() {
	//Determines the number of samples in row and column to take for computing the cross convolution. 
  this->row_samples_ = (this->height_in_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->col_samples_ = (this->width_in_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}


template <typename Dtype>
void CrossConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	//Initializw the Convolution Layer
	ConvolutionParameter conv_param = this->layer_param_.convolution_param(); 
	bool bias_term = this->layer_param_.convolution_param().bias_term();
	CHECK(!bias_term) << "Bias Term should be false for CrossConvolution Layer"; 

	CHECK(conv_param.group()==1) << "Only Group size 1 supported for now";
	CHECK(conv_param.num_output()==0) << "Number of outputs is automatically determined \
				and should be set to zero";

	CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;

		//Reshape conv_bottom_ and conv_top_
  num_in_      = bottom[0]->num();
	channels_in_ = bottom[0]->channels(); 
  height_in_   = bottom[0]->height();
  width_in_    = bottom[0]->width();
  int num2      = bottom[1]->num();
	int channels2 = bottom[1]->channels(); 
  int height2   = bottom[1]->height();
  int width2    = bottom[1]->width();
	CHECK_EQ(num_in_, num2) << "Number of examples mismatch";
	CHECK_EQ(channels_in_, channels2) << "Number of examples mismatch";
	CHECK_EQ(height_in_, height2) << "Height-MisMatch";
	CHECK_EQ(width_in_, width2)   << "Width-MisMatch";

	LOG(INFO) << num_in_ << "\t" <<  channels_in_ << "\t" <<
		height_in_ << "\t" << width_in_;

	num_out_    = num_in_;
	size_in_    = channels_in_ * height_in_ * width_in_; 
	this->compute_conv_num_samples();	
	channels_out_ = row_samples_ * col_samples_; 

	//Register the Convolution Layer
	ConvolutionParameter* tmp_conv_param = this->layer_param_.mutable_convolution_param(); 
	tmp_conv_param->set_num_output(channels_out_);
	LayerParameter conv_param_layer(this->layer_param_);
	conv_param_layer.set_type("Convolution");
  conv_layer_ = LayerRegistry<Dtype>::CreateLayer(conv_param_layer);

	//Reshape the bottom blob 
	conv_bottom_.Reshape(1, channels_in_, height_in_, width_in_);

  conv_bottom_vec_.clear();
  conv_bottom_vec_.push_back(&conv_bottom_);
  conv_top_vec_.clear();
  conv_top_vec_.push_back(&conv_top_);
  conv_layer_->SetUp(conv_bottom_vec_, conv_top_vec_);

	this->param_propagate_down_.resize(2, true);

	/*
	this->blobs_.resize(2);
	this->blobs_[0].reset(new Blob<Dtype>(
			channels_out_, channels_in_, kernel_h_, kernel_w_));
	this->blobs_[1].reset(new Blob<Dtype>(
			1, channels_in_, height_in_, width_in_));
	*/

}


template <typename Dtype>
void CrossConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
	//Reshape the bottom blob 
	conv_bottom_.Reshape(1, channels_in_, height_in_, width_in_);
	conv_layer_->Reshape(conv_bottom_vec_, conv_top_vec_);

	//Shape the top
	//Assume that size of top is 1.
	int num      = conv_top_vec_[0]->num();
	int channels = conv_top_vec_[0]->channels();
	int height   = conv_top_vec_[0]->height();
	int width    = conv_top_vec_[0]->width();
	CHECK_EQ(num, 1) << "Something is weird";
	CHECK_EQ(channels, channels_out_) << "Channels Mismatch";	

	height_out_ = height;
	width_out_  = width;
	size_out_   = channels_out_ * height_out_ * width_out_;

	//Set the shape of output blobs
	for (int i=0; i<top.size(); i++){
		top[i]->Reshape(num_out_, channels_out_, height_out_, width_out_);
	}

	//Set shape of col_buffer
	col_buffer_.Reshape(channels_out_, channels_in_, kernel_h_, kernel_w_);
	
}

template <typename Dtype>
void CrossConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	//Construct the filter from one sample location at a time.
	vector<shared_ptr<Blob<Dtype> > > conv_layer_blobs = conv_layer_->blobs();
	const Dtype* bottom_data_0 = bottom[0]->cpu_data();
	const Dtype* bottom_data_1 = bottom[1]->cpu_data();
	Dtype* ipData       = conv_bottom_vec_[0]->mutable_cpu_data();
	Dtype* opData       = top[0]->mutable_cpu_data();
	for (int n=0; n < num_in_; n++){
		//Load the weights from the bottom[0]
		crossconv_getweights_cpu(bottom_data_0 + bottom[0]->offset(n), conv_layer_blobs[0]->mutable_cpu_data());
		//Copy the data of bottom[1]
		caffe_copy(size_in_, bottom_data_1 + bottom[1]->offset(n), ipData);  
		//Perform the convolution
		conv_layer_->Forward(conv_bottom_vec_, conv_top_vec_);
		//Copy the result back into the top blob 
		caffe_copy(size_out_, conv_top_vec_[0]->cpu_data(), opData + top[0]->offset(n)); 
	}
}

template <typename Dtype>
void CrossConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		//This is for weights
		//Construct the filter from one sample location at a time.
		const Dtype* bottom_data_0 = bottom[0]->cpu_data();
		const Dtype* bottom_data_1 = bottom[1]->cpu_data();
		const Dtype* top_diff      = top[0]->cpu_diff();
		Dtype* bottom_diff_0 = bottom[0]->mutable_cpu_diff();
		Dtype* bottom_diff_1 = bottom[1]->mutable_cpu_diff();
		Dtype* ipData       = conv_bottom_vec_[0]->mutable_cpu_data();
		Dtype* col_buff     = col_buffer_.mutable_cpu_diff();

		vector<shared_ptr<Blob<Dtype> > > conv_layer_blobs = conv_layer_->blobs();

		for (int n=0; n < num_in_; n++){
			//Load the weights from the bottom[0]
			crossconv_getweights_cpu(bottom_data_0, conv_layer_blobs[0]->mutable_cpu_data());
			//Copy the data of bottom[1]
			caffe_copy(size_in_, bottom_data_1, ipData);
			//Copy the diff from the top
 			caffe_copy(size_out_, top_diff, conv_top_vec_[0]->mutable_cpu_diff());
			//Perform the backward pass
			conv_layer_->Backward(conv_top_vec_, propagate_down, conv_bottom_vec_);
			//Copy the gradients wrt to weights, i.e. bottom[0] into a buffer 
			caffe_copy(conv_layer_blobs[0]->count(), conv_layer_blobs[0]->cpu_diff(), col_buff);
			//Transform the buffer into diff
			crossconv_col2im_cpu(col_buff, bottom_diff_0);
			if (propagate_down[1]){
				//Copy the gradients for bottom[1]
				caffe_copy(size_in_, conv_bottom_vec_[0]->cpu_diff(), bottom_diff_1);
			}
			bottom_data_0 += bottom[0]->offset(1);
			bottom_data_1 += bottom[1]->offset(1);
			bottom_diff_0 += bottom[0]->offset(1);
			bottom_diff_1 += bottom[1]->offset(1);
			top_diff      += top[0]->offset(1);
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(CrossConvolutionLayer);
#endif

INSTANTIATE_CLASS(CrossConvolutionLayer);
REGISTER_LAYER_CLASS(CrossConvolution);

}  // namespace caffe
