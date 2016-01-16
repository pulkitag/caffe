#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/imchannel2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/filter_kernel.hpp"
#include "caffe/layers/topography_layer.hpp"

namespace caffe {

template <typename Dtype>
void TopographyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  TopographyParameter top_param = this->layer_param_.topography_param();
  CHECK(top_param.has_kernel_size())
      << "kernel_size for topography is required.";
	kernel_h_ = kernel_w_ = top_param.kernel_size();
  CHECK_GT(kernel_h_, 0) << "Topography Kernel dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Topography Kernel dimensions cannot be zero.";
	CHECK_EQ(kernel_h_ % 2,1) << "Topography Kernel should be odd.";
	CHECK_EQ(kernel_w_ % 2,1) << "Topography Kernel should be odd.";
 
	pad_h_ = pad_w_ = (kernel_h_ - 1)/2;
	stride_h_ = stride_w_ = 1;  

  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  group_ = this->layer_param_.topography_param().group();
  CHECK_EQ(channels_ % group_, 0)
      << "Number of channels in prevous layer should be multiples of group.";

	chHeight_ = sqrt(channels_ / group_);
	chWidth_ 	= chHeight_;
	//Ensure that channels can form a square grid.
	CHECK_EQ(chHeight_ * chWidth_, channels_ / group_)
			<< "Number of groups or channels is inappropriate";


	smooth_output_ = top_param.smooth_output();

  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
			this->blobs_.resize(1);
    // Initialize and fill the weights:
    // 1 x 1 x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        1, 1, kernel_h_, kernel_w_));

  	if (top_param.is_gaussian_topo()){
			LOG(INFO)<<"Gaussian Weight init";	
			Dtype* weights = this->blobs_[0]->mutable_cpu_data();
			caffe_gaussian_kernel(weights, (Dtype)top_param.gaussian_sd(), 
							top_param.kernel_size()); 
		}else{
			LOG(INFO)<<"Random Weight Init";	
			shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
					this->layer_param_.topography_param().weight_filler()));
			weight_filler->Fill(this->blobs_[0].get());
		}
	}
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void TopographyLayer<Dtype>::PrintWeights() const {
	const Dtype* weight = this->blobs_[0]->cpu_data();
	int sz = this->blobs_[0]->count();
	LOG(INFO)<<"Topography Layer Weights " << sz;
	for (int i=0; i<sz; i++){
		LOG(INFO) << weight[i] << " ";
	}
}

template <typename Dtype>
void TopographyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  height_out_ = height_;
  width_out_  = width_;
	//The size of channel grid.
	//std::cout << "groups: " << group_ << " pad_w: " << pad_w_ 
 	//					<< " stride_w_: " << stride_w_ << " \n "; 
	width_col_  = (chWidth_ + 2 * pad_w_ - kernel_w_)/stride_w_ + 1;
	height_col_ = (chHeight_ + 2 * pad_h_ - kernel_h_)/stride_h_ + 1;
	for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, width_col_ * height_col_ * group_ , 
						height_out_, width_out_);
  }

  // Prepare the matrix multiplication computation.
  // Each input will be convolved as a single GEMM.
  M_ = 1;
  K_ = kernel_h_ * kernel_w_;
  N_ = height_out_ * width_out_ * width_col_ * height_col_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage.
  col_buffer_.Reshape(
      1, width_col_ * height_col_ * kernel_h_ * kernel_w_, height_out_, width_out_);
  
  top_col_buffer_.Reshape(
      1, 1, 1, width_col_ * height_col_ * height_out_ * width_out_);

	for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, channels_, height_out_, width_out_);
  }
}

template <typename Dtype>
void TopographyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
		
		if (smooth_output_){
			Dtype* col_data = col_buffer_.mutable_cpu_data();
			const Dtype* weight = this->blobs_[0]->cpu_data();
			int top_offset = M_ * N_;  // number of values in an output region / column
			for (int n = 0; n < num_; ++n) {
				// im2col transformation: unroll input regions for filtering
				// into column matrix for multplication.
						 // Take inner products for groups.
				for (int g = 0; g < group_; ++g) {
					 imchannel2col_cpu(bottom_data + bottom[i]->offset(n) + 
						 g * height_ *  width_ * channels_ / group_ , 
						 channels_/group_,
						 height_, width_, chHeight_, chWidth_, 
						 kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
						 col_data);
					
					caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
						(Dtype)1., weight, col_data,
						(Dtype)0., top_data + top[i]->offset(n) + top_offset * g);

					//Copy data from top_col buffer to top;
				}
			}
		}else{
			caffe_copy(top[i]->count(), bottom_data, top_data); 
		}	
  }
}

template <typename Dtype>
void TopographyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
	weight = this->blobs_[0]->cpu_data();
	weight_diff = this->blobs_[0]->mutable_cpu_diff();
	caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  
	const int top_offset = M_ * N_;
  
	for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
		if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->cpu_diff();
      }
      Dtype* col_data = col_buffer_.mutable_cpu_data();
      Dtype* col_diff = col_buffer_.mutable_cpu_diff();
      const Dtype* bottom_data = (bottom)[i]->cpu_data();
      Dtype* bottom_diff = (bottom)[i]->mutable_cpu_diff();
      
			for (int n = 0; n < num_; ++n) {
					
				// gradient w.r.t. weight. Note that we will accumulate diffs.
				if (this->param_propagate_down_[0]) {
					for (int g = 0; g < group_; ++g) {
						imchannel2col_cpu(bottom_data + bottom[i]->offset(n) 
								+ g * height_ * width_ * channels_ / group_, 
								channels_ / group_, height_, width_, chHeight_, chWidth_,
								kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
								col_data);
						
						caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
								(Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
								col_data, (Dtype)1.,
								weight_diff);
					}
				}

				// gradient w.r.t. bottom data, if necessary.
				if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->cpu_data();
          }
					for (int g = 0; g < group_; ++g) {
            caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                (Dtype)1., weight,
                top_diff + top[i]->offset(n) + top_offset * g,
                (Dtype)0., col_diff);
          
						// col2im back to the data
						colchannel2im_cpu(col_diff, channels_ / group_, height_, width_,
								chHeight_, chWidth_,kernel_h_, kernel_w_, pad_h_, pad_w_,
								stride_h_, stride_w_, bottom_diff + bottom[i]->offset(n) +
								g * (channels_ / group_) * height_ * width_);
					}
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TopographyLayer);
#endif

INSTANTIATE_CLASS(TopographyLayer);
REGISTER_LAYER_CLASS(Topography);
}  // namespace caffe
