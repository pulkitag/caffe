#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/imchannel2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/// @brief refer to CPU forward -- the BLAS implementation is the same.
template <typename Dtype>
void TopographyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    
		if (smooth_output_){
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			const Dtype* weight = this->blobs_[0]->gpu_data();
			int top_offset = M_ * N_;
			for (int n = 0; n < num_; ++n) {
				// Take inner products for groups.
				for (int g = 0; g < group_; ++g) {
				// im2col transformation: unroll input regions for filtering
				// into column matrix for multplication.
					imchannel2col_gpu(bottom_data + bottom[i]->offset(n) + 
							g * height_ * width_ * channels_ / group_, 
							channels_ / group_, height_, width_, chHeight_, chWidth_,
							kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
							col_data);
				 
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
						(Dtype)1., weight, col_data,
						(Dtype)0., top_data + top[i]->offset(n) + top_offset * g);
					
				 }
			}
		}else{
			caffe_copy(top[i]->count(), bottom_data, top_data); 
		}
  }
}

/// @brief refer to CPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void TopographyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  //if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  //}
  
  const int top_offset = M_ * N_;
  for (int i = 0; i < top.size(); ++i) {
		const Dtype* top_diff = NULL;
		if (this->param_propagate_down_[0] || propagate_down[i]) {
			if (!top_diff) {
				top_diff = top[i]->gpu_diff();
			}
			Dtype* col_data = col_buffer_.mutable_gpu_data();
			Dtype* col_diff = col_buffer_.mutable_gpu_diff();
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
		
			for (int n = 0; n < num_; ++n) {
				
				// gradient w.r.t. weight. Note that we will accumulate diffs.
				if (this->param_propagate_down_[0]) {
					for (int g = 0; g < group_; ++g) {
						// Since we sory in the forward pass by not storing all col
						// data, we will need to recompute them.
					  imchannel2col_gpu(bottom_data + bottom[i]->offset(n) 
							+ g * height_ * width_ * channels_ / group_, 
							channels_ / group_, height_, width_, chHeight_, chWidth_,
							kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
							col_data);
					
						caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
								(Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
								col_data, (Dtype)1.,
								weight_diff);
					}
				}

				// gradient w.r.t. bottom data, if necessary
				if (propagate_down[i]) {
					if (weight == NULL) {
						weight = this->blobs_[0]->gpu_data();
					}
					for (int g = 0; g < group_; ++g) {
						caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
								(Dtype)1., weight,
								top_diff + top[i]->offset(n) + top_offset * g,
								(Dtype)0., col_diff);

						// col2im back to the data
						colchannel2im_gpu(col_diff, channels_ / group_ , height_, width_,
							 chHeight_, chWidth_, kernel_h_, kernel_w_, pad_h_, pad_w_,
							 stride_h_, stride_w_,
								bottom_diff + bottom[i]->offset(n) + 
								g * ( channels_ / group_ ) * height_ * width_ );
					}
				}
			}
		}
	}
}


INSTANTIATE_LAYER_GPU_FUNCS(TopographyLayer);

}  // namespace caffe
