#ifndef CAFFE_CROSS_CONV_LAYER_HPP_
#define CAFFE_CROSS_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class CrossConvolutionLayer : public Layer<Dtype> {
 public:
  explicit CrossConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CrossConvolution"; }
  
	virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
	// Compute height_out_ and width_out_ from other parameters.
  virtual void compute_conv_num_samples();
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int num_in_, num_out_;
  int size_in_, size_out_;
	int channels_in_, channels_out_;
  int height_in_, width_in_;
  int height_out_, width_out_;
  int pad_h_, pad_w_;
  int num_output_;
  int row_samples_, col_samples_;
	bool bias_term_;
  bool is_1x1_;

  Blob<Dtype> conv_bottom_;
  Blob<Dtype> conv_top_;
  vector<Blob<Dtype>*> conv_bottom_vec_;
  vector<Blob<Dtype>*> conv_top_vec_;

  /// The internal Convolution Layer
  shared_ptr<Layer<Dtype> > conv_layer_;
 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void crossconv_getweights_cpu(const Dtype* data, Dtype* col_buff) {
    im2filtercol_cpu(data, channels_in_, height_in_, width_in_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_buff);
  }
  inline void crossconv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    filtercol2im_cpu(col_buff, channels_in_, height_in_, width_in_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, data);
  }
#ifndef CPU_ONLY
  inline void crossconv_getweights_gpu(const Dtype* data, Dtype* col_buff) {
    im2filtercol_gpu(data, channels_in_, height_in_, width_in_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_buff);
  }
  inline void crossconv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    filtercol2im_gpu(col_buff, channels_in_, height_in_, width_in_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, data);
  }
#endif

	Blob<Dtype> col_buffer_;
};


}  // namespace caffe

#endif  // CAFFE_CROSS_CONV_LAYER_HPP_
