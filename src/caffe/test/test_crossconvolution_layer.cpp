#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.

template <typename Dtype>
void caffe_crossconv(const Blob<Dtype>* in1, ConvolutionParameter* conv_param,
    const Blob<Dtype>* in2,
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_size()) {
    kernel_h = kernel_w = conv_param->kernel_size();
  } else {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  }
  int pad_h, pad_w;
  if (!conv_param->has_pad_h()) {
    pad_h = pad_w = conv_param->pad();
  } else {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  }
  int stride_h, stride_w;
  if (!conv_param->has_stride_h()) {
    stride_h = stride_w = conv_param->stride();
  } else {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  }
	int num       = in1->num();
	int height    = in1->height();
	int width     = in1->width();
	int channels  = in1->channels();
	//int filter_sz = channels * kernel_h * kernel_w;
	CHECK_EQ(num, in2->num()) << "Num Mismatch";
	CHECK_EQ(height, in2->height()) << "Height Mismatch";
	CHECK_EQ(width,  in2->width()) << "Width Mismatch";
	CHECK_EQ(channels, in2->channels()) << "Channels Mismatch";

	int height_out   = (height + 2 * pad_h - kernel_h)/stride_h + 1;
	int width_out    = (width  + 2 * pad_w - kernel_w)/stride_w + 1;
	int channels_out = height_out * width_out;
	int imSz         = channels * height * width;  
  // Groups
	// Convolution
  const Dtype* in_data_1 = in1->cpu_data();
	const Dtype* in_data_2 = in2->cpu_data();
	int offset_im1, offset_im2;
  Dtype* out_data = out->mutable_cpu_data();
	caffe_set(out->count(), Dtype(0), out_data);
	std::cout << out->count() << "\n";
	int top_idx = 0;

	for (int n=0; n < num; n++){
		//std::cout << n << "\n";
		for (int h_o =0; h_o < height_out; h_o++){
		for (int w_o =0; w_o < width_out; w_o++){
			int    h_im1 = h_o * stride_h - pad_h;
			int    w_im1 = w_o * stride_w - pad_w;
			for (int h_i = 0; h_i < height_out; h_i++){
			for (int w_i = 0; w_i < width_out; w_i++){
				int h_im2 = h_i * stride_h - pad_h;
				int w_im2 = w_i * stride_w - pad_w;
				int pos = h_i * width_out + w_i;
				Dtype val = 0;
				for (int c = 0; c < channels; c++){
					offset_im1 = c * height * width;
					offset_im2 = c * height * width;
				for (int h = 0; h < kernel_h; h++){
				for (int w = 0; w < kernel_w; w++){
					//Take the kernel from im1 and convolve with im2
					int h_off1 = h_im1 + h;
					int h_off2 = h_im2 + h;
					int w_off1 = w_im1 + w;
					int w_off2 = w_im2 + w;
					bool c1 = (h_im1 + h >=0) && (h_im1 + h < height) && (w_im1 + w >=0) && (w_im1 + w < width); 	
					bool c2 = (h_im2 + h >=0) && (h_im2 + h < height) && (w_im2 + w >=0) && (w_im2 + w < width);
					if (c1 && c2){
						int off1 = h_off1 * width + w_off1;
						int off2 = h_off2 * width + w_off2;
						val += in_data_1[offset_im1 + off1] * in_data_2[offset_im2 + off2];
						//std::cout << "pos: " << top_idx + pos << "\t val: " << val << "\n";
						//std::cout << "off1, val: " << off1 << "," << in_data_1[offset_im1 + off1] << 
						//		 "\t off2: " << off2 << "\n";
					}				
				}
				}
				}
				out_data[0] = val;
				out_data += 1;
			}
			}
		}
		}
		top_idx += height_out * width_out;
		in_data_1 += imSz;
		in_data_2 += imSz;
	}
}

template void caffe_crossconv(const Blob<float>* in1,
    ConvolutionParameter* conv_param,
    const Blob<float>* in2,
    Blob<float>* out);
template void caffe_crossconv(const Blob<double>* in1,
    ConvolutionParameter* conv_param,
    const Blob<double>* in2,
    Blob<double>* out);

template <typename TypeParam>
class CrossConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CrossConvolutionLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_bottom_2_(new Blob<Dtype>(2, 3, 6, 4)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CrossConvolutionLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CrossConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(CrossConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
	int height_in = 6;
	int width_in  = 4;
	int kernel_size = 3;
	int stride      = 2;
  convolution_param->set_kernel_size(kernel_size);
  convolution_param->set_stride(stride);
  convolution_param->set_num_output(0);
  convolution_param->set_bias_term(false);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  shared_ptr<Layer<Dtype> > layer(
      new CrossConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	
	int height_out, width_out;
	height_out = (height_in - kernel_size)/stride + 1;
	width_out  = (width_in - kernel_size)/stride + 1;
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), height_out * width_out);
  EXPECT_EQ(this->blob_top_->height(), height_out);
  EXPECT_EQ(this->blob_top_->width(), width_out);
}


TYPED_TEST(CrossConvolutionLayerTest, TestSimpleCrossConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(0);
  convolution_param->set_bias_term(false);
	shared_ptr<Layer<Dtype> > layer(
      new CrossConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_crossconv(this->blob_bottom_, convolution_param, this->blob_bottom_2_,
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
		//std::cout << i << ": " << top_data[i] << "\n";
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }

  /*
	int num       = this->blob_bottom_->num();
	int height    = this->blob_bottom_->height();
	int width     = this->blob_bottom_->width();
	int channels  = this->blob_bottom_->channels();
	int count = 0;	
	const Dtype* in1 = this->blob_bottom_->cpu_data();
	for (int n = 0; n < num; n++){
		for (int ch = 0; ch < channels; ch++){
			for (int h = 0; h < height; h++){
				std::string s = "";
				for (int w = 0; w < width; w ++){
					char buffer[50];
					int nChar = sprintf(buffer, "\t %f", in1[count]);
					std::string s1(buffer);
					s = s + s1;
					count += 1;
				}
				s = s + "\n";
				std::cout << s;
			}
		}
	} 

	std::cout << "\n";
	count = 0;
	const Dtype* in2 = this->blob_bottom_2_->cpu_data();
	for (int n = 0; n < num; n++){
		for (int ch = 0; ch < channels; ch++){
			for (int h = 0; h < height; h++){
				std::string s = "";
				for (int w = 0; w < width; w ++){
					char buffer[50];
					int nChar = sprintf(buffer, "\t %f", in2[count]);
					std::string s1(buffer);
					s = s + s1;
					count += 1;
				}
				s = s + "\n";
				std::cout << s;
			}
		}
	}

	std::cout << "Top_Data \n";
	count = 0;
	for (int n = 0; n < num; n++){
		for (int ch = 0; ch < channels; ch++){
			for (int h = 0; h < height; h++){
				std::string s = "";
				for (int w = 0; w < width; w ++){
					char buffer[50];
					int nChar = sprintf(buffer, "\t %f", top_data[count]);
					std::string s1(buffer);
					s = s + s1;
					count += 1;
				}
				s = s + "\n";
				std::cout << s;
			}
		}
	}

	std::cout << "Ref_Top_Data \n";
	count = 0;
	for (int n = 0; n < num; n++){
		for (int ch = 0; ch < channels; ch++){
			for (int h = 0; h < height; h++){
				std::string s = "";
				for (int w = 0; w < width; w ++){
					char buffer[50];
					int nChar = sprintf(buffer, "\t %f", ref_top_data[count]);
					std::string s1(buffer);
					s = s + s1;
					count += 1;
				}
				s = s + "\n";
				std::cout << s;
			}
		}
	}
*/

}

TYPED_TEST(CrossConvolutionLayerTest, TestCrossConvolutionPad) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_pad(1);
  convolution_param->set_num_output(0);
  convolution_param->set_bias_term(false);
	shared_ptr<Layer<Dtype> > layer(
      new CrossConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_crossconv(this->blob_bottom_, convolution_param, this->blob_bottom_2_,
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
		//std::cout << i << ": " << top_data[i] << "\n";
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}


TYPED_TEST(CrossConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(0);
  convolution_param->set_bias_term(false);
  CrossConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(CrossConvolutionLayerTest, TestGradientPad) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  convolution_param->set_kernel_size(5);
  convolution_param->set_stride(2);
  convolution_param->set_pad(1);
  convolution_param->set_num_output(0);
  convolution_param->set_bias_term(false);
  CrossConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


TYPED_TEST(CrossConvolutionLayerTest, Test1x1Gradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  convolution_param->set_kernel_size(1);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(0);
  convolution_param->set_bias_term(false);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  CrossConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

/*
TYPED_TEST(CrossConvolutionLayerTest, TestGradientGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
  convolution_param->set_num_output(3);
  convolution_param->set_group(3);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  CrossConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
*/

}  // namespace caffe
