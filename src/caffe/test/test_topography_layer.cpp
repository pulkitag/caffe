#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/util/math_functions.hpp"
#include "caffe/util/filter_kernel.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_topography(const Blob<Dtype>* in, TopographyParameter* topography_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
	
 	int kernel_h, kernel_w;
	CHECK(topography_param->has_kernel_size())
      << "kernel_size for topography is required.";
	kernel_h = kernel_w = topography_param->kernel_size();
  CHECK_GT(kernel_h, 0) << "Topography Kernel dimensions cannot be zero.";
  CHECK_GT(kernel_w, 0) << "Topography Kernel dimensions cannot be zero.";
	CHECK_EQ(kernel_h % 2,1) << "Topography Kernel should be odd.";
	CHECK_EQ(kernel_w % 2,1) << "Topography Kernel should be odd.";
 
  int pad_h, pad_w;
  int stride_h, stride_w;
	pad_h = pad_w = (kernel_h - 1)/2;
	stride_h = stride_w = 1;  
  
  // Groups
  int groups = topography_param->group();
  int o_g = out->channels() / groups;
  int k_g = in->channels() / groups;
  int o_head, k_head;

	int ch_h = sqrt(k_g);
	int ch_w = ch_h;

	int nH_steps = (ch_h + 2 * pad_h - kernel_h) / stride_h + 1;  
	int nW_steps = (ch_w + 2 * pad_w - kernel_w) / stride_w + 1;  
  // Convolution
  const Dtype* in_data = in->cpu_data();
  const Dtype* weight_data = weights[0]->cpu_data();
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->num(); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
			int out_ch = 0;
      for (int h = 0; h < nH_steps; h++) {
        for (int w = 0; w < nW_steps; w++) {
					out_ch = out_ch + 1;
          for (int y = 0; y < out->height(); y++) {
            for (int x = 0; x < out->width(); x++) {
              for (int p = 0; p < kernel_h; p++) {
                for (int q = 0; q < kernel_w; q++) {
               		int h_pad = h * stride_h + p - pad_h;
									int w_pad = w * stride_w + q - pad_w;
									//Toroidal wrapping
									h_pad = caffe_cpu_modulus(h_pad, ch_h);
									w_pad = caffe_cpu_modulus(w_pad, ch_w);
									//
									int ch    = h_pad * ch_w + w_pad;
                  if (h_pad >= 0 && h_pad < ch_h
                    && w_pad >= 0 && w_pad < ch_w) {
                    out_data[out->offset(n, out_ch - 1 + o_head, y, x)] +=
                        in_data[in->offset(n, ch, y, x)]
                        * weight_data[weights[0]->offset(0, 0, p, q)];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template void caffe_topography(const Blob<float>* in,
    TopographyParameter* topography_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_topography(const Blob<double>* in,
    TopographyParameter* topography_param,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class TopographyLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
/*
template <typename Dtype>
class TopographyLayerTest : public ::testing::Test {
*/
 protected:
  TopographyLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 9, 3, 2)),
        blob_bottom_2_(new Blob<Dtype>(2, 9, 3, 2)),
        blob_bottom_3_(new Blob<Dtype>(2, 12, 1, 1)),
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()),
        blob_top_3_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    //SequenceFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
	 }

	virtual void ResetVecs() {
		int sz = blob_bottom_vec_.size();
		for (int i=0; i < sz; ++i){
			blob_bottom_vec_.pop_back();
			blob_top_vec_.pop_back();
		}
	}

  virtual ~TopographyLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
		delete blob_bottom_3_;
    delete blob_top_;
    delete blob_top_2_;
 		delete blob_top_3_;
	}

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  Blob<Dtype>* const blob_top_3_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TopographyLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(TopographyLayerTest, TestDtypes);

/*
TYPED_TEST(TopographyLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  //typedef TypeParam Dtype;
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(2);
  topography_param->set_group(1);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<Dtype> > layer(
      new TopographyLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 6);
  EXPECT_EQ(this->blob_top_2_->width(), 4);

  // setting group should not change the shape
  topography_param->set_group(3);
  this->ResetVecs();
  this->blob_bottom_vec_.push_back(this->blob_bottom_3_);
  this->blob_top_vec_.push_back(this->blob_top_3_);
 
  layer.reset(new TopographyLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
	EXPECT_EQ(this->blob_top_3_->num(), 2);
  EXPECT_EQ(this->blob_top_3_->channels(), 12);
  EXPECT_EQ(this->blob_top_3_->height(), 6);
  EXPECT_EQ(this->blob_top_3_->width(), 4);
}
*/


TYPED_TEST(TopographyLayerTest, TestSimpleConvolution) {
  // We will simply see if the convolution layer carries out averaging well.
  typedef typename TypeParam::Dtype Dtype;
	std::cout << "Entering \n";
  //typedef TypeParam Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(1);
	topography_param->set_is_gaussian_topo(false);
  topography_param->mutable_weight_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new TopographyLayer<Dtype>(layer_param));
 
	std::cout << "Seeting Layer \n";
	 layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

	std::cout << "Forward Pass \n";
	layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
 
	std::cout <<"Reference calculation \n";	
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_topography(this->blob_bottom_, 
			topography_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
	for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }

  caffe_topography(this->blob_bottom_2_, topography_param,
		   layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(TopographyLayerTest, TestNonSmoothOp) {
  // We will simply see if the convolution layer carries out averaging well.
  typedef typename TypeParam::Dtype Dtype;
	std::cout << "Entering \n";
  //typedef TypeParam Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(1);
	topography_param->set_smooth_output(false);
	topography_param->set_is_gaussian_topo(false);
  topography_param->mutable_weight_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new TopographyLayer<Dtype>(layer_param));
 
	std::cout << "Seeting Layer \n";
	 layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

	std::cout << "Forward Pass \n";
	layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
 
	std::cout <<"Reference calculation \n";	
  // Check against reference convolution.
  const Dtype* top_data;
	const Dtype* ref_top_data;
	top_data = this->blob_top_->cpu_data();
  ref_top_data = this->blob_bottom_->cpu_data();
	for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }

  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->blob_bottom_2_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}




// */
/*
TYPED_TEST(TopographyLayerTest, TestSimpleConvolutionGroup) {
  // We will simply see if the convolution layer carries out averaging well.
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(2);
  topography_param->set_num_output(3);
  topography_param->set_group(3);
  topography_param->mutable_weight_filler()->set_type("gaussian");
  shared_ptr<Layer<Dtype> > layer(
      new TopographyLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_topography(this->blob_bottom_, topography_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}
*/


/*
TYPED_TEST(TopographyLayerTest, TestSobelConvolution) {
  // Test separable convolution by computing the Sobel operator
  // as a single filter then comparing the result
  // as the convolution of two rectangular filters.
  typedef typename TypeParam::Dtype Dtype;
  // Fill bottoms with identical Gaussian noise.
  shared_ptr<GaussianFiller<Dtype> > filler;
  FillerParameter filler_param;
  filler_param.set_value(1.);
  filler.reset(new GaussianFiller<Dtype>(filler_param));
  filler->Fill(this->blob_bottom_);
  this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
  // Compute Sobel G_x operator as 3 x 3 convolution.
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(2);
  topography_param->set_num_output(1);
  topography_param->set_bias_term(false);
  shared_ptr<Layer<Dtype> > layer(
      new TopographyLayer<Dtype>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 3, 3));
  Dtype* weights = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 9;  // 3 x 3 filter
    weights[i +  0] = -1;
    weights[i +  1] =  0;
    weights[i +  2] =  1;
    weights[i +  3] = -2;
    weights[i +  4] =  0;
    weights[i +  5] =  2;
    weights[i +  6] = -1;
    weights[i +  7] =  0;
    weights[i +  8] =  1;
  }
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 convolutions.
  // (1) the [1 2 1] column filter
  vector<Blob<Dtype>*> sep_blob_bottom_vec;
  vector<Blob<Dtype>*> sep_blob_top_vec;
  shared_ptr<Blob<Dtype> > blob_sep(new Blob<Dtype>());
  sep_blob_bottom_vec.push_back(this->blob_bottom_2_);
  sep_blob_top_vec.push_back(this->blob_top_2_);
  topography_param->clear_kernel_size();
  topography_param->clear_stride();
  topography_param->set_kernel_h(3);
  topography_param->set_kernel_w(1);
  topography_param->set_stride_h(2);
  topography_param->set_stride_w(1);
  topography_param->set_num_output(1);
  topography_param->set_bias_term(false);
  layer.reset(new TopographyLayer<Dtype>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 3, 1));
  Dtype* weights_1 = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 3 x 1 filter
    weights_1[i +  0] = 1;
    weights_1[i +  1] = 2;
    weights_1[i +  2] = 1;
  }
  layer->SetUp(sep_blob_bottom_vec, &(sep_blob_top_vec));
  layer->Forward(sep_blob_bottom_vec, &(sep_blob_top_vec));
  // (2) the [-1 0 1] row filter
  blob_sep->CopyFrom(*this->blob_top_2_, false, true);
  sep_blob_bottom_vec.clear();
  sep_blob_bottom_vec.push_back(blob_sep.get());
  topography_param->set_kernel_h(1);
  topography_param->set_kernel_w(3);
  topography_param->set_stride_h(1);
  topography_param->set_stride_w(2);
  topography_param->set_num_output(1);
  topography_param->set_bias_term(false);
  layer.reset(new TopographyLayer<Dtype>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 3, 1, 3));
  Dtype* weights_2 = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 1 x 3 filter
    weights_2[i +  0] = -1;
    weights_2[i +  1] =  0;
    weights_2[i +  2] =  1;
  }
  layer->SetUp(sep_blob_bottom_vec, &(sep_blob_top_vec));
  layer->Forward(sep_blob_bottom_vec, &(sep_blob_top_vec));
  // Test equivalence of full and separable filters.
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype* sep_top_data = this->blob_top_2_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
  }
}
*/

TYPED_TEST(TopographyLayerTest, TestRotation) {
  typedef typename TypeParam::Dtype Dtype;
  std::cout << "here Loc 0 \n";
  
	LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(1);
	topography_param->set_is_gaussian_topo(false);
  topography_param->mutable_weight_filler()->set_type("sequence");
  std::cout << "here Loc 1 \n";

	 shared_ptr<Layer<Dtype> > layer(
      new TopographyLayer<Dtype>(layer_param));
	 layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

	layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 1, 3, 3));
  Dtype* weights = layer->blobs()[0]->mutable_cpu_data();
	caffe_gaussian_kernel(weights, (Dtype)0.5, 3);
	for (int c=0; c < 9; ++c){
		std::cout <<weights[c] <<" \n";
	}
	/*for (int c = 0; c < 1; ++c) {
    int i = c * 9;  // 3 x 3 filter
    weights[i +  0] =  1;
    weights[i +  1] =  0;
    weights[i +  2] =  0;
    weights[i +  3] =  0;
    weights[i +  4] =  0;
    weights[i +  5] =  0;
    weights[i +  6] =  0;
    weights[i +  7] =  0;
    weights[i +  8] =  0;
  }*/
  std::cout << "Weights Filled \n";

	std::cout << "Forward Pass \n";
	layer->Forward((this->blob_bottom_vec_), &(this->blob_top_vec_));

	std::cout << "###### Bottom ########## \n";
	const Dtype* bottom = this->blob_bottom_vec_[0]->cpu_data();
	const int nc        = this->blob_bottom_vec_[0]->channels();
	for (int count=0; count < this->blob_bottom_vec_[0]->count(); ++count){
		if ((count % nc)==0)
			std::cout << " \n c: ";
		std::cout << bottom[count] << " ";
	}

	std::cout << " \n ###### Top  ########## \n";
	const Dtype* top = this->blob_top_vec_[0]->cpu_data();
	for (int count=0; count < this->blob_top_vec_[0]->count(); ++count){
		if ((count % nc)==0)
			std::cout << " \n c: ";
		std::cout << top[count] << " ";
	}
 
	std::cout<<"\n";

}
// */



/*
TYPED_TEST(TopographyLayerTest, TestGradientSimple) {
  typedef typename TypeParam::Dtype Dtype;
  std::cout << "here Loc 0 \n";
  
	LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(1);
	topography_param->set_is_gaussian_topo(false);
  topography_param->mutable_weight_filler()->set_type("sequence");
  std::cout << "here Loc 1 \n";

	 shared_ptr<Layer<Dtype> > layer(
      new TopographyLayer<Dtype>(layer_param));
	 layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

	layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<Dtype>(1, 1, 3, 3));
  Dtype* weights = layer->blobs()[0]->mutable_cpu_data();
	for (int c = 0; c < 1; ++c) {
    int i = c * 9;  // 3 x 3 filter
    weights[i +  0] =  1;
    weights[i +  1] =  2;
    weights[i +  2] =  3;
    weights[i +  3] =  4;
    weights[i +  4] =  5;
    weights[i +  5] =  6;
    weights[i +  6] =  7;
    weights[i +  7] =  8;
    weights[i +  8] =  9;
  }
  std::cout << "Weights Filled \n";

	vector<bool>  prop_down;
	prop_down.push_back(true);

	std::cout << "Backward Pass \n";
	caffe_set(this->blob_top_vec_[0]->count(), Dtype(1), 
						this->blob_top_vec_[0]->mutable_cpu_diff());
	layer->Backward( (this->blob_top_vec_), prop_down, &(this->blob_bottom_vec_));

	//std::cout<<"Size is "<<this->blob_bottom_vec_.size();
	std::cout << "Gradient ############### \n \n";
	const Dtype* bottom = this->blob_bottom_vec_[0]->cpu_diff();
	for (int count=0; count < this->blob_bottom_vec_[0]->count(); ++count){
			//std::cout << " \n St: ";
		  std::cout << bottom[count] << " ";
	}
	std::cout<<"\n";

	///
 // GradientChecker<Dtype> checker(1e-2, 1e-3);
 // checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
 //     &(this->blob_top_vec_));
	
}
*/


TYPED_TEST(TopographyLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
	std::cout <<"Test Gradient \n";
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  topography_param->set_kernel_size(3);
  topography_param->set_stride(1);
	topography_param->set_is_gaussian_topo(false);
  topography_param->mutable_weight_filler()->set_type("gaussian");
  TopographyLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}
// */

/*
TYPED_TEST(TopographyLayerTest, TestGradientGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(2);
  topography_param->set_num_output(3);
  topography_param->set_group(3);
  topography_param->mutable_weight_filler()->set_type("gaussian");
  topography_param->mutable_bias_filler()->set_type("gaussian");
  TopographyLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}
*/

//CuDNN test - for later. 
/*

#ifdef USE_CUDNN

template <typename Dtype>
class CuDNNTopographyLayerTest : public ::testing::Test {
 protected:
  CuDNNTopographyLayerTest()
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

  virtual ~CuDNNTopographyLayerTest() {
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

TYPED_TEST_CASE(CuDNNTopographyLayerTest, TestDtypes);

TYPED_TEST(CuDNNTopographyLayerTest, TestSetupCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(2);
  topography_param->set_num_output(4);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  shared_ptr<Layer<TypeParam> > layer(
      new CuDNNTopographyLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 4);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
  // setting group should not change the shape
  topography_param->set_num_output(3);
  topography_param->set_group(3);
  layer.reset(new CuDNNTopographyLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 3);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
}

TYPED_TEST(CuDNNTopographyLayerTest, TestSimpleConvolutionCuDNN) {
  // We will simply see if the convolution layer carries out averaging well.
  Caffe::set_mode(Caffe::GPU);
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(2);
  topography_param->set_num_output(4);
  topography_param->mutable_weight_filler()->set_type("gaussian");
  topography_param->mutable_bias_filler()->set_type("constant");
  topography_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new CuDNNTopographyLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Check against reference convolution.
  const TypeParam* top_data;
  const TypeParam* ref_top_data;
  caffe_topography(this->blob_bottom_, topography_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  caffe_topography(this->blob_bottom_2_, topography_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_2_));
  top_data = this->blob_top_2_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(CuDNNTopographyLayerTest, TestSimpleConvolutionGroupCuDNN) {
  // We will simply see if the convolution layer carries out averaging well.
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(2);
  topography_param->set_num_output(3);
  topography_param->set_group(3);
  topography_param->mutable_weight_filler()->set_type("gaussian");
  topography_param->mutable_bias_filler()->set_type("constant");
  topography_param->mutable_bias_filler()->set_value(0.1);
  shared_ptr<Layer<TypeParam> > layer(
      new CuDNNTopographyLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Check against reference convolution.
  const TypeParam* top_data;
  const TypeParam* ref_top_data;
  caffe_topography(this->blob_bottom_, topography_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
}

TYPED_TEST(CuDNNTopographyLayerTest, TestSobelConvolutionCuDNN) {
  // Test separable convolution by computing the Sobel operator
  // as a single filter then comparing the result
  // as the convolution of two rectangular filters.
  Caffe::set_mode(Caffe::GPU);
  // Fill bottoms with identical Gaussian noise.
  shared_ptr<GaussianFiller<TypeParam> > filler;
  FillerParameter filler_param;
  filler_param.set_value(1.);
  filler.reset(new GaussianFiller<TypeParam>(filler_param));
  filler->Fill(this->blob_bottom_);
  this->blob_bottom_2_->CopyFrom(*this->blob_bottom_);
  // Compute Sobel G_x operator as 3 x 3 convolution.
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(2);
  topography_param->set_num_output(1);
  topography_param->set_bias_term(false);
  shared_ptr<Layer<TypeParam> > layer(
      new CuDNNTopographyLayer<TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 3, 3));
  TypeParam* weights = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 9;  // 3 x 3 filter
    weights[i +  0] = -1;
    weights[i +  1] =  0;
    weights[i +  2] =  1;
    weights[i +  3] = -2;
    weights[i +  4] =  0;
    weights[i +  5] =  2;
    weights[i +  6] = -1;
    weights[i +  7] =  0;
    weights[i +  8] =  1;
  }
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Compute Sobel G_x operator as separable 3 x 1 and 1 x 3 convolutions.
  // (1) the [1 2 1] column filter
  vector<Blob<TypeParam>*> sep_blob_bottom_vec;
  vector<Blob<TypeParam>*> sep_blob_top_vec;
  shared_ptr<Blob<TypeParam> > blob_sep(new Blob<TypeParam>());
  sep_blob_bottom_vec.push_back(this->blob_bottom_2_);
  sep_blob_top_vec.push_back(this->blob_top_2_);
  topography_param->clear_kernel_size();
  topography_param->clear_stride();
  topography_param->set_kernel_h(3);
  topography_param->set_kernel_w(1);
  topography_param->set_stride_h(2);
  topography_param->set_stride_w(1);
  topography_param->set_num_output(1);
  topography_param->set_bias_term(false);
  layer.reset(new CuDNNTopographyLayer<TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 3, 1));
  TypeParam* weights_1 = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 3 x 1 filter
    weights_1[i +  0] = 1;
    weights_1[i +  1] = 2;
    weights_1[i +  2] = 1;
  }
  layer->SetUp(sep_blob_bottom_vec, &(sep_blob_top_vec));
  layer->Forward(sep_blob_bottom_vec, &(sep_blob_top_vec));
  // (2) the [-1 0 1] row filter
  blob_sep->CopyFrom(*this->blob_top_2_, false, true);
  sep_blob_bottom_vec.clear();
  sep_blob_bottom_vec.push_back(blob_sep.get());
  topography_param->set_kernel_h(1);
  topography_param->set_kernel_w(3);
  topography_param->set_stride_h(1);
  topography_param->set_stride_w(2);
  topography_param->set_num_output(1);
  topography_param->set_bias_term(false);
  layer.reset(new CuDNNTopographyLayer<TypeParam>(layer_param));
  layer->blobs().resize(1);
  layer->blobs()[0].reset(new Blob<TypeParam>(1, 3, 1, 3));
  TypeParam* weights_2 = layer->blobs()[0]->mutable_cpu_data();
  for (int c = 0; c < 3; ++c) {
    int i = c * 3;  // 1 x 3 filter
    weights_2[i +  0] = -1;
    weights_2[i +  1] =  0;
    weights_2[i +  2] =  1;
  }
  layer->SetUp(sep_blob_bottom_vec, &(sep_blob_top_vec));
  layer->Forward(sep_blob_bottom_vec, &(sep_blob_top_vec));
  // Test equivalence of full and separable filters.
  const TypeParam* top_data = this->blob_top_->cpu_data();
  const TypeParam* sep_top_data = this->blob_top_2_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], sep_top_data[i], 1e-4);
  }
}

TYPED_TEST(CuDNNTopographyLayerTest, TestGradientCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  topography_param->set_kernel_size(3);
  topography_param->set_stride(2);
  topography_param->set_num_output(2);
  topography_param->mutable_weight_filler()->set_type("gaussian");
  topography_param->mutable_bias_filler()->set_type("gaussian");
  CuDNNTopographyLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(CuDNNTopographyLayerTest, TestGradientGroupCuDNN) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  TopographyParameter* topography_param =
      layer_param.mutable_topography_param();
  topography_param->set_kernel_size(3);
  topography_param->set_stride(2);
  topography_param->set_num_output(3);
  topography_param->set_group(3);
  topography_param->mutable_weight_filler()->set_type("gaussian");
  topography_param->mutable_bias_filler()->set_type("gaussian");
  CuDNNTopographyLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

#endif
*/
}  // namespace caffe
