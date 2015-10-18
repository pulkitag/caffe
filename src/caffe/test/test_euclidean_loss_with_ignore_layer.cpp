#include <cmath>
#include <cstdlib>
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

template <typename TypeParam>
class EuclideanLossWithIgnoreLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  EuclideanLossWithIgnoreLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 6, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
		Dtype* labels = this->blob_bottom_label_->mutable_cpu_data();
		int bCount     = this->blob_bottom_data_->count(1,4);
		int lbCount    = this->blob_bottom_label_->count(1,4);
		for (int n=0; n<blob_bottom_label_->num(); n++){
			if (n%3==0){
				labels[bCount] = Dtype(0);
			}else{
				labels[bCount] = Dtype(1);
			}
			labels += lbCount;
		}	
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);

		/*
		std::cout << "Bottom Data \t";
		for (int i=0; i<this->blob_bottom_data_->count(); i++){
			std::cout << this->blob_bottom_data_->cpu_data()[i] << "\t";
		}
		std::cout << "\n";
		*/
		/*
		std::cout << "Bottom Label \n";
		for (int i=0; i<this->blob_bottom_label_->num(); i++){
			for (int j=0; j<lbCount; j++){
				std::cout << this->blob_bottom_label_->cpu_data()[i*lbCount + j] << "\t";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
		*/
  }
  virtual ~EuclideanLossWithIgnoreLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
	  LayerParameter layer_param1;
		const Dtype lossWeight1 = 1.0; 
    layer_param1.add_loss_weight(lossWeight1);
    EuclideanLossWithIgnoreLayer<Dtype> layer_weight_1(layer_param1);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
	  LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    EuclideanLossWithIgnoreLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-2;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(EuclideanLossWithIgnoreLayerTest, TestDtypesAndDevices);

TYPED_TEST(EuclideanLossWithIgnoreLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(EuclideanLossWithIgnoreLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  EuclideanLossWithIgnoreLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

/*
TYPED_TEST(EuclideanLossWithIgnoreLayerTest, TestGradientNrmlz) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
	EuclideanLossParameter* euclid_param = layer_param.mutable_euclideanloss_param();
	euclid_param->set_is_normalized(true);
  euclid_param->set_normalize_choice(0);
	const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  EuclideanLossWithIgnoreLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(EuclideanLossWithIgnoreLayerTest, TestGradientNrmlz2) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
	EuclideanLossParameter* euclid_param = layer_param.mutable_euclideanloss_param();
	euclid_param->set_is_normalized(true);
  euclid_param->set_normalize_choice(1);
	const Dtype kLossWeight = 2.7;
  layer_param.add_loss_weight(kLossWeight);
  EuclideanLossWithIgnoreLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
*/
}  // namespace caffe
