#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/imchannel2col.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {


// Forward declare kernel functions
template <typename Dtype>
__global__ void imchannel2col_gpu_kernel(const int n, const Dtype* data_im,
    const int imHeight, const int imWidth, 
    const int chHeight, const int chWidth, 
		const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_col);

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class ImChannel2ColKernelTest : public ::testing::Test {
 protected:
  ImChannel2ColKernelTest()
        // big so launches > 1024 threads
      : blob_bottom_(new Blob<Dtype>(5, 121, 131, 149)),
        blob_top_(new Blob<Dtype>()),
        blob_top_cpu_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    SequenceFiller<Dtype> filler(filler_param);
    //GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

		show_cpu_output_ = false;
		show_gpu_output_ = false;
    imHeight_ = blob_bottom_->height();
    imWidth_ = blob_bottom_->width();
    channels_ = blob_bottom_->channels();
		chHeight_ = sqrt(channels_);
		chWidth_  = chHeight_;
    pad_ = 2;
    stride_ = 2;
    kernel_size_ = 3;
    height_col_  = (chHeight_ + 2 * pad_ - kernel_size_) / stride_ + 1;
    width_col_   = (chWidth_  + 2 * pad_ - kernel_size_) / stride_ + 1;
 		total_patches_ = imHeight_ * imWidth_ * height_col_ * width_col_;
	 }

  virtual ~ImChannel2ColKernelTest() {
      delete blob_bottom_;
      delete blob_top_;
      delete blob_top_cpu_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_cpu_;
  int imHeight_;
  int imWidth_;
	int chHeight_;
	int chWidth_;
  int channels_;
  int pad_;
  int stride_;
  int kernel_size_;
  int height_col_;
  int width_col_;
	int total_patches_;
	bool show_cpu_output_;
	bool show_gpu_output_;
};

TYPED_TEST_CASE(ImChannel2ColKernelTest, TestDtypes);

TYPED_TEST(ImChannel2ColKernelTest, TestGPU) {
  Caffe::set_mode(Caffe::GPU);

  // Reshape the blobs to correct size for im2col output
  this->blob_top_->Reshape(this->blob_bottom_->num(),
   this->imHeight_ * this->imWidth_ * this->kernel_size_ * this->kernel_size_,
     this->height_col_,
     this->width_col_);

  this->blob_top_cpu_->Reshape(this->blob_bottom_->num(),
   this->imHeight_ * this->imWidth_ * this->kernel_size_ * this->kernel_size_,
      this->height_col_,
      this->width_col_);

  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();
  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  TypeParam* cpu_data = this->blob_top_cpu_->mutable_cpu_data();

	std::cout<<"Size is: " << this->blob_bottom_->count() <<" \n";
	//std::cout<<"Size is: " << this->blob_bottom_->count() <<" \n";
 
	
  // CPU Version
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    imchannel2col_cpu(this->blob_bottom_->cpu_data() + this->blob_bottom_->offset(n),
      this->channels_, 
			this->imHeight_, this->imWidth_,
			this->chHeight_, this->chWidth_,
      this->kernel_size_, this->kernel_size_, 
			this->pad_, this->pad_,
      this->stride_, this->stride_,
      cpu_data + this->blob_top_cpu_->offset(n));
  }
 
	std::cout<<"Size is: " << this->blob_bottom_->count() 
		<<" chnls: "<<this->blob_bottom_->channels()
		<<" h: " << this->blob_bottom_->height()
		<<" w: " << this->blob_bottom_->width() <<" \n";
	
	const TypeParam* cpu_data_bottom = this->blob_bottom_->cpu_data();
	/*for (int count=0; count < this->blob_bottom_->count(); ++count){
		std::cout << cpu_data_bottom[count] << "\t";
	}*/

	if (this->show_cpu_output_){
		std::cout<<"CPU Output \n";
		std::cout << "\n" <<"\n"<<"\n";;
		for (int count=0; count < this->blob_top_->count(); ++count){
			if ((count % this->total_patches_)==0)
				std::cout << " \n St: ";
			std::cout << cpu_data[count] << " ";
		}
		std::cout<<"\n";
	}
	//

	LOG(INFO) << "CPU Version computed";
  // GPU version
  int num_kernels = this->imHeight_ * this->imWidth_ * this->height_col_ * this->width_col_;
  int default_grid_dim = CAFFE_GET_BLOCKS(num_kernels);

  // Launch with different grid sizes
  for (int grid_div = 1; grid_div <= 8; grid_div++) {
	 //LOG(INFO) << "grid_div" << grid_div;	
    for (int n = 0; n < this->blob_bottom_->num(); ++n) {
      int grid_dim = default_grid_dim/grid_div;
			if (grid_dim ==0)
				continue;
			//LOG(INFO)<<"Grid_dim: " << grid_dim <<" num_kernel: " <<num_kernels
			//				 <<"Num_Threads: " << CAFFE_CUDA_NUM_THREADS;
      // NOLINT_NEXT_LINE(whitespace/operators)
      imchannel2col_gpu_kernel<TypeParam><<<grid_dim, CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, bottom_data + this->blob_bottom_->offset(n),
        this->imHeight_, this->imWidth_,
        this->chHeight_, this->chWidth_,
			  this->kernel_size_, this->kernel_size_,
        this->pad_, this->pad_, this->stride_, this->stride_,
        this->height_col_, this->width_col_,
        top_data + this->blob_top_->offset(n));
      CUDA_POST_KERNEL_CHECK;
    }

		if (this->show_gpu_output_){
			std::cout<<"GPU Output \n";
			for (int count=0; count < this->blob_top_->count(); ++count){
				if ((count % this->total_patches_)==0)
					std::cout << " \n St: ";
				std::cout << this->blob_top_->cpu_data()[count] << " ";
			}
			std::cout<<"\n";
		}

    // Compare results against CPU version
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      TypeParam cpuval = cpu_data[i];
      TypeParam gpuval = this->blob_top_->cpu_data()[i];
      EXPECT_EQ(cpuval, gpuval);
      if (cpuval != gpuval) {
        break;
      }
    }
  }
}

}  // namespace caffe
