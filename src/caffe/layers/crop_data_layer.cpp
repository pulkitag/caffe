#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>
#include <boost/thread.hpp>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > CropDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
CropDataLayer<Dtype>::~CropDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void CropDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  LOG(INFO) << "CropData layer:" << std::endl
      << "  cache_images: "
      << this->layer_param_.generic_window_data_param().cache_images() << std::endl;

  is_ready_          = this->layer_param_.generic_window_data_param().is_ready();
  cache_images_      = this->layer_param_.generic_window_data_param().cache_images();
  string root_folder = this->layer_param_.generic_window_data_param().root_folder();
	is_random_crop_    = this->layer_param_.generic_window_data_param().random_crop();
	max_jitter_        = this->layer_param_.generic_window_data_param().max_jitter();

	if (is_random_crop_ || max_jitter_ > 0){
    prefetch_rng_.reset(new Caffe::RNG(rand_seed_));
	}else{
		prefetch_rng_.reset();
	}

	LOG(INFO) << "Random cropping: " << is_random_crop_;
	LOG(INFO) << "Crop Size: " 
      << this->layer_param_.generic_window_data_param().crop_size();
	LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.generic_window_data_param().context_pad();
  LOG(INFO) << "Crop mode: "
      << this->layer_param_.generic_window_data_param().crop_mode();
	LOG(INFO) << "Maximum Jitter: "
      << this->layer_param_.generic_window_data_param().max_jitter();
  // image
  const int crop_size = this->layer_param_.generic_window_data_param().crop_size();
  CHECK_GT(crop_size, 0);
  const int batch_size = this->layer_param_.generic_window_data_param().batch_size();
  const int channels = top[0]->channels();
	channels_ = channels;
	for (int i = 0; i < this->PREFETCH_COUNT; ++i){
		this->prefetch_[i].data_.Reshape(batch_size, channels, crop_size, crop_size);
	}
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  // data mean
  has_mean_file_   = this->transform_param_.has_mean_file();
  has_mean_values_ = this->transform_param_.mean_value_size() > 0;
  if (has_mean_file_) {
    const string& mean_file =
          this->transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  if (has_mean_values_) {
    CHECK(has_mean_file_ == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < this->transform_param_.mean_value_size(); ++c) {
      mean_values_.push_back(this->transform_param_.mean_value(c));
    }
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
	
	//Initialize read_count_
	read_count_ = 0;
	LOG(INFO) << " #### I am INSIDE Crop, I am Ready: "
						<< is_ready_;
}

template <typename Dtype>
unsigned int CropDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void CropDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif
	while (!is_ready_){
		LOG_EVERY_N(INFO, 1000000) << "RESCUE ME";
		//Do Nothing
	}

	LOG(INFO) << "I AM STARTING >>> YEAH";
  try {
    while (!this->must_stop()) {
      Batch<Dtype>* batch = this->prefetch_free_.pop();
  		//LOG(INFO) << "BATCH LOADING ROUTINE";
	    load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      this->prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}


// Thread fetching the data
template <typename Dtype>
void CropDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows
  
	//LOG(INFO) << "###### READY AT ENTRY ##### " << is_ready_;

	if (!is_ready_){
		LOG(INFO) << "###### CropDataLayer is not ready ######";
		//THis is necessary as otherwise the layer will stall. 
		//this->prefetch_free_.push(batch);
		return;
	}

	// /*
	CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  Dtype* top_data       = batch->data_.mutable_cpu_data();
  //Dtype* top_label      = batch->label_.mutable_cpu_data();
  const Dtype scale     = this->layer_param_.generic_window_data_param().scale();
  const int batch_size  = this->layer_param_.generic_window_data_param().batch_size();
  const int context_pad = this->layer_param_.generic_window_data_param().context_pad();
  const int crop_size   = this->layer_param_.generic_window_data_param().crop_size();
  const bool mirror     = this->layer_param_.generic_window_data_param().mirror();
  //const int crop_size   = this->transform_param_.crop_size();
  //const bool mirror     = this->transform_param_.mirror();
  Dtype* mean     = NULL;
  int mean_off    = 0;
  int mean_width  = 0;
  int mean_height = 0;
  if (this->has_mean_file_) {
    mean        = this->data_mean_.mutable_cpu_data();
    mean_off    = (this->data_mean_.width() - crop_size) / 2;
    mean_width  = this->data_mean_.width();
    mean_height = this->data_mean_.height();
  }
  cv::Size cv_crop_size(crop_size, crop_size);
  const string& crop_mode = this->layer_param_.generic_window_data_param().crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  caffe_set(batch->data_.count(), Dtype(0), top_data);
  const int num_samples = static_cast<int>(static_cast<float>(batch_size));

	if (windows_.size() < num_examples_){
		LOG(INFO) << "###### CRASHING: WINDOWS ARE NOT INITALIZED ######";
		LOG(INFO) << windows_.size() << " " << num_examples_;
	}	
	//Assert windows_ are not empty
	CHECK_EQ(windows_.size(), num_examples_);
  int item_id = 0;
	for (int dummy = 0; dummy < num_samples; ++dummy) {
	
		// sample a window
		timer.Start();
		//LOG(INFO) << "Size of windows: " <<  windows_.size() 
		//					<< " Read Count: " << read_count_;
		vector<float> window = windows_[read_count_];

		bool do_mirror = mirror && PrefetchRand() % 2;

		//Get the image
		cv::Mat cv_img;
		int imHeight, imWidth;
		if (this->cache_images_) {
			Datum image_cached = image_database_cache_[read_count_];
			cv_img = DecodeDatumToCVMat(image_cached, true);
		} else {
			// load the image containing the window
			pair<std::string, vector<int> > image = image_database_[read_count_];
			imHeight = image.second[1];
			imWidth  = image.second[2];
			//LOG(INFO) << image.first;
			if (channels_ == 1){ 
				cv_img = cv::imread(image.first, CV_LOAD_IMAGE_GRAYSCALE);
			}
			else{
				cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
			}
			if (!cv_img.data) {
				LOG(ERROR) << "Could not open or find file " << image.first;
				return;
			}
		}
		read_time += timer.MicroSeconds();
		timer.Start();
		const int channels = cv_img.channels();
		CHECK_EQ(channels, channels_);

		// crop window out of image and warp it
		int x1, y1, x2, y2;
		if (is_random_crop_){
			int maxHeightSt = std::max(0, imHeight - crop_size);
			int maxWidthSt  = std::max(0, imWidth - crop_size);
			const unsigned int rand_index_h = PrefetchRand();
			const unsigned int rand_index_w = PrefetchRand();
			x1 = rand_index_w % maxWidthSt;
			y1 = rand_index_h % maxHeightSt;
			x2 = x1 + crop_size - 1;
			y2 = y1 + crop_size - 1; 
			//LOG(INFO) << "Crop Layer num: " << layer_num_ << " dims: " 
			//					<< x1 <<", " << x2 << ", " << y1 <<", " << ", " << y2;
		}else{
			x1 = window[CropDataLayer<Dtype>::X1];
			y1 = window[CropDataLayer<Dtype>::Y1];
			x2 = window[CropDataLayer<Dtype>::X2];
			y2 = window[CropDataLayer<Dtype>::Y2];
			if (this->phase_ == TRAIN && max_jitter_ > 0){
				const unsigned int rand_index_h = PrefetchRand();
				const unsigned int rand_index_w = PrefetchRand();
				const unsigned int pos_index_h = PrefetchRand();
				const unsigned int pos_index_w = PrefetchRand();
				int x_jit = rand_index_w % max_jitter_;
				int y_jit = rand_index_h % max_jitter_;
				int x_sgn = pos_index_w  % 2;
				int y_sgn = pos_index_h % 2;			
				if (x_sgn==0)
					x_jit = -x_jit;
				if (y_sgn==0)
					y_jit = -y_jit;
				x1 += x_jit;
				x2 += x_jit;
				y1 += y_jit;
				y2 += y_jit;
				x1  = std::max(0, x1);
				y1  = std::max(0, y1);
				x2  = std::min(imWidth-1,  x2);
				y2  = std::max(imHeight-1, y2);
		}	
	}
		
		int pad_w = 0;
		int pad_h = 0;
		if (context_pad > 0 || use_square) {
			// scale factor by which to expand the original region
			// such that after warping the expanded region to crop_size x crop_size
			// there's exactly context_pad amount of padding on each side
			Dtype context_scale = static_cast<Dtype>(crop_size) /
					static_cast<Dtype>(crop_size - 2*context_pad);

			// compute the expanded region
			Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
			Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
			Dtype center_x = static_cast<Dtype>(x1) + half_width;
			Dtype center_y = static_cast<Dtype>(y1) + half_height;
			if (use_square) {
				if (half_height > half_width) {
					half_width = half_height;
				} else {
					half_height = half_width;
				}
			}
			x1 = static_cast<int>(round(center_x - half_width*context_scale));
			x2 = static_cast<int>(round(center_x + half_width*context_scale));
			y1 = static_cast<int>(round(center_y - half_height*context_scale));
			y2 = static_cast<int>(round(center_y + half_height*context_scale));

			// the expanded region may go outside of the image
			// so we compute the clipped (expanded) region and keep track of
			// the extent beyond the image
			int unclipped_height = y2-y1+1;
			int unclipped_width = x2-x1+1;
			int pad_x1 = std::max(0, -x1);
			int pad_y1 = std::max(0, -y1);
			int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
			int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
			// clip bounds
			x1 = x1 + pad_x1;
			x2 = x2 - pad_x2;
			y1 = y1 + pad_y1;
			y2 = y2 - pad_y2;
			CHECK_GT(x1, -1);
			CHECK_GT(y1, -1);
			CHECK_LT(x2, cv_img.cols);
			CHECK_LT(y2, cv_img.rows);

			int clipped_height = y2-y1+1;
			int clipped_width = x2-x1+1;

			// scale factors that would be used to warp the unclipped
			// expanded region
			Dtype scale_x =
					static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
			Dtype scale_y =
					static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

			// #### No Padding Change #####
			
			// size to warp the clipped expanded region to
			cv_crop_size.width =
					static_cast<int>(round(static_cast<Dtype>(unclipped_width)*scale_x));
			cv_crop_size.height =
					static_cast<int>(round(static_cast<Dtype>(unclipped_height)*scale_y));

			cv_crop_size.width  = std::min(cv_crop_size.width, crop_size);
			cv_crop_size.height = std::min(cv_crop_size.height, crop_size);
			//##### End No Padding Change #####
		}

		//Clip the image
		//LOG(INFO) << x1 << ", " << x2 <<", " << y1 << ", " << y2;
		cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
		cv::Mat cv_cropped_img = cv_img(roi);

		//Resize the image. 
		cv::resize(cv_cropped_img, cv_cropped_img,
				cv_crop_size, 0, 0, cv::INTER_LINEAR);

		// horizontal flip at random
		if (do_mirror) {
			cv::flip(cv_cropped_img, cv_cropped_img, 1);
		}

		//COPY WITHOUT PADDING
		// copy the warped window into top_data
		for (int h = 0; h < cv_cropped_img.rows; ++h) {
			const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
			int img_index = 0;
			for (int w = 0; w < cv_cropped_img.cols; ++w) {
				for (int c = 0; c < channels; ++c) {
					int top_index = ((item_id * channels + c) * crop_size + h)
									 * crop_size + w ;
					Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
					if (this->has_mean_file_) {
						int mean_index = (c * mean_height + h + mean_off)
												 * mean_width + w + mean_off;
						top_data[top_index] = (pixel - mean[mean_index]) * scale;
					} else {
						if (this->has_mean_values_) {
							top_data[top_index] = (pixel - this->mean_values_[c]) * scale;
						} else {
							top_data[top_index] = pixel * scale;
						}
					}
				}
			}
		}
	//END WITHOUT PADDING
		
		trans_time += timer.MicroSeconds();
		item_id++;
		read_count_++;
		//If the end of the window file is reached. 
		if (read_count_ == num_examples_){
			read_count_ = 0;
			//LOG(INFO) << "Resetting read_count";
		}
	}
  batch_timer.Stop();
  DLOG(INFO) << "CropDataLayer Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";

}

template <typename Dtype>
void CropDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	
	//LOG(INFO) << "I AM AT WORK";
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }
  this->prefetch_free_.push(batch);
	//fwdCount_ += top[0]->num()
}


INSTANTIATE_CLASS(CropDataLayer);
REGISTER_LAYER_CLASS(CropData);

}  // namespace caffe
