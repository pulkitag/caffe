#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

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

// caffe.proto > LayerParameter > GenericWindowDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
GenericWindowData2Layer<Dtype>::~GenericWindowData2Layer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void GenericWindowData2Layer<Dtype>::ReadMean(){
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
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels_) <<
     "Specify either 1 mean_value or as many as channels: " << channels_;
    if (channels_ > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels_; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

template <typename Dtype>
void GenericWindowData2Layer<Dtype>::ReadWindowFile(){
  string root_folder = this->layer_param_.generic_window_data_param().root_folder();
	std::ifstream infile(this->layer_param_.generic_window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.generic_window_data_param().source() << std::endl;

  string hashtag, checkName;
  int image_index;
  if (!(infile >> hashtag >> checkName)) {
    LOG(FATAL) << "Window file is empty";
  }
	CHECK_EQ(hashtag, "#");
	CHECK_EQ(checkName, "GenericDataLayer");
	infile >> num_examples_ >> img_group_size_ >> label_size_;
	LOG(INFO) << "num examples: "   << num_examples_ << ","
						<< "img_group_size: " << img_group_size_ << ","
						<< "label_size: "     << label_size_;

	//Initialize the windows
	all_windows_.clear();
	all_windows_.resize(img_group_size_);

	//Initialize the database
	all_image_database_.clear();
	all_image_database_.resize(img_group_size_);
	all_image_database_cache_.clear();
	all_image_database_cache_.resize(img_group_size_);

	//Intialize the store for labels
	labels_.reset(new Blob<Dtype>(num_examples_, label_size_,1,1));
	
	LOG(INFO) << "Processing Image and labels";
	Dtype* label_data = labels_->mutable_cpu_data(); 
	string tmp_hash;
  int tmp_num;
	infile >> tmp_hash >> tmp_num;
	int count_examples = 0;  
	do {
    CHECK_EQ(hashtag, "#");
		//LOG(INFO) << "I am here";
		for (int i=0; i < img_group_size_; i++){
			// read image path
			string image_path;
			infile >> image_path;
			image_path = root_folder + image_path;

			// read image dimensions
			vector<int> image_size(3);
			infile >> image_size[0] >> image_size[1] >> image_size[2];
			all_image_database_[i].push_back(std::make_pair(image_path, image_size));
			
			//Store the image if needed. 
			if (cache_images_) {
				Datum datum;
				if (!ReadFileToDatum(image_path, &datum)) {
					LOG(ERROR) << "Could not open or find file " << image_path;
					return;
				}
				all_image_database_cache_[i].push_back(datum);
			}
    
			//Read the window. 
		  int x1, y1, x2, y2;
			infile >> x1 >> y1 >> x2 >> y2;
			CHECK_GE(x1, 0) << "x1 is less than 0";
			CHECK_GE(y1, 0) << "y1 is less than 0";
			CHECK_LT(x2, image_size[2]) << "x2 is greater than imgSz";
			CHECK_LT(y2, image_size[1]) << "y2 is greater than imgSz";
			
			//LOG(INFO) << x1 << "\t" << y1 << "\t" << x2 << "\t" << y2;
      vector<float> window(CropDataLayer<Dtype>::NUM);
      window[CropDataLayer<Dtype>::IMAGE_INDEX] = image_index;
      window[CropDataLayer<Dtype>::X1] = x1;
      window[CropDataLayer<Dtype>::Y1] = y1;
      window[CropDataLayer<Dtype>::X2] = x2;
      window[CropDataLayer<Dtype>::Y2] = y2;
			all_windows_[i].push_back(window);

			if (image_index % 1000 == 0) {
				LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_examples_;
			}
    }
		//Read the labels
		float lbl;
		for (int l=0; l < label_size_; l++){
			infile >> lbl;
			label_data[0] = lbl;
			label_data += 1;
		}
		count_examples += 1;
		CHECK_GE(num_examples_, count_examples);
  } while (infile >> hashtag >> image_index);
	infile.close();
	CHECK_EQ(num_examples_, count_examples);
	
	for (int i=0; i<img_group_size_; i++){
		LOG(INFO) << "Number of windows: "
			        << all_windows_[i].size();
	}

}

template <typename Dtype>
void GenericWindowData2Layer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the window_file and creates structures
  // which store the filenames. 
	// Format
	// # GenericDataLayer2
	//	Num Examples
	//	Num Images per Example
	//  Num Labels
	// # <Example Number>
	// ImName-1 channels height width x1 y1 x2 y2
	// ImName-2 ...
	// .
	// .
	// ImName-k ...
	// Label (N-D as floats)
	
  LOG(INFO) << "Generic Window data layer 2:" << std::endl
			<< "  cache_images: "
      << this->layer_param_.generic_window_data_param().cache_images() << std::endl
      << "  root_folder: "
      << this->layer_param_.generic_window_data_param().root_folder();

	cache_images_      = this->layer_param_.generic_window_data_param().cache_images();
	is_random_crop_    = this->layer_param_.generic_window_data_param().random_crop();

	LOG(INFO) << "Random cropping: " << is_random_crop_;
	LOG(INFO) << "Crop Size: " 
      << this->layer_param_.generic_window_data_param().crop_size();
	LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.generic_window_data_param().context_pad();
  LOG(INFO) << "Crop mode: "
      << this->layer_param_.generic_window_data_param().crop_mode();

  //Crop Size
  const int crop_size = this->layer_param_.generic_window_data_param().crop_size();
  CHECK_GT(crop_size, 0);

	//Size of the batch. 
  batch_size_ = this->layer_param_.generic_window_data_param().batch_size();

	//Set size of the tops.
	if (this->layer_param_.generic_window_data_param().is_gray()){
		channels_ = 1;
	}
	else{
		channels_ = 3;
	}
	
	ReadWindowFile();	
	ReadMean();
	prefetch_rng_.clear();
	prefetch_rng_.resize(img_group_size_);
	for (int ii=0; ii < img_group_size_; ii++){ 
		if (is_random_crop_){
			prefetch_rng_[ii].reset(new Caffe::RNG(rand_seed_));
		}else{
			prefetch_rng_[ii].reset();
		} 
	}

	//Data
	top[0]->Reshape(batch_size_, img_group_size_ * channels_, crop_size, crop_size);
 	//Labels
  top[1]->Reshape(batch_size_, label_size_, 1, 1);
  
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(batch_size_, 
								img_group_size_ * channels_, crop_size, crop_size);
	} 
	LOG(INFO) << "output data size: " << top[0]->num() << ", "
      << top[0]->channels() << "," << top[0]->height() << ", "
      << top[0]->width();
 
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].label_.Reshape(batch_size_, label_size_, 1, 1);
  }
	
	//Initialize the read_count to zero. 
	read_count_ = 0;
	stream_data_.clear();
	for (int ii=0; ii<img_group_size_; ii++){
		Blob<Dtype>* dummy = new Blob<Dtype>(batch_size_, channels_, crop_size, crop_size);
		stream_data_.push_back(dummy);
	} 
}


template <typename Dtype>
unsigned int GenericWindowData2Layer<Dtype>::PrefetchRand(int streamNum) {
  CHECK(prefetch_rng_[streamNum]);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_[streamNum]->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void GenericWindowData2Layer<Dtype>::ReadData(int streamNum, Dtype* top_data) {
	CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
	LOG(INFO) << "READ 1";

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

	LOG(INFO) << "READ 2";
  const int num_samples = static_cast<int>(static_cast<float>(batch_size));
	//Get the data for this stream
	LOG(INFO) << "L1";
	vector<Datum> image_database_cache_ = all_image_database_cache_[streamNum];
	LOG(INFO) << "L2";
	vector<vector<float> > windows_     = all_windows_[streamNum];
	LOG(INFO) << "L3";
	vector<std::pair<std::string, vector<int> > >image_database_ 
                                      = all_image_database_[streamNum]; 	
	LOG(INFO) << "L4";
	if (windows_.size() < num_examples_){
		LOG(INFO) << "###### CRASHING: WINDOWS ARE NOT INITALIZED ######";
	}	
	//Assert windows_ are not empty
	CHECK_EQ(windows_.size(), num_examples_);

	LOG(INFO) << "READ 3";
  int item_id = 0;
	for (int dummy = 0; dummy < num_samples; ++dummy) {
		// sample a window
		timer.Start();
		//LOG(INFO) << "Size of windows: " <<  windows_.size() 
		//					<< " Read Count: " << read_count_;
		vector<float> window = windows_[read_count_];

		bool do_mirror = mirror && PrefetchRand(streamNum) % 2;

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
			const unsigned int rand_index_h = PrefetchRand(streamNum);
			const unsigned int rand_index_w = PrefetchRand(streamNum);
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
			int top_index;
			for (int w = 0; w < cv_cropped_img.cols; ++w) {
				for (int c = 0; c < channels; ++c) {
					top_index = ((item_id * channels + c) * crop_size + h)
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
			LOG(INFO) << top_data[top_index];
		}
	//END WITHOUT PADDING
		/*
		// COPY WITH PADDING 
		// copy the warped window into top_data
		for (int h = 0; h < cv_cropped_img.rows; ++h) {
			const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
			int img_index = 0;
			for (int w = 0; w < cv_cropped_img.cols; ++w) {
				for (int c = 0; c < channels; ++c) {
					int top_index = ((item_id * channels + c) * crop_size + h + pad_h)
									 * crop_size + w + pad_w;
					// int top_index = (c * height + h) * width + w;
					Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
					if (this->has_mean_file_) {
						int mean_index = (c * mean_height + h + mean_off + pad_h)
												 * mean_width + w + mean_off + pad_w;
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
	  */
		trans_time += timer.MicroSeconds();

		#if 0
		// useful debugging code for dumping transformed windows to disk
		string file_id;
		std::stringstream ss;
		ss << PrefetchRand();
		ss >> file_id;
		std::ofstream inf((string("dump/") + file_id +
				string("_info.txt")).c_str(), std::ofstream::out);
		inf << image.first << std::endl
				<< window[CropDataLayer<Dtype>::X1]+1 << std::endl
				<< window[CropDataLayer<Dtype>::Y1]+1 << std::endl
				<< window[CropDataLayer<Dtype>::X2]+1 << std::endl
				<< window[CropDataLayer<Dtype>::Y2]+1 << std::endl
				<< do_mirror << std::endl
		inf.close();
		std::ofstream top_data_file((string("dump/") + file_id +
				string("_data.txt")).c_str(),
				std::ofstream::out | std::ofstream::binary);
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < crop_size; ++h) {
				for (int w = 0; w < crop_size; ++w) {
					top_data_file.write(reinterpret_cast<char*>(
							&top_data[((item_id * channels + c) * crop_size + h)
												* crop_size + w]),
							sizeof(Dtype));
				}
			}
		}
		top_data_file.close();
		#endif

		item_id++;
		read_count_++;
		//If the end of the window file is reached. 
		if (read_count_ == num_examples_){
			read_count_ = 0;
			LOG(INFO) << "Resetting read_count";
		}
	}
	LOG(INFO) << "IMAGES READ!!";
}


// Thread fetching the data
template <typename Dtype>
void GenericWindowData2Layer<Dtype>::load_batch(Batch<Dtype>* batch) {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

	LOG(INFO) << "CHECK 1";
	//Intialize the variables. 	
	Dtype* top_data    = batch->data_.mutable_cpu_data();
  Dtype* top_label   = batch->label_.mutable_cpu_data();
	const Dtype* label = labels_->cpu_data();	

	// zero out batch
	std::vector<Blob<Dtype>*> dummy_bottom;
	dummy_bottom.clear();
  caffe_set(batch->data_.count(), Dtype(0), top_data);

	LOG(INFO) << "CHECK 2";
	// Copy the labels
	for (int n=0; n < batch_size_; n++){
		for (int l=0; l < label_size_; l++){
			top_label[0] = label[label_size_ * read_count_ + l];
			top_label += 1;
		}
		read_count_ += 1;
		if (read_count_ >= num_examples_){
			read_count_ = 0;
			LOG(INFO) << "Resetting read_count";
		}
	}
	
	LOG(INFO) << "CHECK 3";
	//Do a forward pass on the CropData layers
	for (int i=0; i<img_group_size_; i++){
		ReadData(i, stream_data_[i]->mutable_cpu_data());
	}

	// Copy and interleave the data appropriately.
	const int imSz = channels_ *  crop_size_ * crop_size_;
	for (int n=0; n < batch_size_; n++){
		for (int i=0; i < img_group_size_; i++){
			caffe_copy(imSz, stream_data_[i]->cpu_data() + n * imSz,
									top_data);
			top_data += imSz;
		}	
	}
	
	//Start Threads for CropData layers.
	//for (int i = 0; i < img_group_size_; i++){
	//	crop_data_layers_[i]->CreatePrefetchThread();
	//}
	
  batch_timer.Stop();
  DLOG(INFO) << "Generic Window-2 Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(GenericWindowData2Layer);
REGISTER_LAYER_CLASS(GenericWindowData2);

}  // namespace caffe
