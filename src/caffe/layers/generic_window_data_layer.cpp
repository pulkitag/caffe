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
GenericWindowDataLayer<Dtype>::~GenericWindowDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void GenericWindowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the window_file and creates structures
  // which store the filenames. 
	// Format
	// # GenericDataLayer
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
	
  LOG(INFO) << "Generic Window data layer:" << std::endl
			<< "  cache_images: "
      << this->layer_param_.generic_window_data_param().cache_images() << std::endl
      << "  root_folder: "
      << this->layer_param_.generic_window_data_param().root_folder();

  cache_images_ = this->layer_param_.generic_window_data_param().cache_images();
  string root_folder = this->layer_param_.generic_window_data_param().root_folder();

  const bool prefetch_needs_rand =
      this->transform_param_.mirror() ||
      this->transform_param_.crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

	// image
  const int crop_size = this->transform_param_.crop_size();
  CHECK_GT(crop_size, 0);
	//Size of the batch. 
  batch_size_ = this->layer_param_.generic_window_data_param().batch_size();
	
  std::ifstream infile(this->layer_param_.generic_window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.generic_window_data_param().source() << std::endl;


  string hashtag, checkName;
  int image_index, channels;
  if (!(infile >> hashtag >> checkName)) {
    LOG(FATAL) << "Window file is empty";
  }
	CHECK_EQ(hashtag, "#");
	CHECK_EQ(checkName, "GenericDataLayer");
	infile >> num_examples_ >> img_group_size_ >> label_size_;

	//Set the label size.
	labels_.reset(new Blob<Dtype>(num_examples_, label_size_,1,1));

	//Set size of the tops.
	const int top_channels = 3; 
	top[0]->Reshape(batch_size_, img_group_size_ * top_channels, crop_size, crop_size);
	this->prefetch_data_.Reshape(batch_size_, img_group_size_ * top_channels, crop_size, crop_size);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  top[img_group_size_]->Reshape(batch_size_, label_size_, 1, 1);
  this->prefetch_label_.Reshape(batch_size_, label_size_, 1, 1);

	//Initialize the read_count to zero. 
	read_count_ = 0;	

	//Setup the CropData 
	crop_tops_vec_.clear();
	vector<Blob<Dtype>*> dummy_bottom;
	
	for (int i=0; i < img_group_size_; i++){
		//Create the new crop layer. 
		LayerParameter crop_data_param(this->layer_param_);
		crop_data_param.set_type("CropData");
		shared_ptr<CropDataLayer<Dtype> > crop_layer;
		crop_layer.reset(new CropDataLayer<Dtype>(crop_data_param));
		//Create the top for the new layer. 
		vector<Blob<Dtype>*> top_vec;
		Blob<Dtype>* top_dummy;
		top_dummy->Reshape(batch_size_, top_channels, crop_size, crop_size);
		top_vec.push_back(top_dummy);
		crop_tops_vec_.push_back(top_vec);
		//Setup the new layer.  
		crop_layer->SetUp(dummy_bottom, top_vec); 
		crop_data_layers_.push_back(crop_layer);
	}	

	
	Dtype* label_data = labels_->mutable_cpu_data(); 
  do {
    CHECK_EQ(hashtag, "#");
		for (int i=0; i < img_group_size_; i++){
			// read image path
			string image_path;
			infile >> image_path;
			image_path = root_folder + image_path;
			
			// read image dimensions
			vector<int> image_size(3);
			infile >> image_size[0] >> image_size[1] >> image_size[2];
			channels = image_size[0];
			crop_data_layers_[i]->image_database_.push_back(std::make_pair(image_path, image_size));
			
			//Store the image if needed. 
			if (cache_images_) {
				Datum datum;
				if (!ReadFileToDatum(image_path, &datum)) {
					LOG(ERROR) << "Could not open or find file " << image_path;
					return;
				}
				crop_data_layers_[i]->image_database_cache_.push_back(datum);
			}
    
			//Read the window. 
		  int x1, y1, x2, y2;
			infile >> x1 >> y1 >> x2 >> y2;
      vector<float> window(CropDataLayer<Dtype>::NUM);
      window[CropDataLayer<Dtype>::IMAGE_INDEX] = image_index;
      window[CropDataLayer<Dtype>::X1] = x1;
      window[CropDataLayer<Dtype>::Y1] = y1;
      window[CropDataLayer<Dtype>::X2] = x2;
      window[CropDataLayer<Dtype>::Y2] = y2;
			crop_data_layers_[i]->windows_.push_back(window);

			//Read the labels
			float lbl;
			for (int l=0; l < label_size_; l++){
				infile >> lbl;
				label_data[0] = lbl;
				label_data += 1;
			}
			if (image_index % 10000 == 0) {
				LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_examples_;
			}
    }
  } while (infile >> hashtag >> image_index);
  
}

template <typename Dtype>
unsigned int GenericWindowDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void GenericWindowDataLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  Dtype* top_data    = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label   = this->prefetch_label_.mutable_cpu_data();
	const Dtype* label = labels_->cpu_data();	

	// zero out batch
	std::vector<Blob<Dtype>*> dummy_bottom;
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);
	
	for (int i=0; i<img_group_size_; i++){
		crop_data_layers_[i]->Forward(dummy_bottom, crop_tops_vec_[i]);
		caffe_copy(crop_tops_vec_[i][0]->count(), crop_tops_vec_[i][0]->cpu_data(),
								top_data);
		top_data += crop_tops_vec_[i][0]->count();	
	}
	
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

  batch_timer.Stop();
  DLOG(INFO) << "Generic Window Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(GenericWindowDataLayer);
REGISTER_LAYER_CLASS(GenericWindowData);

}  // namespace caffe
