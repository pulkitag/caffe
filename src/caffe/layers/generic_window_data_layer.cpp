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
  this->StopInternalThread();
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
	prefetch_rng_.reset();

	// image
  const int crop_size = this->layer_param_.generic_window_data_param().crop_size();
	crop_size_          = crop_size;
  CHECK_GT(crop_size, 0);
	//Size of the batch. 
  batch_size_ = this->layer_param_.generic_window_data_param().batch_size();
	
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

	//Set the label size.
	labels_.reset(new Blob<Dtype>(num_examples_, label_size_,1,1));

	//Set size of the tops.
	if (this->layer_param_.generic_window_data_param().is_gray()){
		channels_ = 1;
	}
	else{
		channels_ = 3;
	} 
	top[0]->Reshape(batch_size_, img_group_size_ * channels_, crop_size, crop_size);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(batch_size_, img_group_size_ * channels_, crop_size, crop_size);
	} 
 LOG(INFO) << "output data size: " << top[0]->num() << ", "
      << top[0]->channels() << "," << top[0]->height() << ", "
      << top[0]->width();
 
	// label
  top[1]->Reshape(batch_size_, label_size_, 1, 1);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].label_.Reshape(batch_size_, label_size_, 1, 1);
  }

	//Initialize the read_count to zero. 
	read_count_ = 0;	

	//Setup the CropData 
	crop_tops_vec_.clear();
	crop_data_layers_.clear();

	LOG(INFO) << "Setting up crop layers";	
	for (int i=0; i < img_group_size_; i++){
		//Create the new crop layer.
		LayerParameter crop_data_param(this->layer_param_);
		crop_data_param.set_type("CropData");
		shared_ptr<CropDataLayer<Dtype> > crop_layer;
		crop_layer.reset(new CropDataLayer<Dtype>(crop_data_param));
		//Create the top for the new layer. 
		vector<Blob<Dtype>*> top_vec;
		Blob<Dtype>* top_dummy = new Blob<Dtype>(batch_size_, channels_, crop_size, crop_size);
		top_vec.push_back(top_dummy);
		crop_tops_vec_.push_back(top_vec);
		//Setup the new layer. 
		LOG(INFO) << "I am Crop" << i <<" , I am Ready: "
  						<< crop_layer->layer_param_.generic_window_data_param().is_ready();
		crop_layer->layer_num_ = i;
		crop_layer->rand_seed_ = 3;
		crop_layer->SetUp(bottom, top_vec);
		crop_layer->windows_.clear();
		crop_layer->image_database_.clear();
		crop_layer->image_database_cache_.clear();
		crop_layer->num_examples_ = num_examples_;
		crop_data_layers_.push_back(crop_layer);
	}	

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
			crop_data_layers_[i]->windows_.push_back(window);

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
			        << crop_data_layers_[i]->windows_.size();
		crop_data_layers_[i]->is_ready_ = true;
	
	/*
		LOG(INFO) << "Popping " << "Size: " << 
			crop_data_layers_[i]->prefetch_free_.size() << 
			"READINESS " << crop_data_layers_[i]->is_ready_;
		Batch<Dtype>* batch = crop_data_layers_[i]->prefetch_free_.pop();
		LOG(INFO) << "Loading batch";
		crop_data_layers_[i]->load_batch(batch);
		LOG(INFO) << "Pushing";
		crop_data_layers_[i]->prefetch_full_.push(batch);
		LOG(INFO) << "This is done";
	*/
	}
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
void GenericWindowDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;

	//Intialize the variables. 	
	Dtype* top_data    = batch->data_.mutable_cpu_data();
  Dtype* top_label   = batch->label_.mutable_cpu_data();
	const Dtype* label = labels_->cpu_data();	

	// zero out batch
	std::vector<Blob<Dtype>*> dummy_bottom;
	dummy_bottom.clear();
  caffe_set(batch->data_.count(), Dtype(0), top_data);

	//First make sure that threads of CropDataLayer have done some work.
	for (int i = 0; i < img_group_size_; i++){
		//LOG(INFO) << "WAITING";
		Batch<Dtype>* batch = crop_data_layers_[i]->prefetch_full_.pop("STUCK :( ");
		crop_data_layers_[i]->prefetch_full_.push(batch);
		//Batch<Dtype>* tmpBatch;
		//bool isData = crop_data_layers_[i]->prefetch_full_.try_peek(&tmpBatch);
		//LOG(INFO) << "PEEKING";
	} 	 

	// Copy the labels
	for (int n=0; n < batch_size_; n++){
		for (int l=0; l < label_size_; l++){
			top_label[0] = label[label_size_ * read_count_ + l];
			top_label += 1;
		}
		read_count_ += 1;
		//LOG(INFO) << "READ COUNT " << read_count_; 
		if (read_count_ >= num_examples_){
			read_count_ = 0;
			//LOG(INFO) << "Resetting read_count";
		}
	}
	
	//Do a forward pass on the CropData layers
	for (int i=0; i<img_group_size_; i++){
		crop_data_layers_[i]->Forward(dummy_bottom, crop_tops_vec_[i]);
		CHECK_EQ(read_count_, crop_data_layers_[i]->read_count_);
	}

	// Copy and interleave the data appropriately.
	const int imSz = channels_ *  crop_size_ * crop_size_;
	for (int n=0; n < batch_size_; n++){
		for (int i=0; i < img_group_size_; i++){
			caffe_copy(imSz, crop_tops_vec_[i][0]->cpu_data() + n * imSz,
									top_data);
			top_data += imSz;
		}	
	}
	
	//Start Threads for CropData layers.
	//for (int i = 0; i < img_group_size_; i++){
	//	crop_data_layers_[i]->CreatePrefetchThread();
	//}
	
  batch_timer.Stop();
  DLOG(INFO) << "Generic Window Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(GenericWindowDataLayer);
REGISTER_LAYER_CLASS(GenericWindowData);

}  // namespace caffe
