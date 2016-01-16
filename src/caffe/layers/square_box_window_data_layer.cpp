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
#include "caffe/layers/square_box_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > SquareBoxDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
SquareBoxDataLayer<Dtype>::~SquareBoxDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void SquareBoxDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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

  LOG(INFO) << "SquareBox Window data layer:" << std::endl
      << "  root_folder: "
      << this->layer_param_.square_box_data_param().root_folder();

  string root_folder = this->layer_param_.square_box_data_param().root_folder();

	const unsigned int prefetch_rng_seed = caffe_rng_rand();
	prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
	//prefetch_rng_.reset();

  std::ifstream infile(this->layer_param_.square_box_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.square_box_data_param().source() << std::endl;


  string hashtag, checkName;
  if (!(infile >> hashtag >> checkName)) {
    LOG(FATAL) << "Window file is empty";
  }
	CHECK_EQ(hashtag, "#");
	CHECK_EQ(checkName, "SqBoxWindowDataLayer");
	infile >> num_im_;
	int label_size = 3;
  
  int image_index, channels;
	infile >> hashtag >> image_index;
	do {
    CHECK_EQ(hashtag, "#");
    int num_windows;
    string image_path;
    // read image path and number of windows
    infile >> num_windows >> image_path;
    image_path = root_folder + image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
    image_database_.push_back(std::make_pair(image_path, image_size));

		//Read all windows in an image 
    for (int i = 0; i < num_windows; ++i) {
      int obj_cx, obj_cy, box_cx, box_cy, box_sz;
      infile >> obj_cx >> obj_cy >> box_cx >> box_cy >> box_sz;
     	//Make a window
			vector<int> window(SquareBoxDataLayer::NUM);
      window[SquareBoxDataLayer::IMAGE_INDEX] = image_index;
      window[SquareBoxDataLayer::OBJ_X] = obj_cx;
      window[SquareBoxDataLayer::OBJ_Y] = obj_cy;
      window[SquareBoxDataLayer::BOX_CX] = box_cx;
      window[SquareBoxDataLayer::BOX_CY] = box_cy;
      window[SquareBoxDataLayer::BOX_SZ] = box_sz;
			windows_.push_back(window);
		}

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  } while (infile >> hashtag >> image_index);

  LOG(INFO) << "Number of images: " << image_index+1;

	// image
  const int crop_size = this->layer_param_.square_box_data_param().crop_size();
  CHECK_GT(crop_size, 0);
  const int batch_size = this->layer_param_.square_box_data_param().batch_size();
  top[0]->Reshape(batch_size, channels, crop_size, crop_size);

  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(batch_size, channels, crop_size, crop_size);
	}
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  top[1]->Reshape(batch_size, 1, 1, 1);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(batch_size, 1, 1, 1);
  }

  // data mean
  has_mean_file_ = this->transform_param_.has_mean_file();
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
}

template <typename Dtype>
unsigned int SquareBoxDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void SquareBoxDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  Dtype* top_data  = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();
  const int batch_size = this->layer_param_.square_box_data_param().batch_size();
 	const Dtype mn_bxsz_by_imsz = this->layer_param_.square_box_data_param().mn_bxsz_by_imsz();
 	const Dtype mx_bxsz_by_imsz = this->layer_param_.square_box_data_param().mx_bxsz_by_imsz();
 	const Dtype mx_dist         = this->layer_param_.square_box_data_param().mx_dist();
  const int crop_size = this->layer_param_.square_box_data_param().crop_size();
  const bool mirror = this->transform_param_.mirror();
  Dtype* mean = NULL;
  int mean_off = 0;
  int mean_width = 0;
  int mean_height = 0;
  if (this->has_mean_file_) {
    mean = this->data_mean_.mutable_cpu_data();
    mean_off = (this->data_mean_.width() - crop_size) / 2;
    mean_width = this->data_mean_.width();
    mean_height = this->data_mean_.height();
  }
  cv::Size cv_crop_size(crop_size, crop_size);

	LOG(INFO) << "LOC 0";
  // zero out batch
  caffe_set(batch->data_.count(), Dtype(0), top_data);

  int item_id = 0;
	for (int dummy = 0; dummy < batch_size; ++dummy) {
		// sample a window
		timer.Start();
		LOG(INFO) << "LOC RAND";
		const unsigned int rand_index = PrefetchRand();
		LOG(INFO) << "LOC RAND PASSED" << rand_index;
		LOG(INFO) << "Size: " << windows_.size();
		vector<int> window = windows_[rand_index % windows_.size()];
	 
		LOG(INFO) << "LOC RAND BETWEEN";
		//If mirroring 
		bool do_mirror = mirror && PrefetchRand() % 2;

		// load the image containing the window
		pair<std::string, vector<int> > image =
				image_database_[window[SquareBoxDataLayer<Dtype>::IMAGE_INDEX]];

		LOG(INFO) << "LOC RAND AFTER";
		//Load the image
		cv::Mat cv_img;
		cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
		if (!cv_img.data) {
				LOG(ERROR) << "Could not open or find file " << image.first;
				return;
		}
		read_time += timer.MicroSeconds();
		timer.Start();
		const int channels = cv_img.channels();
		const int imHeight = image.second[1];
		const int imWidth  = image.second[2];
		const int maxSide  = std::max(imHeight, imWidth);    

		LOG(INFO) << "LOC 1";

		 // crop window out of image and warp it
		int o_x  = window[SquareBoxDataLayer<Dtype>::OBJ_X];
		int o_y  = window[SquareBoxDataLayer<Dtype>::OBJ_Y];
		int b_cx = window[SquareBoxDataLayer<Dtype>::BOX_CX];
		int b_cy = window[SquareBoxDataLayer<Dtype>::BOX_CY];
		int sz   = window[SquareBoxDataLayer<Dtype>::BOX_SZ];

		//This is only used for computing the gt overlap.
		int bx1, bx2, by1, by2;
		bx1 = std::max(0, static_cast<int>(round(b_cx - sz / 2.0)));
		bx2 = std::min(imWidth, static_cast<int>(round(b_cx + sz / 2.0))); 
		by1 = std::max(0, static_cast<int>(round(b_cy - sz / 2.0)));
		by2 = std::min(imHeight, static_cast<int>(round(b_cy + sz / 2.0))); 
		Dtype areaGtBox = (by2 - by1) * (bx2 - bx1);		

		Dtype sample_bsz, sample_bcx, sample_bcy;
		bool sampleFlag = true;
		LOG(INFO) << "LOC 2";
		while (sampleFlag){
			//Sample the scale ratio
			Dtype scale_ratio;
			caffe_rng_uniform(1, mn_bxsz_by_imsz, mx_bxsz_by_imsz, &scale_ratio);
			sample_bsz = maxSide * scale_ratio;  	

			//Sample the distance 
			Dtype dist1, dist2, coinFlip1, coinFlip2;
			caffe_rng_uniform(1, Dtype(0), Dtype(mx_dist), &dist1);
			caffe_rng_uniform(1, Dtype(0), Dtype(mx_dist), &dist2);
			caffe_rng_uniform(1, Dtype(0), Dtype(1), &coinFlip1);
			caffe_rng_uniform(1, Dtype(0), Dtype(1), &coinFlip2);
			if (coinFlip1 < 0.5)
				dist1 = -dist1;
			if (coinFlip2 < 0.5)
				dist2 = -dist2;
			sample_bcx = o_x + dist1;
			sample_bcy = o_y + dist2;

			sampleFlag = sampleFlag && (sample_bcx > 0) && (sample_bcx < imWidth);
			sampleFlag = sampleFlag && (sample_bcy > 0) && (sample_bcy < imHeight);
			//Keep the object within the box
			sampleFlag = sampleFlag && (dist1 < sample_bsz) && (-dist1 < sample_bsz);
			sampleFlag = sampleFlag && (dist2 < sample_bsz) && (-dist2 < sample_bsz);
			sampleFlag = !sampleFlag;
		 }
		LOG(INFO) << "LOC 3";
		//Predict the displacement
		Dtype delx = sample_bcx - o_x;
		Dtype dely = sample_bcy - o_y;
		//Predict the overlap
		int x1, x2, y1, y2;
		x1 = static_cast<int>(round(sample_bcx - sample_bsz / 2.0));
		x2 = static_cast<int>(round(sample_bcx + sample_bsz / 2.0));
		y1 = static_cast<int>(round(sample_bcy - sample_bsz / 2.0));
		y2 = static_cast<int>(round(sample_bcy + sample_bsz / 2.0));

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
		Dtype areaBox = (x2 - x1) * (y2 - y1);

		//Get the coordinates of the intersection box.
		int ix1, ix2, iy1, iy2;
		ix1 = std::max(x1, bx1);
		iy1 = std::max(y1, by1);
		ix2 = std::min(x2, bx2);
		iy2 = std::min(y2, by2);
		Dtype ovArea = 0;	
		if ((iy1 > iy2) || (ix1 > ix2)){
			ovArea = 0;
		}
		else{
			ovArea = (iy2 - iy1) * (ix2 - ix1);
		}
			
		int pad_w = 0;
		int pad_h = 0;
		int clipped_height = y2-y1+1;
		int clipped_width = x2-x1+1;

		// scale factors that would be used to warp the unclipped
		// expanded region
		Dtype scale_x =
				static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
		Dtype scale_y =
				static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

		// size to warp the clipped expanded region to
		cv_crop_size.width =
				static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
		cv_crop_size.height =
				static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
		pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
		pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
		pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
		pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

		pad_h = pad_y1;
		// if we're mirroring, we mirror the padding too (to be pedantic)
		if (do_mirror) {
			pad_w = pad_x2;
		} else {
			pad_w = pad_x1;
		}

		// ensure that the warped, clipped region plus the padding fits in the
		// crop_size x crop_size image (it might not due to rounding)
		if (pad_h + cv_crop_size.height > crop_size) {
			cv_crop_size.height = crop_size - pad_h;
		}
		if (pad_w + cv_crop_size.width > crop_size) {
			cv_crop_size.width = crop_size - pad_w;
		}

		cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
		cv::Mat cv_cropped_img = cv_img(roi);
		cv::resize(cv_cropped_img, cv_cropped_img,
				cv_crop_size, 0, 0, cv::INTER_LINEAR);

		// horizontal flip at random
		if (do_mirror) {
			cv::flip(cv_cropped_img, cv_cropped_img, 1);
		}

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
						top_data[top_index] = (pixel - mean[mean_index]);
					} else {
						if (this->has_mean_values_) {
							top_data[top_index] = (pixel - this->mean_values_[c]);
						} else {
							top_data[top_index] = pixel;
						}
					}
				}
			}
		}
		trans_time += timer.MicroSeconds();
		// get window label

		#if 0
		// useful debugging code for dumping transformed windows to disk
		string file_id;
		std::stringstream ss;
		ss << PrefetchRand();
		ss >> file_id;
		std::ofstream inf((string("dump/") + file_id +
				string("_info.txt")).c_str(), std::ofstream::out);
		inf << image.first << std::endl
				<< window[SquareBoxDataLayer<Dtype>::X1]+1 << std::endl
				<< window[SquareBoxDataLayer<Dtype>::Y1]+1 << std::endl
				<< window[SquareBoxDataLayer<Dtype>::X2]+1 << std::endl
				<< window[SquareBoxDataLayer<Dtype>::Y2]+1 << std::endl
				<< do_mirror << std::endl
				<< top_label[item_id] << std::endl
				<< is_fg << std::endl;
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
	}
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(SquareBoxDataLayer);
REGISTER_LAYER_CLASS(SquareBoxData);

}  // namespace caffe
