#ifndef CAFFE_GENERIC_WINDOW_DATA_LAYER_HPP_
#define CAFFE_GENERIC_WINDOW_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
//#include "caffe/layers/crop_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class CropDataLayer;

template <typename Dtype>
class GenericWindowDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit GenericWindowDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~GenericWindowDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GenericWindowData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 2; }

 protected:
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<Caffe::RNG> prefetch_rng_;
	vector<vector<Blob<Dtype>* > > crop_tops_vec_;
	//The crop_data_layers_ used to fetch the data.
  vector<shared_ptr<CropDataLayer<Dtype> > > crop_data_layers_;
  shared_ptr<Blob<Dtype> > labels_;
	//Statistics of the data to be used. 
	int num_examples_, img_group_size_, label_size_; 
	//The size of the batch. 
	int batch_size_;
	//How many elements have been read.
	int read_count_;
  //Cache images or not. 
	bool cache_images_;
	int channels_;
	int crop_size_;
};

template <typename Dtype>
class GenericWindowData2Layer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit GenericWindowData2Layer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~GenericWindowData2Layer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GenericWindowData2"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

	//virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
 
	protected:
  virtual unsigned int PrefetchRand(int streamNum);
  virtual void load_batch(Batch<Dtype>* batch);
	virtual void ReadMean();
	virtual void ReadWindowFile();
	virtual void ReadData(int streamNum, Dtype* data);

  vector<shared_ptr<Caffe::RNG> > prefetch_rng_;
	//Contains a list of images and the size of the image in channels * h * w. 
  vector<vector<std::pair<std::string, vector<int> > > > all_image_database_;
  enum WindowField {IMAGE_INDEX, X1, Y1, X2, Y2, NUM};
  vector<vector<vector<float> > > all_windows_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
	//If the images are cached then store them here. 
  vector<vector<Datum> > all_image_database_cache_;
	bool is_random_crop_;
	unsigned int rand_seed_;
	int channels_;

	shared_ptr<Blob<Dtype> > labels_;
	//Statistics of the data to be used. 
	int num_examples_, img_group_size_, label_size_; 
	//The size of the batch. 
	int batch_size_;
	//How many elements have been read.
	int read_count_;
  //Cache images or not. 
	bool cache_images_;
	int crop_size_;

	//Data of each of the streams that are concatenated together
	vector<Blob<Dtype>* > stream_data_;
};

/**
 * @brief Helper for Crop Data Layer
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class CropDataLayer : public BasePrefetchingDataLayer<Dtype> {
	friend class GenericWindowDataLayer<Dtype>;
	friend class GenericWindowData2Layer<Dtype>;
 public:
  explicit CropDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~CropDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CropData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 
	protected:
  virtual void InternalThreadEntry();
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<Caffe::RNG> prefetch_rng_;
	//Contains a list of images and the size of the image in channels * h * w. 
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField {IMAGE_INDEX, X1, Y1, X2, Y2, NUM};
  vector<vector<float> > windows_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
  bool cache_images_;
	//If the images are cached then store them here. 
  vector<Datum> image_database_cache_;
	//The number of images that have been read. 
	int read_count_;
	int fwd_count_;
	int num_examples_;
	bool is_ready_;
	bool is_random_crop_;
	unsigned int rand_seed_;
	//Useful for debugging the layer as it tells what is the layer number. 
	int layer_num_;
	int channels_;
	//Maximum Jitter while loading images
	int max_jitter_;
};



}  // namespace caffe

#endif  // CAFFE_GENERIC_WINDOW_DATA_LAYER_HPP_
