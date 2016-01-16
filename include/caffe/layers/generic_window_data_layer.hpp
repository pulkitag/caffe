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
#include "caffe/layers/crop_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

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

}  // namespace caffe

#endif  // CAFFE_GENERIC_WINDOW_DATA_LAYER_HPP_
