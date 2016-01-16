#ifndef CAFFE_SQUARE_BOX_DATA_LAYER_HPP_
#define CAFFE_SQUARE_BOX_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
template <typename Dtype>
class SquareBoxDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit SquareBoxDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~SquareBoxDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SquareBox"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, OBJ_X, OBJ_Y, BOX_CX, BOX_CY, BOX_SZ, NUM };
  vector<vector<int> > windows_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
	//Number of images
	int num_im_;
  //labels
	shared_ptr<Blob<Dtype> > labels_;
};



}  // namespace caffe

#endif  // CAFFE_SQUARE_BOX_DATA_LAYER_HPP_
