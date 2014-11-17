// Copyright Ross Girshick and Yangqing Jia 2013
//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#include "mex.h"
#include "caffe/caffe.hpp"
#include <iostream>
#include <string>
#include <vector>

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace caffe;

// The pointer to the internal caffe::Net instance
static shared_ptr<Net<float> > net_;

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order 
//   [batch_images, channels, height, width] 
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array. 

static void find_layer_idx(vector<string> names, int* nameIdx){
  const vector<string>& layer_names = net_->layer_names();
  for (int i=0;i<(int)names.size();i++){
     nameIdx[i] = -1;
     for (int j=0;j<(int)layer_names.size();j++){
	 if (layer_names[j].compare(names[i])==0)
		nameIdx[i] = j;
     }
  }  
}

static mxArray* do_forward(const mxArray* const bottom) {
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(static_cast<unsigned int>(mxGetDimensions(bottom)[0]), 
      input_blobs.size());
  std::cout<< "Sizes\t"<<input_blobs.size()<<"\t"<<mxGetDimensions(bottom)[0]<<std::endl;
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
	 if (mxGetClassID(elem)!=mxSINGLE_CLASS)
		mexErrMsgTxt("forward needs inputs of type single"); 
    const float* const data_ptr = 
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(input_blobs[i]->mutable_cpu_data(), data_ptr,
          sizeof(float) * input_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(input_blobs[i]->mutable_gpu_data(), data_ptr,
          sizeof(float) * input_blobs[i]->count(), cudaMemcpyHostToDevice);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    mxArray* mx_blob = mxCreateNumericMatrix(output_blobs[i]->count(), 
        1, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(data_ptr, output_blobs[i]->cpu_data(),
          sizeof(float) * output_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(data_ptr, output_blobs[i]->gpu_data(),
          sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }

  return mx_out;
}

//Set the labels
void do_set_labels(const mxArray* const bottom) {
	//Set labels for the data - useful when doing back-prop with soft-max with loss layer.
  vector<Blob<float>*>& input_blobs = net_->input_blobs();
  CHECK_EQ(input_blobs.size(),2);
  std::cout<< "Sizes\t"<<input_blobs.size()<<"\t"<<mxGetDimensions(bottom)[0]<<std::endl;
	const mxArray* const elem = mxGetCell(bottom, 0);
	const float* const data_ptr = 
			reinterpret_cast<const float* const>(mxGetPr(elem));
	switch (Caffe::mode()) {
	case Caffe::CPU:
		memcpy(input_blobs[1]->mutable_cpu_data(), data_ptr,
				sizeof(float) * input_blobs[1]->count());
		break;
	case Caffe::GPU:
		cudaMemcpy(input_blobs[1]->mutable_gpu_data(), data_ptr,
				sizeof(float) * input_blobs[1]->count(), cudaMemcpyHostToDevice);
		break;
	default:
		LOG(FATAL) << "Unknown Caffe mode.";
	}  // switch (Caffe::mode())
}

// Get Layer Data
static mxArray* do_get_data(){
  const vector<shared_ptr<Blob<float> > >& output_blobs = net_->blobs();
  mxArray* mx_out = mxCreateCellMatrix(output_blobs.size(), 1);
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    mxArray* mx_blob = mxCreateNumericMatrix(output_blobs[i]->count(), 
        1, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(data_ptr, output_blobs[i]->cpu_data(),
          sizeof(float) * output_blobs[i]->count());
      break;
    case Caffe::GPU:
      cudaMemcpy(data_ptr, output_blobs[i]->gpu_data(),
          sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode.";
    }  // switch (Caffe::mode())
  }
  return mx_out;
}

static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {layer_blobs.size(), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells); 
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};
        mxArray* mx_weights = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

//        mexPrintf("layer: %s (%d) blob: %d  %d: (%d, %d, %d) %d\n", 
//            layer_names[i].c_str(), i, j, layer_blobs[j]->num(), 
//            layer_blobs[j]->height(), layer_blobs[j]->width(), 
//            layer_blobs[j]->channels(), layer_blobs[j]->count());

        switch (Caffe::mode()) {
        case Caffe::CPU:
          memcpy(weights_ptr, layer_blobs[j]->cpu_data(), 
              sizeof(float) * layer_blobs[j]->count());
          break;
        case Caffe::GPU:
          CUDA_CHECK(cudaMemcpy(weights_ptr, layer_blobs[j]->gpu_data(),
              sizeof(float) * layer_blobs[j]->count(), cudaMemcpyDeviceToHost));
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
      }
    }
  }

  return mx_layers;
}


void set_weights(MEX_ARGS) {
	/*
		Set the weights of a desired layer. 
	*/

	const mxArray *namesCell = prhs[0]; 
  const float* weights  = (float *) mxGetData(prhs[1]);
  const int* blobNum     = (int * ) mxGetData(prhs[2]);
	int* dataDims         = (int *) mxGetDimensions(prhs[1]);
  const int numDataDims = (int) mxGetNumberOfDimensions(prhs[1]);

	unsigned int numNames = mxGetNumberOfElements(namesCell);
	if (numNames==0){
		mexPrintf("No names specified");
		return;
	}

	//Find the layer-idx of the layers which correspond to given names.
	vector<string> names;
  string s;
  int idx;
	for (unsigned int i=0;i<numNames;i++){
	 const mxArray* tmp = mxGetCell(namesCell,i);
   s                  = mxArrayToString(tmp);
	 names.push_back(s);
	}
	int* nameIdx = new int[names.size()];
	find_layer_idx(names,nameIdx);

  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();


  // Step 3: copy weights into output
	for (unsigned int i = 0; i < numNames; ++i) {
		idx = nameIdx[i];
		if (idx==-1){
			mexErrMsgTxt("Layer Name not found");
		}
		vector<shared_ptr<Blob<float> > >& layer_blobs = layers[idx]->blobs();
		if (layer_blobs.size() == 0) {
			mexErrMsgTxt("THis layer has no associated weights");
		}

		mexPrintf("Number of blobs in a layer is %d \n",layer_blobs.size());
		int j = blobNum[0];
		if (j >= layer_blobs.size())
			mexPrintf("Number of blobs is greater than all present blobs");
  	//for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
			// internally data is stored as (width, height, channels, num)
			// where width is the fastest dimension
			mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
					layer_blobs[j]->channels(), layer_blobs[j]->num()};

			mexPrintf("Pos1, Layer Dims: %d, %d, %d, %d \n",dims[0],dims[1],dims[2],dims[3]);
			
			bool isEqual = true;
			for (int k=0;k<4;k++)
				isEqual &= dims[k]==dataDims[k];

			mexPrintf("Pos2, Layer Dims: %d, %d, %d, %d \n",dims[0],dims[1],dims[2],dims[3]);
			//get around weird matlab issue of not producing singleton dims.
			bool isEqual2 = true;
			int countDims  = 1;
			int countDims2 = 1;
			for (int k=0;k<numDataDims;k++){
				countDims  *= dataDims[k];
				countDims2 *= dims[k];
			}
			mexPrintf("Pos3, Layer Dims: %d, %d, %d, %d \n",dims[0],dims[1],dims[2],dims[3]);
			for (int k=numDataDims;k<4;k++)
				isEqual2 &= dims[k]==1;

			isEqual2 &= (countDims == countDims2);
			isEqual = isEqual || isEqual2;	
			
		if (!isEqual){
				mexPrintf("Layer Dims: %d, %d, %d, %d \n",dims[0],dims[1],dims[2],dims[3]);
				mexPrintf("Data Dims: %d, %d, %d, %d \n",dataDims[0],dataDims[1],
																					dataDims[2],dataDims[3]);
				mexErrMsgTxt("Dimension Mismatch");
			}

			switch (Caffe::mode()) {
			case Caffe::CPU:
				memcpy(layer_blobs[j]->mutable_cpu_data(), weights,
						sizeof(float) * layer_blobs[j]->count());
				break;
			case Caffe::GPU:
				CUDA_CHECK(cudaMemcpy(layer_blobs[j]->mutable_gpu_data(),weights,
						sizeof(float) * layer_blobs[j]->count(), cudaMemcpyHostToDevice));
				break;
			default:
				LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
			}
		}
	//}
}



static void get_weights(MEX_ARGS) {
  plhs[0] = do_get_weights();
}

// The caffe::Caffe utility functions.
static void set_mode_cpu(MEX_ARGS) { 
  Caffe::set_mode(Caffe::CPU); 
}

static void set_mode_gpu(MEX_ARGS) { 
  Caffe::set_mode(Caffe::GPU); 
}

static void set_phase_train(MEX_ARGS) { 
  Caffe::set_phase(Caffe::TRAIN); 
}

static void set_phase_test(MEX_ARGS) { 
  Caffe::set_phase(Caffe::TEST); 
}

static void set_device(MEX_ARGS) { 
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id); 
}

static void init(MEX_ARGS) {
  if (nrhs != 2) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);

  net_.reset(new Net<float>(string(param_file)));
  net_->CopyTrainedLayersFrom(string(model_file));

  mxFree(param_file);
  mxFree(model_file);
}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    LOG(ERROR) << "Only given " << nrhs << " arguments";
    mexErrMsgTxt("Wrong number of arguments");
  }

 
  plhs[0] = do_forward(prhs[0]);
}

static void set_labels(MEX_ARGS) {
  do_set_labels(prhs[0]);
}

static void disable_dropout_backprop(MEX_ARGS) {
  /*
	Need to disable dropout backprop when interested in extracting class conditional gradients.
	*/
	const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
	const vector<string>& layer_names = net_->layer_names();
	int numLayers = layers.size();
	vector<bool>& layer_need_backward_ = net_->layer_need_backward();
	string s = "dropout";
	for (int i=0; i<numLayers; i++){
		if(s.compare((layers[i]->layer_param()).type())==0){
			layer_need_backward_[i] = false;
			LOG(INFO) << "Disabling Backward computation for "<<layer_names[i];
		}
	}
}

static void backward(MEX_ARGS) {
	float loss = net_->Backward();
}


static void get_data(MEX_ARGS) {

  plhs[0] = do_get_data();
}

static void get_blob_names(MEX_ARGS){
  const vector<string>& blobNames = net_->blob_names();
  const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
  int numBlobs = blobNames.size();
  mexPrintf("Number of Blobs is %d \n",numBlobs);
  string name;

  for (int i=0; i<numBlobs; i++){
      name = blobNames.at(i);
      //iblob = blobs.at(i);
      //mexPrintf("%s \n", name);
      std::cout << name <<"\t" <<blobs[i]->count()<<"\t"<<blobs[i]->num()<<"\t" 
	<< blobs[i]->channels()<<"\t"<<blobs[i]->height()<<std::endl;
  }
}




static void get_features(MEX_ARGS){
	const mxArray *namesCell = prhs[0]; 
	mwSize numNames = mxGetNumberOfElements(namesCell);
	if (numNames==0){
	mexPrintf("No names specified");
	return;
	}
	vector<string> names;
  string s;
	for (int i=0;i<numNames;i++){
	 const mxArray* tmp = mxGetCell(namesCell,i);
   s                  = mxArrayToString(tmp);
	 names.push_back(s);
	}
			 
	int* nameIdx = new int[names.size()];
	find_layer_idx(names,nameIdx);
	/*for (int i=0;i<(int)names.size();i++){
	 std::cout<<nameIdx[i]<<std::endl;
	}*/

  const vector<vector<Blob<float>*> >& top_vecs = net_->top_vecs();
  
  const mwSize dims[2]   = {numNames,1};
	const char*  fnames[2] = {"feat","name"};
  mxArray* mx_feat       = mxCreateStructArray(2,dims,2,fnames);
  int idx;

  for (int i=0;i<numNames;i++){
    idx = nameIdx[i];
		if (idx==-1){
			mexPrintf("Layer not found %s",names[i].c_str());
			continue;
		}
		mxArray* mx_layer_cells = NULL;
		const mwSize dims[2]    = {top_vecs[idx].size(), 1};
    mx_layer_cells          = mxCreateCellArray(2,dims);
    mxSetField(mx_feat,i,"feat",mx_layer_cells);
	  mxSetField(mx_feat,i,"name",mxCreateString(names[i].c_str()));
    for (unsigned int j=0;j<top_vecs[idx].size();j++){
			mwSize dims[4] = {top_vecs[idx][j]->width(), top_vecs[idx][j]->height(),
												top_vecs[idx][j]->channels(),top_vecs[idx][j]->num()};
			mxArray* mx_features = mxCreateNumericArray(4,dims,mxSINGLE_CLASS,mxREAL);
      mxSetCell(mx_layer_cells,j,mx_features);
			float* features_ptr = reinterpret_cast<float*>(mxGetPr(mx_features));
			switch (Caffe::mode()){
				case Caffe::CPU:
					memcpy(features_ptr, top_vecs[idx][j]->cpu_data(), sizeof(float) * top_vecs[idx][j]->count());
				  break;
				case Caffe::GPU:
					CUDA_CHECK(cudaMemcpy(features_ptr, top_vecs[idx][j]->gpu_data(),
										 sizeof(float) * top_vecs[idx][j]->count(),cudaMemcpyDeviceToHost));
					break;
				default:
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}	
	 }
 }
  	
  plhs[0] = mx_feat; 
	delete nameIdx;
}



static void get_gradients(MEX_ARGS){
	const mxArray *namesCell = prhs[0]; 
	mwSize numNames = mxGetNumberOfElements(namesCell);
	if (numNames==0){
	mexPrintf("No names specified");
	return;
	}
	vector<string> names;
  string s;
	for (int i=0;i<numNames;i++){
	 const mxArray* tmp = mxGetCell(namesCell,i);
   s                  = mxArrayToString(tmp);
	 names.push_back(s);
	}
			 
	int* nameIdx = new int[names.size()];
	find_layer_idx(names,nameIdx);
	/*for (int i=0;i<(int)names.size();i++){
	 std::cout<<nameIdx[i]<<std::endl;
	}*/

  const vector<vector<Blob<float>*> >& top_vecs = net_->top_vecs();
  
  const mwSize dims[2]   = {numNames,1};
	const char*  fnames[2] = {"feat","name"};
  mxArray* mx_feat       = mxCreateStructArray(2,dims,2,fnames);
  int idx;

  for (int i=0;i<numNames;i++){
    idx = nameIdx[i];
		if (idx==-1){
			mexPrintf("Layer not found %s",names[i].c_str());
			continue;
		}
		mxArray* mx_layer_cells = NULL;
		const mwSize dims[2]    = {top_vecs[idx].size(), 1};
    mx_layer_cells          = mxCreateCellArray(2,dims);
    mxSetField(mx_feat,i,"feat",mx_layer_cells);
	  mxSetField(mx_feat,i,"name",mxCreateString(names[i].c_str()));
    for (unsigned int j=0;j<top_vecs[idx].size();j++){
			mwSize dims[4] = {top_vecs[idx][j]->width(), top_vecs[idx][j]->height(),
												top_vecs[idx][j]->channels(),top_vecs[idx][j]->num()};
			mxArray* mx_features = mxCreateNumericArray(4,dims,mxSINGLE_CLASS,mxREAL);
      mxSetCell(mx_layer_cells,j,mx_features);
			float* features_ptr = reinterpret_cast<float*>(mxGetPr(mx_features));
			switch (Caffe::mode()){
				case Caffe::CPU:
					memcpy(features_ptr, top_vecs[idx][j]->cpu_diff(), sizeof(float) * top_vecs[idx][j]->count());
				  break;
				case Caffe::GPU:
					CUDA_CHECK(cudaMemcpy(features_ptr, top_vecs[idx][j]->gpu_diff(),
										 sizeof(float) * top_vecs[idx][j]->count(),cudaMemcpyDeviceToHost));
					break;
				default:
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}	
	 }
 }
  	
  plhs[0] = mx_feat; 
	delete nameIdx;
}

static void get_gradients_bottom(MEX_ARGS){
	const mxArray *namesCell = prhs[0]; 
	mwSize numNames = mxGetNumberOfElements(namesCell);
	if (numNames==0){
	mexPrintf("No names specified");
	return;
	}
	vector<string> names;
  string s;
	for (int i=0;i<numNames;i++){
	 const mxArray* tmp = mxGetCell(namesCell,i);
   s                  = mxArrayToString(tmp);
	 names.push_back(s);
	}
			 
	int* nameIdx = new int[names.size()];
	find_layer_idx(names,nameIdx);
	/*for (int i=0;i<(int)names.size();i++){
	 std::cout<<nameIdx[i]<<std::endl;
	}*/

  const vector<vector<Blob<float>*> >& top_vecs = net_->bottom_vecs();
  
  const mwSize dims[2]   = {numNames,1};
	const char*  fnames[2] = {"feat","name"};
  mxArray* mx_feat       = mxCreateStructArray(2,dims,2,fnames);
  int idx;

  for (int i=0;i<numNames;i++){
    idx = nameIdx[i];
		if (idx==-1){
			mexPrintf("Layer not found %s",names[i].c_str());
			continue;
		}
		mxArray* mx_layer_cells = NULL;
		const mwSize dims[2]    = {top_vecs[idx].size(), 1};
    mx_layer_cells          = mxCreateCellArray(2,dims);
    mxSetField(mx_feat,i,"feat",mx_layer_cells);
	  mxSetField(mx_feat,i,"name",mxCreateString(names[i].c_str()));
    for (unsigned int j=0;j<top_vecs[idx].size();j++){
			mwSize dims[4] = {top_vecs[idx][j]->width(), top_vecs[idx][j]->height(),
												top_vecs[idx][j]->channels(),top_vecs[idx][j]->num()};
			mxArray* mx_features = mxCreateNumericArray(4,dims,mxSINGLE_CLASS,mxREAL);
      mxSetCell(mx_layer_cells,j,mx_features);
			float* features_ptr = reinterpret_cast<float*>(mxGetPr(mx_features));
			switch (Caffe::mode()){
				case Caffe::CPU:
					memcpy(features_ptr, top_vecs[idx][j]->cpu_diff(), sizeof(float) * top_vecs[idx][j]->count());
				  break;
				case Caffe::GPU:
					CUDA_CHECK(cudaMemcpy(features_ptr, top_vecs[idx][j]->gpu_diff(),
										 sizeof(float) * top_vecs[idx][j]->count(),cudaMemcpyDeviceToHost));
					break;
				default:
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}	
	 }
 }
  	
  plhs[0] = mx_feat; 
	delete nameIdx;
}

static void set_top_diff(MEX_ARGS){
	//Set the top different to the desired value and then compute the gradient with respect to it.

  if (mxGetClassID(prhs[1])!=mxSINGLE_CLASS)
		mexErrMsgTxt("Input to top_diff should be of type single");

	const mxArray *namesCell = prhs[0]; 
  const float* topData = (float *) mxGetData(prhs[1]);
	int* dataDims        = (int *) mxGetDimensions(prhs[1]);

	mwSize numNames = mxGetNumberOfElements(namesCell);
	if (numNames==0){
		mexPrintf("No names specified");
		return;
	}
	if (numNames > 1){
		mexErrMsgTxt("You should not set diff values for more than 1 layers");
	}

	vector<string> names;
  string s;
  int idx;
	for (int i=0;i<numNames;i++){
	 const mxArray* tmp = mxGetCell(namesCell,i);
   s                  = mxArrayToString(tmp);
	 names.push_back(s);
	}
	int* nameIdx = new int[names.size()];
	find_layer_idx(names,nameIdx);

  const vector<vector<Blob<float>*> >& top_vecs = net_->top_vecs();

  for (int i=0;i<numNames;i++){
    idx = nameIdx[i];
		if (idx==-1){
			mexPrintf("Layer not found %s",names[i].c_str());
			continue;
		}
    for (unsigned int j=0;j<top_vecs[idx].size();j++){
	
			int dims[4] = {top_vecs[idx][j]->width(), top_vecs[idx][j]->height(),
												top_vecs[idx][j]->channels(),top_vecs[idx][j]->num()};
			bool isEqual = true;
			LOG(INFO)<<isEqual;
			for (int k=0;k<4;k++){
				isEqual &= dims[k]==dataDims[k];
				LOG(INFO)<<dims[k]<<" "<<dataDims[k]<<" "<<isEqual;
			}		

			if (!isEqual)
				mexErrMsgTxt("Dimension Mismatch");

			switch (Caffe::mode()){
				case Caffe::CPU:
					memcpy(top_vecs[idx][j]->mutable_cpu_diff(),topData, sizeof(float) * top_vecs[idx][j]->count());
				  break;
				case Caffe::GPU:
					CUDA_CHECK(cudaMemcpy(top_vecs[idx][j]->mutable_gpu_diff(),topData,
										 sizeof(float) * top_vecs[idx][j]->count(),cudaMemcpyHostToDevice));
					break;
				default:
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}	
	 }
 }
  	
	delete nameIdx;
}

static void set_top_data(MEX_ARGS){
	//Set the top data to the desired value and then compute the gradient with respect to it.
	
  if (mxGetClassID(prhs[1])!=mxSINGLE_CLASS)
		mexErrMsgTxt("Inputs to set_top_data should be of type single");

	const mxArray *namesCell = prhs[0]; 
    const float* topData = (float *) mxGetData(prhs[1]);
	int* dataDims        = (int *) mxGetDimensions(prhs[1]);

	mwSize numNames = mxGetNumberOfElements(namesCell);
	if (numNames==0){
		mexPrintf("No names specified");
		return;
	}
	if (numNames > 1){
		mexErrMsgTxt("You should not set data values for more than 1 layers");
	}

	vector<string> names;
  string s;
  int idx;
	for (int i=0;i<numNames;i++){
	 const mxArray* tmp = mxGetCell(namesCell,i);
   s                  = mxArrayToString(tmp);
	 names.push_back(s);
	}
	int* nameIdx = new int[names.size()];
	find_layer_idx(names,nameIdx);

  const vector<vector<Blob<float>*> >& top_vecs = net_->top_vecs();

  for (int i=0;i<numNames;i++){
    idx = nameIdx[i];
		if (idx==-1){
			mexPrintf("Layer not found %s",names[i].c_str());
			continue;
		}
    for (unsigned int j=0;j<top_vecs[idx].size();j++){
	
			int dims[4] = {top_vecs[idx][j]->width(), top_vecs[idx][j]->height(),
												top_vecs[idx][j]->channels(),top_vecs[idx][j]->num()};
			bool isEqual = true;
			for (int k=0;k<4;k++){
				isEqual &= dims[k]==dataDims[k];
				LOG(INFO)<<dims[k]<<" "<<dataDims[k]<<" "<<isEqual;
			}		

			if (!isEqual)
				mexErrMsgTxt("Dimension Mismatch");

			switch (Caffe::mode()){
				case Caffe::CPU:
					memcpy(top_vecs[idx][j]->mutable_cpu_data(),topData, sizeof(float) * top_vecs[idx][j]->count());
				  break;
				case Caffe::GPU:
					CUDA_CHECK(cudaMemcpy(top_vecs[idx][j]->mutable_gpu_data(),topData,
										 sizeof(float) * top_vecs[idx][j]->count(),cudaMemcpyHostToDevice));
					break;
				default:
					LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
		}	
	 }
 }
  	
	delete nameIdx;
}




static void get_layer_info(MEX_ARGS){
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();
  const vector<vector<Blob<float>*> >& top_vecs = net_->top_vecs();
  const vector<vector<Blob<float>*> >& bottom_vecs = net_->bottom_vecs();
  std::cout<<"Layers: "<<layers.size()<<" top_vecs "<<top_vecs.size()<<" bottom_vecs "<< 
  bottom_vecs.size()<< std::endl;
  int numLayers     = layers.size();
  
  //mxArray* mx_blob = mxCreateNumericMatrix(output_blobs[i]->count(), 

  //Find number of layers 
  for (int i=0;i<numLayers;i++){
	std::cout<<"Layer "<<layer_names[i]<<" top_vec_count "<<top_vecs[i][0]->count()<<"\t"<<top_vecs[i][0]->width()<<"\t"<<top_vecs[i][0]->height()<<"\t"<<
	top_vecs[i][0]->channels()<<"\t"<<top_vecs[i][0]->num()<<std::endl;

	if ((layers[i]->blobs()).size() !=0){
		std::cout<<(layers[i]->layer_param()).name()<<"\t"<<layer_names[i]<<"\t"<<
		(layers[i]->blobs()).size()<<std::endl;
        }
  }   

}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "forward",            forward         },
  { "backward",           backward        },
  { "init",               init            },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_phase_train",    set_phase_train },
  { "set_phase_test",     set_phase_test  },
  { "set_device",         set_device      },
  { "get_blob_names",     get_blob_names  },
  { "get_data",           get_data        },
  { "get_layer_info",     get_layer_info  },
  { "get_features",       get_features    },
  { "get_gradients",      get_gradients  },
  { "get_gradients_bottom",      get_gradients_bottom  },
  { "disable_dropout_backprop",  disable_dropout_backprop},
  { "set_top_diff",  			set_top_diff},
  { "set_top_data",  			set_top_data},
  { "set_labels",  			  set_labels},
  { "set_weights",			  set_weights},
  { "get_weights",			  get_weights},
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  if (nrhs == 0) {
    LOG(ERROR) << "No API command given";
    mexErrMsgTxt("An API command is requires");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      LOG(ERROR) << "Unknown command `" << cmd << "'";
      mexErrMsgTxt("API command not recognized");
    }
    mxFree(cmd);
  }
}
