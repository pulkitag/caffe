#include "stdio.h"
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/io.hpp"

/*
This is useful when siamese net has been trained and the mean_files needs to be split. 
*/

int main(int argc, char** argv){

	if (argc<3){
		std::cout <<"Usage: ./split_mean mean_file out_file";
		return 1;
	}

  caffe::BlobProto blob_proto;
	caffe::Blob<float> data_mean;
	const float* mean;
	caffe::ReadProtoFromBinaryFileOrDie(argv[1], &blob_proto);
	data_mean.FromProto(blob_proto);
	mean = data_mean.cpu_data();

	int num = data_mean.num();
	int nc = data_mean.channels();
	int h  = data_mean.height();
	int w  = data_mean.width();
	assert(nc % 2 ==0);
	assert(num == 1);

	nc = nc / 2;
	caffe::BlobProto out_mean;
	out_mean.set_num(num);
	out_mean.set_channels(nc);
	out_mean.set_height(h);
	out_mean.set_width(w);
	
	for (int i=0; i < nc * h * w ; i++)
		out_mean.add_data(mean[i]);

  LOG(INFO) << "Write to " << argv[2];
  WriteProtoToBinaryFile(out_mean, argv[2]);
	return 0;
}
