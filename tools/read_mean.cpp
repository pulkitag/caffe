#include "stdio.h"
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/io.hpp"

int main(int argc, char** argv){

	if (argc<3){
		std::cout <<"Usage: ./read_mean mean_file out_file";
		return 1;
	}

  caffe::BlobProto blob_proto;
	caffe::Blob<float> data_mean;
	const float* mean;
	caffe::ReadProtoFromBinaryFileOrDie(argv[1], &blob_proto);
	data_mean.FromProto(blob_proto);
	mean = data_mean.cpu_data();

	FILE* fid = fopen(argv[2], "w");
	for (int i=0; i < data_mean.count(); i++)
		std::fprintf(fid, "%f \n", mean[i]); 	
	fclose(fid);
	return 0;
}
