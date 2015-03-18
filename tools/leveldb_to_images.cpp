#include <stdint.h>
#include "leveldb/db.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>	
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>


#include "caffe/data_transformer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"


	

void write_image(std::string file_name, float* data, int count);

int main(int argc, char** argv){

	if (argc < 2){
		std::cout << "Usage: ./leveldb_to_images.bin dbDirectoryName numExamples(optional)" <<std::endl;
		return 0;
	}
	int maxCount,count,countFlag;
	countFlag = 0;
	count = 0;
	if (argc >= 3){
		countFlag = 1;
		maxCount     = atoi(argv[2]);
	}

	std::string dbName(argv[1]);
	leveldb::DB* db_temp;
  leveldb::Options options;

	caffe::Datum datum;
	
  options.create_if_missing = false;
  std::cout << "Opening leveldb " << dbName <<std::endl;
  leveldb::Status status = leveldb::DB::Open(
      options, dbName, &db_temp);
	if (!status.ok()){
  	std::cout<< "Failed to open leveldb "
      << dbName << std::endl << status.ToString();
			return 1;
	}
  //db_.reset(db_temp);
  //iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  //iter_->SeekToFirst();

	//Transformation parameters
  caffe::TransformationParameter transform_param;
 	caffe::DataTransformer<float> data_transformer(transform_param, caffe::TEST);

  int nr,nc,nd,label; 
  leveldb::Iterator* it = db_temp->NewIterator(leveldb::ReadOptions());
	it->SeekToFirst();
	datum.ParseFromString(it->value().ToString());
	nr = datum.height();
	nc = datum.width();
	nd = datum.channels();

	//Mean of the data
	float* mean = new float[nr*nc*nd];
	float* data = new float[nr*nc*nd];
	for (int i=0; i<nr*nc*nd; i++){
		mean[i] = 0;
		data[i] = 0;
	}

	std::cout << "Starting Iteration \n";
	int imSz = nr * nc * nd;

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    //std::cout << it->key().ToString() << ": "  << it->value().ToString() << std::endl;
		datum.ParseFromString(it->value().ToString());
		const std::string& datString = datum.data();	
		//data_transformer.Transform(datum, data);
		for (int k=0; k<imSz; k++){
			data[k] = static_cast<float>(static_cast<uint8_t>(datString[k]));
		}	

		std::cout << "Writing File \n";	
		char buffer[50];
		int len = sprintf(buffer, "tmp/count%08d.txt", count);
		std::string file_name(buffer, 0 , len);
		write_image(file_name, data, imSz);

		label = datum.label();
		count = count + 1;
		if (countFlag){
			if (count >=maxCount){
				break;
			}
		}
	}
	assert(it->status().ok());  // Check for any errors found during the scan
	delete it;
	delete mean;
	delete data;
}

void write_image(std::string file_name, float* data, int count){

	FILE* fid;
	fid = std::fopen(file_name.c_str(), "w");
	for(int i=0; i<count; i++)
		std::fprintf(fid, "%f \n", data[i]); 
	
	std::fclose(fid);
}
