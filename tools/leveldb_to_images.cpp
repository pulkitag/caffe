#include <stdint.h>
#include "leveldb/db.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>	

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
 	caffe::DataTransformer<float> data_transformer(transform_param);

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

  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    //std::cout << it->key().ToString() << ": "  << it->value().ToString() << std::endl;
		datum.ParseFromString(it->value().ToString());
		data_transformer.Transform(0, datum, mean, data);
	
		std::cout << "Writing File \n";	
		char buffer[50];
		int len = sprintf(buffer, "tmp/count%08d.txt", count);
		std::string file_name(buffer, 0 , len);
		write_image(file_name, data, nr * nc * nd);

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

	std::ofstream fid;
	fid.open(file_name.c_str(), std::ofstream::out);
	for(int i=0; i<count; i++)
		fid << data[i] << "\n"; 
	
	fid.close();
}
