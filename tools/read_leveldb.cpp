#include <stdint.h>
#include "leveldb/db.h"
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>	

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

int main(int argc, char** argv){

	if (argc < 2){
		std::cout << "Usage: ./read_leveldb dbDirectoryName numExamples(optional)" <<std::endl;
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

  int nr,nc,nd,label; 
  leveldb::Iterator* it = db_temp->NewIterator(leveldb::ReadOptions());
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    //std::cout << it->key().ToString() << ": "  << it->value().ToString() << std::endl;
	std::cout << "I am here \n";
	datum.ParseFromString(it->value().ToString());
	nr = datum.height();
	nc = datum.width();
	nd = datum.channels();
	label = datum.label();
    std::cout << it->key().ToString()  << "\t" << nr << "\t" << nc <<"\t" << nd << "\t" << label << std::endl;
		count = count + 1;
		if (countFlag){
			if (count >=maxCount){
				break;
			}
		}
  }
  assert(it->status().ok());  // Check for any errors found during the scan
  delete it;
	
}
