#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include "stdint.h"

#include "H5Cpp.h"

#include "glog/logging.h"
#include "leveldb/db.h"
#include <leveldb/write_batch.h>"
#include "google/protobuf/text_format.h"

#include "caffe/proto/caffe.pb.h"
void read_data(H5::DataSet& dataset, H5::DataSpace dataspace,
						 H5::DataSpace memspace, float* data_out, 
						unsigned long offset, unsigned long nx );


int main(int argc, char** argv){

	if (argc < 3){
		std::cout << "Usage: ./convert_dataset_rotation HDF5_FILE_NAME LEVELDB_NAME \n";
		return 1;
	}

	int labelSz = 9;
	std::string filePath(argv[1]);
	std::cout << filePath << "\n";
	const H5std_string fileName(filePath);
	const H5std_string dataLbl("labels");

	//Load the daasets
	H5::H5File file(fileName, H5F_ACC_RDONLY);
	H5::DataSet lbl = file.openDataSet(dataLbl);

	//Check Type
	H5T_class_t type_class = lbl.getTypeClass();	
	assert(type_class == H5T_NATIVE_UCHAR);

	//Get dimensions
	int ndims;
	unsigned long N;
	H5::DataSpace dataspaceLbl  = lbl.getSpace();
	hsize_t dims_out[1];
	ndims = dataspaceLbl.getSimpleExtentDims( dims_out, NULL);
	N     = (unsigned long)dims_out[0];
	
	//Define memspaces
	hsize_t memDims[1];
	memDims[0] = N;
	H5::DataSpace memspaceLbl(1, memDims);

	
	//leveldb
	//std::string db_path = dataPath + "mnist_leveldb";
	std::string db_path(argv[2]);
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch = NULL;
	LOG(INFO) << "Opening leveldb " << db_path;
    leveldb::Status status = leveldb::DB::Open(
        options, db_path.c_str(), &db);
    CHECK(status.ok()) << "Failed to open leveldb " << db_path
        << ". Is it already existing?";
  batch = new leveldb::WriteBatch();

  // Storing to db
	int Nr = labelSz;
  float* labels = new float[Nr];
  int count = 0;
	unsigned long num_items = N / labelSz;
  const int kMaxKeyLength = 10;
  char key_cstr[kMaxKeyLength];
  std::string value;

  caffe::Datum datum;
  datum.set_channels(1);
  datum.set_height(1);
  datum.set_width(labelSz);
  LOG(INFO) << "A total of " << num_items << " items.";
  for (int i=0; i<labelSz; i++)
		datum.add_float_data(0.0);
	for (int item_id = 0; item_id < num_items; ++item_id) {
    read_data(lbl, dataspaceLbl, memspaceLbl, labels, item_id, labelSz);
		for (int i=0; i< labelSz; i++){
			datum.set_float_data(i, labels[i]);
  		//std::cout << labels[i] << "\n"; 
		}
		snprintf(key_cstr, kMaxKeyLength, "%08d", item_id);
    datum.SerializeToString(&value);
    std::string keystr(key_cstr);

    // Put in db
		batch->Put(keystr, value);

    if (++count % 1000 == 0) {
      // Commit txn
			db->Write(leveldb::WriteOptions(), batch);
			delete batch;
			batch = new leveldb::WriteBatch();
		}
	}
  // write the last batch
  if (count % 1000 != 0) {
		db->Write(leveldb::WriteOptions(), batch);
		delete batch;
		delete db;
		LOG(ERROR) << "Processed " << count << " files.";
  }
  delete labels;
	return 0;
}


void read_data(H5::DataSet& dataset, H5::DataSpace dataspace,
						 H5::DataSpace memspace, float* data_out, 
						unsigned long offset, unsigned long nx ){

	//Select data in dataspace
	hsize_t    dataOffset[1];   // hyperslab offset in the file
	hsize_t    dataCount[1];    // size of the hyperslab in the file
	dataOffset[0] = nx * offset;
	dataCount[0]  = nx;
	dataspace.selectHyperslab( H5S_SELECT_SET, dataCount, dataOffset );	


	//Select data in memory
	hsize_t      offset_out[1];   // hyperslab offset in memory
	hsize_t      count_out[1];    // size of the hyperslab in memory
	offset_out[0] = 0;
	count_out[0]  = nx;
	memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out);	
	dataset.read( data_out, H5::PredType::NATIVE_FLOAT, memspace, dataspace );
	//dataset.read( data_out, H5T_NATIVE_UCHAR, memspace, dataspace );
}
