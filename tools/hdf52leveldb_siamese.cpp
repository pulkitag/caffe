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
						 H5::DataSpace memspace, unsigned char* data_out, 
						unsigned long offset, unsigned long nx );


int main(int argc, char** argv){

	if (argc < 3){
		std::cout << "Usage: ./convert_dataset_rotation HDF5_FILE_NAME LEVELDB_NAME \n";
		return 1;
	}

	int rows, cols;
	rows = 28;
	cols = 28;
	//std::string dataPath = "/work4/pulkitag/data_sets/digits/";
	//std::string filePath = dataPath + "mnist_train.hdf5";
	std::string filePath(argv[1]);
	std::cout << filePath << "\n";
	const H5std_string fileName(filePath);
	const H5std_string dataIm1("images1");
	const H5std_string dataIm2("images2");
	const H5std_string dataLbl("labels");

	//Load the daasets
	H5::H5File file(fileName, H5F_ACC_RDONLY);
	H5::DataSet im1 = file.openDataSet(dataIm1);
	H5::DataSet im2 = file.openDataSet(dataIm2);
	H5::DataSet lbl = file.openDataSet(dataLbl);

	//Check Type
	hid_t dt = H5Tcopy(H5T_STD_U8LE);
	H5T_class_t type_class = im1.getTypeClass();	
	assert(type_class == H5T_NATIVE_UCHAR);
	type_class = im2.getTypeClass();	
	assert(type_class == H5T_Integer);
	type_class = lbl.getTypeClass();	
	assert(type_class == H5T_Integer);

	//Get dimensions
	int ndims;
	unsigned long N,imsz;
	H5::DataSpace dataspace1    = im1.getSpace();
	H5::DataSpace dataspace2    = im2.getSpace();
	H5::DataSpace dataspaceLbl  = lbl.getSpace();
	hsize_t dims_out[1];
	ndims = dataspace1.getSimpleExtentDims( dims_out, NULL);
	imsz = (unsigned long)dims_out[0];
	ndims = dataspace2.getSimpleExtentDims( dims_out, NULL);
	assert(imsz==(unsigned long)dims_out[0]);
	ndims = dataspaceLbl.getSimpleExtentDims( dims_out, NULL);
	N     = (unsigned long)dims_out[0];
	assert(imsz == rows * cols * N);
	std::cout << "Num Images: " << N << " \n";


	//Define memspaces
	hsize_t memDims[1];
	memDims[0] = imsz;
	H5::DataSpace memspace1(1, memDims);
	H5::DataSpace memspace2(1, memDims);
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
	int Nr = rows * cols;
  unsigned char* pixels = new unsigned char[2 * Nr];
	unsigned char labels;
  int count = 0;
	unsigned long num_items = N;
  const int kMaxKeyLength = 10;
  char key_cstr[kMaxKeyLength];
  std::string value;

  caffe::Datum datum;
  datum.set_channels(2);
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int item_id = 0; item_id < num_items; ++item_id) {
    read_data(im1, dataspace1, memspace1, pixels, item_id, Nr);
    read_data(im2, dataspace2, memspace2, pixels + Nr, item_id, Nr);
    read_data(lbl, dataspaceLbl, memspaceLbl, &labels, item_id, 1);
		datum.set_data(pixels, 2 * rows * cols);
    datum.set_label(labels);
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
  delete pixels;
	return 0;
}


void read_data(H5::DataSet& dataset, H5::DataSpace dataspace,
						 H5::DataSpace memspace, unsigned char* data_out, 
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
	dataset.read( data_out, H5::PredType::NATIVE_UCHAR, memspace, dataspace );
	//dataset.read( data_out, H5T_NATIVE_UCHAR, memspace, dataspace );
}
