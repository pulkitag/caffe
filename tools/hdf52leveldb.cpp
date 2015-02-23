#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include "stdint.h"

//#include "H5Cpp.h"
#include "hdf5.h"
#include "hdf5_hl.h"
#include "h5hl_region.h"

#include "glog/logging.h"
#include "leveldb/db.h"
#include <leveldb/write_batch.h>"
#include "google/protobuf/text_format.h"

#include "caffe/proto/caffe.pb.h"
void read_data(H5::DataSet& dataset, H5::DataSpace dataspace,
						 H5::DataSpace memspace, unsigned char* data_out, 
						unsigned long offset, unsigned long nx );


int main(int argc, char** argv){
	/*
		Assumes hdf5 file has two datafields:
			1. images - containing images - flattened out in (ch, height, width) - fastest dimension is imRow
			2. labels - as many labels as the number of images
	*/

	if (argc < 3){
		std::cout << "Usage: ./hdf52leveldb HDF5_FILE_NAME LEVELDB_NAME NUM_ROWS NUM_COLS NUM_CHANNELS\n";
		return 1;
	}
	//The size of the image
	int rows, cols;
	if (argc < 5){
		rows = 28;
		cols = 28;
	}
	else{
		rows = atoi(argv[3]);
		cols = atoi(argv[4]);
	}
	//Number of channels - grayscale or color image
	int nCh = 3;
	if (argc>=6){
		nCh = atoi(argv[5]);
	}

	//Open the File
	std::string filePath(argv[1]);
	std::cout << "Reading from: " <<  filePath << "\n";
	hFid = H5Fopen(filePath, H5F_ACC_RDONLY, H5P_DEFAULT);
	hsize_t dims[1];
	herr_t  status;
	int ndims;
	unsigned long N,imsz;

	//Get the Size
	status = H5LTget_dataset_info(file_id,"/images",dims, NULL, NULL);
	imsz = (unsigned long)dims[0];
	status = H5LTget_dataset_info(file_id,"/labels",dims, NULL, NULL);
	N     = (unsigned long)dims[0];
	assert(imsz == rows * cols * N);
	std::cout << "Num Images: " << N << " \n";
	H5Fclose(hFid);

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
  int count = 0;
	int numPix = nCh * rows * cols;
  unsigned char* pixels = new unsigned char[numPix];
	float labels;
	unsigned long num_items = N;
  const int kMaxKeyLength = 10;
  char key_cstr[kMaxKeyLength];
  std::string value;

  caffe::Datum datum;
  datum.set_channels(nCh);
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
	hsize_t blockCoord[2];
  for (int item_id = 0; item_id < num_items; ++item_id) {
		//Read the image
		blockCoord[0] = item_id * numPix;
		blockCoord[1] = blockCoord[0] + numPix;
  	status = H5LTread_region(filePath, "/images",  H5T_NATIVE_UCHAR, pixels);
   	//Read the label
		blockCoord[0] = item_id;
		blockCoord[1] = blockCoord[0] + 1;
  	status = H5LTread_region(filePath, "/labels",  H5T_NATIVE_FLOAT, pixels);
		datum.set_data(pixels, numPix);
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

/*	
	const H5std_string fileName(filePath);
	const H5std_string dataIm1("images");
	const H5std_string dataLbl("labels");

	//Load the daasets
	H5::H5File file(fileName, H5F_ACC_RDONLY);
	H5::DataSet im1 = file.openDataSet(dataIm1);
	H5::DataSet lbl = file.openDataSet(dataLbl);

	//Check Type
	H5T_class_t type_class = im1.getTypeClass();	
	assert(type_class == H5T_NATIVE_UCHAR);
	type_class = lbl.getTypeClass();	
	assert(type_class == H5T_NATIVE_UCHAR);

	//Get dimensions
	int ndims;
	unsigned long N,imsz;
	H5::DataSpace dataspace1    = im1.getSpace();
	H5::DataSpace dataspaceLbl  = lbl.getSpace();
	hsize_t dims_out[1];
	ndims = dataspace1.getSimpleExtentDims( dims_out, NULL);
	imsz = (unsigned long)dims_out[0];
	ndims = dataspaceLbl.getSimpleExtentDims( dims_out, NULL);
	N     = (unsigned long)dims_out[0];
	assert(imsz == rows * cols * N);
	std::cout << "Num Images: " << N << " \n";

	//Define memspaces
	hsize_t memDims[1];
	memDims[0] = imsz;
	H5::DataSpace memspace1(1, memDims);
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
  unsigned char* pixels = new unsigned char[Nr];
	unsigned char labels;
  int count = 0;
	unsigned long num_items = N;
  const int kMaxKeyLength = 10;
  char key_cstr[kMaxKeyLength];
  std::string value;

  caffe::Datum datum;
  datum.set_channels(1);
  datum.set_height(rows);
  datum.set_width(cols);
  LOG(INFO) << "A total of " << num_items << " items.";
  LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
  for (int item_id = 0; item_id < num_items; ++item_id) {
    read_data(im1, dataspace1, memspace1, pixels, item_id, Nr);
    read_data(lbl, dataspaceLbl, memspaceLbl, &labels, item_id, 1);
		datum.set_data(pixels, rows * cols);
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
*/

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
