#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <opencv2/highgui/highgui_c.h>
#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/cropping_kat_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"






// inputs are an image and  cropping center x,y coordinates! and 

namespace caffe {
  
  template <typename Dtype>
  void CroppingKatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int crop_size= this->layer_param_.transform_param().crop_size();
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  // box coordinates
  CHECK_EQ(bottom[1]->channels(), 2);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
	LOG(INFO)<< bottom[0]->num() << "," << bottom[0]->channels() << "," << crop_size;
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), crop_size, crop_size);

 LOG(INFO)<<"top count:"<<top[0]->count();
}


template <typename Dtype>
void CroppingKatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 const int crop_size = this->layer_param_.transform_param().crop_size();  

    
  int batch_size=bottom[0]->num();
  int imgchannels=bottom[0]->channels();
  singleimage_.Reshape(1, 3,  bottom[0]->height(), bottom[0]->width());
  singleimage2_.Reshape(1, 3,  bottom[0]->height(), bottom[0]->width());
//LOG(INFO)<<"cropsize:"<<crop_size;
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), crop_size, crop_size);
 

 
 
}




template <typename Dtype>
void CroppingKatLayer<Dtype>::Forward_cpu(
   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  //LOG(INFO)<<"CROPPING LAYER FORWARD CPU"<<"width:"<<bottom[0]->width()<<
  //"height:"<<bottom[0]->height()<<"channels:"<<bottom[0]->channels()<<"count:"<<bottom[0]->count();
  const Dtype scale = this->layer_param_.window_data_param().scale();
   const Dtype margin = this->layer_param_.window_data_param().margin();
     const int crop_size = this->layer_param_.transform_param().crop_size();
     int context_pad=1;
  int batch_size=bottom[0]->num();
  const int channels = bottom[0]->channels();
int use_square=0;
//LOG(INFO)<<"batchsize:"<<batch_size<<" crop_size:"<<crop_size;
 cv::Size cv_crop_size(crop_size, crop_size);
 caffe_set(crop_size*crop_size*bottom[0]->channels(), Dtype(165), top[0]->mutable_cpu_data());

  for (int item_id=0;item_id<batch_size;item_id++)
  {
    
    
    //cv::Mat image_alias = image;
   //float* Idata=new float[bottom[0]->height()*bottom[0]->width()*3];
  // LOG(INFO)<<"before copying a single image";
    caffe_copy(channels*bottom[0]->width()*bottom[0]->height(), 
	       bottom[0]->cpu_data()+item_id*channels*bottom[0]->width()*bottom[0]->height(),
	       singleimage_.mutable_cpu_data());
    // LOG(INFO)<<"done copying a single image";
   int index=0;
        for (int h = 0; h < bottom[0]->height(); ++h) {
          for (int w = 0; w < bottom[0]->width(); ++w) {
	     for (int c = 0; c < channels; ++c) {
	    singleimage2_.mutable_cpu_data()[index]=
            singleimage_.cpu_data()[c*bottom[0]->width()*bottom[0]->height()+
	     h*bottom[0]->width()+w];
		         index++;
          }
        }
      }
      
     // LOG(INFO)<<"done with single image 2";
   //  cv::Mat cv_img=converttoopencvimg(bottom[0]->mutable_cpu_data(), bottom[0]->height(), bottom[0]->width() );
    cv::Mat cv_img=converttoopencvimg(singleimage2_.mutable_cpu_data(), bottom[0]->height(), bottom[0]->width() );
  std::stringstream tmpi;
		tmpi<<item_id;	
		
    std::string fileName1 = std::string("./debugDump/img_pre");
	   fileName1 += tmpi.str();
           fileName1 += ".png";
      //imwrite(fileName1.c_str(), cv_img);
    
   
			
  // LOG(INFO)<<"  cv_img.cols:"<<cv_img.cols<<"  cv_img.rows:"<<cv_img.rows;

    float x_center=bottom[1]->cpu_data()[item_id*2+0];
    float y_center=bottom[1]->cpu_data()[item_id*2+1];
    int x1=static_cast<int>(x_center-margin);//bottom[1]->cpu_data()[item_id*4+2];
     int y1=static_cast<int>(y_center-margin);//bottom[1]->cpu_data()[item_id*4+2];
      int x2=static_cast<int>(x_center+margin);//bottom[1]->cpu_data()[item_id*4+2];
       int y2=static_cast<int>(y_center+margin);//bottom[1]->cpu_data()[item_id*4+2];
   // LOG(INFO)<<"x1:"<<x1<<" x2:"<<x2<<" y1:"<<y1<<" y2:"<<y2;
    
    

      
	

      int pad_w = 0;
      int pad_h = 0;
    
        // scale factor by which to expand the original region
        // such that after warping the expanded region to crop_size x crop_size
        // there's exactly context_pad amount of padding on each side
        Dtype context_scale = static_cast<Dtype>(crop_size) /
            static_cast<Dtype>(crop_size - 2*context_pad);
        // LOG(INFO)<<"scale:"<<scale;
        // compute the expanded region
        Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
        Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
        Dtype center_x = static_cast<Dtype>(x1) + half_width;
        Dtype center_y = static_cast<Dtype>(y1) + half_height;
        if (use_square) {
          if (half_height > half_width) {
            half_width = half_height;
          } else {
            half_height = half_width;
          }
        }
        x1 = static_cast<int>(round(center_x - half_width*context_scale));
        x2 = static_cast<int>(round(center_x + half_width*context_scale));
        y1 = static_cast<int>(round(center_y - half_height*context_scale));
        y2 = static_cast<int>(round(center_y + half_height*context_scale));
        // LOG(INFO)<<" x1:"<<x1<<" x2:"<<x2<<" y1:"<<y1<<" y2:"<<y2;
        // the expanded region may go outside of the image
        // so we compute the clipped (expanded) region and keep track of
        // the extent beyond the image
        int unclipped_height = y2-y1+1;
        int unclipped_width = x2-x1+1;
        int pad_x1 = std::max(0, -x1);
        int pad_y1 = std::max(0, -y1);
        int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
        int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
        // clip bounds
        x1 = x1 + pad_x1;
        x2 = x2 - pad_x2;
        y1 = y1 + pad_y1;
        y2 = y2 - pad_y2;
        CHECK_GT(x1, -1);
        CHECK_GT(y1, -1);
        CHECK_LT(x2, cv_img.cols);
        CHECK_LT(y2, cv_img.rows);

        int clipped_height = y2-y1+1;
        int clipped_width = x2-x1+1;

        // scale factors that would be used to warp the unclipped
        // expanded region
        Dtype scale_x =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
        Dtype scale_y =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

        // size to warp the clipped expanded region to
        cv_crop_size.width =
            static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
        cv_crop_size.height =
            static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
        pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
        pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
        pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
        pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

        pad_h = pad_y1;
        // if we're mirroring, we mirror the padding too (to be pedantic)
        
        pad_w = pad_x1;
        

        // ensure that the warped, clipped region plus the padding fits in the
        // crop_size x crop_size image (it might not due to rounding)
        if (pad_h + cv_crop_size.height > crop_size) {
          cv_crop_size.height = crop_size - pad_h;
        }
        if (pad_w + cv_crop_size.width > crop_size) {
          cv_crop_size.width = crop_size - pad_w;
        }
      
   // LOG(INFO)<<"Right before ROI x1:"<<x1<<" x2:"<<x2<<" y1:"<<y1<<" y2:"<<y2;
      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat cv_cropped_img = cv_img(roi);
      cv::resize(cv_cropped_img, cv_cropped_img,
          cv_crop_size, 0, 0, cv::INTER_LINEAR);
          fileName1 = std::string("./debugDump/img_roi");
	   fileName1 += tmpi.str();
		      	  fileName1 += ".png";
      //imwrite(fileName1.c_str(), cv_cropped_img);
      
  // LOG(INFO)<<"done roi and resize";
     
    // LOG(INFO)<<" pad_h:"<<pad_h<<" pad_w:"<<pad_w<<  " cv_cropped_img.cols:"<<cv_cropped_img.cols<<"  cv_cropped_img.rows:"<<cv_cropped_img.rows<<" cv_cropped_img.channels:"<<std::cout<<cv_cropped_img.channels();

      // copy the warped window into top_data
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cv_cropped_img.rows; ++h) {
          for (int w = 0; w < cv_cropped_img.cols; ++w) {
            Dtype pixel =
                static_cast<Dtype>(cv_cropped_img.at<cv::Vec3f>(h, w)[c]);
 //LOG(INFO)<<"index in top: "<<((item_id * channels + c) * crop_size + h + pad_h)
              //       * crop_size + w + pad_w;
		//LOG(INFO)<<"pixel:"<<pixel;
		  //   LOG(INFO)<<"width:"<<top[0]->width()<<"height:"<<top[0]->height()<<"channels:"<<top[0]->channels();
		   //  LOG(INFO)<<" content:"<<top[0]->mutable_cpu_data()[((item_id * channels + c) * crop_size + h + pad_h)
                   //  * crop_size + w + pad_w];
		    // LOG(INFO)<<" num:"<<top[0]->num()<<" channels:"<<top[0]->channels()<<" width:"<<top[0]->width()<<" height:"<<top[0]->height();
            top[0]->mutable_cpu_data()[((item_id * channels + c) * crop_size + h + pad_h)
                     * crop_size + w + pad_w]
                = (pixel-Dtype(90.0))
                  * scale;
		
          }
          
        }
      }

      
    //  LOG(INFO)<<"done copy warping crop_size:"<<crop_size;


		//VISUALIZATION CODE!	
		LOG(INFO)<<"visualize!";

			  std::stringstream tmp,tmp2,tmp3,tmp4;

		
			tmp<<item_id;
			tmp4<<pad_h;
			
		      	 std::string fileName = std::string("./debugDump/imgpostcrop");
		      	 //std::string fileName = printf("./debugDump/%03d.png",tmp);
		      	
			   fileName += tmp.str();
		      	  fileName += ".png";
			  cv::Mat debugImage(crop_size, crop_size, CV_8UC3, cv::Scalar(0));
				  for (int c = 0; c < channels; ++c) 
				    for (int h = 0; h < crop_size; ++h) 
				      for (int w = 0; w < crop_size; ++w) 
					if ((abs(h-ceil(crop_size/2))<5)&&(abs(w-ceil(crop_size/2))<5))
					{
					  if ((c==1)||(c==2))
					    debugImage.at<cv::Vec3b>(h, w)[c] =255;
					  else
					     debugImage.at<cv::Vec3b>(h, w)[c] =0;
					}
					else
					  debugImage.at<cv::Vec3b>(h, w)[c] =  top[0]->cpu_data()[((item_id * channels + c) * crop_size + h) * crop_size + w ] + Dtype(90.0);
		
// 					debugImage.at<cv::Vec3b>(h, w)[c] =  top_data[((place_item_id * channels + c) * crop_size + h + pad_h) * crop_size + w + pad_w] + Dtype(90.0);
// 				  imwrite(fileName.c_str(), debugImage);
// 					debugImage.at<cv::Vec3b>(h, w)[c] =  top_data[((place_item_id * channels + c) * crop_size + h) * crop_size + w] + Dtype(90.0);
// 				LOG(INFO)<<"WRITING IMAGE"<<"f:"<<fileName.c_str();
			//imwrite(fileName.c_str(), debugImage);

				      
				   

		LOG(INFO)<<"done visualize!";
		
      }

     
//LOG(FATAL)<<"end";

}



#ifdef CPU_ONLY
STUB_GPU(CroppingKatLayer);
#endif



INSTANTIATE_CLASS(CroppingKatLayer);
REGISTER_LAYER_CLASS(CroppingKat);


}  // namespace caffe

