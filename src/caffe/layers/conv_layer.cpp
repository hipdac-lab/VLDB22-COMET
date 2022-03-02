#include <vector>

#include "caffe/layers/conv_layer.hpp"
#include "../SZ/sz/include/sz.h"
#include "../SZ/sz/include/rw.h"
    
size_t rr5=0,rr4=0,rr3=0,rr2=0,rr1=0;
size_t outSize2=0;
int time_tool = 0;
int conv_size = 0;
int comp_size = 0;

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }

     /*SZ_Init("../SZ/example/sz.config");
     rr1 = this->top_dim_*this->num_;
     float temp[rr1];
     for (int i = 0; i < rr1; i++) {
         temp[i] = top[0]->mutable_cpu_data()[i];
     }
     time_tool += 1;
     unsigned char *bytes = SZ_compress(SZ_FLOAT, &temp[0], &outSize2, rr5, rr4, rr3, rr2, rr1);    
     if (time_tool % 100 == 0)
       printf("Current compression ratio of Conv_ is from %d to %d\n", rr1*32, outSize2);
     void *decData = SZ_decompress(SZ_FLOAT, bytes, outSize2, rr5, rr4, rr3, rr2, rr1);
     float *decData2 = (float *)decData;
     for (int i = 0; i < rr1; i++) {
         top[0]->mutable_cpu_data()[i] = *(float*)(decData2+i);
     }
     free(bytes);
     //free(decData);
     free(decData2);
     //free(data);
     SZ_Finalize();
*/
  //top[i]->mutable_cpu_data()[310000] = 0.0;
  /*printf("conv-size = %d\n", this->top_dim_*this->num_);
  for (int i = 0; i < this->top_dim_*this->num_; i++) {
    top[0]->mutable_cpu_data()[i] = 0.0;
  }*/
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    //const Dtype* bottom_data = bottom[i]->cpu_data();
    
    const Dtype* bottom_data = bottom[i]->cpu_data();

    Dtype *p_var = NULL;
    p_var = const_cast <Dtype*>(bottom_data); 

    SZ_Init("/opt/SZ/example/sz.config");                                 
    //      rr1 = this->bottom_dim_;
    //      rr2 = this->num_;
    //      float temp[this->bottom_dim_*this->num_];
    //      for (int i = 0; i < this->bottom_dim_*this->num_; i++) {
    rr1 = this->bottom_dim_*this->num_;
    //printf("test to see if working 0003\n"); 
    //printf("this->bottom_dim_ = %d\n", this->bottom_dim_);
    //printf("this->num_ = %d\n", this->num_);
    //printf("Size of p_var = %d\n", sizeof(p_var));
    //printf("top.size() = %d\n", top.size());
    //printf("Time tool = %d\n", time_tool);
    //printf("bottom.size() = %d\n", bottom.size());
    //printf("this->top_dim_ = %d\n", this->top_dim_);
    //printf("Size of data = %d\n", sizeof(bottom[i]->cpu_data()));
    //printf("test value = %f\n", bottom[i]->cpu_data()[64895]);
    //printf("test value = %f\n", bottom[i]->cpu_data()[74895]);
    //printf("test value = %f\n", bottom[i]->cpu_data()[704895]);
    float *temp;
    temp = (float*) malloc (rr1*sizeof(float));
    //printf("test out 0000\n");
    memcpy(temp, p_var, rr1*sizeof(float));
    //printf("test to see if working 0004\n"); 
    time_tool += 1;
    unsigned char *bytes = SZ_compress(SZ_FLOAT, temp, &outSize2, rr5, rr4, rr3, rr2, rr1);    
    //printf("test to see if working 0005\n"); 
    LOG(INFO) << "Current compression ratio of Conv_ is from " << rr1/250.0 << " to " << outSize2/1000.0;

    //printf("test to see if working 0006\n"); 
    void *decData = SZ_decompress(SZ_FLOAT, bytes, outSize2, rr5, rr4, rr3, rr2, rr1);
    float *decData2 = (float *)decData;
    //printf("test to see if working 0007\n"); 
    memcpy(p_var, decData2, rr1*sizeof(float));
    conv_size += rr1/250;
    comp_size += outSize2/1000;
    if (time_tool % 5 == 0) {
        printf("Current compresion ratio of Conv is %f x.\n", float(conv_size)/comp_size);
        conv_size = 0;
        comp_size = 0;
    }

//     conv_count += 1;
//     if (conv_count % 5 != 6) {
//         char filename[32];
//         sprintf(filename, "%d", 10 * (((conv_count - 1)/5)) + 5 - ((conv_count-1) % 5));
//         strcat(filename, ".txt");
//         char str[128] = "/home/jinsian/caffe-master/test_data/conv_";
//         char str01[128] = "/home/jinsian/caffe-master/test_data/conv_ori_";
//         strcat(str, filename);
//         strcat(str01, filename);
//         //FILE *output01 = fopen(str, "w");
//         //for (int i = 0; i < rr1; i++)                                                  
//           //  fprintf(output01, "%f ", p_var[i]);
//         FILE *output02 = fopen(str01, "w");
//         for (int i = 0; i < rr1; i++)
//             fprintf(output02, "%f ", temp[i]);
//         printf("filename = %s\n", str);
//       }


    free(bytes);
    free(decData2);
    free(temp);
    SZ_Finalize();

    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
	    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
	    for (int n = 0; n < this->num_; ++n) {
		    this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
	    }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
	    for (int n = 0; n < this->num_; ++n) {
		    // gradient w.r.t. weight. Note that we will accumulate diffs.
		    if (this->param_propagate_down_[0]) {
			    this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
					    top_diff + n * this->top_dim_, weight_diff);
		    }
		    // gradient w.r.t. bottom data, if necessary.
		    if (propagate_down[i]) {
			    this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
					    bottom_diff + n * this->bottom_dim_);
		    }
	    }
    }
  }
  }

#ifdef CPU_ONLY
  STUB_GPU(ConvolutionLayer);
#endif

  INSTANTIATE_CLASS(ConvolutionLayer);

  }  // namespace caffe
