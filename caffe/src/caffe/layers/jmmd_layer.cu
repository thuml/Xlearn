#include <vector>
#include <cfloat>
#include <algorithm>

#include "caffe/layers/jmmd_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CalculateKernel(const int n, const Dtype* distance2, const Dtype gamma,
        Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
      out[index] = exp(-gamma * distance2[index]);
  }
}

template <typename Dtype>
__global__ void CalculateElewiseSquareDistance(const int n, const Dtype* source, const Dtype* target,
        const int source_num, const int target_num, const int dim, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
      int data_index1 = index / dim / (source_num + target_num);
      int data_index2 = index / dim % (source_num + target_num);
      int dim_offset = index % dim;
      Dtype data1;
      Dtype data2;
      if(data_index1 >= source_num){
          data_index1 -= source_num;
          data1 = target[data_index1 * dim + dim_offset];
      }
      else{
          data1 = source[data_index1 * dim + dim_offset];
      }
      if(data_index2 >= source_num){
          data_index2 -= source_num;
          data2 = target[data_index2 * dim + dim_offset];
      }
      else{
          data2 = source[data_index2 * dim + dim_offset];
      }
      out[index] = (data1 - data2) * (data1 - data2);
  }
}

template <typename Dtype>
__global__ void CalculateLabelProbRBFKernel(const int n, const Dtype* source_p, const Dtype* target_p,
        const int source_num, const int total_num, const int label_dim, const Dtype sigma,
        const int kernel_num, const Dtype kernel_mul, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
      int index1 = index / total_num;
      int index2 = index % total_num;
      if(index1 < source_num && index2 < source_num){
          //source vs. source
          int offset1 = index1 * label_dim;
          int offset2 = index2 * label_dim;
          Dtype sum = Dtype(0);
          for(int j = 0;j < label_dim;++j){
              sum += (source_p[offset1 + j] - source_p[offset2 + j]) * (source_p[offset1 + j] - source_p[offset2 + j]);
          }
          Dtype times = pow(kernel_mul, (Dtype)(kernel_num / 2));
          Dtype temp_sigma = sigma / times;
          out[index] = Dtype(0);
          for(int j = 0;j < kernel_num;++j){
              out[index] += exp(-sum / temp_sigma);
              temp_sigma *= kernel_mul;
          }
      }
      else if(index1 < source_num && index2 >= source_num){
          //source vs. target
          int offset1 = label_dim * index1;
          int offset2 = label_dim * (index2 - source_num);
          Dtype sum = Dtype(0);
          for(int j = 0;j < label_dim;++j){
              sum += (source_p[offset1 + j] - target_p[offset2 + j]) * (source_p[offset1 + j] - target_p[offset2 + j]);
          }
          Dtype times = pow(kernel_mul, (Dtype)(kernel_num / 2));
          Dtype temp_sigma = sigma / times;
          out[index] = Dtype(0);
          for(int j = 0;j < kernel_num;++j){
              out[index] += exp(-sum / temp_sigma);
              temp_sigma *= kernel_mul;
          }
      }
      else if(index1 >= source_num && index2 < source_num){
          //target vs. source
          int offset1 = label_dim * (index1 - source_num);
          int offset2 = label_dim * index2;
          Dtype sum = Dtype(0);
          for(int j = 0;j < label_dim;++j){
              sum += (target_p[offset1 + j] - source_p[offset2 + j]) * (target_p[offset1 + j] - source_p[offset2 + j]);
          }
          Dtype times = pow(kernel_mul, (Dtype)(kernel_num / 2));
          Dtype temp_sigma = sigma / times;
          out[index] = Dtype(0);
          for(int j = 0;j < kernel_num;++j){
              out[index] += exp(-sum / temp_sigma);
              temp_sigma *= kernel_mul;
          }
      }
      else{
          //target vs. target
          int offset1 = label_dim * (index1 - source_num);
          int offset2 = label_dim * (index2 - source_num);
          Dtype sum = Dtype(0);
          for(int j = 0;j < label_dim;++j){
              sum += (target_p[offset1 + j] - target_p[offset2 + j]) * (target_p[offset1 + j] - target_p[offset2 + j]);
          }
          Dtype times = pow(kernel_mul, (Dtype)(kernel_num / 2));
          Dtype temp_sigma = sigma / times;
          out[index] = Dtype(0);
          for(int j = 0;j < kernel_num;++j){
              out[index] += exp(-sum / temp_sigma);
              temp_sigma *= kernel_mul;
          }
      }
  }
}

template <typename Dtype>
void JMMDLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* source = bottom[0]->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    
    //calculate square distance between each data pair -> distance2
    Dtype* elewise_square_distance = diff_.mutable_gpu_data(); // square distance between element pairs
    Dtype* distance2 = diff_.mutable_gpu_diff(); // square distance between data pairs
    int nthreads = total_num_ * total_num_ * dim_;
    CalculateElewiseSquareDistance<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, source, target, source_num_, target_num_, dim_, elewise_square_distance);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, total_num_ * total_num_, dim_, (Dtype)1.,
                         elewise_square_distance, diff_multiplier_.gpu_data(), (Dtype)0., distance2);
    
    //calculate bandwith of RBF kernel -> gamma_
    Dtype bandwidth;
    caffe_gpu_asum(total_num_ * total_num_, distance2, &bandwidth);
    gamma_ = (total_num_ * total_num_ - total_num_) / bandwidth;
    
    //calculate each kernel of data
    Dtype gamma_times = pow(kernel_mul_, (Dtype)(kernel_num_ / 2));
    Dtype kernel_gamma = gamma_ / gamma_times;
    nthreads = total_num_ * total_num_;
    for(int i = 0;i < kernel_num_;++i){
        CalculateKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, distance2, kernel_gamma, kernel_val_[i]->mutable_gpu_data());
        kernel_gamma *= kernel_mul_;
    }

    //calculate each kernel of label
    const Dtype* source_label = bottom[2]->gpu_data();
    const Dtype* target_label = bottom[3]->gpu_data();
    nthreads = total_num_ * total_num_;
    int label_dim = bottom[3]->count() / bottom[3]->count(0, 1);
    CalculateLabelProbRBFKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, bottom[2]->gpu_data(), bottom[3]->gpu_data(),
            source_num_, total_num_, label_dim, sigma_, label_kernel_num_, label_kernel_mul_, delta_.mutable_gpu_data());
    
    //set mmd loss to zero in forward
    top[0]->mutable_cpu_data()[0] = Dtype(0);
}

template <typename Dtype>
__global__ void CalculateDiff(const int n, const Dtype* source, const Dtype* target,
        const int source_num, const int total_num, const int dim, const Dtype* kernel_val, 
        const Dtype gamma, const Dtype* label_kernel, const Dtype* data, const int data_index, 
        Dtype* source_diff, Dtype* target_diff) {
  CUDA_KERNEL_LOOP(index, n) {
      int oppo_index = index / dim;
      int dim_offset = index % dim;
      Dtype data1 = data[dim_offset];
      Dtype data2 = (oppo_index >= source_num) ?
          target[dim * (oppo_index - source_num) + dim_offset]: 
          source[dim * oppo_index + dim_offset];
      int kernel_index = data_index * total_num + oppo_index;
      Dtype factor_of_diff = -2 * gamma * kernel_val[kernel_index] * label_kernel[kernel_index];
      if(oppo_index >= source_num){
          oppo_index -= source_num;
          if(data_index >= source_num)
              target_diff[oppo_index * dim + dim_offset] += factor_of_diff * (data2 - data1);
          else
              target_diff[oppo_index * dim + dim_offset] += -0.5 * factor_of_diff * (data2 - data1); 
      }
      else{
          if(data_index < source_num)
              source_diff[oppo_index * dim + dim_offset] += 0.25 * factor_of_diff * (data2 - data1);
          else
              source_diff[oppo_index * dim + dim_offset] += -0.5 * factor_of_diff * (data2 - data1);
      }
  }
}

template <typename Dtype>
void JMMDLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (train_iter_num_ >= 0){
    Dtype* source_diff = bottom[0]->mutable_gpu_diff();
    Dtype* target_diff = bottom[1]->mutable_gpu_diff();
    Dtype* label_source_diff = bottom[2]->mutable_gpu_diff();
    Dtype* label_target_diff = bottom[3]->mutable_gpu_diff();
    int label_dim = bottom[2]->count() / bottom[2]->count(0, 1);
    caffe_gpu_set(source_num_ * label_dim, Dtype(0), label_source_diff);
    caffe_gpu_set(target_num_ * label_dim, Dtype(0), label_target_diff);
    
    caffe_gpu_set(source_num_ * dim_, Dtype(0), source_diff);
    caffe_gpu_set(target_num_ * dim_, Dtype(0), target_diff);
    if(source_num_ <= 1 || target_num_ <= 1) return;
    int sample_num = (source_num_ > target_num_) ? source_num_ : target_num_;
    int s1, s2, t1, t2;
    Dtype* tempX1 = diff_.mutable_gpu_diff() + total_num_ * total_num_;
    Dtype* tempX2 = diff_.mutable_gpu_diff() + total_num_ * total_num_ + dim_;
    Dtype* tempY1 = NULL, *tempY2 = NULL;
    tempY1 = diff_.mutable_gpu_diff() + total_num_ * total_num_ + dim_ + dim_;
    tempY2 = diff_.mutable_gpu_diff() + total_num_ * total_num_ + dim_ + dim_ + label_dim;
    for(int i = 0;i < sample_num;++i){
        s1 = rand() % source_num_;
        s2 = rand() % source_num_;
        s2 = (s1 != s2) ? s2 : (s2 + 1) % source_num_;
        
        t1 = rand() % target_num_;
        t2 = rand() % target_num_;
        t2 = (t1 != t2) ? t2 : (t2 + 1) % target_num_;
        
        Dtype square_sum = 0;
        Dtype factor_for_diff = 0;
        const Dtype* x_s1 = bottom[0]->gpu_data() + s1 * dim_;
        const Dtype* x_s2 = bottom[0]->gpu_data() + s2 * dim_;
        const Dtype* x_t1 = bottom[1]->gpu_data() + t1 * dim_;
        const Dtype* x_t2 = bottom[1]->gpu_data() + t2 * dim_;
        const Dtype* y_s1 = bottom[2]->gpu_data() + s1 * label_dim;
        const Dtype* y_s2 = bottom[2]->gpu_data() + s2 * label_dim;
        const Dtype* y_t1 = bottom[3]->gpu_data() + t1 * label_dim;
        const Dtype* y_t2 = bottom[3]->gpu_data() + t2 * label_dim;
        
        caffe_gpu_sub<Dtype>(dim_, x_s1, x_s2, tempX1);
        caffe_gpu_sub<Dtype>(dim_, x_s2, x_s1, tempX2);
        caffe_gpu_dot<Dtype>(dim_, tempX1, tempX1, &square_sum);
        Dtype times = pow(kernel_mul_, (Dtype)(kernel_num_ / 2));
        Dtype temp_gamma = gamma_ / times;
        Dtype x_kernel = Dtype(0);
        for(int j = 0; j < kernel_num_; j++){
            Dtype temp_n = (0.0 - temp_gamma) * square_sum;
            x_kernel += exp(temp_n);
            temp_n = exp(temp_n) * delta_.cpu_data()[s1 * total_num_ + s2];
            temp_n = (-2) * temp_gamma * temp_n;
            factor_for_diff += temp_n;
            temp_gamma = temp_gamma * kernel_mul_;
        }
        caffe_gpu_scal(dim_, loss_weight_ * factor_for_diff / sample_num, tempX1);
        caffe_gpu_scal(dim_, loss_weight_ * factor_for_diff / sample_num, tempX2);
        caffe_gpu_add(dim_, tempX1, source_diff + s1 * dim_, source_diff + s1 * dim_);
        caffe_gpu_add(dim_, tempX2, source_diff + s2 * dim_, source_diff + s2 * dim_);

        caffe_gpu_sub<Dtype>(label_dim, y_s1, y_s2, tempY1);
        caffe_gpu_sub<Dtype>(label_dim, y_s2, y_s1, tempY2);
        factor_for_diff = (-2) / sigma_ * x_kernel * delta_.cpu_data()[s1 * total_num_ + s2];
        caffe_gpu_scal(label_dim, loss_weight_ * factor_for_diff / sample_num, tempY1);
        caffe_gpu_scal(label_dim, loss_weight_ * factor_for_diff / sample_num, tempY2);
        caffe_gpu_add(label_dim, tempY1, label_source_diff + s1 * label_dim, label_source_diff + s1 * label_dim);
        caffe_gpu_add(label_dim, tempY2, label_source_diff + s2 * label_dim, label_source_diff + s2 * label_dim);
         
        factor_for_diff = 0;
        caffe_gpu_sub<Dtype>(dim_, x_s1, x_t2, tempX1);
        caffe_gpu_sub<Dtype>(dim_, x_t2, x_s1, tempX2);
        caffe_gpu_dot<Dtype>(dim_, tempX1, tempX1, &square_sum);
        temp_gamma = gamma_ / times;
        x_kernel = Dtype(0);
        for(int j = 0; j < kernel_num_; j++){
            Dtype temp_n = (0.0 - temp_gamma) * square_sum;
            x_kernel += exp(temp_n);
            temp_n = exp(temp_n) * Dtype(-1) * delta_.cpu_data()[s1 * total_num_ + source_num_ + t2];
            temp_n = (-2) * temp_gamma * temp_n;
            factor_for_diff += temp_n;
            temp_gamma = temp_gamma * kernel_mul_;
        }
        caffe_gpu_scal(dim_, loss_weight_ * factor_for_diff / sample_num, tempX1);
        caffe_gpu_scal(dim_, loss_weight_ * factor_for_diff / sample_num, tempX2);
        caffe_gpu_add(dim_, tempX1, source_diff + s1 * dim_, source_diff + s1 * dim_);
        caffe_gpu_add(dim_, tempX2, target_diff + t2 * dim_, target_diff + t2 * dim_);

        caffe_gpu_sub<Dtype>(label_dim, y_s1, y_t2, tempY1);
        caffe_gpu_sub<Dtype>(label_dim, y_t2, y_s1, tempY2);
        factor_for_diff = 2 / sigma_ * x_kernel * delta_.cpu_data()[s1 * total_num_ + source_num_ + t2];
        caffe_gpu_scal(label_dim, loss_weight_ * factor_for_diff / sample_num, tempY1);
        caffe_gpu_scal(label_dim, loss_weight_ * factor_for_diff / sample_num, tempY2);
        caffe_gpu_add(label_dim, tempY1, label_source_diff + s1 * label_dim, label_source_diff + s1 * label_dim);
        caffe_gpu_add(label_dim, tempY2, label_target_diff + t2 * label_dim, label_target_diff + t2 * label_dim);
         
        factor_for_diff = 0;
        caffe_gpu_sub<Dtype>(dim_, x_t1, x_s2, tempX1);
        caffe_gpu_sub<Dtype>(dim_, x_s2, x_t1, tempX2);
        caffe_gpu_dot<Dtype>(dim_, tempX1, tempX1, &square_sum);
        temp_gamma = gamma_ / times;
        x_kernel = Dtype(0);
        for(int j = 0; j < kernel_num_; j++){
            Dtype temp_n = (0.0 - temp_gamma) * square_sum;
            x_kernel += exp(temp_n);
            temp_n = exp(temp_n) * Dtype(-1) * delta_.cpu_data()[(t1 + source_num_) * total_num_ + s2];
            temp_n = (-2) * temp_gamma * temp_n;
            factor_for_diff += temp_n;
            temp_gamma = temp_gamma * kernel_mul_;
        }
        caffe_gpu_scal(dim_, loss_weight_ * factor_for_diff / sample_num, tempX1);
        caffe_gpu_scal(dim_, loss_weight_ * factor_for_diff / sample_num, tempX2);
        caffe_gpu_add(dim_, tempX1, target_diff + t1 * dim_, target_diff + t1 * dim_);
        caffe_gpu_add(dim_, tempX2, source_diff + s2 * dim_, source_diff + s2 * dim_);

        caffe_gpu_sub<Dtype>(label_dim, y_s2, y_t1, tempY1);
        caffe_gpu_sub<Dtype>(label_dim, y_t1, y_s2, tempY2);
        factor_for_diff = 2 / sigma_ * x_kernel * delta_.cpu_data()[(t1 + source_num_) * total_num_ + s2];
        caffe_gpu_scal(label_dim, loss_weight_ * factor_for_diff / sample_num, tempY1);
        caffe_gpu_scal(label_dim, loss_weight_ * factor_for_diff / sample_num, tempY2);
        caffe_gpu_add(label_dim, tempY1, label_source_diff + s2 * label_dim, label_source_diff + s2 * label_dim);
        caffe_gpu_add(label_dim, tempY2, label_target_diff + t1 * label_dim, label_target_diff + t1 * label_dim);
        
        factor_for_diff = 0;
        caffe_gpu_sub<Dtype>(dim_, x_t1, x_t2, tempX1);
        caffe_gpu_sub<Dtype>(dim_, x_t2, x_t1, tempX2);
        caffe_gpu_dot<Dtype>(dim_, tempX1, tempX1, &square_sum);
        temp_gamma = gamma_ / times;
        x_kernel = Dtype(0);
        for(int j = 0; j < kernel_num_; j++){
            Dtype temp_n = (0.0 - temp_gamma) * square_sum;
            x_kernel += exp(temp_n);
            temp_n = exp(temp_n) * delta_.cpu_data()[(t1 + source_num_) * total_num_ + t2 + source_num_];
            temp_n = (-2) * temp_gamma * temp_n;
            factor_for_diff += temp_n;
            temp_gamma = temp_gamma * kernel_mul_;
        }
        caffe_gpu_scal(dim_, loss_weight_ * factor_for_diff / sample_num, tempX1);
        caffe_gpu_scal(dim_, loss_weight_ * factor_for_diff / sample_num, tempX2);
        caffe_gpu_add(dim_, tempX1, target_diff + t1 * dim_, target_diff + t1 * dim_);
        caffe_gpu_add(dim_, tempX2, target_diff + t2 * dim_, target_diff + t2 * dim_);

        caffe_gpu_sub<Dtype>(label_dim, y_t1, y_t2, tempY1);
        caffe_gpu_sub<Dtype>(label_dim, y_t2, y_t1, tempY2);
        factor_for_diff = (-2) / sigma_ * x_kernel * delta_.cpu_data()[(t1 + source_num_) * total_num_ + t2 + source_num_];
        caffe_gpu_scal(label_dim, loss_weight_ * factor_for_diff / sample_num, tempY1);
        caffe_gpu_scal(label_dim, loss_weight_ * factor_for_diff / sample_num, tempY2);
        caffe_gpu_add(label_dim, tempY1, label_target_diff + t1 * label_dim, label_target_diff + t1 * label_dim);
        caffe_gpu_add(label_dim, tempY2, label_target_diff + t2 * label_dim, label_target_diff + t2 * label_dim);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(JMMDLossLayer);


}  // namespace caffe
