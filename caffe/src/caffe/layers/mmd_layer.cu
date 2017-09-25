#include <vector>
#include <cfloat>
#include <algorithm>

#include "caffe/layers/mmd_layer.hpp"
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
__global__ void CalculateSpreadDistance2(const int n, const Dtype* source, const Dtype* target,
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
void MMDLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* source = bottom[0]->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    
    //calculate square distance
    Dtype* spread_distance2 = diff_.mutable_gpu_data();
    Dtype* distance2 = diff_.mutable_gpu_diff();
    int nthreads = total_num_ * total_num_ * dim_;
    CalculateSpreadDistance2<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, source, target, source_num_, target_num_, dim_, spread_distance2);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, total_num_ * total_num_, dim_, (Dtype)1.,
                         spread_distance2, diff_multiplier_.gpu_data(), (Dtype)0., distance2);
    
    //calculate bandwith
    Dtype bandwidth;
    caffe_gpu_asum(total_num_ * total_num_, distance2, &bandwidth);
    if(fix_gamma_){
        gamma_ = gamma_ < 0 ? (total_num_ * total_num_ - total_num_) / bandwidth : gamma_;
    }
    else{
        gamma_ = (total_num_ * total_num_ - total_num_) / bandwidth;
    }
    
    //calculate each kernel of data
    Dtype gamma_times = pow(kernel_mul_, (Dtype)(kernel_num_ / 2));
    Dtype kernel_gamma = gamma_ / gamma_times;
    
    nthreads = total_num_ * total_num_;
    for(int i = 0;i < kernel_num_;++i){
        CalculateKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            nthreads, distance2, kernel_gamma, kernel_val_[i]->mutable_gpu_data());
        kernel_gamma *= kernel_mul_;
    }

    Dtype loss = 0;
    int sample_num = (source_num_ > target_num_) ? source_num_ : target_num_;
    int s1, s2, t1, t2;
    for(int i = 0;i < sample_num;++i){
        s1 = rand() % source_num_;
        s2 = rand() % source_num_;
        s2 = (s1 != s2) ? s2 : (s2 + 1) % source_num_;
        
        t1 = rand() % target_num_;
        t2 = rand() % target_num_;
        t2 = (t1 != t2) ? t2 : (t2 + 1) % target_num_;
        
        for(int i = 0;i < kernel_num_;++i){
            loss += kernel_val_[i]->cpu_data()[s1 * total_num_ + s2];
            loss += kernel_val_[i]->cpu_data()[(source_num_ + t1) * total_num_ + source_num_ + t2];
            loss -= kernel_val_[i]->cpu_data()[s1 * total_num_ + source_num_ + t2];
            loss -= kernel_val_[i]->cpu_data()[s2 * total_num_ + source_num_ + t1];
        }
    }

    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void calculate_diff(const int dim, const Dtype* f1, const Dtype* f2, const Dtype coeff,
        const Dtype gamma, const int kernel_num, const Dtype kernel_mul, const Dtype loss_weight, 
        const int sample_num, Dtype* tmp1, Dtype* tmp2, Dtype* diff1, Dtype* diff2) {
    Dtype square_sum = 0;
    Dtype factor_for_diff = 0;
    caffe_gpu_sub<Dtype>(dim, f1, f2, tmp1);
    caffe_gpu_sub<Dtype>(dim, f2, f1, tmp2);
    caffe_gpu_dot<Dtype>(dim, tmp1, tmp1, &square_sum);
    Dtype times = pow(kernel_mul, (Dtype)(kernel_num / 2));
    Dtype temp_gamma = gamma / times;
    
    for(int i = 0; i < kernel_num; i++){
        Dtype temp_n = (0.0 - temp_gamma) * square_sum;
        temp_n = exp(temp_n) * coeff;
        temp_n = (-2) * temp_gamma * temp_n;
        factor_for_diff += temp_n;
        temp_gamma = temp_gamma * kernel_mul;
    }
    caffe_gpu_scal(dim, loss_weight * factor_for_diff / sample_num, tmp1);
    caffe_gpu_scal(dim, loss_weight * factor_for_diff / sample_num, tmp2);
    caffe_gpu_add(dim, tmp1, diff1, diff1);
    caffe_gpu_add(dim, tmp2, diff2, diff2);
}

template <typename Dtype>
void MMDLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    Dtype* source_diff = bottom[0]->mutable_gpu_diff();
    Dtype* target_diff = bottom[1]->mutable_gpu_diff();
    caffe_gpu_set(source_num_ * dim_, Dtype(0), source_diff);
    caffe_gpu_set(target_num_ * dim_, Dtype(0), target_diff);
    
    if(source_num_ <= 1 || target_num_ <= 1) return;
    int sample_num = (source_num_ > target_num_) ? source_num_ : target_num_;
    int s1, s2, t1, t2;
    Dtype* tempX1 = diff_.mutable_gpu_diff() + total_num_ * total_num_;
    Dtype* tempX2 = diff_.mutable_gpu_diff() + total_num_ * total_num_ + dim_;
    for(int i = 0;i < sample_num;++i){
        s1 = rand() % source_num_;
        s2 = rand() % source_num_;
        s2 = (s1 != s2) ? s2 : (s2 + 1) % source_num_;
        
        t1 = rand() % target_num_;
        t2 = rand() % target_num_;
        t2 = (t1 != t2) ? t2 : (t2 + 1) % target_num_;
        
        const Dtype* x_s1 = bottom[0]->gpu_data() + s1 * dim_;
        const Dtype* x_s2 = bottom[0]->gpu_data() + s2 * dim_;
        const Dtype* x_t1 = bottom[1]->gpu_data() + t1 * dim_;
        const Dtype* x_t2 = bottom[1]->gpu_data() + t2 * dim_;

        calculate_diff(dim_, x_s1, x_s2, Dtype(1), gamma_, kernel_num_, 
                kernel_mul_, loss_weight_, sample_num, tempX1, tempX2, 
                source_diff + s1 * dim_, source_diff + s2 * dim_);

        calculate_diff(dim_, x_s1, x_t2, Dtype(-1), gamma_, kernel_num_, 
                kernel_mul_, loss_weight_, sample_num, tempX1, tempX2, 
                source_diff + s1 * dim_, target_diff + t2 * dim_);
        
        calculate_diff(dim_, x_t1, x_s2, Dtype(-1), gamma_, kernel_num_, 
                kernel_mul_, loss_weight_, sample_num, tempX1, tempX2, 
                target_diff + t1 * dim_, source_diff + s2 * dim_);
        
        calculate_diff(dim_, x_t1, x_t2, Dtype(1), gamma_, kernel_num_, 
                kernel_mul_, loss_weight_, sample_num, tempX1, tempX2, 
                target_diff + t1 * dim_, target_diff + t2 * dim_);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(MMDLossLayer);


}  // namespace caffe
