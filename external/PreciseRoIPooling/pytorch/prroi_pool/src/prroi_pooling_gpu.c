#include <math.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "prroi_pooling_gpu_impl.cuh"

// 前向计算
at::Tensor prroi_pooling_forward_cuda(const at::Tensor &features, const at::Tensor &rois, int pooled_height, int pooled_width, float spatial_scale) {
    int nr_rois = rois.size(0);
    int nr_channels = features.size(1);
    int height = features.size(2);
    int width = features.size(3);
    int top_count = nr_rois * nr_channels * pooled_height * pooled_width;
    auto output = at::zeros({nr_rois, nr_channels, pooled_height, pooled_width}, features.options());

    if (output.numel() == 0) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        return output;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    PrRoIPoolingForwardGpu(
        stream,
        features.data_ptr<float>(),
        rois.data_ptr<float>(),
        output.data_ptr<float>(),
        nr_channels,
        height,
        width,
        pooled_height,
        pooled_width,
        spatial_scale,
        top_count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    return output;
}

// 反向计算（特征的梯度）
at::Tensor prroi_pooling_backward_cuda(
    const at::Tensor &features, const at::Tensor &rois, const at::Tensor &output, const at::Tensor &output_diff,
    int pooled_height, int pooled_width, float spatial_scale) {

    auto features_diff = at::zeros_like(features);

    int nr_rois = rois.size(0);
    int batch_size = features.size(0);
    int nr_channels = features.size(1);
    int height = features.size(2);
    int width = features.size(3);
    int top_count = nr_rois * nr_channels * pooled_height * pooled_width;
    int bottom_count = batch_size * nr_channels * height * width;

    if (output.numel() == 0) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        return features_diff;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    PrRoIPoolingBackwardGpu(
        stream,
        features.data_ptr<float>(),
        rois.data_ptr<float>(),
        output.data_ptr<float>(),
        output_diff.data_ptr<float>(),
        features_diff.data_ptr<float>(),
        nr_channels,
        height,
        width,
        pooled_height,
        pooled_width,
        spatial_scale,
        top_count,
        bottom_count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    return features_diff;
}

// 反向计算（坐标的梯度）
at::Tensor prroi_pooling_coor_backward_cuda(
    const at::Tensor &features, const at::Tensor &rois, const at::Tensor &output, const at::Tensor &output_diff,
    int pooled_height, int pooled_width, float spatial_scale) {

    auto coor_diff = at::zeros_like(rois);

    int nr_rois = rois.size(0);
    int nr_channels = features.size(1);
    int height = features.size(2);
    int width = features.size(3);
    int top_count = nr_rois * nr_channels * pooled_height * pooled_width;
    int bottom_count = nr_rois * 5;

    if (output.numel() == 0) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        return coor_diff;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    PrRoIPoolingCoorBackwardGpu(
        stream,
        features.data_ptr<float>(),
        rois.data_ptr<float>(),
        output.data_ptr<float>(),
        output_diff.data_ptr<float>(),
        coor_diff.data_ptr<float>(),
        nr_channels,
        height,
        width,
        pooled_height,
        pooled_width,
        spatial_scale,
        top_count,
        bottom_count
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    return coor_diff;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prroi_pooling_forward_cuda", &prroi_pooling_forward_cuda, "PRRoIPooling_forward");
    m.def("prroi_pooling_backward_cuda", &prroi_pooling_backward_cuda, "PRRoIPooling_backward");
    m.def("prroi_pooling_coor_backward_cuda", &prroi_pooling_coor_backward_cuda, "PRRoIPooling_backward_coor");
}
