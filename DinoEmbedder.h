#pragma once

// 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式
//  DinoEmbedder.h
//  DINOv2 image embedding via ONNX Runtime (C++)
//
//  Input:  BGR crown crop (cv::Mat, any size)
//  Output: L2-normalized embedding vector
//            dinov2_vits14 ⊥ 384-dim
//            dinov2_vitb14 ⊥ 768-dim
//
//  Model file: export with tools/export_dinov2_onnx.py
//              (dynamic batch, input [N,3,224,224], output [N,D])
//
//  Dependencies:
//    - ONNX Runtime 1.16+ (onnxruntime_cxx_api.h, onnxruntime.lib)
//    - OpenCV
//
//  Windows link: onnxruntime.lib (+ onnxruntime.dll next to exe)
//  CUDA (optional): onnxruntime-gpu build, useCUDA = true
// 式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式

#include "TSICommon.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

struct DinoEmbedderOptions
{
	std::string modelPath = "models/dinov2_vits14.onnx";
	int inputSize = 224;					// DINOv2 patch 14 ⊥ 224 = 16x16 patches
	int batchSize = 32;					// crops per ONNX run
	int intraOpThreads = 0;			// 0 = ORT default
	bool useCUDA = false;			// requires onnxruntime-gpu
	bool l2Normalize = true;			// unit-length output (recommended for clustering/RF)
};

class DinoEmbedder
{
public:

	explicit DinoEmbedder(const DinoEmbedderOptions& opts = DinoEmbedderOptions())
		: opts_(opts)
		, env_(ORT_LOGGING_LEVEL_WARNING, "DinoEmbedder")
	{
		Ort::SessionOptions so;
		so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

		if (opts_.intraOpThreads > 0)
		{
			so.SetIntraOpNumThreads(opts_.intraOpThreads);
		}

		if (opts_.useCUDA)
		{
			OrtCUDAProviderOptions cuda;
			cuda.device_id = 0;
			so.AppendExecutionProvider_CUDA(cuda);
		}

#ifdef _WIN32
		std::wstring wpath = std::filesystem::path(opts_.modelPath).wstring();
		session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), so);
#else
		session_ = std::make_unique<Ort::Session>(env_, opts_.modelPath.c_str(), so);
#endif

		Ort::AllocatorWithDefaultOptions alloc;

		inputNameStr_ = session_->GetInputNameAllocated(0, alloc).get();
		outputNameStr_ = session_->GetOutputNameAllocated(0, alloc).get();

		// Output dim from model metadata: [N, D]
		auto outInfo = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
		auto shape = outInfo.GetShape();

		if ((shape.size() != 2) || (shape[1] <= 0))
		{
			throw std::runtime_error("DinoEmbedder: unexpected output shape (expected [N, D])");
		}

		dim_ = (int)shape[1];
	
		std::cout << "=== DinoEmbedder ===\n"
			<< "  Model : " << opts_.modelPath << "\n"
			<< "  Input : " << opts_.inputSize << "x" << opts_.inputSize << "\n"
			<< "  Dim   : " << dim_ << "\n"
			<< "  Batch : " << opts_.batchSize << "\n"
			<< "  EP    : " << (opts_.useCUDA ? "CUDA" : "CPU") << "\n";
	}

	int dim() const
	{
		return dim_;
	}

	int batchSize() const
	{
		return opts_.batchSize;
	}

	// 式式式 Single crop 式式式
	std::vector<float> embed(const cv::Mat& bgr)
	{
		std::vector<cv::Mat> one { bgr };
		auto out = embedBatch(one);

		return out.empty() ? std::vector<float>() : out[0];
	}

	// Batch of crops (any count; internally chunked by opts.batchSize)
	std::vector<std::vector<float>> embedBatch(const std::vector<cv::Mat>& crops)
	{
		std::vector<std::vector<float>> result;
		result.reserve(crops.size());

		for (size_t start = 0 ; start < crops.size() ; start += opts_.batchSize)
		{
			size_t end = std::min(crops.size(), start + (size_t)opts_.batchSize);
			runChunk(crops, start, end, result);
		}

		return result;
	}

private:

	DinoEmbedderOptions opts_;
	Ort::Env env_;
	std::unique_ptr<Ort::Session> session_;
	std::string inputNameStr_;
	std::string outputNameStr_;
	int dim_ = 0;

	// BGR crop ⊥ normalized RGB CHW float, written into dst (3*S*S floats)
	void preprocess(const cv::Mat& bgr, float* dst) const
	{
		const int S = opts_.inputSize;

		// ImageNet normalization (DINOv2 uses the standard values), R G B order
		static const float kMean[3] = { 0.485f, 0.456f, 0.406f };
		static const float kStd[3] = { 0.229f, 0.224f, 0.225f };

		cv::Mat img;

		if (bgr.channels() == 4)
		{
			cv::cvtColor(bgr, img, cv::COLOR_BGRA2BGR);
		}
		else if (bgr.channels() == 1)
		{
			cv::cvtColor(bgr, img, cv::COLOR_GRAY2BGR);
		}
		else
		{
			img = bgr;
		}

		// Crown crops (~120px) are upscaled ⊥ cubic; large crops downscaled ⊥ area
		int interp = ((img.cols < S) || (img.rows < S)) ? cv::INTER_CUBIC : cv::INTER_AREA;

		cv::Mat resized;
		cv::resize(img, resized, cv::Size(S, S), 0, 0, interp);

		cv::Mat rgb;
		cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

		cv::Mat f32;
		rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

		// HWC ⊥ CHW with per-channel normalization
		const int plane = S * S;

		for (int y = 0 ; y < S ; ++y)
		{
			const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);

			for (int x = 0 ; x < S ; ++x)
			{
				int idx = y * S + x;

				dst[0 * plane + idx] = (row[x][0] - kMean[0]) / kStd[0];
				dst[1 * plane + idx] = (row[x][1] - kMean[1]) / kStd[1];
				dst[2 * plane + idx] = (row[x][2] - kMean[2]) / kStd[2];
			}
		}
	}

	void runChunk(const std::vector<cv::Mat>& crops, size_t start, size_t end, std::vector<std::vector<float>>& out)
	{
		const int N = (int)(end - start);
		const int S = opts_.inputSize;
		const size_t perImage = (size_t)(3 * S * S);

		std::vector<float> input(perImage * N);

		for (int i = 0 ; i < N ; ++i)
		{
			preprocess(crops[start + i], input.data() + perImage * i);
		}

		std::array<int64, 4> inShape { (int64_t)N, 3, (int64_t)S, (int64_t)S };

		Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		Ort::Value inTensor = Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), inShape.data(), inShape.size());

		const char* inNames[] = { inputNameStr_.c_str() };
		const char* outNames[] = { outputNameStr_.c_str() };

		auto outputs = session_->Run(Ort::RunOptions { nullptr }, inNames, &inTensor, 1, outNames, 1);

		const float* data = outputs[0].GetTensorData<float>();
		auto outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

		if ((outShape.size() != 2) || (outShape[0] != N) || (outShape[1] != dim_))
		{
			throw std::runtime_error("DinoEmbedder: output shape mismatch");
		}

		for (int i = 0 ; i < N ; ++i)
		{
			std::vector<float> v(data + (size_t)i * dim_, data + (size_t)(i + 1) * dim_);

			if (opts_.l2Normalize)
			{
				double norm = 0.0;

				for (float x : v)
				{
					norm += (double)x * x;
				}

				norm = std::sqrt(norm);

				if (norm > 1e-12)
				{
					for (float& x : v)
					{
						x = (float)(x / norm);
					}
				}
			}

			out.push_back(std::move(v));
		}
	}
};