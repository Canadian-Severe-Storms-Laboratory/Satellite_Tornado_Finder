#pragma once
#include "CPP_CS_Interop.h"
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <iostream>
#include <string>
#include <array>
#include <vector>


EXPORT class OnnxModel{
	
protected:
	//Onnx session boiler plate
	Ort::Env env;
	Ort::RunOptions runOptions;
	Ort::Session session = Ort::Session(nullptr);
	std::array<const char*, 1> inputNames;
	std::array<const char*, 1> outputNames;
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	int64_t batchSize;
	std::string modelName;

	//input / output tensors of model, must be initialized in subclass
	//These are intermediates that reference underlying containers which have the actual data
	Ort::Value inputTensor = Ort::Value(nullptr);
	Ort::Value outputTensor = Ort::Value(nullptr);

	std::vector<int64_t> inputShape;
	std::vector<int64_t> outputShape;

	//Actual input / output containers
	std::vector<float> inputBuffer;
	std::vector<float> outputBuffer;

private:

	static bool attemptToUseDML(OrtApi const& ortApi, Ort::SessionOptions &sessionOptions) {
		try {
			//setup DirectML (DirectX12) options
			OrtDmlApi const* ortDmlApi = nullptr;
			ortApi.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<void const**>(&ortDmlApi));

			sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
			sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL); // For DML EP
			sessionOptions.DisableMemPattern(); // For DML EP

			Ort::ThrowOnError(ortDmlApi->SessionOptionsAppendExecutionProvider_DML(sessionOptions, /*device index*/ 0));
		}
		catch (Ort::Exception& e) {
			return false;
		}
		
		return true;
	}

	void attemptToUseGPU(OrtApi const& ortApi, Ort::SessionOptions& sessionOptions) {
		usingGPU = attemptToUseDML(ortApi, sessionOptions);
	}


public:

	bool usingGPU;

	OnnxModel(std::string modelPath, std::string modelName, size_t batchSize) {

		std::wstring wModelPath = std::wstring(modelPath.begin(), modelPath.end());

		this->batchSize = batchSize;
		this->modelName = modelName;

		const Ort::Env env = Ort::Env{ ORT_LOGGING_LEVEL_FATAL, "" };

		Ort::SessionOptions sessionOptions = Ort::SessionOptions();

		OrtApi const& ortApi = Ort::GetApi();

		attemptToUseGPU(ortApi, sessionOptions);

		//setting a constant batch size can improve performance
		//ortApi.AddFreeDimensionOverrideByName(sessionOptions, "batch_size", batchSize);

		session = Ort::Session(env, wModelPath.c_str(), sessionOptions);

		//default boiler plate
		const Ort::AllocatorWithDefaultOptions ort_alloc;
		Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
		Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
		inputNames = { inputName.get() };
		outputNames = { outputName.get() };
		(void)inputName.release();
		(void)outputName.release();

	}

	void initializeTensors(const int inputSize, const int predicitionSize) {
		inputBuffer.resize(batchSize * inputSize);
		outputBuffer.resize(batchSize * predicitionSize);

		inputTensor = Ort::Value::CreateTensor<float>(memory_info, inputBuffer.data(), inputBuffer.size(), inputShape.data(), inputShape.size());
		outputTensor = Ort::Value::CreateTensor<float>(memory_info, outputBuffer.data(), outputBuffer.size(), outputShape.data(), outputShape.size());
	}

	void predict() {
		session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
	}
};

