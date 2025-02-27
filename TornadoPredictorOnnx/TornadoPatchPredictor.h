#pragma once
#include "OnnxModel.h"

EXPORT class TornadoPatchPredictor : public OnnxModel {

private:
	static constexpr size_t batchSize = 1024;
	static constexpr size_t inputHeight = 32;
	static constexpr size_t inputWidth = 32;
	static constexpr size_t inputChannels = 6;
	static constexpr size_t inputSize = inputWidth * inputHeight * inputChannels;
	static constexpr size_t predicitionSize = 1;
	static constexpr size_t stride = 16;
	static constexpr float norm = (float)(1.0 / 255.0);
	static constexpr std::array<float, 6> normMean = { 16.68440278f, 38.07405572f, 27.43923875f, 22.9263001f, 47.33313112f, 42.46188265f };
	static constexpr std::array<float, 6> normStd_1 = { 0.056274883f, 0.052738555f, 0.04629368f, 0.063687086f, 0.05583825f, 0.04237275f };

public:

	TornadoPatchPredictor(std::string path) : OnnxModel(path, "TornadoPatchPredictor", batchSize) {

		inputShape = { batchSize, inputHeight, inputWidth, inputChannels };
		outputShape = { batchSize, predicitionSize };

		initializeTensors(inputSize, predicitionSize);
	}

	std::vector<unsigned char> analyze(std::vector<unsigned char>& before, std::vector<unsigned char>& after, int width, int height) {

		std::vector<float> predictions;
		predictions.reserve((size_t)(ceil((width / stride)) * ceil((height / stride))));

		auto addPatchToBuffer = [&](int y, int x, size_t& inputIdx) {
			for (int i = y; i < y + 32; i++) {
				for (int j = x; j < x + 32; j++) {
					const size_t imgIdx = 3 * (i * (size_t)width + j);

					inputBuffer[inputIdx]     = ((float)before[imgIdx]     - normMean[0]) * normStd_1[0];
					inputBuffer[inputIdx + 1] = ((float)before[imgIdx + 1] - normMean[1]) * normStd_1[1];
					inputBuffer[inputIdx + 2] = ((float)before[imgIdx + 2] - normMean[2]) * normStd_1[2];

					inputBuffer[inputIdx + 3] = ((float)after[imgIdx] - normMean[3]) * normStd_1[3];
					inputBuffer[inputIdx + 4] = ((float)after[imgIdx + 1] - normMean[4]) * normStd_1[4];
					inputBuffer[inputIdx + 5] = ((float)after[imgIdx + 2] - normMean[5]) * normStd_1[5];

					inputIdx += 6;
				}
			}
		};

		size_t inputIdx = 0;

		for (int i = 0; i < height - inputHeight + stride; i += stride) {
			for (int j = 0; j < width - inputWidth + stride; j += stride) {

				addPatchToBuffer(i, j, inputIdx);

				if (inputIdx < batchSize * inputSize) continue;

				inputIdx = 0;

				predict();

				predictions.insert(predictions.end(), outputBuffer.begin(), outputBuffer.end());
			}
		}

		if (inputIdx > 0) {
			predict();
			predictions.insert(predictions.end(), outputBuffer.begin(), outputBuffer.begin() + inputIdx / inputSize);
		}

		std::vector<float> predMask(height * width, 0);
		std::vector<unsigned char> mask(height * width, 0);

		auto addPredictionToMask = [&](int y, int x, float value) {
			for (int i = y; i < y + 32; i++) {
				for (int j = x; j < x + 32; j++) {
					const size_t imgIdx = i * (size_t)width + j;

					predMask[imgIdx] += value;
					mask[imgIdx] += 1;
				}
			}
		};

		size_t predIdx = 0;

		for (int i = 0; i < height - inputHeight + stride; i += stride) {
			for (int j = 0; j < width - inputWidth + stride; j += stride) {
				addPredictionToMask(i, j, predictions[predIdx++]);
			}
		}

		for (int i = 0; i < height * width; i++) {
			mask[i] = (predMask[i] / (float)mask[i]) < 0.7f ? 0 : 255;
		}

		return mask;
	}
	
};