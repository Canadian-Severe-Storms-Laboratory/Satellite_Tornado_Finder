#pragma once
#include "OnnxModel.h"

EXPORT class TornadoPatchPredictor64 : public OnnxModel {

private:
	static constexpr size_t batchSize = 256;
	static constexpr size_t inputHeight = 64;
	static constexpr size_t inputWidth = 64;
	static constexpr size_t inputChannels = 6;
	static constexpr size_t inputSize = inputWidth * inputHeight * inputChannels;
	static constexpr size_t predicitionSize = 1;
	static constexpr size_t stride = 32;
	static constexpr float norm = (float)(1.0 / 255.0);
	static constexpr std::array<float, 6> normMean = { 20.95997633f, 43.91433058f, 35.6977732f, 24.67067952f, 49.58721762f, 45.24592313f };
	static constexpr std::array<float, 6> normStd_1 = { 0.052520446f, 0.045986537f, 0.03649149f, 0.05021587f, 0.043970615f, 0.03363907f };

public:

	TornadoPatchPredictor64(std::string path) : OnnxModel(path, "TornadoPatchPredictor", batchSize) {
		inputShape = { batchSize, inputChannels, inputHeight, inputWidth };
		outputShape = { batchSize, predicitionSize };

		initializeTensors(inputSize, predicitionSize);
	}

	std::vector<unsigned char> analyze(std::vector<unsigned char>& before, std::vector<unsigned char>& after, int width, int height) {

		std::vector<float> predictions;
		predictions.reserve((size_t)(ceil((width / stride)) * ceil((height / stride))));

		auto addPatchToBuffer = [&](int y, int x, size_t& inputIdx) {
			//before
			for (int c = 0; c < 3; c++) {
				for (int i = y; i < y + inputHeight; i++) {
					for (int j = x; j < x + inputWidth; j++) {
						const size_t imgIdx = 3 * (i * (size_t)width + j);
						
						inputBuffer[inputIdx++] = ((float)before[imgIdx + c] - normMean[c]) * normStd_1[c];
					}
				}
			}
			// after
			for (int c = 3; c < 6; c++) {
				for (int i = y; i < y + inputHeight; i++) {
					for (int j = x; j < x + inputWidth; j++) {
						const size_t imgIdx = 3 * (i * (size_t)width + j);

						inputBuffer[inputIdx++] = ((float)after[imgIdx + (c - 3)] - normMean[c]) * normStd_1[c];
					}
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
			for (int i = y + 16; i < y + inputHeight - 16; i++) {
				for (int j = x + 16; j < x + inputWidth - 16; j++) {
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
			//mask[i] = (predMask[i] / (float)mask[i]) < 0.5f ? 0 : 255;
			mask[i] = (unsigned char) round(255.0f * predMask[i] / (float)mask[i]);
		}

		return mask;
	}
	
};