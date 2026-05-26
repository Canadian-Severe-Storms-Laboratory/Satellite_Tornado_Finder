#pragma once
#include "OnnxModel.h"

EXPORT class TornadoPatchPredictor64_256 : public OnnxModel {

private:
	static constexpr size_t batchSize = 256;
	static constexpr size_t inputHeight = 64;
	static constexpr size_t inputWidth = 64;
	static constexpr size_t inputChannels = 12;
	static constexpr size_t inputSize = inputWidth * inputHeight * inputChannels;
	static constexpr size_t predicitionSize = 1;
	static constexpr size_t stride = 16;
	static constexpr float norm = (float)(1.0 / 255.0);
	static constexpr std::array<float, 6> normMean = { 18.68838881f, 41.17134002f, 31.97962192f, 19.05978276f, 43.07571633f, 35.68422487f };
	static constexpr std::array<float, 6> normStd_1 = { 1.0f / 15.62747299f, 1.0f / 18.76652384f, 1.0f / 23.22864492f, 1.0f / 15.70466111f, 1.0f / 19.25905989f, 1.0f / 24.46462193f };

	//compute mean pixel in 4x4 grid
	float computeMeanPixel4x4(const std::vector<unsigned char>& img, int width, int x, int y, int c) {
		float sum = 0;

		for (int i = y; i < y + 4; i++) {
			for (int j = x; j < x + 4; j++) {
				sum += img[3 * (i * (size_t)width + j) + c];
			}
		}

		return sum / 16.0f;
	}


public:

	TornadoPatchPredictor64_256(std::string path) : OnnxModel(path, "TornadoPatchPredictor64_256", batchSize) {
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
				for (int i = y + 2*inputHeight - inputHeight / 2; i < y + 2*inputHeight + inputHeight / 2; i++) {
					for (int j = x + 2*inputWidth - inputWidth / 2; j < x + 2*inputWidth + inputWidth / 2; j++) {
						const size_t imgIdx = 3 * (i * (size_t)width + j);
						
						inputBuffer[inputIdx++] = ((float)before[imgIdx + c] - normMean[c]) * normStd_1[c];
					}
				}
			}
			// after
			for (int c = 3; c < 6; c++) {
				for (int i = y + 2*inputHeight - inputHeight / 2; i < y + 2*inputHeight + inputHeight / 2; i++) {
					for (int j = x + 2*inputWidth - inputWidth / 2; j < x + 2*inputWidth + inputWidth / 2; j++) {
						const size_t imgIdx = 3 * (i * (size_t)width + j);

						inputBuffer[inputIdx++] = ((float)after[imgIdx + (c - 3)] - normMean[c]) * normStd_1[c];
					}
				}
			}

			//before large 4x4 mean
			for (int c = 0; c < 3; c++) {
				for (int i = y; i < y + 4 * inputHeight; i += 4) {
					for (int j = x; j < x + 4 * inputWidth; j += 4) {
						inputBuffer[inputIdx++] = (computeMeanPixel4x4(before, width, j, i, c) - normMean[c]) * normStd_1[c];
					}
				}
			}

			//after large 4x4 mean
			for (int c = 3; c < 6; c++) {
				for (int i = y; i < y + 4 * inputHeight; i += 4) {
					for (int j = x; j < x + 4 * inputWidth; j += 4) {
						inputBuffer[inputIdx++] = (computeMeanPixel4x4(after, width, j, i, c - 3) - normMean[c]) * normStd_1[c];
					}
				}
			}

		};

		size_t inputIdx = 0;

		for (int i = 0; i < height - 4*inputHeight + stride; i += stride) {
			for (int j = 0; j < width - 4*inputWidth + stride; j += stride) {

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
			for (int i = y + 2 * inputHeight - inputHeight / 2; i < y + 2 * inputHeight + inputHeight / 2; i++) {
				for (int j = x + 2 * inputWidth - inputWidth / 2; j < x + 2 * inputWidth + inputWidth / 2; j++) {
					const size_t imgIdx = i * (size_t)width + j;

					predMask[imgIdx] += value;
					mask[imgIdx] += 1;
				}
			}
		};

		size_t predIdx = 0;

		for (int i = 0; i < height - 4 * inputHeight + stride; i += stride) {
			for (int j = 0; j < width - 4 * inputWidth + stride; j += stride) {
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