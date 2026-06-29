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
	static constexpr size_t stride = 32;
	static constexpr float norm = (float)(1.0 / 255.0);
	//static constexpr std::array<float, 6> normMean = { 18.68838881f, 41.17134002f, 31.97962192f, 19.05978276f, 43.07571633f, 35.68422487f };
	// Constexpr precomputation of normalization constants
	static constexpr std::array<float, 6> normStdInv = { 
		1.0f / 15.62747299f, 1.0f / 18.76652384f, 1.0f / 23.22864492f, 
		1.0f / 15.70466111f, 1.0f / 19.25905989f, 1.0f / 24.46462193f 
	};
	static constexpr std::array<float, 6> normMeanTimesStdInv = {
		18.68838881f / 15.62747299f, 41.17134002f / 18.76652384f, 31.97962192f / 23.22864492f,
		19.05978276f / 15.70466111f, 43.07571633f / 19.25905989f, 35.68422487f / 24.46462193f
	};
	static constexpr size_t channelPlaneSize = inputHeight * inputWidth; // 4096

	/*float computeMeanPixel4x4(unsigned char* img, int width, int x, int y, int c) {
		// Shuffle mask to extract channel c from 4 consecutive RGB pixels (12 bytes)
		// Pixel layout: [R G B R G B R G B R G B ...]
		// We want bytes at offsets: c, c+3, c+6, c+9
		const __m128i shufMask = _mm_setr_epi8(
			(char)c, (char)(c + 3), (char)(c + 6), (char)(c + 9),
			-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
		);

		const unsigned char* row0 = img + 3 * (y * (size_t)width + x);
		const unsigned char* row1 = img + 3 * ((y + 1) * (size_t)width + x);
		const unsigned char* row2 = img + 3 * ((y + 2) * (size_t)width + x);
		const unsigned char* row3 = img + 3 * ((y + 3) * (size_t)width + x);

		// Load 16 bytes from each row (only first 12 matter, rest masked by shuffle)
		__m128i r0 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row0)), shufMask);
		__m128i r1 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row1)), shufMask);
		__m128i r2 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row2)), shufMask);
		__m128i r3 = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row3)), shufMask);

		// Combine: pack 4 bytes from each row into a single 16-byte vector
		__m128i combined = _mm_or_si128(
			_mm_or_si128(r0, _mm_slli_si128(r1, 4)),
			_mm_or_si128(_mm_slli_si128(r2, 8), _mm_slli_si128(r3, 12))
		);

		// Sum all 16 bytes using SAD against zero
		__m128i zero = _mm_setzero_si128();
		__m128i sad = _mm_sad_epu8(combined, zero);

		// SAD produces two 64-bit sums (low 8 bytes and high 8 bytes)
		int sum = _mm_extract_epi16(sad, 0) + _mm_extract_epi16(sad, 4);

		constexpr float scale = 1.0f / 16.0f; // 16 pixels in 4x4 block

		return (float)sum * scale;
	}*/

	// Compute mean of 3 channels simultaneously for a 4x4 block
	void computeMeanPixel4x4_3ch(const unsigned char* img, int imgWidth, int x, int y, float& outR, float& outG, float& outB) {
		// Shuffle masks to deinterleave RGB from 4 pixels (12 bytes) into separate channels
		// Input layout per row: R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 [garbage bytes 12-15]
		const __m128i shufR = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
		const __m128i shufG = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
		const __m128i shufB = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
		const __m128i zero = _mm_setzero_si128();

		__m128i accumR = zero;
		__m128i accumG = zero;
		__m128i accumB = zero;

		for (int dy = 0; dy < 4; dy++) {
			const unsigned char* row = img + 3 * ((y + dy) * (size_t)imgWidth + x);
			__m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row));

			// Extract each channel's 4 bytes into low positions, rest zeroed
			__m128i r = _mm_shuffle_epi8(data, shufR);
			__m128i g = _mm_shuffle_epi8(data, shufG);
			__m128i b = _mm_shuffle_epi8(data, shufB);

			// SAD against zero sums the bytes horizontally into a 16-bit result
			accumR = _mm_add_epi16(accumR, _mm_sad_epu8(r, zero));
			accumG = _mm_add_epi16(accumG, _mm_sad_epu8(g, zero));
			accumB = _mm_add_epi16(accumB, _mm_sad_epu8(b, zero));
		}

		// All sums are in the lowest 16-bit lane of the 128-bit accumulators
		constexpr float scale = 1.0f / 16.0f;
		outR = (float)_mm_extract_epi16(accumR, 0) * scale;
		outG = (float)_mm_extract_epi16(accumG, 0) * scale;
		outB = (float)_mm_extract_epi16(accumB, 0) * scale;
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
			const size_t patchBase = inputIdx;

			// Process all 3 channels of a 64x64 region from an RGB image simultaneously
			auto processSmallPatch3Ch = [&](const unsigned char* img, int normBase) {
				const __m256 vStdInv0 = _mm256_set1_ps(normStdInv[normBase + 0]);
				const __m256 vStdInv1 = _mm256_set1_ps(normStdInv[normBase + 1]);
				const __m256 vStdInv2 = _mm256_set1_ps(normStdInv[normBase + 2]);
				const __m256 vMeanStd0 = _mm256_set1_ps(normMeanTimesStdInv[normBase + 0]);
				const __m256 vMeanStd1 = _mm256_set1_ps(normMeanTimesStdInv[normBase + 1]);
				const __m256 vMeanStd2 = _mm256_set1_ps(normMeanTimesStdInv[normBase + 2]);

				const size_t ch0Base = patchBase + (size_t)(normBase + 0) * channelPlaneSize;
				const size_t ch1Base = patchBase + (size_t)(normBase + 1) * channelPlaneSize;
				const size_t ch2Base = patchBase + (size_t)(normBase + 2) * channelPlaneSize;

				const int startY = y + 2 * (int)inputHeight - (int)inputHeight / 2;
				const int startX = x + 2 * (int)inputWidth - (int)inputWidth / 2;

				size_t pixelOffset = 0;
				for (int i = startY; i < startY + (int)inputHeight; i++) {
					const unsigned char* row = img + 3 * (i * (size_t)width + startX);
					for (int j = 0; j < (int)inputWidth; j += 8) {
						const unsigned char* p = row + j * 3;
						alignas(32) float vR[8], vG[8], vB[8];
						for (int k = 0; k < 8; k++) {
							vR[k] = (float)p[k * 3 + 0];
							vG[k] = (float)p[k * 3 + 1];
							vB[k] = (float)p[k * 3 + 2];
						}

						__m256 r = _mm256_load_ps(vR);
						__m256 g = _mm256_load_ps(vG);
						__m256 b = _mm256_load_ps(vB);

						__m256 rNorm = _mm256_fmsub_ps(r, vStdInv0, vMeanStd0);
						__m256 gNorm = _mm256_fmsub_ps(g, vStdInv1, vMeanStd1);
						__m256 bNorm = _mm256_fmsub_ps(b, vStdInv2, vMeanStd2);

						_mm256_storeu_ps(&inputBuffer[ch0Base + pixelOffset], rNorm);
						_mm256_storeu_ps(&inputBuffer[ch1Base + pixelOffset], gNorm);
						_mm256_storeu_ps(&inputBuffer[ch2Base + pixelOffset], bNorm);

						pixelOffset += 8;
					}
				}
			};

			// Process all 3 channels of 4x4 mean downsampled region simultaneously
			auto processLargePatch3Ch = [&](const unsigned char* img, int normBase) {
				const __m256 vStdInv0 = _mm256_set1_ps(normStdInv[normBase + 0]);
				const __m256 vStdInv1 = _mm256_set1_ps(normStdInv[normBase + 1]);
				const __m256 vStdInv2 = _mm256_set1_ps(normStdInv[normBase + 2]);
				const __m256 vMeanStd0 = _mm256_set1_ps(normMeanTimesStdInv[normBase + 0]);
				const __m256 vMeanStd1 = _mm256_set1_ps(normMeanTimesStdInv[normBase + 1]);
				const __m256 vMeanStd2 = _mm256_set1_ps(normMeanTimesStdInv[normBase + 2]);

				// Large patch channels come after the 6 small patch channels
				const size_t ch0Base = patchBase + (size_t)(normBase + 6) * channelPlaneSize;
				const size_t ch1Base = patchBase + (size_t)(normBase + 7) * channelPlaneSize;
				const size_t ch2Base = patchBase + (size_t)(normBase + 8) * channelPlaneSize;

				size_t pixelOffset = 0;
				for (int i = y; i < y + 4 * (int)inputHeight; i += 4) {
					for (int jj = 0; jj < (int)(4 * inputWidth); jj += 32) {
						alignas(32) float vR[8], vG[8], vB[8];
						for (int k = 0; k < 8; k++) {
							computeMeanPixel4x4_3ch(img, width, x + jj + k * 4, i, vR[k], vG[k], vB[k]);
						}

						__m256 r = _mm256_load_ps(vR);
						__m256 g = _mm256_load_ps(vG);
						__m256 b = _mm256_load_ps(vB);

						__m256 rNorm = _mm256_fmsub_ps(r, vStdInv0, vMeanStd0);
						__m256 gNorm = _mm256_fmsub_ps(g, vStdInv1, vMeanStd1);
						__m256 bNorm = _mm256_fmsub_ps(b, vStdInv2, vMeanStd2);

						_mm256_storeu_ps(&inputBuffer[ch0Base + pixelOffset], rNorm);
						_mm256_storeu_ps(&inputBuffer[ch1Base + pixelOffset], gNorm);
						_mm256_storeu_ps(&inputBuffer[ch2Base + pixelOffset], bNorm);

						pixelOffset += 8;
					}
				}
			};

			// before image: small patch channels 0,1,2
			processSmallPatch3Ch(before.data(), 0);
			// after image: small patch channels 3,4,5
			processSmallPatch3Ch(after.data(), 3);

			// before large 4x4 mean: channels 6,7,8
			processLargePatch3Ch(before.data(), 0);
			// after large 4x4 mean: channels 9,10,11
			processLargePatch3Ch(after.data(), 3);

			inputIdx += inputSize;
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