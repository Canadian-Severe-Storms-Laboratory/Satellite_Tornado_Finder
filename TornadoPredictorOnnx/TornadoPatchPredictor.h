#pragma once
#include "OnnxModel.h"
#include <immintrin.h>
#include <chrono>

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
			// Precompute FMA constants: result = val * std_1 - (mean * std_1)
			const __m256 vStd0 = _mm256_set1_ps(normStd_1[0]);
			const __m256 vStd1 = _mm256_set1_ps(normStd_1[1]);
			const __m256 vStd2 = _mm256_set1_ps(normStd_1[2]);
			const __m256 vStd3 = _mm256_set1_ps(normStd_1[3]);
			const __m256 vStd4 = _mm256_set1_ps(normStd_1[4]);
			const __m256 vStd5 = _mm256_set1_ps(normStd_1[5]);

			const __m256 vMeanStd0 = _mm256_set1_ps(normMean[0] * normStd_1[0]);
			const __m256 vMeanStd1 = _mm256_set1_ps(normMean[1] * normStd_1[1]);
			const __m256 vMeanStd2 = _mm256_set1_ps(normMean[2] * normStd_1[2]);
			const __m256 vMeanStd3 = _mm256_set1_ps(normMean[3] * normStd_1[3]);
			const __m256 vMeanStd4 = _mm256_set1_ps(normMean[4] * normStd_1[4]);
			const __m256 vMeanStd5 = _mm256_set1_ps(normMean[5] * normStd_1[5]);

			const __m256i zeroVec = _mm256_setzero_si256();

			for (int i = y; i < y + inputHeight; i++) {
				const size_t rowBase = 3 * (i * (size_t)width + x);
				const unsigned char* bRow = &before[rowBase];
				const unsigned char* aRow = &after[rowBase];

				// Process 8 pixels at a time (32 pixels per row = 4 iterations)
				for (int jj = 0; jj < inputWidth; jj += 8) {
					const unsigned char* bPtr = bRow + jj * 3; // 24 bytes for 8 RGB pixels
					const unsigned char* aPtr = aRow + jj * 3;

					// Load 32 bytes (only first 24 matter) from before and after
					// Use unaligned load; extra bytes beyond 24 are harmless as long as memory is accessible
					__m128i bLo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bPtr));      // bytes 0..15
					__m128i bHi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bPtr + 12)); // bytes 12..27 (overlap is fine)
					__m128i aLo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(aPtr));
					__m128i aHi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(aPtr + 12));

					// Deinterleave RGB from 8 pixels (24 bytes)
					// Pixels 0-3 are in bLo (bytes 0..11), pixels 4-7 are in bHi starting at offset 0 (bytes 12..23 of original)
					// Shuffle masks to extract R, G, B from first 4 pixels (12 bytes in a 16-byte register)
					const __m128i shufR = _mm_setr_epi8(0, 3, 6, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
					const __m128i shufG = _mm_setr_epi8(1, 4, 7, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
					const __m128i shufB = _mm_setr_epi8(2, 5, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

					// For the high part (pixels 4-7), same shuffle but source starts at byte 0 of bHi
					// bHi was loaded from bPtr+12, so bHi[0..11] = pixels 4-7

					// Extract R,G,B for before pixels 0-3
					__m128i bR03 = _mm_shuffle_epi8(bLo, shufR); // R0 R1 R2 R3 in bytes 0-3
					__m128i bG03 = _mm_shuffle_epi8(bLo, shufG);
					__m128i bB03 = _mm_shuffle_epi8(bLo, shufB);

					// Extract R,G,B for before pixels 4-7
					__m128i bR47 = _mm_shuffle_epi8(bHi, shufR);
					__m128i bG47 = _mm_shuffle_epi8(bHi, shufG);
					__m128i bB47 = _mm_shuffle_epi8(bHi, shufB);

					// Combine into 8-byte vectors: R0..R7 in low 8 bytes
					__m128i bR = _mm_unpacklo_epi32(bR03, bR47); // R0 R1 R2 R3 R4 R5 R6 R7 in bytes 0-7
					__m128i bG = _mm_unpacklo_epi32(bG03, bG47);
					__m128i bB = _mm_unpacklo_epi32(bB03, bB47);

					// Convert bytes to 32-bit ints then to floats
					__m256 bRf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bR));
					__m256 bGf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bG));
					__m256 bBf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bB));

					// Same for after
					__m128i aR03 = _mm_shuffle_epi8(aLo, shufR);
					__m128i aG03 = _mm_shuffle_epi8(aLo, shufG);
					__m128i aB03 = _mm_shuffle_epi8(aLo, shufB);
					__m128i aR47 = _mm_shuffle_epi8(aHi, shufR);
					__m128i aG47 = _mm_shuffle_epi8(aHi, shufG);
					__m128i aB47 = _mm_shuffle_epi8(aHi, shufB);

					__m128i aR = _mm_unpacklo_epi32(aR03, aR47);
					__m128i aG = _mm_unpacklo_epi32(aG03, aG47);
					__m128i aB = _mm_unpacklo_epi32(aB03, aB47);

					__m256 aRf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(aR));
					__m256 aGf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(aG));
					__m256 aBf = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(aB));

					// Apply normalization: val * std_1 - mean*std_1 using FMA
					__m256 ch0 = _mm256_fmsub_ps(bRf, vStd0, vMeanStd0);
					__m256 ch1 = _mm256_fmsub_ps(bGf, vStd1, vMeanStd1);
					__m256 ch2 = _mm256_fmsub_ps(bBf, vStd2, vMeanStd2);
					__m256 ch3 = _mm256_fmsub_ps(aRf, vStd3, vMeanStd3);
					__m256 ch4 = _mm256_fmsub_ps(aGf, vStd4, vMeanStd4);
					__m256 ch5 = _mm256_fmsub_ps(aBf, vStd5, vMeanStd5);

					// Interleave and store: for each pixel p, store ch0[p], ch1[p], ch2[p], ch3[p], ch4[p], ch5[p]
					// We have 8 pixels, each producing 6 floats = 48 floats total
					// Extract each lane and store sequentially
					alignas(32) float c0[8], c1[8], c2[8], c3[8], c4[8], c5[8];
					_mm256_store_ps(c0, ch0);
					_mm256_store_ps(c1, ch1);
					_mm256_store_ps(c2, ch2);
					_mm256_store_ps(c3, ch3);
					_mm256_store_ps(c4, ch4);
					_mm256_store_ps(c5, ch5);

					for (int p = 0; p < 8; p++) {
						inputBuffer[inputIdx]     = c0[p];
						inputBuffer[inputIdx + 1] = c1[p];
						inputBuffer[inputIdx + 2] = c2[p];
						inputBuffer[inputIdx + 3] = c3[p];
						inputBuffer[inputIdx + 4] = c4[p];
						inputBuffer[inputIdx + 5] = c5[p];
						inputIdx += 6;
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
			//mask[i] = (predMask[i] / (float)mask[i]) < 0.5f ? 0 : 255;
			mask[i] = (unsigned char) round(255.0f * predMask[i] / (float)mask[i]);
		}

		return mask;
	}
	
};