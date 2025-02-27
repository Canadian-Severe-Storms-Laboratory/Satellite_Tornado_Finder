#pragma once
#include <vector>
#include <string>
#include <span>

#if _NOEXPORT 
#define EXPORT 
#else
#define EXPORT __declspec(dllexport)
#endif

EXPORT std::span<unsigned char> createSpan(unsigned char* spanArray, int size) {
	return std::span<unsigned char>(spanArray, size);
}

EXPORT std::span<float> createSpan(float* spanArray, int size) {
	return std::span<float>(spanArray, size);
}

EXPORT size_t getSpanSize(std::span<unsigned char>& span) {
	return span.size();
}

EXPORT size_t getSpanSize(std::span<float>& span) {
	return span.size();
}

EXPORT unsigned char* getSpanPtr(std::span<unsigned char>& span) {
	return span.data();
}

EXPORT float* getSpanPtr(std::span<float>& span) {
	return span.data();
}

EXPORT void freeSpan(std::span<unsigned char>& span) {
	delete[] span.data();
}

EXPORT void freeSpan(std::span<float>& span) {
	delete[] span.data();
}

EXPORT std::vector<unsigned char> createVector(unsigned char* sourceArray, int size) {
	std::vector<unsigned char> result(sourceArray, sourceArray + size);

	return result;
}

EXPORT std::vector<float> createVector(float* sourceArray, int size) {
	std::vector<float> result(sourceArray, sourceArray + size);

	return result;
}

EXPORT std::vector<unsigned char> createByteVector(int size) {
	std::vector<unsigned char> result(size);

	return result;
}

EXPORT std::vector<float> createFloatVector(int size) {
	std::vector<float> result(size);

	return result;
}

EXPORT unsigned char* getVector(std::vector<unsigned char>& sourceVector) {
	return sourceVector.data();
}

EXPORT float* getVector(std::vector<float>& sourceVector) {
	return sourceVector.data();
}