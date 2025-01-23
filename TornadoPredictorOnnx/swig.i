%module TornadoPredictorOnnx //name of C# class to access C++ code through

//code included in Generated C++ wrapper
%{
#include "CPP_CS_Interop.h"
#include "OnnxModel.h"
#include "TornadoPatchPredictor.h" //C++ headers
%}

%include "typemaps.i"

// make partial c# classes for wrapper code
%typemap(csclassmodifiers) SWIGTYPE "public partial class"

// make a partial c# class for wrapper code of specific type
%typemap(csclassmodifiers) std::span<char>* "public partial class"
%typemap(csclassmodifiers) std::span<float>* "public partial class"

//build in templates for standard types
%include "std_string.i"
%include "std_vector.i"
%include "carrays.i"
%template(ByteVector) std::vector<unsigned char>;
%template(FloatVector) std::vector<float>;

//handling arrays as pointers rather than wrapped type
%include "arrays_csharp.i"
%apply unsigned char FIXED[] {unsigned char *sourceArray} 
%apply unsigned char* {unsigned char *spanArray} 
%apply float FIXED[] {float *sourceArray} 
%apply float* {float *spanArray} 

//making methods unsafe as to allow pointers
%csmethodmodifiers createVector "public unsafe";
%csmethodmodifiers createSpan "public unsafe";

//indicate operating system
%include <windows.i>

//define header files to Generate Wrapper for
%include "CPP_CS_Interop.h"
%include "OnnxModel.h"
%include "TornadoPatchPredictor.h"

%begin %{
#ifndef _NOEXPORT
%}

%init %{
#endif
%}