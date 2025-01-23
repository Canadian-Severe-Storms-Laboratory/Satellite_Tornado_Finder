using System;
using System.Linq;
using System.Runtime.InteropServices;

public partial class FloatVector : global::System.IDisposable, global::System.Collections.IEnumerable, global::System.Collections.Generic.IList<float>
{
    [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
    public static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);

    // cast from C# array to vector
    public static implicit operator FloatVector(float[] inVal)
    {
        return TornadoPredictorOnnx.createVector(inVal, inVal.Length);
    }

    // cast to C# array from vector
    public static unsafe implicit operator float[](FloatVector inVal)
    {
        if (inVal.Count == 0) return [];

        var p = TornadoPredictorOnnx.getVector(inVal);

        IntPtr ptr = SWIGTYPE_p_float.getCPtr(p).Handle;

        float[] ret = new float[inVal.Count];
        Marshal.Copy(ptr, ret, 0, ret.Length);

        return ret;
    }

    public static unsafe implicit operator Span<float>(FloatVector inVal)
    {
        var p = TornadoPredictorOnnx.getVector(inVal);
        float* ptr = (float*)SWIGTYPE_p_float.getCPtr(p).Handle;

        return new Span<float>(ptr, inVal.Count);
    }
}