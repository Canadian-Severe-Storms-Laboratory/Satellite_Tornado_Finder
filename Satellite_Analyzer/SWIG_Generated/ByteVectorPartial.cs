using System;
using System.Linq;
using System.Runtime.InteropServices;
using ArcGIS.Core.Data.UtilityNetwork.Trace;
using OpenCvSharp;

public partial class ByteVector : global::System.IDisposable, global::System.Collections.IEnumerable, global::System.Collections.Generic.IList<byte>
{
    [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
    public static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);

    // cast from C# array to vector
    public static implicit operator ByteVector(byte[] inVal)
    {
        return TornadoPredictorOnnx.createVector(inVal, inVal.Length);
    }

    public static unsafe implicit operator ByteVector(Mat mat)
    {
        int length = mat.Height * mat.Width * mat.Channels();

        ByteVector bv = TornadoPredictorOnnx.createByteVector(length);
        var p = TornadoPredictorOnnx.getVector(bv);
        IntPtr ptr = SWIGTYPE_p_unsigned_char.getCPtr(p).Handle;

        var srcSpan = new Span<byte>(mat.DataPointer, length);
        var dstSpan = new Span<byte>(ptr.ToPointer(), length);

        srcSpan.CopyTo(dstSpan);

        return bv;
    }

    // cast to C# array from vector
    public static unsafe implicit operator byte[](ByteVector inVal)
    {
        if (inVal.Count == 0) return [];

        var p = TornadoPredictorOnnx.getVector(inVal);

        IntPtr ptr = SWIGTYPE_p_unsigned_char.getCPtr(p).Handle;

        byte[] ret = new byte[inVal.Count];
        Marshal.Copy(ptr, ret, 0, ret.Length);

        return ret;
    }

    public static unsafe implicit operator Span<byte>(ByteVector inVal)
    {
        var p = TornadoPredictorOnnx.getVector(inVal);
        byte* ptr = (byte*)SWIGTYPE_p_unsigned_char.getCPtr(p).Handle;

        return new Span<byte>(ptr, inVal.Count);
    }

    public static unsafe Mat ToMat(ByteVector bv, Size size)
    {
        Mat mat = new(size, MatType.CV_8UC1);

        var p = TornadoPredictorOnnx.getVector(bv);
        IntPtr ptr = SWIGTYPE_p_unsigned_char.getCPtr(p).Handle;

        var srcSpan = new Span<byte>(ptr.ToPointer(), size.Width * size.Height);
        var dstSpan = new Span<byte>(mat.DataPointer, size.Width * size.Height);

        srcSpan.CopyTo(dstSpan);

        return mat;
    }
}
