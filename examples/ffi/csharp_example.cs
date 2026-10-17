// Zero-allocation .NET 8+ (`Span<byte>` / `ReadOnlySpan<byte>`) P/Invoke binding for VXG v4.
// Pins caller spans in place without intermediate managed heap copies.

using System;
using System.Runtime.InteropServices;

namespace VxgCodec;

public static unsafe partial class Vxg
{
    private const string LibName = "vxg";

    [LibraryImport(LibName)]
    private static partial nint vxg_compress_bound(nuint rawLen, nuint blockSize);

    [LibraryImport(LibName)]
    private static partial nint vxg_frame_raw_size(byte* src, nuint srcLen);

    [LibraryImport(LibName)]
    private static partial nint vxg_compress_into(
        byte* src, nuint srcLen,
        byte* dst, nuint dstCap,
        nuint threads, nuint blockSize, int checksum);

    [LibraryImport(LibName)]
    private static partial nint vxg_decompress_into(
        byte* src, nuint srcLen,
        byte* dst, nuint dstCap,
        nuint threads, int verifyChecksum);

    [LibraryImport(LibName)]
    private static partial nint vxg_extract_block_into(
        byte* src, nuint srcLen,
        nuint blockIndex,
        byte* dst, nuint dstCap,
        int verifyChecksum);

    public static int MaxCompressedSize(int rawLen, int blockSize = 0)
    {
        nint res = vxg_compress_bound((nuint)rawLen, (nuint)blockSize);
        if (res < 0) throw new InvalidOperationException($"vxg_compress_bound failed ({res})");
        return (int)res;
    }

    public static int CompressInto(ReadOnlySpan<byte> src, Span<byte> dst, int threads = 0, int blockSize = 0, bool checksum = true)
    {
        fixed (byte* srcPtr = src)
        fixed (byte* dstPtr = dst)
        {
            nint written = vxg_compress_into(
                srcPtr, (nuint)src.Length,
                dstPtr, (nuint)dst.Length,
                (nuint)threads, (nuint)blockSize, checksum ? 1 : 0);
            if (written < 0) throw new InvalidOperationException($"vxg_compress_into failed ({written})");
            return (int)written;
        }
    }

    public static int FrameRawSize(ReadOnlySpan<byte> frame)
    {
        fixed (byte* srcPtr = frame)
        {
            nint size = vxg_frame_raw_size(srcPtr, (nuint)frame.Length);
            if (size < 0) throw new InvalidOperationException($"vxg_frame_raw_size failed ({size})");
            return (int)size;
        }
    }

    public static int DecompressInto(ReadOnlySpan<byte> frame, Span<byte> dst, int threads = 0, bool verifyChecksum = true)
    {
        fixed (byte* srcPtr = frame)
        fixed (byte* dstPtr = dst)
        {
            nint written = vxg_decompress_into(
                srcPtr, (nuint)frame.Length,
                dstPtr, (nuint)dst.Length,
                (nuint)threads, verifyChecksum ? 1 : 0);
            if (written < 0) throw new InvalidOperationException($"vxg_decompress_into failed ({written})");
            return (int)written;
        }
    }
}
