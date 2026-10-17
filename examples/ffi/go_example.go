// Zero-copy Go (`cgo`) integration example for VXG v4.
// Links directly against `libvxg.a` (static) or `-lvxg` (dynamic) and passes Go slice pointers
// straight to `vxg_compress_into` / `vxg_decompress_into` without intermediate allocations.
//
// Run from repository root:
//   cargo build --release
//   CGO_CFLAGS="-I$(pwd)/include" CGO_LDFLAGS="$(pwd)/target/release/libvxg.a -lpthread -ldl -lm" go run examples/ffi/go_example.go

package main

/*
#include "vxg.h"
*/
import "C"
import (
	"bytes"
	"fmt"
	"unsafe"
)

func Compress(src []byte, threads int, checksum bool) ([]byte, error) {
	bound := C.vxg_compress_bound(C.size_t(len(src)), 0)
	if bound < 0 {
		return nil, fmt.Errorf("vxg_compress_bound: %s", C.GoString(C.vxg_error_message(bound)))
	}
	dst := make([]byte, int(bound))
	var srcPtr *C.uint8_t
	if len(src) > 0 {
		srcPtr = (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(src)))
	}
	dstPtr := (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(dst)))
	csum := C.int32_t(0)
	if checksum {
		csum = 1
	}
	written := C.vxg_compress_into(
		srcPtr,
		C.size_t(len(src)),
		dstPtr,
		C.size_t(len(dst)),
		C.size_t(threads),
		0,
		csum,
	)
	if written < 0 {
		return nil, fmt.Errorf("vxg_compress_into: %s", C.GoString(C.vxg_error_message(written)))
	}
	return dst[:int(written)], nil
}

func Decompress(frame []byte, threads int, verifyChecksum bool) ([]byte, error) {
	if len(frame) == 0 {
		return nil, fmt.Errorf("empty frame")
	}
	srcPtr := (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(frame)))
	rawLen := C.vxg_frame_raw_size(srcPtr, C.size_t(len(frame)))
	if rawLen < 0 {
		return nil, fmt.Errorf("vxg_frame_raw_size: %s", C.GoString(C.vxg_error_message(rawLen)))
	}
	dst := make([]byte, int(rawLen))
	var dstPtr *C.uint8_t
	if len(dst) > 0 {
		dstPtr = (*C.uint8_t)(unsafe.Pointer(unsafe.SliceData(dst)))
	}
	verify := C.int32_t(0)
	if verifyChecksum {
		verify = 1
	}
	written := C.vxg_decompress_into(
		srcPtr,
		C.size_t(len(frame)),
		dstPtr,
		C.size_t(len(dst)),
		C.size_t(threads),
		verify,
	)
	if written < 0 {
		return nil, fmt.Errorf("vxg_decompress_into: %s", C.GoString(C.vxg_error_message(written)))
	}
	return dst[:int(written)], nil
}

func main() {
	var buf bytes.Buffer
	for i := 0; i < 1000; i++ {
		fmt.Fprintf(&buf, "{\"ts\":\"2026-10-17T19:00:%02dZ\",\"seq\":%d,\"status\":\"ok\"}\n", i%60, i)
	}
	raw := buf.Bytes()
	compressed, err := Compress(raw, 0, true)
	if err != nil {
		panic(err)
	}
	restored, err := Decompress(compressed, 0, true)
	if err != nil {
		panic(err)
	}
	if !bytes.Equal(raw, restored) {
		panic("roundtrip mismatch")
	}
	fmt.Printf(
		"VXG v%s Go cgo OK: %d B -> %d B (ratio=%.4f)\n",
		C.GoString(C.vxg_version_string()),
		len(raw),
		len(compressed),
		float64(len(compressed))/float64(len(raw)),
	)
}
