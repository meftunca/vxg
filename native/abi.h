#pragma once
#include <stddef.h>
#include <stdint.h>
// run is synchronous on return, context buffers/pipelines are cached between calls.
// operation 0=encode, 1=decode. See docs/GPU.md for layout and validation contracts.
struct VxgParams { uint32_t total_size, block_size, block_count, operation; };
#define VXG_DECLARE(name) \
extern "C" void* vxg_##name##_create(char* error, size_t cap); \
extern "C" int vxg_##name##_run(void*,const uint8_t*,size_t,uint8_t*,size_t,uint32_t*,size_t,const VxgParams*,double*,char*,size_t); \
extern "C" void vxg_##name##_destroy(void*);
