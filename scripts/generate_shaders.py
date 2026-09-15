#!/usr/bin/env python3
"""Generate real MSL, GLSL and CUDA sources from the reviewed shared algorithm.
No shader compiler is required for generation. Compilation happens in build.rs.
"""
from pathlib import Path
import argparse
root = Path(__file__).resolve().parents[1]
core = (root / "native/shaders/core.gpu").read_text()
common = """
#define WG 64u
struct Params { uint total_size; uint block_size; uint block_count; uint operation; };
"""
metal = r'''#include <metal_stdlib>
using namespace metal;
#define GPU_INLINE inline
#define CONTEXT thread const Buffers& buf,
#define CTX buf,
#define INPUT buf.input
#define OUTPUT buf.output
#define WORK buf.work
#define DESC buf.desc
#define REF_UINT(n) thread uint& n
#define HASH_PARAM threadgroup atomic_uint* table,
#define HASH_STORE(h,v) atomic_store_explicit(&table[h],v,memory_order_relaxed)
#define HASH_MIN(h,v) atomic_fetch_min_explicit(&table[h],v,memory_order_relaxed)
#define HASH_LOAD(h) atomic_load_explicit(&table[h],memory_order_relaxed)
#define BARRIER() threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)
struct Buffers { const device uint* input; device uint* output; device uint* work; device uint* desc; };
''' + common + core + r'''
kernel void vxg_kernel(const device uint* input [[buffer(0)]], device uint* output [[buffer(1)]],
 device uint* work [[buffer(2)]], device uint* desc [[buffer(3)]], constant Params& p [[buffer(4)]],
 uint tid [[thread_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint table[2048];
    Buffers buf={input,output,work,desc};
    uint bid=group.x;
    if (bid>=p.block_count) { return; }
    if (p.operation==0u) { encode_core(buf,table,tid,bid,p.total_size,p.block_size); }
    else { decode_core(buf,tid,bid,p.block_size); }
}
'''
cuda = r'''#pragma once
#include <cuda_runtime.h>
typedef unsigned int uint;
#define GPU_INLINE __device__ __forceinline__
#define CONTEXT const Buffers& buf,
#define CTX buf,
#define INPUT buf.input
#define OUTPUT buf.output
#define WORK buf.work
#define DESC buf.desc
#define REF_UINT(n) uint& n
#define HASH_PARAM uint* table,
#define HASH_STORE(h,v) table[h]=(v)
#define HASH_MIN(h,v) atomicMin(&table[h],v)
#define HASH_LOAD(h) table[h]
#define BARRIER() __syncthreads()
struct Buffers { const uint* input; uint* output; uint* work; uint* desc; };
''' + common + core + r'''
__global__ void vxg_kernel(const uint* input, uint* output, uint* work, uint* desc, Params p) {
    __shared__ uint table[2048];
    Buffers buf={input,output,work,desc};
    uint bid=blockIdx.x, tid=threadIdx.x;
    if (bid>=p.block_count) { return; }
    if (p.operation==0u) { encode_core(buf,table,tid,bid,p.total_size,p.block_size); }
    else { decode_core(buf,tid,bid,p.block_size); }
}
'''
glsl = r'''#version 450
layout(local_size_x=64,local_size_y=1,local_size_z=1) in;
layout(set=0,binding=0,std430) readonly buffer InBuffer { uint input_data[]; };
layout(set=0,binding=1,std430) coherent buffer OutBuffer { uint output_data[]; };
layout(set=0,binding=2,std430) coherent buffer WorkBuffer { uint work_data[]; };
layout(set=0,binding=3,std430) coherent buffer DescBuffer { uint descriptor_data[]; };
layout(push_constant) uniform Push { uint total_size; uint block_size; uint block_count; uint operation; } p;
shared uint table[2048];
#define GPU_INLINE
#define CONTEXT
#define CTX
#define INPUT input_data
#define OUTPUT output_data
#define WORK work_data
#define DESC descriptor_data
#define REF_UINT(n) inout uint n
#define HASH_PARAM
#define HASH_STORE(h,v) table[h]=(v)
#define HASH_MIN(h,v) atomicMin(table[h],v)
#define HASH_LOAD(h) table[h]
#define BARRIER() memoryBarrierBuffer(); memoryBarrierShared(); barrier()
''' + common + core + r'''
void main() {
    uint bid=gl_WorkGroupID.x, tid=gl_LocalInvocationIndex;
    if (bid>=p.block_count) { return; }
    if (p.operation==0u) { encode_core(tid,bid,p.total_size,p.block_size); }
    else { decode_core(tid,bid,p.block_size); }
}
'''
files = {"vxg.metal": metal, "vxg.cuh": cuda, "vxg.comp": glsl}
parser=argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args=parser.parse_args()
for name, data in files.items():
    path=root/"native/shaders"/name
    if args.check:
        if not path.exists() or path.read_text()!=data: raise SystemExit(f"stale shader: {path}")
    else: path.write_text(data)
print("shader sources consistent" if args.check else "generated Metal, CUDA, GLSL sources")
