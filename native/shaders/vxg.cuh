#pragma once
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

#define WG 64u
struct Params { uint total_size; uint block_size; uint block_count; uint operation; };
// Shared algorithm source: compiled as CUDA C++, Metal C++, GLSL, and CPU emulation.
// No byte-expanded u32 representation: device buffers hold packed bytes.
// Host validates all decode commands before dispatch. WG must be a uniform constant.
GPU_INLINE uint vx_min(uint a, uint b) { return a < b ? a : b; }
GPU_INLINE uint in_byte(CONTEXT uint p) { return (INPUT[p >> 2u] >> ((p & 3u)*8u)) & 255u; }
GPU_INLINE uint out_byte(CONTEXT uint p) { return (OUTPUT[p >> 2u] >> ((p & 3u)*8u)) & 255u; }
GPU_INLINE uint work_byte(CONTEXT uint p) { return (WORK[p >> 2u] >> ((p & 3u)*8u)) & 255u; }
GPU_INLINE uint in_u32(CONTEXT uint p) {
    uint s = (p & 3u)*8u;
    uint lo = INPUT[p >> 2u];
    if (s == 0u) { return lo; }
    return (lo >> s) | (INPUT[(p >> 2u)+1u] << (32u-s));
}
GPU_INLINE uint hash8(CONTEXT uint p) {
    uint a = in_u32(CTX p), b = in_u32(CTX p+4u);
    b = (b << 13u) | (b >> 19u);
    return ((a * 0x9e3779b1u) ^ b) >> 21u;
}
// Only lane 0 writes scratch streams, in increasing byte order. A new word is
// initialized on its first byte rather than read-modify-writing uninitialized memory.
GPU_INLINE void work_put(CONTEXT uint p, uint value) {
    uint shift = (p & 3u)*8u;
    uint previous = shift == 0u ? 0u : WORK[p >> 2u];
    WORK[p >> 2u] = (previous & ~(255u << shift)) | ((value & 255u) << shift);
}
GPU_INLINE void work_put32(CONTEXT uint p, uint value) {
    for (uint k=0u;k<4u;++k) { work_put(CTX p+k, value >> (8u*k)); }
}
GPU_INLINE void emit_control(CONTEXT uint base, REF_UINT(cp), uint op, uint lit, uint run) {
    uint token = op | (vx_min(lit,31u) << 3u) | (vx_min(run,255u) << 8u);
    work_put(CTX base+cp, token); work_put(CTX base+cp+1u, token >> 8u); cp += 2u;
    if (lit >= 31u) { work_put32(CTX base+cp, lit); cp += 4u; }
    if (run >= 255u) { work_put32(CTX base+cp, run); cp += 4u; }
}
GPU_INLINE void encode_core(CONTEXT HASH_PARAM uint tid, uint bid, uint total, uint block_size) {
    uint raw_base = bid*block_size;
    uint rn = vx_min(block_size, total-raw_base);
    uint wb = bid*(4u*block_size+256u);
    uint cb = wb+16u, ab = wb+16u+2u*block_size+64u, lb = wb+16u+3u*block_size+128u;
    for (uint h=tid;h<2048u;h+=WG) { HASH_STORE(h, 0xffffffffu); }
    BARRIER();
    // Parallel, deterministic first-anchor discovery. Atomic order cannot change min.
    for (uint p=tid;p+8u<=rn;p+=WG) { HASH_MIN(hash8(CTX raw_base+p), p); }
    BARRIER();
    if (tid == 0u) {
        uint cp=0u, ap=0u, lp=0u, pos=0u, anchor=0u, recent=0u, count=0u;
        bool abandoned=false;
        // Greedy token selection is serialized per block in v0.1, not per file.
        // It is intentionally a correct GPU baseline, not a claim of optimal occupancy.
        while (pos+8u<=rn) {
            uint candidate = HASH_LOAD(hash8(CTX raw_base+pos));
            if (candidate>=pos || in_u32(CTX raw_base+pos)!=in_u32(CTX raw_base+candidate)
                || in_u32(CTX raw_base+pos+4u)!=in_u32(CTX raw_base+candidate+4u)) { ++pos; continue; }
            uint run=8u;
            while (pos+run+4u<=rn && in_u32(CTX raw_base+pos+run)==in_u32(CTX raw_base+candidate+run)) { run+=4u; }
            while (pos+run<rn && in_byte(CTX raw_base+pos+run)==in_byte(CTX raw_base+candidate+run)) { ++run; }
            uint distance=pos-candidate;
            uint op=1u;
            if (distance==1u) { op=3u; } else if (distance==2u) { op=4u; }
            else if (distance==4u) { op=5u; } else if (distance==8u) { op=6u; }
            else if (distance==recent) { op=2u; }
            uint lit=pos-anchor;
            emit_control(CTX cb, cp, op, lit, run);
            if (op==1u) { work_put32(CTX ab+ap, distance); ap+=4u; recent=distance; }
            else if (op>=3u) {
                uint seed=in_u32(CTX raw_base+pos);
                if (op==3u) { seed &=255u; } else if (op==4u) { seed &=65535u; }
                work_put32(CTX ab+ap, seed); ap+=4u;
                if (op==6u) { work_put32(CTX ab+ap, in_u32(CTX raw_base+pos+4u)); ap+=4u; }
            }
            for (uint j=0u;j<lit;++j) { work_put(CTX lb+lp+j, in_byte(CTX raw_base+anchor+j)); }
            lp+=lit; ++count; pos+=run; anchor=pos;
            if (16u+cp+ap+lp>=rn) { abandoned=true; break; }
        }
        uint tail=rn-anchor;
        uint predicted=16u+cp+ap+lp+tail+(tail==0u?0u:(2u+(tail>=31u?4u:0u)));
        uint kind=0u, size=rn;
        if (!abandoned && predicted<rn) {
            if (tail>0u) {
                emit_control(CTX cb, cp, 0u, tail, 0u);
                for (uint j=0u;j<tail;++j) { work_put(CTX lb+lp+j, in_byte(CTX raw_base+anchor+j)); }
                lp+=tail; ++count;
            }
            WORK[wb/4u]=count; WORK[wb/4u+1u]=cp;
            WORK[wb/4u+2u]=ap; WORK[wb/4u+3u]=lp;
            kind=1u; size=16u+cp+ap+lp;
        }
        DESC[4u*bid]=size; DESC[4u*bid+1u]=kind; DESC[4u*bid+2u]=0u; DESC[4u*bid+3u]=0u;
    }
    BARRIER();
    uint used=DESC[4u*bid], kind=DESC[4u*bid+1u];
    uint cp=0u, ap=0u;
    if (kind==1u) { cp=WORK[wb/4u+1u]; ap=WORK[wb/4u+2u]; }
    // Coalesced packed-u32 stores. All slot bytes, including unused bytes, are initialized.
    for (uint word=tid;word<block_size/4u;word+=WG) {
        uint value=0u;
        for (uint k=0u;k<4u;++k) {
            uint j=word*4u+k, b=0u;
            if (j<used) {
                if (kind==0u) { b=in_byte(CTX raw_base+j); }
                else if (j<16u) { b=work_byte(CTX wb+j); }
                else if (j<16u+cp) { b=work_byte(CTX cb+j-16u); }
                else if (j<16u+cp+ap) { b=work_byte(CTX ab+j-16u-cp); }
                else { b=work_byte(CTX lb+j-16u-cp-ap); }
            }
            value |= b << (8u*k);
        }
        OUTPUT[raw_base/4u+word]=value;
    }
}
// source_kind: 0=input literals, 1=periodic LZ seed in already-produced output,
// 2=inline seed. Each lane owns a whole destination word. The two barriers prevent
// a packed-word read/write race when a match seed shares a word with its destination.
GPU_INLINE void segment(CONTEXT uint tid, uint block_base, uint pos, uint len,
    uint source_kind, uint source, uint period, uint seed0, uint seed1) {
    if (len==0u) { return; } // Uniform across this workgroup.
    uint first=pos/4u, words=(pos+len+3u)/4u-first;
    for (uint wave=0u;wave<words;wave+=WG) {
        uint w=wave+tid;
        uint value=0u;
        if (w<words) {
            uint word_index=first+w;
            uint previous=word_index*4u<pos ? OUTPUT[block_base/4u+word_index] : 0u;
            for (uint k=0u;k<4u;++k) {
                uint p=word_index*4u+k, b=0u;
                if (p<pos) { b=(previous>>(8u*k))&255u; }
                else if (p<pos+len) {
                    uint q=p-pos;
                    if (source_kind==0u) { b=in_byte(CTX source+q); }
                    else if (source_kind==1u) { b=out_byte(CTX block_base+source+(q<period?q:q%period)); }
                    else {
                        uint index=q%period;
                        b=index<4u ? ((seed0>>(index*8u))&255u) : ((seed1>>((index-4u)*8u))&255u);
                    }
                }
                value |= b<<(8u*k);
            }
        }
        BARRIER();
        if (w<words) { OUTPUT[block_base/4u+first+w]=value; }
        BARRIER();
    }
}
GPU_INLINE void decode_core(CONTEXT uint tid, uint bid, uint block_size) {
    uint ib=DESC[4u*bid], rn=DESC[4u*bid+2u], kind=DESC[4u*bid+3u];
    uint ob=bid*block_size;
    if (kind==0u) {
        for (uint word=tid;word<(rn+3u)/4u;word+=WG) {
            uint value=0u;
            for (uint k=0u;k<4u;++k) { uint j=word*4u+k; if (j<rn) { value |= in_byte(CTX ib+j)<<(8u*k); } }
            OUTPUT[ob/4u+word]=value;
        }
        return;
    }
    uint count=in_u32(CTX ib), cs=in_u32(CTX ib+4u), ds=in_u32(CTX ib+8u);
    uint ci=ib+16u, ai=ci+cs, li=ai+ds, pos=0u, recent=0u;
    for (uint command=0u;command<count;++command) {
        uint token=in_byte(CTX ci)|(in_byte(CTX ci+1u)<<8u); ci+=2u;
        uint op=token&7u, lit=(token>>3u)&31u, run=token>>8u;
        if (lit==31u) { lit=in_u32(CTX ci); ci+=4u; }
        if (run==255u) { run=in_u32(CTX ci); ci+=4u; }
        uint distance=0u, seed0=0u, seed1=0u;
        if (op==1u) { distance=in_u32(CTX ai); ai+=4u; recent=distance; }
        else if (op==2u) { distance=recent; }
        else if (op>=3u) { seed0=in_u32(CTX ai); ai+=4u; if (op==6u) { seed1=in_u32(CTX ai); ai+=4u; } }
        segment(CTX tid, ob, pos, lit, 0u, li, 1u, 0u, 0u); pos+=lit; li+=lit;
        if (op==1u || op==2u) { segment(CTX tid, ob, pos, run, 1u, pos-distance, distance, 0u, 0u); }
        else if (op>=3u) { segment(CTX tid, ob, pos, run, 2u, 0u, 1u<<(op-3u), seed0, seed1); }
        pos+=run;
    }
}

__global__ void vxg_kernel(const uint* input, uint* output, uint* work, uint* desc, Params p) {
    __shared__ uint table[2048];
    Buffers buf={input,output,work,desc};
    uint bid=blockIdx.x, tid=threadIdx.x;
    if (bid>=p.block_count) { return; }
    if (p.operation==0u) { encode_core(buf,table,tid,bid,p.total_size,p.block_size); }
    else { decode_core(buf,tid,bid,p.block_size); }
}
