// Runs the exact common GPU algorithm under a CPU SIMT emulator. This does NOT
// compile/execute MSL, GLSL/SPIR-V, CUDA or any Rust source. See VALIDATION.md.
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
using uint=uint32_t;
#ifndef EMU_WG
#define EMU_WG 8
#endif
#define WG uint(EMU_WG)
struct Buffers {
    const std::vector<uint>& input;
    std::vector<uint>& output; std::vector<uint>& work; std::vector<uint>& desc;
    std::barrier<>& sync;
};
static void amin(uint& value,uint x) {
    std::atomic_ref<uint> a(value); uint old=a.load(std::memory_order_relaxed);
    while(old>x&&!a.compare_exchange_weak(old,x,std::memory_order_relaxed)) {}
}
#define GPU_INLINE inline
#define CONTEXT const Buffers& buf,
#define CTX buf,
#define INPUT buf.input
#define OUTPUT buf.output
#define WORK buf.work
#define DESC buf.desc
#define REF_UINT(n) uint& n
#define HASH_PARAM uint* table,
#define HASH_STORE(h,v) table[h]=(v)
#define HASH_MIN(h,v) amin(table[h],v)
#define HASH_LOAD(h) table[h]
#define BARRIER() buf.sync.arrive_and_wait()
#include "../native/shaders/core.gpu"

std::vector<uint8_t> read(const char* path) {
    std::ifstream f(path,std::ios::binary); if(!f)throw std::runtime_error("input file");
    return {std::istreambuf_iterator<char>(f),{}};
}
void write(const char*path,const std::vector<uint8_t>&v) {
    std::ofstream f(path,std::ios::binary); f.write(reinterpret_cast<const char*>(v.data()),v.size());
    if(!f)throw std::runtime_error("output file");
}
uint load32(const std::vector<uint8_t>& v,size_t p) {
    if(p+4>v.size())throw std::runtime_error("truncated");
    return uint(v[p])|(uint(v[p+1])<<8)|(uint(v[p+2])<<16)|(uint(v[p+3])<<24);
}
void put32(std::vector<uint8_t>&v,uint x){for(int i=0;i<4;++i)v.push_back(uint8_t(x>>(i*8)));}
void put64(std::vector<uint8_t>&v,uint64_t x){for(int i=0;i<8;++i)v.push_back(uint8_t(x>>(i*8)));}
std::vector<uint> packed(const std::vector<uint8_t>&v) {
    std::vector<uint> a((v.size()+3)/4+4,0); if(!v.empty())std::memcpy(a.data(),v.data(),v.size()); return a;
}
void launch(Buffers&buf,uint n,uint bs,uint nb,uint op) {
    for(uint bid=0;bid<nb;++bid) {
        uint table[2048]={}; std::vector<std::thread> workers;
        for(uint tid=0;tid<WG;++tid) workers.emplace_back([&,tid,bid]{
            if(op==0)encode_core(buf,table,tid,bid,n,bs);else decode_core(buf,tid,bid,bs);
        });
        for(auto&t:workers)t.join();
    }
}
std::vector<uint8_t> encode(const std::vector<uint8_t>&input,uint bs) {
    uint n=input.size(),nb=(n+bs-1)/bs;
    auto a=packed(input); std::vector<uint> out(size_t(nb)*bs/4+4,0),work(size_t(nb)*(4*bs+256)/4+4,0),desc(size_t(nb)*4,0);
    std::barrier sync(EMU_WG); Buffers b{a,out,work,desc,sync}; launch(b,n,bs,nb,0);
    std::vector<uint8_t> frame={'V','X','G','1',2,0,0,0};put64(frame,n);put32(frame,bs);put32(frame,nb);put64(frame,0);
    uint64_t offset=32+32ull*nb;
    for(uint i=0;i<nb;++i){put64(frame,uint64_t(i)*bs);put64(frame,offset);put32(frame,std::min(bs,n-i*bs));put32(frame,desc[4*i]);put32(frame,0);put32(frame,desc[4*i+1]);offset+=desc[4*i];}
    auto bytes=reinterpret_cast<const uint8_t*>(out.data());
    for(uint i=0;i<nb;++i)frame.insert(frame.end(),bytes+size_t(i)*bs,bytes+size_t(i)*bs+desc[4*i]);
    return frame;
}
std::vector<uint8_t> decode(const std::vector<uint8_t>&frame) {
    // Only accepts verifier-generated/validated frames. Deliberately not an untrusted decoder.
    uint n=load32(frame,8),bs=load32(frame,16),nb=load32(frame,20);
    auto a=packed(frame);std::vector<uint>out((size_t(n)+3)/4+4,0),work(4),desc(size_t(nb)*4);
    for(uint i=0;i<nb;++i){size_t p=32+32ull*i;desc[4*i]=load32(frame,p+8);desc[4*i+1]=load32(frame,p+20);desc[4*i+2]=load32(frame,p+16);desc[4*i+3]=load32(frame,p+28);}
    std::barrier sync(EMU_WG);Buffers b{a,out,work,desc,sync};launch(b,n,bs,nb,1);
    auto ptr=reinterpret_cast<const uint8_t*>(out.data());return{ptr,ptr+n};
}
uint64_t random64(uint64_t& s){s^=s<<13;s^=s>>7;s^=s<<17;return s;}
int main(int argc,char**argv)try {
    if(argc==4){auto v=read(argv[2]);if(std::string(argv[1])=="c")write(argv[3],encode(v,4096));else if(std::string(argv[1])=="d")write(argv[3],decode(v));else throw std::runtime_error("c or d");return 0;}
    uint64_t seed=0x123456789abcdefull;size_t cases=0;
    for(uint n:{1u,8u,31u,63u,257u,1024u,4095u,4096u,4101u,8193u})for(uint period:{1u,2u,3u,4u,7u,8u,17u,63u,65u,127u,0u}){
        std::vector<uint8_t> v(n);std::vector<uint8_t>pat(period?period:1);
        for(auto&x:pat)x=uint8_t(random64(seed));
        for(uint i=0;i<n;++i)v[i]=period?pat[i%period]:uint8_t(random64(seed));
        auto f=encode(v,4096);auto d=decode(f);if(d!=v)throw std::runtime_error("GPU emulation roundtrip mismatch");
        if(f.size()>v.size()+32+32*((v.size()+4095)/4096))throw std::runtime_error("size bound");
        ++cases;
    }
    std::cout<<"PASS: "<<cases<<" exact shared-kernel encode/decode cases, WG="<<EMU_WG<<"\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
