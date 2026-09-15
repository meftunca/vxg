// Actual installed C-library sanity benchmark. NOT a VXG benchmark.
// No Rust, GPU, Python-call-loop, or emulator performance is measured here.
#include <lz4.h>
#include <zstd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using Bytes=std::vector<char>;
using Clock=std::chrono::steady_clock;
constexpr size_t BS=65536, N=4*1024*1024;
struct Chunk{Bytes bytes;size_t raw_size;bool raw;};
volatile uint64_t observation=0;
std::vector<Chunk> encode(const Bytes& in,bool zstd,int level){
    std::vector<Chunk> result;
    std::vector<uint64_t> state((LZ4_sizeofState()+7)/8);
    ZSTD_CCtx*ctx=zstd?ZSTD_createCCtx():nullptr;
    if(zstd&&!ctx)throw std::runtime_error("ZSTD context");
    for(size_t off=0;off<in.size();off+=BS){
        size_t n=std::min(BS,in.size()-off);
        size_t cap=zstd?ZSTD_compressBound(n):size_t(LZ4_compressBound(int(n)));
        Bytes out(cap);
        size_t used;
        if(zstd){used=ZSTD_compressCCtx(ctx,out.data(),cap,in.data()+off,n,level);if(ZSTD_isError(used))throw std::runtime_error(ZSTD_getErrorName(used));}
        else{int v=LZ4_compress_fast_extState(state.data(),in.data()+off,out.data(),int(n),int(cap),level);if(v<=0)throw std::runtime_error("LZ4 encode");used=size_t(v);}
        bool raw=used>=n;
        if(raw)out.assign(in.begin()+off,in.begin()+off+n);else out.resize(used);
        result.push_back({std::move(out),n,raw});
    }
    if(ctx)ZSTD_freeCCtx(ctx);return result;
}
void decode(const std::vector<Chunk>&chunks,Bytes&out,bool zstd){
    ZSTD_DCtx*ctx=zstd?ZSTD_createDCtx():nullptr;
    if(zstd&&!ctx)throw std::runtime_error("ZSTD context");
    size_t off=0;
    for(const auto&c:chunks){
        if(c.raw)std::copy(c.bytes.begin(),c.bytes.end(),out.begin()+off);
        else{
            size_t n;
            if(zstd){n=ZSTD_decompressDCtx(ctx,out.data()+off,c.raw_size,c.bytes.data(),c.bytes.size());if(ZSTD_isError(n))throw std::runtime_error(ZSTD_getErrorName(n));}
            else{int v=LZ4_decompress_safe(c.bytes.data(),out.data()+off,int(c.bytes.size()),int(c.raw_size));if(v<0)throw std::runtime_error("LZ4 decode");n=size_t(v);}
            if(n!=c.raw_size)throw std::runtime_error("decoded length");
        }
        off+=c.raw_size;
    }
    if(ctx)ZSTD_freeDCtx(ctx);
}
template<class F>std::vector<double> measure(F f){
    f();f();std::vector<double> ns;
    for(int sample=0;sample<5;++sample){
        auto start=Clock::now();uint64_t count=0;double elapsed;
        do{f();++count;elapsed=std::chrono::duration<double>(Clock::now()-start).count();}while(elapsed<0.025);
        ns.push_back(elapsed*1e9/double(count));
    }
    std::sort(ns.begin(),ns.end());return ns;
}
int main()try{
    std::cerr<<"REFERENCE C LIBRARIES ONLY: liblz4 "<<LZ4_versionString()<<", libzstd "<<ZSTD_versionString()<<"\n";
    std::cout<<"scope,dataset,codec,operation,raw_bytes,encoded_bytes,ratio,median_ns,min_ns,max_ns,GBps,threads,block_size,checksum,samples\n";
    for(const std::string name:{"zeros","period37","json","random","low_entropy_random"}){
        Bytes in(N),out(N);uint64_t s=0x123456789abcdefull;
        const std::string line="{\"tenant\":17,\"event\":\"checklist.completed\",\"ok\":true,\"id\":1001}\n";
        for(size_t i=0;i<N;++i){s^=s<<13;s^=s>>7;s^=s<<17;
            in[i]=name=="zeros"?0:name=="period37"?char((i%37)*7):name=="json"?line[i%line.size()]:name=="random"?char(s):char(s&15);}
        for(int config=0;config<5;++config){
            bool zs=config>=2;int level=config==0?1:config==1?4:config==2?-3:config==3?1:3;
            std::string codec=std::string(zs?"zstd-l":"lz4-a")+std::to_string(level);
            auto chunks=encode(in,zs,level);decode(chunks,out,zs);if(out!=in)throw std::runtime_error("roundtrip");
            size_t encoded=32+chunks.size()*32;for(auto&c:chunks)encoded+=c.bytes.size();
            auto en=measure([&]{auto c=encode(in,zs,level);observation=c.front().bytes.size();});
            auto de=measure([&]{decode(chunks,out,zs);observation=uint8_t(out[0]);});
            if(out!=in)throw std::runtime_error("post-timing correctness");
            for(int operation=0;operation<2;++operation){auto&a=operation?de:en;
                std::cout<<"C_ONLY_NOT_VXG,"<<name<<','<<codec<<','<<(operation?"decompress":"compress")<<','<<N<<','<<encoded<<','<<double(encoded)/N<<','<<a[2]<<','<<a[0]<<','<<a[4]<<','<<double(N)/a[2]<<",1,"<<BS<<",off,5\n";
            }
        }
    }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
