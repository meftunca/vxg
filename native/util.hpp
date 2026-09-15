#pragma once
#include "abi.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <limits>
inline void vxg_message(char* dst,size_t cap,const std::string& message) {
    if(dst && cap) std::snprintf(dst,cap,"%s",message.c_str());
}
inline size_t vxg_align(size_t n,size_t a=16) {
    if(n>std::numeric_limits<size_t>::max()-(a-1))throw std::runtime_error("size overflow");
    return std::max(a,(n+a-1)&~(a-1));
}
inline size_t vxg_work_size(const VxgParams&p) {
    return p.operation==0 ? size_t(p.block_count)*(size_t(p.block_size)*4+256) : 16;
}
inline void vxg_check(const VxgParams*p,size_t input_size,size_t output_size,size_t desc_words) {
    if(!p || p->operation>1 || !p->block_count || p->block_size<4096 || p->block_size>1048576 || p->block_size%64)
        throw std::runtime_error("invalid GPU dispatch parameters");
    if(desc_words!=size_t(p->block_count)*4)throw std::runtime_error("descriptor size");
    uint64_t count=(uint64_t(p->total_size)+p->block_size-1)/p->block_size;
    if(count!=p->block_count || input_size>0xffffffffull)throw std::runtime_error("GPU size/count overflow");
    size_t expected=p->operation==0?size_t(p->block_count)*p->block_size:vxg_align(p->total_size,4);
    if(output_size!=expected || (p->operation==0 && input_size!=p->total_size))throw std::runtime_error("GPU buffer size mismatch");
    if(vxg_work_size(*p)>0xffffffffull)throw std::runtime_error("GPU scratch exceeds 32-bit addressing");
}
