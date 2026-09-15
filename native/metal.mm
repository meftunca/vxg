#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "util.hpp"
#include <memory>
#include "vxg_metal_source.h"

struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> input, output, work, desc;
};
VXG_DECLARE(metal)
extern "C" void* vxg_metal_create(char*error,size_t cap) {
    @autoreleasepool {
        try {
            auto c=std::make_unique<MetalContext>();
            c->device=MTLCreateSystemDefaultDevice();
            if(!c->device)throw std::runtime_error("no Metal device");
            c->queue=[c->device newCommandQueue];
            if(!c->queue)throw std::runtime_error("Metal command queue allocation failed");
            NSString* source=[NSString stringWithUTF8String:VXG_METAL_SOURCE];
            NSError* err=nil;
            MTLCompileOptions* options=[MTLCompileOptions new];
            options.fastMathEnabled=NO; // Lossless integer codec; no relaxed arithmetic contract.
            id<MTLLibrary> library=[c->device newLibraryWithSource:source options:options error:&err];
            if(!library)throw std::runtime_error(err ? [[err localizedDescription] UTF8String] : "MSL compilation failed");
            id<MTLFunction> function=[library newFunctionWithName:@"vxg_kernel"];
            if(!function)throw std::runtime_error("Metal entry point missing");
            c->pipeline=[c->device newComputePipelineStateWithFunction:function error:&err];
            if(!c->pipeline)throw std::runtime_error(err ? [[err localizedDescription] UTF8String] : "Metal pipeline failed");
            if(c->pipeline.maxTotalThreadsPerThreadgroup<64)throw std::runtime_error("Metal workgroup limit <64");
            vxg_message(error,cap,[[c->device name] UTF8String]);
            return c.release();
        } catch(const std::exception&e) { vxg_message(error,cap,e.what());return nullptr; }
    }
}
extern "C" int vxg_metal_run(void*handle,const uint8_t*input,size_t input_size,uint8_t*output,size_t output_size,
 uint32_t*desc,size_t desc_words,const VxgParams*p,double*kernel_ms,char*error,size_t cap) {
    @autoreleasepool {
        try {
            if(!handle)throw std::runtime_error("null Metal context");
            vxg_check(p,input_size,output_size,desc_words);
            auto&c=*static_cast<MetalContext*>(handle);
            auto reserve=[&](id<MTLBuffer> previous,size_t bytes)->id<MTLBuffer>{
                size_t n=vxg_align(bytes);
                if(previous && previous.length>=n)return previous;
                id<MTLBuffer> b=[c.device newBufferWithLength:n options:MTLResourceStorageModeShared];
                if(!b)throw std::runtime_error("Metal buffer allocation failed");
                return b;
            };
            c.input=reserve(c.input,input_size+16);c.output=reserve(c.output,output_size);
            c.work=reserve(c.work,vxg_work_size(*p));c.desc=reserve(c.desc,desc_words*4);
            std::memcpy(c.input.contents,input,input_size);
            std::memset(static_cast<uint8_t*>(c.input.contents)+input_size,0,16);
            std::memcpy(c.desc.contents,desc,desc_words*4);
            id<MTLCommandBuffer> command=[c.queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            if(!command||!encoder)throw std::runtime_error("Metal encoder allocation failed");
            [encoder setComputePipelineState:c.pipeline];
            [encoder setBuffer:c.input offset:0 atIndex:0];[encoder setBuffer:c.output offset:0 atIndex:1];
            [encoder setBuffer:c.work offset:0 atIndex:2];[encoder setBuffer:c.desc offset:0 atIndex:3];
            [encoder setBytes:p length:sizeof(*p) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(p->block_count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status==MTLCommandBufferStatusError)throw std::runtime_error(command.error ? [[command.error localizedDescription] UTF8String] : "Metal dispatch failed");
            std::memcpy(output,c.output.contents,output_size);std::memcpy(desc,c.desc.contents,desc_words*4);
            if(kernel_ms)*kernel_ms=command.GPUEndTime>command.GPUStartTime?(command.GPUEndTime-command.GPUStartTime)*1000.0:-1.0;
            return 0;
        }catch(const std::exception&e){vxg_message(error,cap,e.what());return -1;}
    }
}
extern "C" void vxg_metal_destroy(void*handle) { @autoreleasepool { delete static_cast<MetalContext*>(handle); } }
