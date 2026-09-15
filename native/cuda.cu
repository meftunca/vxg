#include "util.hpp"
#include "shaders/vxg.cuh"
#include <memory>
#include <cstdlib>

static void checked(cudaError_t result,const char*operation) {
    if(result!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(result));
}
struct CudaBuffer {
    void*ptr=nullptr;size_t capacity=0;
    ~CudaBuffer(){if(ptr)cudaFree(ptr);}
    void reserve(size_t n){
        n=vxg_align(n);
        if(capacity>=n)return;
        void*next=nullptr;checked(cudaMalloc(&next,n),"cudaMalloc");
        if(ptr)cudaFree(ptr);ptr=next;capacity=n;
    }
};
struct CudaContext {
    int device=0;
    cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;
    CudaBuffer input,output,work,desc;
    ~CudaContext(){
        cudaSetDevice(device);
        if(stream)cudaStreamSynchronize(stream);
        if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);
        if(stream)cudaStreamDestroy(stream);
    }
};
VXG_DECLARE(cuda)
extern "C" void* vxg_cuda_create(char*error,size_t cap) {
    try {
        auto c=std::make_unique<CudaContext>();
        if(const char*env=std::getenv("VXG_CUDA_DEVICE"))c->device=std::stoi(env);
        checked(cudaSetDevice(c->device),"cudaSetDevice");
        cudaDeviceProp prop{};checked(cudaGetDeviceProperties(&prop,c->device),"cudaGetDeviceProperties");
        if(prop.maxThreadsPerBlock<64||prop.sharedMemPerBlock<8192)throw std::runtime_error("insufficient CUDA workgroup/shared memory limit");
        checked(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking),"cudaStreamCreate");
        checked(cudaEventCreate(&c->begin),"cudaEventCreate");checked(cudaEventCreate(&c->end),"cudaEventCreate");
        vxg_message(error,cap,prop.name);return c.release();
    }catch(const std::exception&e){vxg_message(error,cap,e.what());return nullptr;}
}
extern "C" int vxg_cuda_run(void*handle,const uint8_t*input,size_t input_size,uint8_t*output,size_t output_size,
 uint32_t*desc,size_t desc_words,const VxgParams*p,double*kernel_ms,char*error,size_t cap) {
    try {
        if(!handle)throw std::runtime_error("null CUDA context");vxg_check(p,input_size,output_size,desc_words);
        auto&c=*static_cast<CudaContext*>(handle);checked(cudaSetDevice(c.device),"cudaSetDevice");
        c.input.reserve(input_size+16);c.output.reserve(output_size);c.work.reserve(vxg_work_size(*p));c.desc.reserve(desc_words*4);
        checked(cudaMemcpyAsync(c.input.ptr,input,input_size,cudaMemcpyHostToDevice,c.stream),"upload input");
        checked(cudaMemsetAsync(static_cast<uint8_t*>(c.input.ptr)+input_size,0,16,c.stream),"input padding");
        checked(cudaMemcpyAsync(c.desc.ptr,desc,desc_words*4,cudaMemcpyHostToDevice,c.stream),"upload descriptors");
        checked(cudaEventRecord(c.begin,c.stream),"start event");
        Params params{p->total_size,p->block_size,p->block_count,p->operation};
        vxg_kernel<<<p->block_count,64,0,c.stream>>>(static_cast<const uint*>(c.input.ptr),static_cast<uint*>(c.output.ptr),
            static_cast<uint*>(c.work.ptr),static_cast<uint*>(c.desc.ptr),params);
        checked(cudaGetLastError(),"kernel launch");checked(cudaEventRecord(c.end,c.stream),"stop event");
        checked(cudaMemcpyAsync(output,c.output.ptr,output_size,cudaMemcpyDeviceToHost,c.stream),"download output");
        checked(cudaMemcpyAsync(desc,c.desc.ptr,desc_words*4,cudaMemcpyDeviceToHost,c.stream),"download descriptors");
        checked(cudaStreamSynchronize(c.stream),"CUDA synchronization");
        float ms=0;checked(cudaEventElapsedTime(&ms,c.begin,c.end),"event timing");if(kernel_ms)*kernel_ms=ms;
        return 0;
    }catch(const std::exception&e){
        // A failed call must never leave DMA targeting the caller's Rust buffers.
        if(handle){auto*c=static_cast<CudaContext*>(handle);if(c->stream)cudaStreamSynchronize(c->stream);}
        vxg_message(error,cap,e.what());return -1;
    }
}
extern "C" void vxg_cuda_destroy(void*handle){delete static_cast<CudaContext*>(handle);}
