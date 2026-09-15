#include <vulkan/vulkan.h>
#include "util.hpp"
#include "vxg_spirv.h"
#include <memory>
#include <vector>

static void checked(VkResult result,const char*operation){
    if(result!=VK_SUCCESS)throw std::runtime_error(std::string(operation)+": VkResult="+std::to_string(result));
}
struct VulkanBuffer {
    VkDevice device=VK_NULL_HANDLE;VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;
    void*mapped=nullptr;VkDeviceSize capacity=0;bool coherent=false;
    ~VulkanBuffer(){reset();}
    void reset(){
        if(mapped&&device)vkUnmapMemory(device,memory);mapped=nullptr;
        if(buffer&&device)vkDestroyBuffer(device,buffer,nullptr);buffer=VK_NULL_HANDLE;
        if(memory&&device)vkFreeMemory(device,memory,nullptr);memory=VK_NULL_HANDLE;capacity=0;
    }
    void reserve(VkDevice dev,VkPhysicalDevice physical,size_t size,bool host){
        if(capacity>=size)return;
        reset();device=dev;
        VkBufferCreateInfo bi{};bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;bi.size=vxg_align(size);
        bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;checked(vkCreateBuffer(device,&bi,nullptr,&buffer),"vkCreateBuffer");
        VkMemoryRequirements req{};vkGetBufferMemoryRequirements(device,buffer,&req);
        VkPhysicalDeviceMemoryProperties props{};vkGetPhysicalDeviceMemoryProperties(physical,&props);
        uint32_t type=UINT32_MAX;
        for(int pass=0;pass<2&&type==UINT32_MAX;++pass)for(uint32_t i=0;i<props.memoryTypeCount;++i){
            auto flags=props.memoryTypes[i].propertyFlags;
            VkMemoryPropertyFlags needed=host?VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT:0;
            VkMemoryPropertyFlags preferred=host?VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            if((req.memoryTypeBits&(1u<<i))&&(flags&needed)==needed&&(pass==1||(flags&preferred)==preferred)){type=i;break;}
        }
        if(type==UINT32_MAX)throw std::runtime_error("no compatible Vulkan memory type");
        coherent=(props.memoryTypes[type].propertyFlags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)!=0;
        VkMemoryAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;ai.allocationSize=req.size;ai.memoryTypeIndex=type;
        checked(vkAllocateMemory(device,&ai,nullptr,&memory),"vkAllocateMemory");checked(vkBindBufferMemory(device,buffer,memory,0),"vkBindBufferMemory");
        if(host)checked(vkMapMemory(device,memory,0,VK_WHOLE_SIZE,0,&mapped),"vkMapMemory");
        capacity=bi.size;
    }
    void flush(){if(!coherent){VkMappedMemoryRange r{};r.sType=VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;r.memory=memory;r.size=VK_WHOLE_SIZE;checked(vkFlushMappedMemoryRanges(device,1,&r),"flush staging memory");}}
    void invalidate(){if(!coherent){VkMappedMemoryRange r{};r.sType=VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;r.memory=memory;r.size=VK_WHOLE_SIZE;checked(vkInvalidateMappedMemoryRanges(device,1,&r),"invalidate readback memory");}}
};
struct VulkanContext {
    VkInstance instance=VK_NULL_HANDLE;VkPhysicalDevice physical=VK_NULL_HANDLE;VkDevice device=VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};VkQueue queue=VK_NULL_HANDLE;uint32_t family=0,timestamp_bits=0;
    VkDescriptorSetLayout set_layout=VK_NULL_HANDLE;VkPipelineLayout pipeline_layout=VK_NULL_HANDLE;
    VkPipeline pipeline=VK_NULL_HANDLE;VkDescriptorPool pool=VK_NULL_HANDLE;VkDescriptorSet set=VK_NULL_HANDLE;
    VkCommandPool command_pool=VK_NULL_HANDLE;VkCommandBuffer command=VK_NULL_HANDLE;VkFence fence=VK_NULL_HANDLE;
    VkQueryPool queries=VK_NULL_HANDLE;
    VulkanBuffer input,output,work,desc,upload_input,upload_desc,download_output,download_desc;
    ~VulkanContext(){
        if(device){
            vkDeviceWaitIdle(device);
            input.reset();output.reset();work.reset();desc.reset();upload_input.reset();upload_desc.reset();download_output.reset();download_desc.reset();
            if(queries)vkDestroyQueryPool(device,queries,nullptr);if(fence)vkDestroyFence(device,fence,nullptr);
            if(command_pool)vkDestroyCommandPool(device,command_pool,nullptr);
            if(pool)vkDestroyDescriptorPool(device,pool,nullptr);if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);
            if(pipeline_layout)vkDestroyPipelineLayout(device,pipeline_layout,nullptr);
            if(set_layout)vkDestroyDescriptorSetLayout(device,set_layout,nullptr);vkDestroyDevice(device,nullptr);
        }
        if(instance)vkDestroyInstance(instance,nullptr);
    }
};
VXG_DECLARE(vulkan)
extern "C" void* vxg_vulkan_create(char*error,size_t cap){
    try{
        auto c=std::make_unique<VulkanContext>();
        VkApplicationInfo app{};app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;app.pApplicationName="VXG";app.apiVersion=VK_API_VERSION_1_1;
        VkInstanceCreateInfo ii{};ii.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;ii.pApplicationInfo=&app;
        checked(vkCreateInstance(&ii,nullptr,&c->instance),"vkCreateInstance");
        uint32_t count=0;checked(vkEnumeratePhysicalDevices(c->instance,&count,nullptr),"enumerate devices");
        std::vector<VkPhysicalDevice> devices(count);checked(vkEnumeratePhysicalDevices(c->instance,&count,devices.data()),"enumerate devices");
        // Prefer actual GPUs over CPU software ICDs. The selected device name is reported.
        int best=-1;
        for(auto physical:devices){
            VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(physical,&props);
            if(props.apiVersion<VK_API_VERSION_1_1||props.limits.maxComputeWorkGroupInvocations<64||props.limits.maxComputeSharedMemorySize<8192)continue;
            uint32_t fc=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&fc,nullptr);
            std::vector<VkQueueFamilyProperties> fs(fc);vkGetPhysicalDeviceQueueFamilyProperties(physical,&fc,fs.data());
            for(uint32_t i=0;i<fc;++i)if(fs[i].queueCount&&(fs[i].queueFlags&VK_QUEUE_COMPUTE_BIT)){
                int score=props.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU?3:props.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU?2:1;
                if(score>best){best=score;c->physical=physical;c->properties=props;c->family=i;c->timestamp_bits=fs[i].timestampValidBits;}
                break;
            }
        }
        if(!c->physical)throw std::runtime_error("no compatible Vulkan 1.1 compute device");
        float priority=1.0f;VkDeviceQueueCreateInfo qi{};qi.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;qi.queueFamilyIndex=c->family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{};di.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;
        checked(vkCreateDevice(c->physical,&di,nullptr,&c->device),"vkCreateDevice");vkGetDeviceQueue(c->device,c->family,0,&c->queue);
        VkDescriptorSetLayoutBinding bindings[4]{};
        for(uint32_t i=0;i<4;++i){bindings[i].binding=i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
        VkDescriptorSetLayoutCreateInfo sli{};sli.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;sli.bindingCount=4;sli.pBindings=bindings;
        checked(vkCreateDescriptorSetLayout(c->device,&sli,nullptr,&c->set_layout),"descriptor set layout");
        VkPushConstantRange push{};push.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;push.size=sizeof(VxgParams);
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.setLayoutCount=1;pli.pSetLayouts=&c->set_layout;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&push;
        checked(vkCreatePipelineLayout(c->device,&pli,nullptr,&c->pipeline_layout),"pipeline layout");
        VkShaderModule shader=VK_NULL_HANDLE;VkShaderModuleCreateInfo smi{};smi.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;smi.codeSize=sizeof(VXG_SPIRV);smi.pCode=VXG_SPIRV;
        checked(vkCreateShaderModule(c->device,&smi,nullptr,&shader),"SPIR-V module");
        VkComputePipelineCreateInfo pci{};pci.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;pci.layout=c->pipeline_layout;
        pci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;pci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pci.stage.module=shader;pci.stage.pName="main";
        VkResult pres=vkCreateComputePipelines(c->device,VK_NULL_HANDLE,1,&pci,nullptr,&c->pipeline);vkDestroyShaderModule(c->device,shader,nullptr);checked(pres,"compute pipeline");
        VkDescriptorPoolSize ps{};ps.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;ps.descriptorCount=4;
        VkDescriptorPoolCreateInfo dpi{};dpi.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=&ps;
        checked(vkCreateDescriptorPool(c->device,&dpi,nullptr,&c->pool),"descriptor pool");
        VkDescriptorSetAllocateInfo sai{};sai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;sai.descriptorPool=c->pool;sai.descriptorSetCount=1;sai.pSetLayouts=&c->set_layout;
        checked(vkAllocateDescriptorSets(c->device,&sai,&c->set),"descriptor allocation");
        VkCommandPoolCreateInfo cpi{};cpi.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;cpi.queueFamilyIndex=c->family;cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        checked(vkCreateCommandPool(c->device,&cpi,nullptr,&c->command_pool),"command pool");
        VkCommandBufferAllocateInfo cai{};cai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;cai.commandPool=c->command_pool;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;
        checked(vkAllocateCommandBuffers(c->device,&cai,&c->command),"command buffer");
        VkFenceCreateInfo fi{};fi.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;checked(vkCreateFence(c->device,&fi,nullptr,&c->fence),"fence");
        if(c->timestamp_bits){VkQueryPoolCreateInfo qpi{};qpi.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;qpi.queryType=VK_QUERY_TYPE_TIMESTAMP;qpi.queryCount=2;checked(vkCreateQueryPool(c->device,&qpi,nullptr,&c->queries),"timestamp queries");}
        std::string name=c->properties.deviceName;if(c->properties.deviceType==VK_PHYSICAL_DEVICE_TYPE_CPU)name+=" (software Vulkan, not a hardware GPU)";
        vxg_message(error,cap,name);return c.release();
    }catch(const std::exception&e){vxg_message(error,cap,e.what());return nullptr;}
}
extern "C" int vxg_vulkan_run(void*handle,const uint8_t*input,size_t input_size,uint8_t*output,size_t output_size,
 uint32_t*desc,size_t desc_words,const VxgParams*p,double*kernel_ms,char*error,size_t cap){
    try{
        if(!handle)throw std::runtime_error("null Vulkan context");vxg_check(p,input_size,output_size,desc_words);
        auto&c=*static_cast<VulkanContext*>(handle);
        if(p->block_count>c.properties.limits.maxComputeWorkGroupCount[0])throw std::runtime_error("Vulkan dispatch exceeds max workgroup count; reduce GPU batch size");
        size_t ni=vxg_align(input_size+16),no=vxg_align(output_size),nw=vxg_align(vxg_work_size(*p)),nd=vxg_align(desc_words*4);
        for(size_t bytes:{ni,no,nw,nd})if(bytes>c.properties.limits.maxStorageBufferRange)throw std::runtime_error("Vulkan storage buffer range exceeded; reduce GPU input size");
        c.input.reserve(c.device,c.physical,ni,false);c.output.reserve(c.device,c.physical,no,false);c.work.reserve(c.device,c.physical,nw,false);c.desc.reserve(c.device,c.physical,nd,false);
        c.upload_input.reserve(c.device,c.physical,ni,true);c.upload_desc.reserve(c.device,c.physical,nd,true);
        c.download_output.reserve(c.device,c.physical,no,true);c.download_desc.reserve(c.device,c.physical,nd,true);
        std::memcpy(c.upload_input.mapped,input,input_size);std::memset(static_cast<uint8_t*>(c.upload_input.mapped)+input_size,0,ni-input_size);
        std::memcpy(c.upload_desc.mapped,desc,desc_words*4);std::memset(static_cast<uint8_t*>(c.upload_desc.mapped)+desc_words*4,0,nd-desc_words*4);
        c.upload_input.flush();c.upload_desc.flush();
        VkDescriptorBufferInfo infos[4]={{c.input.buffer,0,ni},{c.output.buffer,0,no},{c.work.buffer,0,nw},{c.desc.buffer,0,nd}};
        VkWriteDescriptorSet writes[4]{};for(uint32_t i=0;i<4;++i){writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=c.set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];}
        vkUpdateDescriptorSets(c.device,4,writes,0,nullptr);
        checked(vkResetCommandBuffer(c.command,0),"reset command buffer");VkCommandBufferBeginInfo bi{};bi.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;checked(vkBeginCommandBuffer(c.command,&bi),"begin command buffer");
        VkBufferCopy copy{};copy.size=ni;vkCmdCopyBuffer(c.command,c.upload_input.buffer,c.input.buffer,1,&copy);copy.size=nd;vkCmdCopyBuffer(c.command,c.upload_desc.buffer,c.desc.buffer,1,&copy);
        VkMemoryBarrier mb{};mb.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;mb.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&mb,0,nullptr,0,nullptr);
        if(c.queries){vkCmdResetQueryPool(c.command,c.queries,0,2);vkCmdWriteTimestamp(c.command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,c.queries,0);}
        vkCmdBindPipeline(c.command,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline);vkCmdBindDescriptorSets(c.command,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline_layout,0,1,&c.set,0,nullptr);
        vkCmdPushConstants(c.command,c.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(*p),p);vkCmdDispatch(c.command,p->block_count,1,1);
        if(c.queries)vkCmdWriteTimestamp(c.command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,c.queries,1);
        mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;mb.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&mb,0,nullptr,0,nullptr);
        copy.size=output_size;vkCmdCopyBuffer(c.command,c.output.buffer,c.download_output.buffer,1,&copy);copy.size=desc_words*4;vkCmdCopyBuffer(c.command,c.desc.buffer,c.download_desc.buffer,1,&copy);
        mb.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;mb.dstAccessMask=VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(c.command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&mb,0,nullptr,0,nullptr);
        checked(vkEndCommandBuffer(c.command),"end command buffer");checked(vkResetFences(c.device,1,&c.fence),"reset fence");
        VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;si.commandBufferCount=1;si.pCommandBuffers=&c.command;checked(vkQueueSubmit(c.queue,1,&si,c.fence),"queue submit");
        checked(vkWaitForFences(c.device,1,&c.fence,VK_TRUE,UINT64_MAX),"GPU completion");
        c.download_output.invalidate();c.download_desc.invalidate();std::memcpy(output,c.download_output.mapped,output_size);std::memcpy(desc,c.download_desc.mapped,desc_words*4);
        if(kernel_ms){
            *kernel_ms=-1.0;
            if(c.queries){uint64_t t[2]{};checked(vkGetQueryPoolResults(c.device,c.queries,0,2,sizeof(t),t,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT),"timestamp readback");
                uint64_t mask=c.timestamp_bits==64?UINT64_MAX:((1ull<<c.timestamp_bits)-1);*kernel_ms=double((t[1]-t[0])&mask)*c.properties.limits.timestampPeriod/1e6;}
        }
        return 0;
    }catch(const std::exception&e){
        if(handle){auto*c=static_cast<VulkanContext*>(handle);if(c->device)vkDeviceWaitIdle(c->device);}
        vxg_message(error,cap,e.what());return -1;
    }
}
extern "C" void vxg_vulkan_destroy(void*handle){delete static_cast<VulkanContext*>(handle);}
