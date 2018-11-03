#include "proxyCL.h"

#include <sstream>
#include <iosfwd>
#include <fstream>
#include <numeric>

template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v)
{
    out << "[";
    size_t last = v.size() - 1;
    for (size_t i = 0; i < v.size(); ++i)
    {
        out << v[i];
        if (i != last)
            out << ", ";
    }
    out << "]";
    return out;
}

void proxyCL::enable_cuda_build_cache(bool enable) const
{
#ifdef _MSC_VER
    if (enable)
        _putenv("CUDA_CACHE_DISABLE=0");
    else
        _putenv("CUDA_CACHE_DISABLE=1");
#else
    if (enable)
        putenv("CUDA_CACHE_DISABLE=0");
    else
        putenv("CUDA_CACHE_DISABLE=1");
#endif
}

std::vector <std::string> proxyCL::available_vendors_str(std::string &error) const
{
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    if (err != CL_SUCCESS)
    {
        error = "Platform::get() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return std::vector<std::string>();
    }

    std::vector<std::string> vendors;
    for (std::vector<cl::Platform>::iterator i = platforms.begin(); i != platforms.end(); ++i)
        vendors.push_back((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str());

    if (err != CL_SUCCESS)
    {
        error = "Platform::getInfo() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return std::vector<std::string>();
    }

    return vendors;
}

std::vector <std::string> proxyCL::available_vendors_str_ex(std::string &error) const
{
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    if (err != CL_SUCCESS)
    {
        error = "Platform::get() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return std::vector<std::string>();
    }

    std::vector<std::string> vendors;
    for (std::vector<cl::Platform>::iterator i = platforms.begin(); i != platforms.end(); ++i)
        vendors.push_back((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str() + std::string(" ") + (*i).getInfo<CL_PLATFORM_VERSION>(&err).c_str());

    if (err != CL_SUCCESS)
    {
        error = "Platform::getInfo() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return std::vector<std::string>();
    }

    return vendors;
}

std::vector <vendor> proxyCL::available_vendors(std::string &error) const
{
    std::vector <vendor> vendors;
    std::vector<std::string> vendors_str = available_vendors_str(error);
    if (vendors_str.empty())
        return vendors;

    const std::string amd_str("Advanced Micro Devices, Inc.");
    const std::string intel_str("Intel(R) Corporation");
    const std::string nvidia_str("NVIDIA Corporation");
    const std::string apple_str("Apple");
    for (size_t v = 0; v < vendors_str.size(); ++v)
    {
        if (vendors_str[v].compare(amd_str) == 0)
            vendors.push_back(amd);
        else if (vendors_str[v].compare(intel_str) == 0)
            vendors.push_back(intel);
        else if (vendors_str[v].compare(nvidia_str) == 0)
            vendors.push_back(nvidia);
        else if (vendors_str[v].compare(apple_str) == 0) {
            vendors.push_back(apple);
        } else {
            vendors.push_back(other);
        }
    }

    return vendors;
}

bool proxyCL::select_vendor(const std::string &vendor, std::string &error)
{
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    if (err != CL_SUCCESS)
    {
        error = "Platform::get() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return false;
    }

    const std::string amd_str("Advanced Micro Devices, Inc.");
    const std::string intel_str("Intel(R) Corporation");
    const std::string nvidia_str("NVIDIA Corporation");
    const std::string apple_str("Apple"); 
    for (std::vector<cl::Platform>::iterator i = platforms.begin(); i != platforms.end(); ++i)
    {
        if (!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>(&err).c_str(), vendor.c_str()))
        {
            platform_ = *i;

            if (vendor == amd_str)
                active_vendor_ = amd;
            else if (vendor == intel_str)
                active_vendor_ = intel;
            else if (vendor == nvidia_str)
                active_vendor_ = nvidia;
            else if (vendor == apple_str)
                active_vendor_ = apple;
            else
                active_vendor_ = other;

            return true;
        }
    }

    if (err != CL_SUCCESS)
    {
        error = "Platform::getInfo() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return false;
    }

    return false;
}

bool proxyCL::select_vendor(vendor v, std::string &error)
{
    const std::string amd_str("Advanced Micro Devices, Inc.");
    const std::string intel_str("Intel(R) Corporation");
    const std::string nvidia_str("NVIDIA Corporation");
    const std::string apple_str("Apple");

    std::string vendor_str;
    switch (v)
    {
    case amd:
        vendor_str = amd_str;
        break;
    case intel:
        vendor_str = intel_str;
        break;
    case nvidia:
        vendor_str = nvidia_str;
        break;
    case apple:
        vendor_str = apple_str;
        break;
    default:
        error = "Unknown vendor!";
        return false;
    }
   
    return select_vendor(vendor_str, error);
}

bool proxyCL::create_context(device_type type, std::string &error)
{
    cl_int err;
    cl_device_type device_type = CL_DEVICE_TYPE_ALL;
    switch (type)
    {
    case cpu:
        device_type = CL_DEVICE_TYPE_CPU;
        break;
    case gpu:
        device_type = CL_DEVICE_TYPE_GPU;
        break;
    default:
        device_type = CL_DEVICE_TYPE_ALL;
    }

    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_)(), 0 };

    context_ = cl::Context(device_type, cps, NULL, NULL, &err);
    if (err != CL_SUCCESS) 
    {
        error = "Context::Context() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return false;
    }

    return true;
}

bool proxyCL::create_context(std::string &error)
{
    cl_int err;
    if (device_() == 0)
    {
        error = "device_ not initialized -- cannot create context";
	return false;
    }
    
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_)(), 0 };

    std::vector <cl::Device> device_vector = { device_ };
    context_ = cl::Context(device_vector, cps, NULL, NULL, &err);
    if (err != CL_SUCCESS) 
    {
        error = "Context::Context() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return false;
    }

    return true;
}

std::vector <device_info> proxyCL::available_devices(std::string &error) const
{
    cl_int err;
    std::vector <device_info> di_vec;
    std::vector <cl::Device> devices;
    err = platform_.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (err != CL_SUCCESS) 
    {
        error = "cl::Platform::getDevices() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return di_vec;
    }

    for (size_t d = 0; d < devices.size(); ++d)
    {
        device_info di;
        di.id = d;
        
        switch (devices[d].getInfo<CL_DEVICE_TYPE>(&err))
        {
        case CL_DEVICE_TYPE_CPU:
            di.type = cpu;
            break;
        case CL_DEVICE_TYPE_GPU:
            di.type = gpu;
            break;
        default:
            di.type = all;
        }

        di.name = devices[d].getInfo<CL_DEVICE_NAME>(&err);
        di_vec.push_back(di);
    }

    return di_vec;
}

bool proxyCL::select_device(size_t id, std::string &error)
{
    cl_int err;
    std::vector <cl::Device> devices;
    err = platform_.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (err != CL_SUCCESS)
    {
        error = "cl::Platform::getDevices() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return false;
    }

    if (id >= devices.size())
    {
        error = "Non-existing device id!";
        return false;
    }

    device_ = devices[id];

    max_global_mem_size_ = device_.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    max_local_mem_size_ = device_.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    max_work_group_size_ = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    work_group_size_ = max_work_group_size_;

    return true;
}

bool proxyCL::load_kernel_from_source_file(const std::string &file_path, std::string &error)
{
    cl_int err;
    std::ifstream cl_file(file_path.c_str());
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, std::make_pair(cl_string.c_str(), cl_string.length()));

    program_ = cl::Program(context_, sources, &err);
    if (err != CL_SUCCESS) {
        error = "Program::Program() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return false;
    }
    return true;
}

bool proxyCL::load_kernel_from_binary_file(const std::string &file_path,
    const std::string &build_options, std::string &error)
{
    std::ifstream cl_file(file_path.c_str(), std::ios_base::binary | std::ios_base::in);

    size_t optionsSize;
    cl_file.read(reinterpret_cast<char*>(&optionsSize), sizeof(optionsSize));
    if (!cl_file.good() || optionsSize != build_options.size()) {
        error = cl_file.good() ? "Build options string does not match" : "File read error";
        return false;
    }

    std::vector<char> cl_binary(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    if (optionsSize >= cl_binary.size()) {
        error = "Build options size corrupted";
        return false;
    }
    if (memcmp(build_options.c_str(), cl_binary.data(), build_options.size()) != 0) {
        error = "Build options string does not match: " + build_options + ", "
            + std::string(cl_binary.data(), build_options.size());
        return false;
    }

    cl::Program::Binaries binaries(1, std::make_pair(cl_binary.data() + optionsSize, cl_binary.size() - optionsSize));

    cl_int err;
    std::vector<cl::Device> devices(1, device_);
    program_ = cl::Program(context_, devices, binaries, nullptr, &err);
    if (err != CL_SUCCESS) {
        error = "Program::Program() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return false;
    }
    return true;
}

bool proxyCL::load_kernel_from_source(const std::string &src, std::string &error)
{
    cl_int err;
    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length()));

    program_ = cl::Program(context_, sources, &err);
    if (err != CL_SUCCESS)
    {
        error = "Program::Program() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return false;
    }

    return true;
}

bool proxyCL::build_kernel(const std::string &build_options, std::string &build_log)
{
    std::ostringstream ostr;
    std::vector<cl::Device> devices(1, device_);
    cl_int err = program_.build(devices, build_options.c_str());
    if (err != CL_SUCCESS) 
    {
        if (err == CL_BUILD_PROGRAM_FAILURE)
        {
            std::string str = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_);

            ostr << " \n\t\t\tBUILD LOG\n";
            ostr << " ************************************************\n";
            ostr << str.c_str() << std::endl;
            ostr << " ************************************************\n";
        }

        ostr << "Program::build() failed (" << ocl_error_to_string(err) << ")\n";
        build_log = ostr.str();
        return false;
    }

    std::string str = program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_);

    ostr << " \n\t\t\tBUILD LOG\n";
    ostr << " ************************************************\n";
    ostr << str.c_str() << std::endl;
    ostr << " ************************************************\n";

    //build_log = ostr.str();

    return true;
}

bool proxyCL::output_binary(const std::string &path, const std::string &build_options, std::string &error)
{
    // Allocate some memory for all the kernel binary data
    const std::vector<size_t> binSizes = program_.getInfo<CL_PROGRAM_BINARY_SIZES>();
    std::vector<char> binData(std::accumulate(binSizes.begin(), binSizes.end(), 0));
    if (binData.size() == 0) { return false; }
    char* binChunk = &binData[0];

    //A list of pointers to the binary data
    std::vector<char*> binaries;
    for (unsigned int i = 0; i<binSizes.size(); ++i) {
        binaries.push_back(binChunk);
        binChunk += binSizes[i];
    }

    program_.getInfo(CL_PROGRAM_BINARIES, &binaries[0]);
    std::ofstream binaryfile(path, std::ios::binary);
    size_t buildOptionsSize = build_options.size();
    binaryfile.write(reinterpret_cast<char*>(&buildOptionsSize), sizeof(buildOptionsSize));
    binaryfile.write(build_options.c_str(), buildOptionsSize);
    for (unsigned int i = 0; i < binaries.size(); ++i)
        binaryfile.write(binaries[i], binSizes[i]);

    return true;
}

bool proxyCL::create_command_queue(bool enable_profiling, bool enable_out_of_order_exec_mode, std::string &error)
{
    cl_int err;
    cl_command_queue_properties prop = 0;
    if (enable_profiling)
    {
        enable_profiling_ = enable_profiling;
        prop |= CL_QUEUE_PROFILING_ENABLE;
    }
    if (enable_out_of_order_exec_mode)
        prop |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

    queue_ = cl::CommandQueue(context_, device_, prop, &err);
    if (err != CL_SUCCESS) 
    {
        error = "CommandQueue::CommandQueue() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return false;
    }

    return true;
}

device_info_ex proxyCL::get_device_info_ex(size_t id, std::string &error) const
{
    cl_int err;
    device_info_ex di;
    std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>(&err);
    if (err != CL_SUCCESS)
    {
        error = "Context::getInfo() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return di;
    }

    if (id >= devices.size())
    {
        error = "Non-existing device id!";
        return di;
    }

    di.id = id;
    switch (devices[id].getInfo<CL_DEVICE_TYPE>(&err))
    {
    case CL_DEVICE_TYPE_CPU:
        di.type = cpu;
        break;
    case CL_DEVICE_TYPE_GPU:
        di.type = gpu;
        break;
    default:
        di.type = all;
    }

    di.name = devices[id].getInfo<CL_DEVICE_NAME>(&err);
    di.max_compute_units = devices[id].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    di.max_work_item_dimensions = devices[id].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
    di.max_work_group_size = devices[id].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    std::vector <size_t> max_work_item_sizes = devices[id].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    di.max_work_items_sizes_x = max_work_item_sizes[0];
    di.max_work_items_sizes_y = max_work_item_sizes[1];
    di.max_work_items_sizes_z = max_work_item_sizes[2];
    di.max_clock_frequency = devices[id].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
    di.max_parameter_size = devices[id].getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>();
    di.global_mem_cache_type = devices[id].getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>();
    di.global_mem_cacheline_size = devices[id].getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>();
    di.global_mem_cache_size = devices[id].getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
    di.global_mem_size = devices[id].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    di.max_constant_buffer_size = devices[id].getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
    di.local_mem_type = devices[id].getInfo<CL_DEVICE_LOCAL_MEM_TYPE>();
    di.local_mem_size = devices[id].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    di.preferred_vector_width_char = devices[id].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>();
    di.preferred_vector_width_short = devices[id].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>();
    di.preferred_vector_width_int = devices[id].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>();
    di.preferred_vector_width_long = devices[id].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>();
    di.preferred_vector_width_float = devices[id].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>();

    return di;
}

std::string proxyCL::get_device_info(size_t id, std::string &error) const
{
    cl_int err;
    device_info_ex di;
    std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>(&err);
    if (err != CL_SUCCESS)
    {
        error = "Context::getInfo() failed (" + std::string(ocl_error_to_string(err)) + ")";
        return "";
    }

    if (id >= devices.size())
    {
        error = "Non-existing device id!";
        return "";
    }

    std::ostringstream ostr;

    ostr.imbue(std::locale(""));
    ostr << std::fixed <<
        "CL_DEVICE_NAME:\t\t\t\t" << devices[id].getInfo<CL_DEVICE_NAME>() << std::endl <<
        "CL_DEVICE_VERSION:\t\t\t" << devices[id].getInfo<CL_DEVICE_VERSION>() << std::endl <<
        "CL_DEVICE_PROFILE:\t\t\t" << devices[id].getInfo<CL_DEVICE_PROFILE>() << std::endl <<
        "CL_DEVICE_VENDOR:\t\t\t" << devices[id].getInfo<CL_DEVICE_VENDOR>() << std::endl <<
        "CL_DRIVER_VERSION:\t\t\t" << devices[id].getInfo<CL_DRIVER_VERSION>() << std::endl <<
        "CL_DEVICE_MAX_COMPUTE_UNITS:\t\t" << devices[id].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl <<
        "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t" << devices[id].getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl <<
        "CL_DEVICE_MAX_WORK_GROUP_SIZE:\t\t" << devices[id].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl <<
        "CL_DEVICE_MAX_WORK_ITEM_SIZES:\t\t" << devices[id].getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>() << std::endl <<
        "CL_DEVICE_MAX_CLOCK_FREQUENCY:\t\t" << devices[id].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl <<
        "CL_DEVICE_MAX_PARAMETER_SIZE:\t\t" << devices[id].getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>() << std::endl <<
        "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:\t" << devices[id].getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>() << std::endl <<
        "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:\t" << devices[id].getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>() << std::endl <<
        "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:\t" << devices[id].getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>() << std::endl <<
        "CL_DEVICE_GLOBAL_MEM_SIZE:\t\t" << devices[id].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl <<
        "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t" << devices[id].getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << std::endl <<
        "CL_DEVICE_LOCAL_MEM_TYPE:\t\t" << devices[id].getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() << std::endl <<
        "CL_DEVICE_LOCAL_MEM_SIZE:\t\t" << devices[id].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl <<
        "CL_DEVICE_PREFERRED_VECTOR_WIDTH_[CHAR,SHORT,INT,LONG,FLOAT]:\t" << devices[id].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>() <<
        " " << devices[id].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>() <<
        " " << devices[id].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>() <<
        " " << devices[id].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>() <<
        " " << devices[id].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>() << std::endl;

    return ostr.str();
}

bool proxyCL::double_fp_support(std::string &error) const
{
    cl_int err;
    cl_device_fp_config double_info = device_.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>(&err);
    if (err != CL_SUCCESS) {
        error = "getInfo<CL_DEVICE_DOUBLE_FP_CONFIG> failed (" + std::string(ocl_error_to_string(err)) + ")";
        return false;
    }

    return double_info ? true : false;
}

std::string proxyCL::get_device_extensions(std::string &error) const
{
    cl_int err;
    std::string exensions = device_.getInfo<CL_DEVICE_EXTENSIONS>(&err);
    if (err != CL_SUCCESS) {
        error = "getInfo<CL_DEVICE_EXTENSIONS> failed (" + std::string(ocl_error_to_string(err)) + ")";
        return "";
    }

    return exensions;
}

bool proxyCL::fp64_extension_support(std::string &error) const
{
    std::string extensions = get_device_extensions(error);
    if (extensions.find("cl_khr_fp64") != std::string::npos)
        return true;
    return false;
}

const char* proxyCL::ocl_error_to_string(cl_int err) const
{
    switch (err)
    {

#ifdef CL_SUCCESS         
    case CL_SUCCESS: return "CL_SUCCESS";
#endif
#ifdef CL_DEVICE_NOT_FOUND 
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
#endif
#ifdef CL_DEVICE_NOT_AVAILABLE 
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
#endif
#ifdef CL_COMPILER_NOT_AVAILABLE 
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
#endif
#ifdef CL_MEM_OBJECT_ALLOCATION_FAILURE 
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
#endif
#ifdef CL_OUT_OF_RESOURCES 
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
#endif
#ifdef CL_OUT_OF_HOST_MEMORY 
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
#endif
#ifdef CL_PROFILING_INFO_NOT_AVAILABLE 
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
#endif
#ifdef CL_MEM_COPY_OVERLAP 
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
#endif
#ifdef CL_IMAGE_FORMAT_MISMATCH 
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
#endif
#ifdef CL_IMAGE_FORMAT_NOT_SUPPORTED 
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
#endif
#ifdef CL_BUILD_PROGRAM_FAILURE 
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
#endif
#ifdef CL_MAP_FAILURE 
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
#endif
#ifdef CL_MISALIGNED_SUB_BUFFER_OFFSET 
    case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
#endif
#ifdef CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST 
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
#ifdef CL_COMPILE_PROGRAM_FAILURE 
    case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
#endif
#ifdef CL_LINKER_NOT_AVAILABLE 
    case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
#endif
#ifdef CL_LINK_PROGRAM_FAILURE 
    case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
#endif
#ifdef CL_DEVICE_PARTITION_FAILED 
    case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
#endif
#ifdef CL_KERNEL_ARG_INFO_NOT_AVAILABLE 
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif
#ifdef CL_INVALID_VALUE 
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
#endif
#ifdef CL_INVALID_DEVICE_TYPE 
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
#endif
#ifdef CL_INVALID_PLATFORM 
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
#endif
#ifdef CL_INVALID_DEVICE 
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
#endif
#ifdef CL_INVALID_CONTEXT 
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
#endif
#ifdef CL_INVALID_QUEUE_PROPERTIES 
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
#endif
#ifdef CL_INVALID_COMMAND_QUEUE 
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
#endif
#ifdef CL_INVALID_HOST_PTR 
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
#endif
#ifdef CL_INVALID_MEM_OBJECT 
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
#endif
#ifdef CL_INVALID_IMAGE_FORMAT_DESCRIPTOR 
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
#endif
#ifdef CL_INVALID_IMAGE_SIZE 
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
#endif
#ifdef CL_INVALID_SAMPLER 
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
#endif
#ifdef CL_INVALID_BINARY 
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
#endif
#ifdef CL_INVALID_BUILD_OPTIONS 
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
#endif
#ifdef CL_INVALID_PROGRAM 
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
#endif
#ifdef CL_INVALID_PROGRAM_EXECUTABLE 
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
#endif
#ifdef CL_INVALID_KERNEL_NAME 
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
#endif
#ifdef CL_INVALID_KERNEL_DEFINITION 
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
#endif
#ifdef CL_INVALID_KERNEL 
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
#endif
#ifdef CL_INVALID_ARG_INDEX 
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
#endif
#ifdef CL_INVALID_ARG_VALUE 
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
#endif
#ifdef CL_INVALID_ARG_SIZE 
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
#endif
#ifdef CL_INVALID_KERNEL_ARGS 
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
#endif
#ifdef CL_INVALID_WORK_DIMENSION 
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
#endif
#ifdef CL_INVALID_WORK_GROUP_SIZE 
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
#endif
#ifdef CL_INVALID_WORK_ITEM_SIZE 
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
#endif
#ifdef CL_INVALID_GLOBAL_OFFSET 
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
#endif
#ifdef CL_INVALID_EVENT_WAIT_LIST 
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
#endif
#ifdef CL_INVALID_EVENT 
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
#endif
#ifdef CL_INVALID_OPERATION 
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
#endif
#ifdef CL_INVALID_GL_OBJECT 
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
#endif
#ifdef CL_INVALID_BUFFER_SIZE 
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
#endif
#ifdef CL_INVALID_MIP_LEVEL 
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
#endif
#ifdef CL_INVALID_GLOBAL_WORK_SIZE 
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
#endif
#ifdef CL_INVALID_PROPERTY 
    case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
#endif
#ifdef CL_INVALID_IMAGE_DESCRIPTOR 
    case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
#endif
#ifdef CL_INVALID_COMPILER_OPTIONS 
    case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
#endif
#ifdef CL_INVALID_LINKER_OPTIONS 
    case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
#endif
#ifdef CL_INVALID_DEVICE_PARTITION_COUNT 
    case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
    }

    return "missing description for error code";
}
