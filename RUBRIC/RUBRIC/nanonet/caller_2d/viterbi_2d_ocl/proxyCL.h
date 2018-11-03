#ifndef PROXY_CL_H
#define PROXY_CL_H

#include <CL/cl.hpp>
#ifdef max // it is defined in cl.hpp
#undef max
#endif //max

#include <string>
#include <vector>

enum vendor
{
    amd,
    intel,
    nvidia,
    apple,
    other
};

enum device_type
{
    cpu,
    gpu,
    all,
    undefined
};

struct device_info
{
    size_t id;
    std::string name;
    device_type type;

    bool operator==(const device_info &data)
    {
        return id == data.id && name == data.name && type == data.type;
    }
};

struct device_info_ex
{
    size_t id;
    std::string name;
    device_type type;
    size_t max_compute_units;
    size_t max_work_item_dimensions;
    size_t max_work_group_size;
    size_t max_work_items_sizes_x;
    size_t max_work_items_sizes_y;
    size_t max_work_items_sizes_z;
    size_t max_clock_frequency;
    size_t max_parameter_size;
    size_t global_mem_cache_type;
    size_t global_mem_cacheline_size;
    size_t global_mem_cache_size;
    size_t global_mem_size;
    size_t max_constant_buffer_size;
    size_t local_mem_type;
    size_t local_mem_size;
    size_t preferred_vector_width_char;
    size_t preferred_vector_width_short;
    size_t preferred_vector_width_int;
    size_t preferred_vector_width_long;
    size_t preferred_vector_width_float;
};

class proxyCL
{
public:
    proxyCL(){};
    ~proxyCL(){};

    bool profiling_enabled() const { return enable_profiling_; }
    void enable_cuda_build_cache(bool enable) const;

    size_t get_max_global_mem_size() const { return max_global_mem_size_; }
    size_t get_max_local_mem_size() const { return max_local_mem_size_; }
    size_t get_max_work_group_size() const { return max_work_group_size_; }
    size_t get_work_group_size() const { return work_group_size_; }
    void set_work_group_size(size_t value) { work_group_size_ = value; }

    std::vector <std::string> available_vendors_str(std::string &error) const;
    std::vector <std::string> available_vendors_str_ex(std::string &error) const;
    std::vector <vendor> available_vendors(std::string &error) const;
    bool select_vendor(const std::string &vendor, std::string &error);
    bool select_vendor(vendor v, std::string &error);
    vendor get_selected_vendor() const { return active_vendor_; }

    bool create_context(device_type type, std::string &error);
    bool create_context(std::string &error);

    std::vector <device_info> available_devices(std::string &error) const;
    bool select_device(size_t id, std::string &error);
    device_info_ex get_device_info_ex(size_t id, std::string &error) const;
    std::string get_device_info(size_t id, std::string &error) const;

    std::string get_device_extensions(std::string &error) const;
    bool fp64_extension_support(std::string &error) const;

    bool double_fp_support(std::string &error) const;

    bool load_kernel_from_source_file(const std::string &file_path, std::string &error);
    bool load_kernel_from_binary_file(const std::string &file_path, const std::string &build_options, std::string &error);
    bool load_kernel_from_source(const std::string &src, std::string &error);
    bool build_kernel(const std::string &build_options, std::string &error);
    bool output_binary(const std::string &path, const std::string &build_options, std::string &error);

    bool create_command_queue(bool enable_profiling, bool enable_out_of_order_exec_mode, std::string &error);

    cl::Program& get_program() { return program_; }
    cl::Context& get_context() { return context_; }
    cl::CommandQueue& get_command_queue() { return queue_; }

    const char* ocl_error_to_string(cl_int err) const;

private:
    cl::Platform platform_;
    cl::Context context_;
    cl::Device device_;
    cl::Program program_;
    cl::CommandQueue queue_;
    size_t max_global_mem_size_{};
    size_t max_local_mem_size_{};
    size_t max_work_group_size_{};
    size_t work_group_size_{};
    bool enable_profiling_{};
    vendor active_vendor_ = other;

};

#endif // PROXY_CL_H
