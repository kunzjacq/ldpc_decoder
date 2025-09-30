#pragma once

#include "config.h"

#define CL_TARGET_OPENCL_VERSION 120

#include <CL/cl.h>
#include <string>
#include <list>

using namespace std;

class cl_manager
{
  public:

  enum _bufType{rw, r, w} bufType;

  class arg_k
  {
  public:
    size_t  m_size;
    const void*   m_ptr;
    arg_k(size_t p_size, const void* p_ptr):m_size(p_size), m_ptr(p_ptr){}
  };


    cl_context        m_context;        // CL context 
    cl_device_id*     m_devices;        // CL device list 
    cl_command_queue  m_command_queue;  // CL command queue
    cl_program        m_program;        // CL program

    size_t        m_max_workgroup_size;
    cl_uint       m_max_dimensions;
    size_t*       m_max_work_item_sizes;

    cl_ulong      m_total_local_memory;
    cl_ulong      m_used_local_memory;
    cl_ulong      m_available_local_memory;
    cl_ulong      m_needed_local_memory;

    bool m_initialized;
    bool m_program_built;

    cl_manager();
    ~cl_manager();
    void build_program(const std::string & p_program_name);
    cl_kernel create_kernel(std::string p_kernel_entry_point);
    void release_kernel(cl_kernel kernel);
    static cl_uchar& getuchar_ref(cl_uchar4& p_vect, unsigned int p_coordinate);
    static void check_output(int res, const char* p_error_message, unsigned int argument_index = -1);
    void set_args(cl_kernel p_kernel, list<arg_k>& p_argument_list);
    void enqueue_call_with_args(cl_kernel p_kernel, std::list<arg_k>& p_argumentList,
      uint32_t p_num_dimensions, size_t* p_item_sizes, size_t* p_group_sizes, unsigned int num_previous_events,
      cl_event* p_previous_events, cl_event* p_next_event);
    void enqueue_call(cl_kernel p_kernel, uint32_t p_num_dimensions, size_t* p_item_sizes,
      size_t* p_group_sizes, unsigned int num_previous_events, cl_event* p_previous_events, cl_event* p_next_event);
    template<class T>
    void create_host_buffer_from_pointer(enum _bufType p_dev_buf_type, unsigned int p_num_elements,
      cl_mem& po_buf, T* p_buf_addr);
    template<class T>
    void create_host_constant_buffer_from_pointer(unsigned int p_num_elements, cl_mem& po_buf,
      T* p_buf_addr);
    template<class T>
    void create_host_pinned_buffer(enum _bufType p_dev_buf_type, size_t p_buffer_size,
      cl_mem& po_buf, T*& po_mem_area);
    template<class T>
    cl_mem create_device_buffer(enum _bufType p_dev_buf_type, size_t p_buffer_size);
    template<class T>
    void create_device_buffer_from_host_data(enum _bufType p_dev_buf_type, size_t p_buffer_size,
      cl_mem& po_buf, T* p_buf_addr);
    bool check_needed_local_memory(cl_kernel p_kernel, size_t p_needed_local_memory);
    template<class T>
    void enqueue_write(
        cl_mem& p_buf, cl_bool p_blocking_write, size_t p_offset, size_t p_copy_size,
        const T* p_ptr, cl_uint p_num_previous_event, cl_event* p_ptr_previous_event,
        cl_event* p_ptr_next_event);
    template<class T>
    void enqueue_read(
        cl_mem& p_buf, cl_bool p_blocking_read, size_t p_offset, size_t p_copy_size,
        T* p_ptr, cl_uint p_num_previous_event, const cl_event* p_ptr_previous_event,
        cl_event* p_ptr_next_event);
    template<class T>
    void enqueue_clear(
        cl_mem& p_buf, size_t p_offset, size_t p_copy_size, cl_uint p_num_previous_event,
        cl_event* p_ptr_previous_event, cl_event* p_ptr_next_event);
    void enqueue_barrier();
    void finish();
    cl_platform_id get_platform_id();
    cl_device_id get_device_id(cl_platform_id p_platform_id, bool p_gpu, int& status);
    cl_ulong get_total_global_memory () const;
};

template<class T>
void cl_manager::create_host_buffer_from_pointer(
    enum _bufType p_dev_buf_type, unsigned int p_num_elements, cl_mem& po_buf, T* p_buf_addr)
{
  cl_int status = 0;
  cl_mem_flags l_flags = 0;
  switch(p_dev_buf_type)
  {
  case rw:
    l_flags = CL_MEM_READ_WRITE;
    break;
  case r:
    l_flags = CL_MEM_READ_ONLY;
    break;
  case w:
    l_flags = CL_MEM_WRITE_ONLY;
  }

  po_buf = clCreateBuffer(m_context, l_flags  | CL_MEM_USE_HOST_PTR, p_num_elements,
    (void*) p_buf_addr, &status);
  check_output(status, "clCreateBuffer failed.");
}

template<class T>
void cl_manager::create_host_constant_buffer_from_pointer(
    unsigned int p_numElements, cl_mem& po_buf, T* p_buf_addr)
{
  create_host_buffer_from_pointer(r, p_numElements, po_buf, p_buf_addr);
}

template<class T>
void cl_manager::create_host_pinned_buffer(
    enum _bufType p_dev_buf_type, size_t p_buffer_size, cl_mem& po_buf, T*& po_mem_area)
{
  cl_int status = 0;
  cl_mem_flags flags = 0;
  cl_map_flags map_flags = 0;
  switch(p_dev_buf_type)
  {
  case rw:
    flags = CL_MEM_READ_WRITE;
    map_flags = CL_MAP_READ | CL_MAP_WRITE;
    break;
  case r:
    flags = CL_MEM_READ_ONLY;
    map_flags = CL_MAP_WRITE;
    break;
  case w:
    flags = CL_MEM_WRITE_ONLY;
    map_flags = CL_MAP_READ;
  }

  po_buf = clCreateBuffer(m_context, flags  | CL_MEM_ALLOC_HOST_PTR, p_buffer_size*sizeof(T),
    nullptr, &status);
  check_output(status, "clCreateBuffer failed.");
  po_mem_area  = static_cast<T*>(clEnqueueMapBuffer(m_command_queue, po_buf, CL_TRUE, map_flags, 0,
    p_buffer_size*sizeof(T), 0, nullptr, nullptr, &status));
  check_output(status, "clEnqueueMapBuffer failed.");
}


template<class T>
void cl_manager::create_device_buffer_from_host_data(
    enum _bufType p_dev_buf_type, size_t p_buffer_size, cl_mem& po_buf, T* p_buf_addr)
{
  cl_int status = 0;
  cl_mem_flags flags = 0;
  switch(p_dev_buf_type)
  {
  case rw:
    flags = CL_MEM_READ_WRITE;
    break;
  case r:
    flags = CL_MEM_READ_ONLY;
    break;
  case w:
    flags = CL_MEM_WRITE_ONLY;
  }
  po_buf = clCreateBuffer(m_context, flags | CL_MEM_COPY_HOST_PTR,
    p_buffer_size * sizeof(T), (void*) p_buf_addr, &status);
  check_output(status, "clCreateBuffer failed.");
}

template<class T>
void cl_manager::enqueue_write(
    cl_mem& p_buf, cl_bool p_blocking_write, size_t p_offset, size_t p_copy_size,
    const T* p_ptr, cl_uint p_num_previous_event, cl_event* p_ptr_previous_event,
    cl_event* p_ptr_next_event)
{
  cl_int status = clEnqueueWriteBuffer(
        m_command_queue, p_buf, p_blocking_write, sizeof(T) * p_offset, sizeof(T) * p_copy_size,
        (const void*) p_ptr, p_num_previous_event, p_ptr_previous_event, p_ptr_next_event);
  cl_manager::check_output(status, "clEnqueueWriteBuffer failed.");
}

template<class T>
void cl_manager::enqueue_read(
    cl_mem& p_buf, cl_bool p_blocking_read, size_t p_offset, size_t p_copy_size,
    T* p_ptr, cl_uint p_num_previous_event, const cl_event* p_ptr_previous_event,
    cl_event* p_ptr_next_event)
{
  cl_int status = clEnqueueReadBuffer(
    m_command_queue, p_buf, p_blocking_read, sizeof(T) * p_offset, sizeof(T) * p_copy_size,
                          static_cast<void *>(p_ptr), p_num_previous_event, p_ptr_previous_event, p_ptr_next_event);
  cl_manager::check_output(status, "clEnqueueReadBuffer failed.");
}

template<class T>
void cl_manager::enqueue_clear(
    cl_mem& p_buf, size_t p_offset, size_t p_copy_size, cl_uint p_num_previous_event,
    cl_event* p_ptr_previous_event, cl_event* p_ptr_next_event)
{
  T pat = 0;
  cl_int status = clEnqueueFillBuffer(
        m_command_queue, p_buf, (void*) &pat, sizeof(T), sizeof(T) * p_offset,
        sizeof(T) * p_copy_size, p_num_previous_event, p_ptr_previous_event,
        p_ptr_next_event);

  cl_manager::check_output(status, "clEnqueueFillBuffer failed.");
}

template <class T> cl_mem cl_manager::create_device_buffer(enum _bufType p_devBufType, size_t p_buffer_size) {
  cl_int status = 0;
  cl_mem_flags flags = 0;
  switch (p_devBufType) {
  case rw:
    flags = CL_MEM_READ_WRITE;
    break;
  case r:
    flags = CL_MEM_READ_ONLY;
    break;
  case w:
    flags = CL_MEM_WRITE_ONLY;
  }
  cl_mem buf = clCreateBuffer(m_context, flags, sizeof(T) * p_buffer_size, nullptr, &status);
  check_output(status, "clCreateBuffer failed.");
  return buf;
}
