#include "config.h"

#ifdef OPENCL_NVIDIA
// may be needed under linux
//#define OPENCL_1_1_COMPAT
#endif

#include "opencl_manager.h"

#include "common.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <memory>

cl_manager::cl_manager() :
m_context(nullptr),
m_devices(nullptr),
m_command_queue(nullptr),
m_max_work_item_sizes(nullptr),
m_initialized(true),
m_program_built(false)
{
  cl_int status = 0;
  size_t deviceListSize = 0;

  try
  {
    cl_platform_id platform = get_platform_id();
    cl_context_properties cps[3] =
    {
      CL_CONTEXT_PLATFORM,
      (cl_context_properties) platform,
      0
    };
    cl_context_properties* cprops = (nullptr == platform) ? nullptr : cps;


#if 0
    m_context = clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, 0, 0, &status);
#else
    cl_device_id deviceId = get_device_id(platform, true, status);
#endif

    // if OpenCL fails to open a context on GPU, then try to fall back to CPU 
    // note: unsupported with NVIDIA
    if (status != CL_SUCCESS)
    {
      cout << "Unsupported GPU device (status =" << status << "); falling back to CPU" << endl;
      //m_context = clCreateContextFromType(cprops, CL_DEVICE_TYPE_CPU, 0, 0, &status);
      deviceId = get_device_id(platform, false, status);
      if (status != CL_SUCCESS)
      {
        cout << "Fallback failed (status = " << status << "), exiting" << endl;
        exit(1);
      }
    }
    m_context = clCreateContext(cprops, 1, &deviceId, nullptr, nullptr, &status);
    check_output(status, "clCreateContext / clCreateContextFromType failed.");

    // FIXME: duplication with getDeviceId
    // get the size of device list data 
    check_output(clGetContextInfo(m_context, CL_CONTEXT_DEVICES, 0, nullptr, &deviceListSize),
        "clGetcontextInfo failed.");

    // Then, allocate memory for device list based on that size
    m_devices = new cl_device_id[deviceListSize];

    // get the device list data
    check_output(clGetContextInfo(m_context, CL_CONTEXT_DEVICES, deviceListSize, m_devices, nullptr),
        "clGetGetContextInfo failed.");

    // Get max work group size
    check_output(clGetDeviceInfo(m_devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (size_t),
        (void *) &m_max_workgroup_size, nullptr), "clGetDeviceInfo CL_DEVICE_MAX_WORK_GROUP_SIZE failed.");

    // get max work group / item dimension
    check_output(clGetDeviceInfo(m_devices[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof (cl_uint),
        (void *) &m_max_dimensions, nullptr), "clGetDeviceInfo CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS failed.");

    m_max_work_item_sizes = new size_t[m_max_dimensions];

    check_output(clGetDeviceInfo(m_devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES,
        sizeof (size_t) * m_max_dimensions, (void *) m_max_work_item_sizes, nullptr),
        "clGetDeviceInfo CL_DEVICE_MAX_WORK_ITEM_SIZES failed.");

    check_output(clGetDeviceInfo(m_devices[0], CL_DEVICE_LOCAL_MEM_SIZE, sizeof (cl_ulong),
        (void *) &m_total_local_memory, nullptr), "clGetDeviceInfo CL_DEVICE_LOCAL_MEM_SIZES failed.");

#if defined(OPENCL_AMD) || defined(OPENCL_NVIDIA)
    // FIXME: AMD should probably use newer call clCreateCommandQueueWithProperties
    cl_command_queue_properties prop = 0;
    prop |= CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    m_command_queue = clCreateCommandQueue(m_context, m_devices[0], prop, &status);
#else
    cl_queue_properties prop = 0;
    m_command_queue = clCreateCommandQueueWithProperties(m_context, m_devices[0], &prop, &status);
#endif
    cl_manager::check_output(status, "clCreateCommandQueue failed.");
  }
  catch (exception & e)
  {
    cout << e.what() << endl;
    m_initialized = false;
  }
}

cl_platform_id cl_manager::get_platform_id()
{
  string platformVendor;
  // set a preferred platform vendor. If no platform matches the preferred 
  // platform name, platform 0 will be selected
#ifdef OPENCL_AMD
  platformVendor = "Advanced Micro Devices, Inc.";
#else
#ifdef OPENCL_NVIDIA
  platformVendor = "NVIDIA Corporation";
#else
#ifdef OPENCL_INTEL
  platformVendor = "Intel(R) Corporation";
#endif
#endif
#endif
  cl_platform_id* platforms = nullptr;
  cl_uint numPlatforms;
  cl_platform_id platform = nullptr;
  int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
  cl_manager::check_output(status, "clGetPlatformIDs failed.");
  unsigned int i_selected = 0;
  cout << "Number of available platforms for OpenCL: " << numPlatforms << endl;
  if (0 < numPlatforms)
  {
    platforms = new cl_platform_id[numPlatforms];
    unique_ptr<cl_platform_id[]> _(platforms);
    status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    cl_manager::check_output(status, "clGetPlatformIDs failed.");
    char vendor[100];
    char version[100];
    for (unsigned int i = 0; i < numPlatforms; ++i)
    {
      status = clGetPlatformInfo(platforms[i],
          CL_PLATFORM_VENDOR,
          sizeof (vendor),
          vendor,
          nullptr);
      cl_manager::check_output(status, "clGetPlatformInfo failed.");
      status = clGetPlatformInfo(platforms[i],
          CL_PLATFORM_VERSION,
          sizeof (version),
          version,
          nullptr);
      cl_manager::check_output(status, "clGetPlatformInfo failed.");
      cout << "platform " << i + 1 << " / " << numPlatforms << ":" << endl;
      cout << " vendor: " << vendor << endl;
      cout << " version: " << version << endl;
      cout.flush();
      if (string(vendor) == platformVendor)
      {
        i_selected = i;
      }
    }
    cout << "Selected platform " << i_selected + 1 << endl << endl;
    platform = platforms[i_selected];
  }
  return platform;
}

cl_device_id cl_manager::get_device_id(cl_platform_id p_platform_id, bool p_gpu, int&status)
{
  const size_t maxNumDevices = 4;
  cl_device_id* devices = new cl_device_id[maxNumDevices];
  unique_ptr<cl_device_id[]> _(devices);
  cl_uint numDevices;
  cl_device_id device = nullptr;
  status = CL_SUCCESS;
  cl_bitfield device_type = p_gpu? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
  status = clGetDeviceIDs(p_platform_id, device_type, maxNumDevices, devices, &numDevices);
  //CLManager::checkOutput(status, "clGetDeviceIDs failed.");
  if(status != CL_SUCCESS) return nullptr;
  unsigned int i_selected = 0;
  cout << "Number of available devices on the selected platform: " << numDevices << endl;
  if (0 < numDevices)
  {
    char deviceName[100];
    size_t workGroupSize;
    size_t workItemSize[3];
    cl_uint maxComputeUnits;
    for (unsigned int i = 0; i < numDevices; ++i)
    {
      status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof (deviceName), deviceName, nullptr);
      //CLManager::checkOutput(status, "clGetDeviceInfo failed.");
      if(status != CL_SUCCESS) return  nullptr;
      status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (workGroupSize),
          &workGroupSize, nullptr);
      //CLManager::checkOutput(status, "clGetDeviceInfo failed.");
      if(status != CL_SUCCESS) return  nullptr;
      status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof (workItemSize),
          workItemSize, nullptr);
      //CLManager::checkOutput(status, "clGetDeviceInfo failed.");
      if(status != CL_SUCCESS) return  nullptr;
      status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof (maxComputeUnits),
          &maxComputeUnits, nullptr);
      //CLManager::checkOutput(status, "clGetDeviceInfo failed.");
      if(status != CL_SUCCESS) return nullptr;

      cout << "device " << i + 1 << " / " << numDevices << ":" << endl;
      cout << " name: " << deviceName << endl;
      cout << " Maximum work group size: " << workGroupSize << endl;
      cout << " Maximum work item sizes: " << workItemSize[0] << ", " << workItemSize[1]
          << ", " << workItemSize[2] << endl;
      cout << " Maximum number of parallel compute units: " << maxComputeUnits << endl;
      cout.flush();
    }
    device = devices[i_selected];
  }
  cout << "Selected device " << i_selected + 1 << endl << endl;
  return device;
}

void cl_manager::build_program(const string & p_program_name)
{
  int status;
  cout << "opening kernel file " << p_program_name << endl;
  ifstream kernelFile(p_program_name.c_str());
  if (!kernelFile.is_open())
  {
    throw error("Kernel file could not be opened");
  }
  string line;
  string prg;
  while (!kernelFile.eof())
  {
    getline(kernelFile, line);
    prg += line + "\n";
  }
  const char* source = prg.c_str();

  size_t sourceSize = prg.size();
  m_program = clCreateProgramWithSource(m_context, 1, &source, &sourceSize, &status);

  cl_manager::check_output(status, "clCreateProgramWithSource failed.");
#if defined(OPENCL_NVIDIA) || defined(OPENCL_INTEL)
  status = clBuildProgram(m_program, 1, m_devices, "-cl-mad-enable -cl-strict-aliasing", nullptr, nullptr);
#else
  // default call
  status = clBuildProgram(m_program, 1, m_devices, "", nullptr, nullptr);
#endif
  const size_t max_program_log_size = 1 << 20;
  char* program_log = new char[max_program_log_size];
  unique_ptr<char[]> _(program_log);
  size_t ret_size = 0;
  clGetProgramBuildInfo(m_program, m_devices[0], CL_PROGRAM_BUILD_LOG, max_program_log_size, program_log, &ret_size);
  if (ret_size > 2)
  {
    cout << ret_size << endl;
    cout << "Program build log:";
    cout << program_log << endl << endl;
  }
  cl_manager::check_output(status, "clBuildProgram failed.");
  m_program_built = true;
}

cl_kernel cl_manager::create_kernel(string p_kernel_entry_point)
{
  cl_int status;
  cl_kernel kernel = clCreateKernel(m_program, p_kernel_entry_point.c_str(), &status);
  cl_manager::check_output(status, "clCreateKernel failed.");
  return kernel;
}

void cl_manager::release_kernel(cl_kernel kernel)
{
  clReleaseKernel(kernel);
}

cl_manager::~cl_manager()
{
  if (m_program_built)
  {
    cl_manager::check_output(clReleaseProgram(m_program), "clReleaseProgram failed.");
  }
  cl_manager::check_output(clReleaseCommandQueue(m_command_queue), "clReleaseCommandQueue failed.");
  cl_manager::check_output(clReleaseContext(m_context), "clReleaseContext failed.");

  delete [] m_max_work_item_sizes;
  m_max_work_item_sizes = nullptr;
  delete [] m_devices;
  m_devices = nullptr;
}

cl_uchar& cl_manager::getuchar_ref(cl_uchar4& p_vect, unsigned int p_coordinate)
{
  return p_vect.s[p_coordinate];
}

void cl_manager::check_output(int res, const char* p_error_message, unsigned int argument_index)
{
  if (res != CL_SUCCESS)
  {
    stringstream s;
    s << p_error_message << endl << "Error code: " << res;
    if(argument_index != -1u) cout << ";argument " << argument_index;
    s << endl;
    error e(s.str().c_str());
    throw e;
  }
}

void cl_manager::set_args(cl_kernel p_kernel, list<arg_k>& p_argument_list)
{
  unsigned int current_arg = 0;
  for (list<arg_k>::iterator it = p_argument_list.begin(); it != p_argument_list.end(); it++)
  {
    int status = clSetKernelArg(p_kernel, current_arg, it->m_size, it->m_ptr);
    check_output(status, "clSetKernelArg failed", current_arg);
    current_arg++;
  }
}

void cl_manager::enqueue_call_with_args(cl_kernel p_kernel, list<arg_k>& p_argument_list,
    uint32_t p_num_dimensions, size_t* p_item_sizes, size_t* p_group_sizes, unsigned int num_previous_events,
    cl_event* p_previous_events, cl_event* p_next_event)
{
  if (p_num_dimensions > m_max_dimensions)
  {
    throw error("Too many dimensions for CL Kernel");
  }
  size_t item_size = 1;
  if (p_item_sizes)
  {
    for (unsigned int i = 0; i < p_num_dimensions; i++)
    {
      if (p_item_sizes[i] > m_max_work_item_sizes[i])
      {
        throw error("At least one work item dimension is too large");
      }
      item_size *= p_item_sizes[i];
    }
    if (item_size > m_max_workgroup_size)
    {
      throw error("Unsupported: The requested number of work items is too large");
    }
  }

  set_args(p_kernel, p_argument_list);
  int status = clEnqueueNDRangeKernel(m_command_queue, p_kernel, p_num_dimensions,
      nullptr, p_group_sizes, p_item_sizes, num_previous_events, p_previous_events, p_next_event);
  check_output(status, "clEnqueueNDRangeKernel failed");
}

void cl_manager::enqueue_call(
    cl_kernel p_kernel, uint32_t p_num_dimensions, size_t* p_item_sizes, size_t* p_group_sizes,
    unsigned int num_previous_events, cl_event* p_previous_events, cl_event* p_next_event)
{
  int status = clEnqueueNDRangeKernel(m_command_queue, p_kernel, p_num_dimensions,
      nullptr, p_group_sizes, p_item_sizes, num_previous_events, p_previous_events, p_next_event);
  check_output(status, "clEnqueueNDRangeKernel failed");
}

void cl_manager::enqueue_barrier()
{
  // For OpenCL 1.1
#ifdef OPENCL_1_1_COMPAT
  clEnqueueBarrier(m_command_queue);
#else
  // For OpenCL 1.2+
  clEnqueueBarrierWithWaitList(m_command_queue, 0, nullptr, nullptr);
#endif
}

void cl_manager::finish()
{
  clFinish(m_command_queue);
}

bool cl_manager::check_needed_local_memory(cl_kernel p_kernel, size_t p_needed_local_memory)
{
  cl_int status = clGetKernelWorkGroupInfo(p_kernel, m_devices[0], CL_KERNEL_LOCAL_MEM_SIZE,
      sizeof (cl_ulong), &m_used_local_memory, nullptr);
  check_output(status, "clGetKernelWorkGroupInfo failed.(usedLocalMemory)");
  m_available_local_memory = m_total_local_memory - m_used_local_memory;
  return p_needed_local_memory <= m_available_local_memory;
}

cl_ulong cl_manager::get_total_global_memory() const
{
  cl_ulong size;
  clGetDeviceInfo(m_devices[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (cl_ulong),
      &size, nullptr);
  return size;
}
