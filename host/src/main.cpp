#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#include "config.h"
#include "test_matrix.h"
using namespace aocl_utils;

// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queue; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernel; // num_devices elements

scoped_array<cl_mem> input_a_buf; // num_devices elements
scoped_array<cl_mem> input_b_buf; // num_devices elements
scoped_array<cl_mem> input_DE_buf; // num_devices elements
scoped_array<cl_mem> output_buf; // num_devices elements
scoped_array<cl_mem> input_sample_buf; // num_devices elements

scoped_array<scoped_aligned_ptr<uint8_t> > input_a, input_b, input_deg, input_sample_idx; // num_devices elements
scoped_array<scoped_aligned_ptr<uint8_t> > output; // num_devices elements

scoped_array<scoped_array<uint8_t> > ref_output; // num_devices elements
scoped_array<unsigned> n_per_device; // num_devices elements
// Problem data.
unsigned N = 1000000; // problem size
// Function prototypes
bool init_opencl();
void init_problem();
void run();
void cleanup();


uint8_t gf_mu_x86(uint8_t a, uint8_t b) {
	uint8_t p = 0; /* the product of the multiplication */

	while (a && b) {
            if (b & 1) /* if b is odd, then add the corresponding a to p (final product = sum of all a's corresponding to odd b's) */
                p ^= a; /* since we're in GF(2^m), addition is an XOR */

            if (a & 0x80) /* GF modulo: if a >= 128, then it will overflow when shifted left, so reduce */
                a = (a << 1) ^ 0x11D; /* XOR with the primitive polynomial x^8 + x^4 + x^3 + x + 1 (0b1_0001_1011) – you can change it but it must be irreducible */
            else
                a <<= 1; /* equivalent to a*2 */
            b >>= 1; /* equivalent to b // 2 */
	}
	return p;
}

// Initializes the OpenCL objects.
bool init_opencl()
{
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("myGEMM2", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create per-device objects.
  queue.reset(num_devices);
  kernel.reset(num_devices);
  n_per_device.reset(num_devices);

  input_a_buf.reset(num_devices);
  input_b_buf.reset(num_devices);
  input_DE_buf.reset(num_devices);
  output_buf.reset(num_devices);
  input_sample_buf.reset(num_devices);


  input_a.reset(num_devices);
  input_b.reset(num_devices);
  input_deg.reset(num_devices);
  output.reset(num_devices);
  ref_output.reset(num_devices);
  input_sample_idx.reset(num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    const char *kernel_name = "myGEMM2";
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of elements processed by this device.
    n_per_device[i] = N / num_devices; // number of elements handled by this device

    // Spread out the remainder of the elements over the first
    // N % num_devices.
    if(i < (N % num_devices)) {
      n_per_device[i]++;
    }
    // Input buffers.
    input_a_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
    FILE_SIZE * sizeof(uint8_t), NULL, &status);
    checkError(status, "Failed to create buffer for input A");
  
    input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
    BATCH_SIZE*MAX_DEGREE*MAX_NUM_BATCH * sizeof(uint8_t), NULL, &status);
    checkError(status, "Failed to create buffer for input B");
  
    input_DE_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
    MAX_NUM_BATCH * sizeof(uint8_t), NULL, &status);
    checkError(status, "Failed to create buffer for input DE");
   

    input_sample_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
    MAX_DEGREE*MAX_NUM_BATCH * sizeof(uint8_t), NULL, &status);
    checkError(status, "Failed to create buffer for input sample idx");
    // Output buffer.
    printf("4\n");

    output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
    PKT_SIZE*BATCH_SIZE* MAX_NUM_BATCH * sizeof(uint8_t), NULL, &status);
    checkError(status, "Failed to create buffer for output");
    
    // Set kernel arguments.

    status = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &input_a_buf[i]);
    checkError(status, "Failed to set argument %d", 0);

    status = clSetKernelArg(kernel[i], 1, sizeof(cl_mem), &input_b_buf[i]);
    checkError(status, "Failed to set argument %d", 1);

    status = clSetKernelArg(kernel[i], 2, sizeof(cl_mem), &output_buf[i]);
    checkError(status, "Failed to set argument %d", 2);

    status = clSetKernelArg(kernel[i], 3, sizeof(cl_mem), &input_DE_buf[i]);
    checkError(status, "Failed to set argument %d", 3);

    status = clSetKernelArg(kernel[i], 4, sizeof(cl_mem), &input_sample_buf[i]);
    checkError(status, "Failed to set argument %d", 3);
    

    }
  printf("Finished Initialization\n");
  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }
  
  printf("starting init problem\n");
  // Generate input vectors A and B and the reference output consisting
  // of a total of N elements.
  // We create separate arrays for each device so that each device has an
  // aligned buffer.

  for(unsigned i = 0; i < num_devices; ++i) {

  // input_a[i].reset(FILE_SIZE);
  // input_b[i].reset(BATCH_SIZE*MAX_DEGREE*MAX_NUM_BATCH);
  // input_deg[i].reset(N_BATCH);
  // input_sample_idx[i].reset(MAX_NUM_BATCH*MAX_DEGREE);

  output[i].reset(PKT_SIZE*BATCH_SIZE*N_BATCH);
  ref_output[i].reset(PKT_SIZE*BATCH_SIZE*N_BATCH);

    // for (int b=0;b<N_BATCH;b++){
    //   input_deg[i][b] = deg_list[b];
    // }

    // for(int j=0;j<FILE_SIZE;j++){
    //   input_a[i][j] = input_file[j];
    // }
    
    // for(int j=0;j<N_BATCH*MAX_DEGREE;j++){
    //   input_sample_idx[i][j] = sample_idx[j];
    // }

    // for(int j=0;j<BATCH_SIZE*MAX_DEGREE*N_BATCH;j++){
    //   input_b[i][j] = B[j];
    // }
    printf("computing golden ref\n");
    // computing golden reference
    // got the offsets
    int offset_list[N_BATCH]={0};
    for(int j=0;j<N_BATCH;j++){
      for(int jj=0;jj<j;jj++){
        offset_list[j] += deg_list[jj];
      }
    }
    
    
    // construct naive matrix A (PKT_SIZE,DEGREE,N_BATCH)s
    int golden_A[MAX_DEGREE*MAX_NUM_BATCH*PKT_SIZE]; // MAX_DEGREE is too large
  
    int count = 0;
    for(int idx=0;idx<N_BATCH;idx++){
      int cur_deg =deg_list[idx];
      int cur_offset = offset_list[idx];
      for(int dd=0;dd<cur_deg;dd++){
        int cur_sample_idx = sample_idx[cur_offset+dd];
        for(int pk=0;pk<PKT_SIZE;pk++){
          golden_A[count] = input_file[pk+cur_sample_idx*PKT_SIZE];
          count++;
        }
      }
    }
   
    // compute golden A * matrix B
    for(int b=0;b<N_BATCH;b++){
      int d = deg_list[b];
      for (int m=0; m<PKT_SIZE; m++) {
          for (int n=0; n<BATCH_SIZE; n++) {
              uint8_t acc = 0;
              for (int k=0; k<d; k++) {
                  acc ^= gf_mu_x86(golden_A[k*PKT_SIZE + m + PKT_SIZE*offset_list[b]],B[n*d + k + BATCH_SIZE*offset_list[b]]);
              }
              ref_output[i][n*PKT_SIZE + m + b*PKT_SIZE*BATCH_SIZE] = acc;
              // printf("%d,",acc);
          }
      }
    }
  }
  printf("finished init problem\n");
}



void run() {
  printf("running\n");
  cl_int status;
 
  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  scoped_array<cl_event> kernel_event(num_devices);
  scoped_array<cl_event> finish_event(num_devices);
  cl_event write_event[4];
  for(unsigned i = 0; i < num_devices; ++i) {


    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    
    status = clEnqueueWriteBuffer(queue[i], input_a_buf[i], CL_FALSE,
        0, FILE_SIZE * sizeof(uint8_t), input_file, 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue[i], input_b_buf[i], CL_FALSE,
        0, sizeof(B) * sizeof(uint8_t), B, 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");

    status = clEnqueueWriteBuffer(queue[i], input_DE_buf[i], CL_FALSE,
        0, sizeof(deg_list)*sizeof(uint8_t), deg_list, 0, NULL, &write_event[2]);
    checkError(status, "Failed to transfer input DE");

    status = clEnqueueWriteBuffer(queue[i], input_sample_buf[i], CL_FALSE,
        0, sizeof(sample_idx)*sizeof(uint8_t), sample_idx, 0, NULL, &write_event[3]);
    checkError(status, "Failed to transfer sampled idx");

    // Enqueue kernel.
   
    const size_t global[3] = { PKT_SIZE, BATCH_SIZE, N_BATCH };
    const size_t local[3] = { TS, TS, N_BATCH };
    // printf("Launching for device %d (%zd elements)\n", i, global_work_size);

    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 3, NULL,
        global, local, 4, write_event, &kernel_event[i]);
    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue[i], output_buf[i], CL_FALSE,
        0, PKT_SIZE*BATCH_SIZE*N_BATCH*sizeof(uint8_t), output[i], 1, &kernel_event[i], &finish_event[i]);

    
  
  }

  // Wait for all devices to finish.
  clWaitForEvents(num_devices, finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  for(unsigned i = 0; i < num_devices; ++i) {
    cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);

    time_ns = getStartEndTime(write_event[0]);
    printf("Write 1: %0.3f ms\n", i, double(time_ns) * 1e-6);
    time_ns = getStartEndTime(write_event[1]);
    printf("Write 2: %0.3f ms\n", i, double(time_ns) * 1e-6);
    time_ns = getStartEndTime(write_event[2]);
    printf("Write 3: %0.3f ms\n", i, double(time_ns) * 1e-6);
    time_ns = getStartEndTime(write_event[3]);
    printf("Write 4: %0.3f ms\n", i, double(time_ns) * 1e-6);
    time_ns = getStartEndTime(finish_event[i]);
    printf("Read: %0.3f ms\n", i, double(time_ns) * 1e-6);
  }

  clReleaseEvent(write_event[0]);
  clReleaseEvent(write_event[1]);
  clReleaseEvent(write_event[2]);
  clReleaseEvent(write_event[3]);
  // Release all events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
    clReleaseEvent(finish_event[i]);
  }

  // Verify results.
  bool pass = true;
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    
      for(unsigned j = 0; j < PKT_SIZE*BATCH_SIZE*N_BATCH ; ++j) {
        // printf("%d,%d\n",output[i][j],ref_output[i][j]);
        if(output[i][j] != ref_output[i][j]) {
          printf("Failed verification @ device %d, index %d\nOutput: %d\nReference: %d\n",
              i, j, output[i][j], ref_output[i][j]);
          pass = false;
        }
       
      }
  }

  printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");
}

void cleanup() {

  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
    if(input_a_buf && input_a_buf[i]) {
      clReleaseMemObject(input_a_buf[i]);
    }
    if(input_b_buf && input_b_buf[i]) {
      clReleaseMemObject(input_b_buf[i]);
    }
    if(output_buf && output_buf[i]) {
      clReleaseMemObject(output_buf[i]);
    }

  }

  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}


// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);

  // Optional argument to specify the problem size.
  if(options.has("n")) {
    N = options.get<unsigned>("n");
  }

  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }
  // Initialize the problem data.
  // Requires the number of devices to be known.
  init_problem();

  // Run the kernel.
  for(int i=0;i<5;i++){
    run();
  }
  

  // Free the resources allocated
  cleanup();

  return 0;
}

