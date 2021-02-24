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
cl_device_id device;; // num_devices elements
cl_context context = NULL;
cl_command_queue queue; // num_devices elements
cl_program program = NULL;
cl_kernel kernel; // num_devices elements

cl_mem input_a_buf; // num_devices elements
cl_mem input_b_buf; // num_devices elements
cl_mem input_DE_buf; // num_devices elements
cl_mem output_buf; // num_devices elements
cl_mem input_sample_buf; // num_devices elements
cl_mem input_DEOFF_buf; // num_devices elements


// scoped_array<scoped_aligned_ptr<uint8_t> > input_a, input_b, input_deg, input_sample_idx, test_out; // num_devices elements
scoped_aligned_ptr<uint8_t>  output; // num_devices elements
scoped_array<uint8_t>  ref_output; // num_devices elements

// Function prototypes
bool init_opencl();
void init_problem();
void run();
void cleanup();

template<int N_BATCH_>
int get_B_size(uint8_t deg_list[N_BATCH_],int n_batch){
    int len = 0;
    for(int i=0;i<n_batch;i++){
        len+=deg_list[i];
    }    
    return len*BATCH_SIZE;
}

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
  device = getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices)[0];
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("myGEMM6", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");


 
  // Command queue.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Kernel.
  const char *kernel_name = "myGEMM6";
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel");

  // Input buffers.
  input_a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
  FILE_SIZE * sizeof(uint8_t), NULL, &status);
  checkError(status, "Failed to create buffer for input A");

  input_b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
  BATCH_SIZE*MAX_DEGREE*MAX_NUM_BATCH * sizeof(uint8_t), NULL, &status);
  checkError(status, "Failed to create buffer for input B");

  input_DE_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
  MAX_NUM_BATCH * sizeof(uint8_t), NULL, &status);
  checkError(status, "Failed to create buffer for input DE");
  

  input_sample_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
  MAX_DEGREE*MAX_NUM_BATCH * sizeof(uint8_t), NULL, &status);
  checkError(status, "Failed to create buffer for input sample idx");
  // Output buffer.

  output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
  PKT_SIZE*BATCH_SIZE* MAX_NUM_BATCH * sizeof(uint8_t), NULL, &status);
  checkError(status, "Failed to create buffer for output");
  
  input_DEOFF_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,
  MAX_NUM_BATCH * sizeof(int), NULL, &status);
  checkError(status, "Failed to create buffer for input DE");

  // Set kernel arguments.

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_a_buf);
  checkError(status, "Failed to set argument %d", 0);

  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &input_b_buf);
  checkError(status, "Failed to set argument %d", 1);

  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buf);
  checkError(status, "Failed to set argument %d", 2);

  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &input_DE_buf);
  checkError(status, "Failed to set argument %d", 3);

  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &input_sample_buf);
  checkError(status, "Failed to set argument %d", 4);

  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &input_DEOFF_buf);
  checkError(status, "Failed to set argument %d", 5);
  
  printf("Finished Initialization\n");
  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }
  
  printf("starting init problem\n");


  output = (uint8_t*) malloc(PKT_SIZE*BATCH_SIZE*N_BATCH);
  ref_output = (uint8_t*) malloc(PKT_SIZE*BATCH_SIZE*N_BATCH);

    printf("computing golden ref\n");
  
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
  int b_size = get_B_size<N_BATCH>(deg_list,N_BATCH);
  uint8_t* B_trans = (uint8_t*)malloc(b_size);
  count = 0;
  for(int b=0;b<N_BATCH;b++){
    int d = deg_list[b];
    for(int j=0;j<BATCH_SIZE;j++){
      for(int k=0;k<d;k++){
        B_trans[count] = B[k*BATCH_SIZE+j + BATCH_SIZE*offset_list[b]];
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
                acc ^= gf_mu_x86(golden_A[k*PKT_SIZE + m + PKT_SIZE*offset_list[b]],B_trans[n*d + k + BATCH_SIZE*offset_list[b]]);
            }
            ref_output[n*PKT_SIZE + m + b*PKT_SIZE*BATCH_SIZE] = acc;
            // printf("%d,",acc);
        }
    }
  }
  free(B_trans);

  printf("finished init problem\n");
  
}



void run() {
  printf("running\n");
  cl_int status;
 
  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  cl_event kernel_event;
  cl_event finish_event;
  

  cl_event write_event[5];


  // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
  // for the host-to-device transfer.
  
  status = clEnqueueWriteBuffer(queue, input_a_buf, CL_FALSE,
      0, FILE_SIZE * sizeof(uint8_t), input_file, 0, NULL, &write_event[0]);
  checkError(status, "Failed to transfer input A");

  status = clEnqueueWriteBuffer(queue, input_b_buf, CL_FALSE,
      0, sizeof(B) , B, 0, NULL, &write_event[1]);
  checkError(status, "Failed to transfer input B");

  status = clEnqueueWriteBuffer(queue, input_DE_buf, CL_FALSE,
      0, sizeof(deg_list), deg_list, 0, NULL, &write_event[2]);
  checkError(status, "Failed to transfer input DE");

  status = clEnqueueWriteBuffer(queue, input_sample_buf, CL_FALSE,
      0, sizeof(sample_idx), sample_idx, 0, NULL, &write_event[3]);
  checkError(status, "Failed to transfer sampled idx");

  status = clEnqueueWriteBuffer(queue, input_DEOFF_buf, CL_FALSE,
      0, sizeof(offset_list), offset_list, 0, NULL, &write_event[4]);
  checkError(status, "Failed to transfer sampled idx");

  // Enqueue kernel.
  
  const size_t global[3] = {  PKT_SIZE/WPTM, BATCH_SIZE/WPTN, N_BATCH }; // 128, 1, 1
  const size_t local[3] = { TSM/WPTM, TSN/WPTN, 1 };

  status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL,
      global, local, 5, write_event, &kernel_event);
  checkError(status, "Failed to launch kernel");

  // Read the result. This the final operation.
  status = clEnqueueReadBuffer(queue, output_buf, CL_FALSE,
      0, PKT_SIZE*BATCH_SIZE*N_BATCH*sizeof(uint8_t), output, 1, &kernel_event, &finish_event);
  checkError(status, "Failed to read");
  

  // Wait for all devices to finish.
  clWaitForEvents(num_devices, &finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  
  cl_ulong time_ns = getStartEndTime(kernel_event);
  printf("Kernel time: %0.3f ms\n", double(time_ns) * 1e-6);

  time_ns = getStartEndTime(write_event[0]);
  printf("Write 1: %0.3f ms\n", double(time_ns) * 1e-6);
  time_ns = getStartEndTime(write_event[1]);
  printf("Write 2: %0.3f ms\n",double(time_ns) * 1e-6);
  time_ns = getStartEndTime(write_event[2]);
  printf("Write 3: %0.3f ms\n", double(time_ns) * 1e-6);
  time_ns = getStartEndTime(write_event[3]);
  printf("Write 4: %0.3f ms\n", double(time_ns) * 1e-6);
  time_ns = getStartEndTime(write_event[4]);
  printf("Write 5: %0.3f ms\n", double(time_ns) * 1e-6);
  time_ns = getStartEndTime(finish_event);
  printf("Read: %0.3f ms\n",  double(time_ns) * 1e-6);


  clReleaseEvent(write_event[0]);
  clReleaseEvent(write_event[1]);
  clReleaseEvent(write_event[2]);
  clReleaseEvent(write_event[3]);
  clReleaseEvent(write_event[4]);
  // Release all events.

  clReleaseEvent(kernel_event);
  clReleaseEvent(finish_event);


  // Verify results.
  bool pass = true;
  for(unsigned j = 0; j < PKT_SIZE*BATCH_SIZE*N_BATCH && pass; ++j) {
    // printf("%d,%d\n",output[i][j],ref_output[i][j]);
    if(output[j] != ref_output[j] ) {
      printf("Failed verification, index %d\nOutput: %d\nReference: %d\n",
           j, output[j], ref_output[j]);
      pass = false;
    }
    
  }
  
  printf("\nVerification: %s\n", pass ? "PASS" : "FAIL");
}

void cleanup() {
  
  if(kernel) {
    clReleaseKernel(kernel);
  }
  if(queue) {
    clReleaseCommandQueue(queue);
  }
  if(input_a_buf) {
    clReleaseMemObject(input_a_buf);
  }
  if(input_b_buf) {
    clReleaseMemObject(input_b_buf);
  }
  if(output_buf) {
    clReleaseMemObject(output_buf);
  }
  if(input_DE_buf) {
    clReleaseMemObject(input_DE_buf);
  }
  if(input_sample_buf) {
    clReleaseMemObject(input_sample_buf);
  }
  if(input_DEOFF_buf) {
    clReleaseMemObject(input_DEOFF_buf);
  }

  // if(program) {
  //   clReleaseProgram(program);
  // }
  if(context) {
    clReleaseContext(context);
  }
}


// Entry point.
int main(int argc, char **argv) {

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

