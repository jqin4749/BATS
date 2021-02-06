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
scoped_array<cl_mem> output_buf; // num_devices elements

scoped_array<scoped_aligned_ptr<uint8_t> > input_a, input_b; // num_devices elements
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
  output_buf.reset(num_devices);


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
    PKT_SIZE*DEGREE * sizeof(uint8_t), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
    BATCH_SIZE*DEGREE * sizeof(uint8_t), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
    PKT_SIZE*BATCH_SIZE * sizeof(uint8_t), NULL, &status);
    checkError(status, "Failed to create buffer for output");
    
    }
  printf("Finished Initialization\n");
  return true;
}

// Initialize the data for the problem. Requires num_devices to be known.
void init_problem() {
  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  input_a.reset(num_devices);
  input_b.reset(num_devices);
  output.reset(num_devices);
  ref_output.reset(num_devices);

  // Generate input vectors A and B and the reference output consisting
  // of a total of N elements.
  // We create separate arrays for each device so that each device has an
  // aligned buffer.
  for(unsigned i = 0; i < num_devices; ++i) {
    input_a[i].reset(PKT_SIZE*DEGREE);
    input_b[i].reset(BATCH_SIZE*DEGREE);
    output[i].reset(PKT_SIZE*BATCH_SIZE);
    ref_output[i].reset(n_per_device[i]);
    int count=0;
    for(int j=0;j<DEGREE;j++){
      for(int jj=0;jj<PKT_SIZE;jj++){
        input_a[i][count] = B[jj][j];
        count++;
      }
    }
    
    count = 0;
    for(int j=0;j<BATCH_SIZE;j++){
      for(int jj=0;jj<DEGREE;jj++){
        input_b[i][count] = G[jj][j];
        count++;
      }
    }

    for (int m=0; m<PKT_SIZE; m++) {
        for (int n=0; n<BATCH_SIZE; n++) {
            uint8_t acc = 0;
            for (int k=0; k<DEGREE; k++) {
                acc ^= gf_mu_x86(input_a[i][k*PKT_SIZE + m],input_b[i][n*DEGREE + k]);
            }
            ref_output[i][n*PKT_SIZE + m] = acc;
            // printf("%d,",acc);
        }
    }
    printf("\n");
  }
}



void run() {
  cl_int status;

  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  scoped_array<cl_event> kernel_event(num_devices);
  scoped_array<cl_event> finish_event(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {


    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
    status = clEnqueueWriteBuffer(queue[i], input_a_buf[i], CL_FALSE,
        0, PKT_SIZE*DEGREE * sizeof(uint8_t), input_a[i], 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue[i], input_b_buf[i], CL_FALSE,
        0, BATCH_SIZE*DEGREE * sizeof(uint8_t), input_b[i], 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");


    // Set kernel arguments.
    unsigned argi = 0;


    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_a_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_b_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &output_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);


    // Enqueue kernel.
   
    const size_t global[2] = { PKT_SIZE, BATCH_SIZE };
    const size_t local[2] = { TS, TS };
    // printf("Launching for device %d (%zd elements)\n", i, global_work_size);


    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 2, NULL,
        global, local, 2, write_event, &kernel_event[i]);

    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue[i], output_buf[i], CL_FALSE,
        0, PKT_SIZE*BATCH_SIZE * sizeof(uint8_t), output[i], 1, &kernel_event[i], &finish_event[i]);

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);

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
  }

  // Release all events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
    clReleaseEvent(finish_event[i]);
  }

  // Verify results.
  bool pass = true;
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    for(unsigned j = 0; j < PKT_SIZE*BATCH_SIZE && pass; ++j) {
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
  run();

  // Free the resources allocated
  cleanup();

  return 0;
}

