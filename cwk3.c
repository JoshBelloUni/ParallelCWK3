//
// Starting point for the OpenCL coursework for COMP3221 Parallel Computation.
//
// Once compiled, execute with the number of rows and columns for the matrix, e.g.
//
// ./cwk3 16 8
//
// This will display the matrix, followed by another matrix that has not been transposed
// correctly. You need to implement OpenCL code so that the transpose is correct.
//
// For this exercise, both the number of rows and columns must be a power of 2,
// i.e. one of 1, 2, 4, 8, 16, 32, ...
//


//
// Includes.
//
#include <stdio.h>
#include <stdlib.h>

// For this coursework, the helper file has 3 routines in addition to simpleOpenContext_GPU() and compileKernelFromFile():
// - getCmdLineArgs(): Gets the command line arguments and checks they are valid.
// - displayMatrix() : Displays the matrix, or just the top-left corner if it is too large.
// - fillMatrix()    : Fills the matrix with random values.
// Do not alter these routines, as they will be replaced with different versions for assessment.
#include "helper_cwk.h"


//
// Main.
//
int main( int argc, char **argv )
{
    //
    // Parse command line arguments and check they are valid. Handled by a routine in the helper file.
    //
    int nRows, nCols;
    getCmdLineArgs( argc, argv, &nRows, &nCols );

    //
    // Initialisation.
    //

    // Set up OpenCL using the routines provided in helper_cwk.h.
    cl_device_id device;
    cl_context context = simpleOpenContext_GPU(&device);

    // Open up a single command queue, with the profiling option off (third argument = 0).
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue( context, device, 0, &status );

    // Allocate memory for the matrix.
    float *hostMatrix = (float*) malloc( nRows*nCols*sizeof(float) );

    // Fill the matrix with random values, and display.
    fillMatrix( hostMatrix, nRows, nCols );
    printf( "Original matrix (only top-left shown if too large):\n" );
    displayMatrix( hostMatrix, nRows, nCols );


    //
    // Transpose the matrix on the GPU.
    //

    // Compile the transpose kernel.
    cl_kernel kernel = compileKernelFromFile( "cwk3.cl", "transpose", context, device );

    // Allocate device buffers for input and output.
    cl_mem d_input  = clCreateBuffer( context, CL_MEM_READ_ONLY,  nRows*nCols*sizeof(float), NULL, &status );
    cl_mem d_output = clCreateBuffer( context, CL_MEM_WRITE_ONLY, nRows*nCols*sizeof(float), NULL, &status );

    // Copy the host matrix to the device input buffer.
    clEnqueueWriteBuffer( queue, d_input, CL_TRUE, 0, nRows*nCols*sizeof(float), hostMatrix, 0, NULL, NULL );

    // Set kernel arguments: input, output, nRows, nCols.
    clSetKernelArg( kernel, 0, sizeof(cl_mem), &d_input  );
    clSetKernelArg( kernel, 1, sizeof(cl_mem), &d_output );
    clSetKernelArg( kernel, 2, sizeof(int),    &nRows    );
    clSetKernelArg( kernel, 3, sizeof(int),    &nCols    );

    // Global work size: round up nCols and nRows to the nearest multiple of TILE_SIZE.
    // Since both are powers of 2, this is only needed when they are smaller than TILE_SIZE.
    #define TILE_SIZE 16
    size_t globalSize[2] = {
        ((nCols + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE,
        ((nRows + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE
    };
    size_t localSize[2] = { TILE_SIZE, TILE_SIZE };

    // Enqueue the kernel and wait for it to finish.
    clEnqueueNDRangeKernel( queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL );
    clFinish( queue );

    // Copy the transposed result back to the host matrix array.
    clEnqueueReadBuffer( queue, d_output, CL_TRUE, 0, nRows*nCols*sizeof(float), hostMatrix, 0, NULL, NULL );

    // Release device resources.
    clReleaseMemObject( d_input  );
    clReleaseMemObject( d_output );
    clReleaseKernel   ( kernel   );

    //
    // Display the final result. This assumes that the transposed matrix was copied back to the hostMatrix array
    // (note the arrays are the same total size before and after transposing - nRows * nCols - so there is no risk
    // of accessing unallocated memory).
    //
    printf( "Transposed matrix (only top-left shown if too large):\n" );
    displayMatrix( hostMatrix, nCols, nRows );


    //
    // Release all resources.
    //
    clReleaseCommandQueue( queue   );
    clReleaseContext     ( context );

    free( hostMatrix );

    return EXIT_SUCCESS;
}

