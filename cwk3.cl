// Kernel for matrix transposition.

__kernel void transpose(
    __global const float *input,   // row-major, nRows x nCols
    __global       float *output,  // row-major, nCols x nRows (transposed)
    int nRows,
    int nCols
)
{
    int globalCol = get_global_id(0);
    int globalRow = get_global_id(1);

    if (globalRow < nRows && globalCol < nCols)
        output[globalCol * nRows + globalRow] = input[globalRow * nCols + globalCol];
}
