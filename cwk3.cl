// Kernel for matrix transposition.
//
// Uses a tiled approach with local memory (Lecture 16) as scratch space:

__kernel void transpose(
    __global const float *input,   // row-major, nRows x nCols
    __global       float *output,  // row-major, nCols x nRows (transposed)
    int nRows,
    int nCols,
    __local float *tile            // tileSize x tileSize scratch space
)
{
    int tileW = get_local_size(0);
    int tileH = get_local_size(1);

    int bx = get_group_id(0);   // work-group block index along columns of input
    int by = get_group_id(1);   // work-group block index along rows of input

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    // Global coordinates in the input matrix.
    int globalCol = bx * tileW + lx;
    int globalRow = by * tileH + ly;

    // Load a tile from global memory into local memory.
    if (globalRow < nRows && globalCol < nCols)
        tile[ly * tileW + lx] = input[globalRow * nCols + globalCol];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write the tile out to the transposed position in global memory.
    int outRow = bx * tileW + ly;
    int outCol = by * tileH + lx;

    if (outRow < nCols && outCol < nRows)
        output[outRow * nRows + outCol] = tile[lx * tileH + ly];
}
