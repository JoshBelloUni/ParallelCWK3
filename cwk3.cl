// Kernel for matrix transposition.
//
// Uses a tiled approach with local memory (Lecture 16) as scratch space:
// each work-group loads a TILE_SIZE x TILE_SIZE block of the input into
// local memory, then writes it out transposed to global memory.

#define TILE_SIZE 16

__kernel void transpose(
    __global const float *input,   // row-major, nRows x nCols
    __global       float *output,  // row-major, nCols x nRows (transposed)
    int nRows,
    int nCols
)
{
    __local float tile[TILE_SIZE][TILE_SIZE];

    int bx = get_group_id(0);   // work-group block index along columns of input
    int by = get_group_id(1);   // work-group block index along rows of input

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    // Global coordinates in the input matrix.
    int globalCol = bx * TILE_SIZE + lx;
    int globalRow = by * TILE_SIZE + ly;

    // Load a tile from global memory into local memory.
    if (globalRow < nRows && globalCol < nCols)
        tile[ly][lx] = input[globalRow * nCols + globalCol];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write the tile out to the transposed position in global memory.
    int outRow = bx * TILE_SIZE + ly;
    int outCol = by * TILE_SIZE + lx;

    if (outRow < nCols && outCol < nRows)
        output[outRow * nRows + outCol] = tile[lx][ly];
}
