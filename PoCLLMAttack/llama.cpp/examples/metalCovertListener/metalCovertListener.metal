#include <metal_stdlib>
using namespace metal;

kernel void covertListenerKernel(
    device float* output [[buffer(0)]],
    constant uint& numValues [[buffer(1)]],
    threadgroup volatile float* shared [[threadgroup(0)]],
    uint local_thread_index [[thread_index_in_threadgroup]],
    uint3 threads_per_threadgroup [[ threads_per_threadgroup ]],
    uint3 threadgroup_position_in_grid [[ threadgroup_position_in_grid ]]
) {
    for (uint i = local_thread_index; i < numValues; i += threads_per_threadgroup.x) {
        uint index = numValues * threadgroup_position_in_grid.x + i;

        // Copy the data
        output[index] = shared[i];

        // Clear it so we don't read it again
        shared[i] = 0;
    }
}