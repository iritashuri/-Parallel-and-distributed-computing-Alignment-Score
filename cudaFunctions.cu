#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

#define THREADS_PER_BLOCKS 512

__global__ void AlignmentScore(char *seq1, char *seq2, int seq1_size,
	int seq2_size, float w[4], int max_ms, int max_offset, double *results,
	int num_of_threads);


// Host functions 
void checkErr(cudaError_t err, const char *err_message, char *var_name) {
	if (err != cudaSuccess) {
		fprintf(stderr, "%s - %s\n, var-> %s", err_message, cudaGetErrorString(err), var_name);
		exit(EXIT_FAILURE);
	}
}

void cudaMallocDoubleArr(double** d_arr,size_t arr_size, cudaError_t err, char* var_name,const char* malloc_err_message){
	err = cudaMalloc((void**)d_arr, arr_size);
	checkErr(err, malloc_err_message, var_name);
}

void cudaMallocFloatArr(float** d_w,size_t w_size, cudaError_t err, char* var_name,const char* malloc_err_message){
	err = cudaMalloc((void**)d_w, w_size);
	checkErr(err, malloc_err_message, var_name);
}

void cudaMallocString(char** d_seq,size_t seq_size, cudaError_t err, char* var_name,const char* malloc_err_message){
	err = cudaMalloc((void**)d_seq, seq_size);
	checkErr(err, malloc_err_message, var_name);
}

void cudaMemcpyHostToDeviceFloat(float* d_w, float* w, size_t w_size, cudaError_t err,const char* copy_err_message,char* var_name){
	err = cudaMemcpy(d_w, w, w_size, cudaMemcpyHostToDevice);
	checkErr(err, copy_err_message, var_name);
}

void cudaMemcpyHostToDeviceString(char* d_seq, char* seq, size_t seq_size, cudaError_t err,const char* copy_err_message,char* var_name){
	err = cudaMemcpy(d_seq, seq, seq_size, cudaMemcpyHostToDevice);
	checkErr(err, copy_err_message, var_name);
}

// Call the GPU and get GPU results 
int computeOnGPU(char *seq1, char *seq2, float w[4], int *bestMS, int *best_offset) {
	
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;
	char var_name[30];

	// Define results arrays
	double *scores_arr;

	// Define parameters size
	size_t seq1_size = strlen(seq1);
	size_t seq2_size = strlen(seq2);
	int max_offset = (int) seq1_size - (int) seq2_size - 1;
	int max_ms = (int) seq2_size;
	int num_of_threads = (int) max_ms * (max_offset + 1);
	size_t scores_size = num_of_threads * sizeof(double);
	size_t w_size = 4 * sizeof(float);

	// Allocate results arrays in host
	scores_arr = (double*) malloc(scores_size);

	// Allocate memory on GPU to copy the data from the host
	const char *malloc_err_message = "Failed to allocate device memory";
	char *d_seq1;
	char *d_seq2;
	double *d_scores_arr;
	float *d_w;
	
	strcpy(var_name, "d_w");
	cudaMallocFloatArr(&d_w, w_size, err, var_name, malloc_err_message);

	strcpy(var_name, "d_scores_arr");
	cudaMallocDoubleArr(&d_scores_arr, scores_size, err, var_name, malloc_err_message);

	strcpy(var_name, "d_seq1");
	cudaMallocString(&d_seq1, seq1_size * sizeof(char), err, var_name, malloc_err_message);


	strcpy(var_name, "d_seq2");
	cudaMallocString(&d_seq2, seq2_size * sizeof(char), err, var_name, malloc_err_message);	


	// Copy data from host to the GPU memory
	const char *copy_err_message = "Failed to copy data from host to device";
	
	strcpy(var_name, "d_w");
	cudaMemcpyHostToDeviceFloat(d_w, w, w_size, err, copy_err_message, var_name);
	
	strcpy(var_name, "d_seq1");
	cudaMemcpyHostToDeviceString(d_seq1, seq1, seq1_size, err, copy_err_message, var_name);

	strcpy(var_name, "d_seq2");
	cudaMemcpyHostToDeviceString(d_seq2, seq2, seq2_size, err, copy_err_message, var_name);


	// Launch the Kernel
	int num_of_blocks_per_grid = num_of_threads / THREADS_PER_BLOCKS;
	
	// Check if we need to add more block
	if (num_of_threads % THREADS_PER_BLOCKS || !num_of_threads) {
		num_of_blocks_per_grid++;
	}

	AlignmentScore<<<num_of_blocks_per_grid, THREADS_PER_BLOCKS>>>(d_seq1, d_seq2, seq1_size, seq2_size, d_w, max_ms, 		max_offset, d_scores_arr, num_of_threads);
	err = cudaGetLastError();

	strcpy(var_name, "No var");
	checkErr(err, "Failed to launch vectorAdd kernel", var_name);

	// Copy the  result from GPU to the host memory.
	const char *copy_res_err_message = "Failed to copy data from decive to host";

	strcpy(var_name, "scores_arr");
	err = cudaMemcpy(scores_arr, d_scores_arr, scores_size,
		cudaMemcpyDeviceToHost);
	checkErr(err, copy_res_err_message, var_name);

	// Initial bests
	*bestMS = 1;
	*best_offset = 0;
	double best_score = scores_arr[0];
	
        // Compute best score ms and offset
	for (int x = 0; x < num_of_threads; x++) {
		if (scores_arr[x] > best_score) {
			best_score = scores_arr[x];
			*bestMS = x % max_ms + 1;
			*best_offset = x / max_ms;
		}
	}
	
	//printf("bestMS = %d, best_offset = %d, best_score=%lf\n",*bestMS, *best_offset, best_score);	

	free(scores_arr);
	// Free allocated memory on GPU

	const char *free_err_message = "Failed to free device data";
	
	
	strcpy(var_name, "d_scores_arr");
	err = cudaFree(d_scores_arr);
	checkErr(err, free_err_message, var_name);

	strcpy(var_name, "d_seq1");
	err = cudaFree(d_seq1);
	checkErr(err, free_err_message, var_name);

	strcpy(var_name, "d_seq2");
	err = cudaFree(d_seq2);
	checkErr(err, free_err_message, var_name);

	return 0;
}


__device__ void mystrlen(int *len, const char *str) {
	// Calculate length of a string
	(*len) = 0;
	while (*str) {
		(*len)++;
		str++;
	}
}

__device__ void checkspecGroup(const char **group_to_check, int size, char c1, char c2, int *is_cons) {
	// Get group of strings and check if 2 characters are in the same string
	int i, j, k, str_len;
	for (i = 0; i < size; i++) {
		mystrlen(&str_len, group_to_check[i]);
		for (j = 0; j < str_len; j++) {
			if (c1 == group_to_check[i][j]) {
				for (k = 0; k < str_len; k++) {
					if (c2 == group_to_check[i][k]) {
						*(is_cons) = 1;
						return;
					}
				}
			}
		}
	}
}

__device__ void checkConserative(int similarityes[4], char c1, char c2) {
	
	if (c1 == c2) {
		similarityes[0]++;
	} else {
		// If c1 != c2 chack if they in the same Conserative Group and update similarityes[1] if yes
		const char *CONSERVATIVE_GROUPS[9] = { "NDEQ", "MILV", "FYW", "NEQK",
		"QHRK", "HY", "STA", "NHQK", "MILF" };
		const int CONSERVATIVE_GROUPS_SIZE = 9;
		int is_cons = 0;
		checkspecGroup(CONSERVATIVE_GROUPS, CONSERVATIVE_GROUPS_SIZE, c1, c2,
			&is_cons);
		if (is_cons) {
			similarityes[1]++;
		}
		if (!is_cons) {
			// If c1 and c2 are not in the same Conserative Group  check if they in the same Semi Conserative Group
			// And update similarityes[2] if yes
			const char *SEMI_CONSERVATIVE_GROUPS[11] = { "SAG", "SGND",
			"NEQHRK", "ATV", "STPA", "NDEQHK", "HFY", "CSA", "STNK",
			"SNDEQK", "FVLIM" };
			const int SEMI_CONSERVATIVE_GROUPS_SIZE = 11;
			checkspecGroup(SEMI_CONSERVATIVE_GROUPS,
				SEMI_CONSERVATIVE_GROUPS_SIZE, c1, c2, &is_cons);
			if (is_cons) {
				similarityes[2]++;

			}
			// If the not in the same group and not equal update similarityes[3]
			if (!is_cons)
				similarityes[3]++;
		}
	}
}

__device__ void calcSimilarity(char *seq1, char *seq2, int len, int similarityes[4], int ms, int offset) {
	len++;// add 1 to len for the ms
	int i = 0;
	
	// Check if chars in same location (according offset and ms location) are equel, conserative or semi conserative
	// Check chars till ms location
	while (i < ms) {
		checkConserative(similarityes, seq1[i + offset], seq2[i]);
		i++;
	}
	// For ms location is not equel, conserative and semi conserative
	similarityes[3]++;
	i++;
	// Check chars from ms location to the end
	while (i < len) {
		checkConserative(similarityes, seq1[i + offset], seq2[i - 1]);
		i++;
	}
}

__device__ void alignmentScoreFunc(double *results, int similarityes[4], float w[4]) {
	*results = (double) (w[0] * similarityes[0] - w[1] * similarityes[1]
		- w[2] * similarityes[2] - w[3] * similarityes[3]);
}

__global__ void AlignmentScore(char *seq1, char *seq2, int seq1_size,
	int seq2_size, float w[4], int max_ms, int max_offset, double *results,
	int num_of_threads) {
	int new_id = threadIdx.x + (blockDim.x * blockIdx.x);

	if (new_id < num_of_threads) {
		int temp_len;
		// Make sure seq1 and seq2 stay as thet were in file
		mystrlen(&temp_len, seq2);
		if (temp_len > seq2_size)
			seq2[seq2_size] = '\0';
		mystrlen(&temp_len, seq1);
		if (temp_len > seq1_size)
			seq1[seq1_size] = '\0';

		// Cumpure ms and offset to compute
		int my_ms = new_id % max_ms + 1;
		int my_offset = new_id / max_ms;
		// Initial similarityes arr - holds amount of each char in similiarity string
		int similarityes[4] = { 0 };
		
		// Update similarityes arr with amount off each char in similarityes string
		calcSimilarity(seq1, seq2, seq2_size, similarityes, my_ms, my_offset);
		// Compute alignmentScoreFunc
		alignmentScoreFunc(&results[new_id], similarityes, w);
	}
}

