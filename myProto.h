#pragma once
 
#define PART  100

#define MAX_SIZE_SEQ_1 3000
#define MAX_SIZE_SEQ_2 2000


// Cuda
int computeOnGPU(char *seq1, char *seq2, float w[4], int *bestMS,int *best_offset);


// C
char* createSEquence(FILE *fp, int max_size);
float readW(FILE *fp);
int readNS2(FILE *fp);
void readFromFile(const char *filename, float w[4], char **seq1, int *NS2,
		char ***seqs2);
void allocateBests(int **best_ms, int **best_offset, int NS2);
int calcSlave_NS2(int NS2);
void freeParams(int **best_ms, int **best_offset, char **seq1, char ***seqs2);
void writeResultstoFile(const char* output_file_name, int num_of_res, int* best_offset, int* best_ms);

// MPI
void sendParamssToSlave(int tag, float w[4], char *seq1, int NS2, int Slave_NS2,
		char **seqs2);
void GetParamsFromMaster(int tag, float w[4], char **seq1, int *NS2,
		char ***seqs2);




