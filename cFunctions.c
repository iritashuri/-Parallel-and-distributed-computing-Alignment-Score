#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "myProto.h"


void readFromFile(const char *filename, float w[4], char **seq1, int *NS2,
	char ***seqs2) {
	FILE *fp = fopen(filename, "r");
	
	if(fp == NULL){
		printf("Faild to open Input File\n");
		exit(1);
	}

	// Read weights and store it in w array 
	w[0] = readW(fp);
	w[1] = readW(fp);
	w[2] = readW(fp);
	w[3] = readW(fp);

	// Read seq1
	char temp[MAX_SIZE_SEQ_1] = { };
	fscanf(fp, "%s", temp);
	int len = 0;
	len = strlen(temp) + 1; //Move over every seq2 and find the best score

	*seq1 = (char*) malloc(len * sizeof(char));
	strcpy(*seq1, temp);

	if (seq1 == NULL) {
		printf("Error while reading seq1");
		return;
	}

	// read NS2 - Number of seq2 
	*NS2 = readNS2(fp);

	// Allocate and read all seqe2
	char temp2[MAX_SIZE_SEQ_2] = { };
	*seqs2 = (char**)malloc((*NS2) * sizeof(char*));
	for (int i = 0; i < *NS2; i++) {
		fscanf(fp, "%s", temp2);
		int si = strlen(temp2);
		// Verify that seq2 length is smaller the seq1 length
		if(len <= si+1){
			printf("sequence2 has to have at list 1 character less then sequence1\n");
			exit(1);
		}			
		(*seqs2)[i] = (char*) malloc(si * sizeof(char));
		strcpy((*seqs2)[i], temp2);
	}

	// Close the file
	fclose(fp);

}

int readNS2(FILE *fp) {
	int temp = 0;
	fscanf(fp, "%d", &temp);
	return temp;

}

float readW(FILE *fp) {
	float temp = 0.0;
	fscanf(fp, "%f", &temp);
	return temp;
}

// Allocate results array
void allocateBests(int **best_ms, int **best_offset, int NS2) {
	*best_ms = (int*) malloc(NS2 * sizeof(int));
	*best_offset = (int*) malloc(NS2 * sizeof(int));
	if (*best_ms == NULL || *best_offset == NULL) {
		printf("Failed to allocate best_ms and best_offset array\n");
		exit(1);
	}
}

// Calculate how many seq2 slave needs to get
int calcSlave_NS2(int NS2) {
	if (NS2 % 2 == 0)
		return NS2 / 2;
	else
		return (NS2 / 2) + 1;

}

void writeResultstoFile(const char* output_file_name,int num_of_res, int* best_offset, int* best_ms){
	FILE *fp;
	fp = fopen(output_file_name, "w");

	if(fp == NULL){
		printf("Faild to open output File\n");
		exit(1);
	}

	for(int i = 0; i < num_of_res; i++){
		fprintf(fp, "best Offset = %d, best MS = % d\n",
			best_offset[i], best_ms[i]);
	}
	fclose(fp);
}

// Free allocated parameters at the end of the program
void freeParams(int **best_ms, int **best_offset, char **seq1, char ***seqs2) {

	free(*seq1);
	free(*seqs2);
	free(*best_ms);
	free(*best_offset);

}
