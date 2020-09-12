#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include "myProto.h"
#include <time.h>

#define MASTER 0
#define SLAVE 1



/*
MPI+OpenMP+CUDA Program to calculate Alignment Score algorithm. 
Gets File with weights array, sequence 1, NS2 - number of sequences to test and NS2 sequences2.
The master Read the file and send to the slave the values and half of the sequences to test.
Each process use openMp library to use multithreading for the calculation.
Each thread that where called is calling GPU to make calculation for a single seq2 at a time.
The results are coming back to the master
The master is writing the results to an input file
*/


int main(int argc, char *argv[]) {
	clock_t begin = clock();
	int my_rank; // Rank of process 
	int p; // Number of processes 
	int tag = 0; // Tag for messages 
	MPI_Status status; // Return status for receive
	const char *filename = "./input.txt";
	const char *output_file_name = "./output.txt";
	float w[4];
	char *seq1 = NULL;
	int NS2;
	int Slave_NS2;
	char **seqs2 = NULL;
	int *best_ms = NULL;
	int *best_offset = NULL;

	// Start up MPI 
	MPI_Init(&argc, &argv);

	// Find out process rank 
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// Find out number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	// Make sure only 2 processes where call
	if (p != 2) {
		printf("this program required 2 processes\n");
		exit(1);
	}

	// Do Master job
	if (my_rank == MASTER) {

		readFromFile(filename, w, &seq1, &NS2, &seqs2);

		// Allocate best_ms and best_offset
		allocateBests(&best_ms, &best_offset, NS2);

		// Calculate how many seqs2 slave needs to get
		Slave_NS2 = calcSlave_NS2(NS2);

		// Send Parameters to the slave
		sendParamssToSlave(tag, w, seq1, NS2, Slave_NS2, seqs2);

		// Do AlignmentScore calculation
		int i = 0;

	    #pragma omp parallel for
			// Move over every seq2 and find the best score
			for (i = Slave_NS2; i < NS2; i++) {
				//printf("i = %d, thread num = %d\n", i, omp_get_thread_num());
				computeOnGPU(seq1, seqs2[i], w, &best_ms[i],&best_offset[i]);
			}

		// Receive from slave the results 
		MPI_Recv(best_offset, Slave_NS2, MPI_INT, SLAVE, tag, MPI_COMM_WORLD,
			&status);
		MPI_Recv(best_ms, Slave_NS2, MPI_INT, SLAVE, tag, MPI_COMM_WORLD,
			&status);

		// Write the results to output.txt
		writeResultstoFile(output_file_name, NS2, best_offset, best_ms);
		

		// Free allocated parameters
		freeParams(&best_ms, &best_offset, &seq1, &seqs2);
		
		// Calculate time
		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		printf("time_spent = %lf\n", time_spent);
	}
	// Do Slave job 
	else {
		GetParamsFromMaster(tag, w, &seq1, &NS2, &seqs2);

		// Allocate best_ms and best_offset
		allocateBests(&best_ms, &best_offset, NS2);

		// Do AlignmentScore calculation
		int i = 0;
		#pragma omp parallel for
			for (i = 0; i < NS2; i++) {
				//printf("i = %d, thread num = %d\n", i, omp_get_thread_num());
				computeOnGPU(seq1, seqs2[i], w, &best_ms[i],&best_offset[i]);
			}

		// Send results to the master
		MPI_Send(best_offset, NS2, MPI_INT, MASTER, tag, MPI_COMM_WORLD);
		MPI_Send(best_ms, NS2, MPI_INT, MASTER, tag, MPI_COMM_WORLD);

		// Free allocated parameters
		freeParams(&best_ms, &best_offset, &seq1, &seqs2);
	}
	// Shut down MPI 
	MPI_Finalize();
	return 0;
}


