#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "myProto.h"


#define MASTER 0
#define SLAVE 1

void sendParamssToSlave(int tag, float w[4], char *seq1, int NS2, int Slave_NS2,
		char **seqs2) {
	// Send slave number of seqs2 to check
	MPI_Send(&Slave_NS2, 1, MPI_INT, SLAVE, tag, MPI_COMM_WORLD);

	int len;
	int i = 0;
	// Send to slave seqs2 to check
	for (i = 0; i < Slave_NS2; i++) {
		len = strlen(seqs2[i]) + 1; /* +1 for the NULL byte */
		MPI_Send(&len, 1, MPI_INT, SLAVE, tag, MPI_COMM_WORLD);
		MPI_Send(seqs2[i], strlen(seqs2[i]), MPI_CHAR, SLAVE, 0,
				MPI_COMM_WORLD);
	}
	// Send slave w array
	MPI_Send(w, 4, MPI_FLOAT, SLAVE, tag, MPI_COMM_WORLD);

	// Send seq1
	len = strlen(seq1) + 1; /* +1 for the NULL byte */
	MPI_Send(&len, 1, MPI_INT, SLAVE, tag, MPI_COMM_WORLD);
	MPI_Send(seq1, strlen(seq1), MPI_CHAR, SLAVE, tag, MPI_COMM_WORLD);

}

void GetParamsFromMaster(int tag, float w[4], char **seq1, int *NS2,
		char ***seqs2) {
	MPI_Status status;
	// Receive from master NS2 and allocate seqs2 in the size of it
	MPI_Recv(NS2, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
	*seqs2 = (char**)malloc((*NS2) * sizeof(char*));
	int i = 0;
	int len;
	// Receive from master seqs2 to check and store it in seqs2 array
	for (i = 0; i < (*NS2); i++) {
		MPI_Recv(&len, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
		(*seqs2)[i] = (char*) malloc(len);
		MPI_Recv((*seqs2)[i], len, MPI_CHAR, MASTER, tag, MPI_COMM_WORLD,
				&status);
		(*seqs2)[i][len - 1] = '\0';
	}
	//Receive w array
	MPI_Recv(w, 4, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD, &status);

	// Receive seq1
	MPI_Recv(&len, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
	*seq1 = (char*) malloc(len);
	MPI_Recv(*seq1, len, MPI_CHAR, MASTER, tag, MPI_COMM_WORLD, &status);
	(*seq1)[len - 1] = '\0';

}
