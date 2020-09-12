# -Parallel-and-distributed-computing-Alignment-Score
MPI+OpenMP+CUDA Program to calculate Alignment Score algorithm. 

### Introduction ###
Gets File with weights array, sequence 1, NS2 - number of sequences to test and NS2 sequences2.
The master Read the file and send to the slave the values and half of the sequences to test.
Each process use openMp library to use multithreading for the calculation.
Each thread that where called is calling GPU to make calculation for a single seq2 at a time.
The results are coming back to the master
The master is writing the results to an input file


### Building and execution instructions ###

The program can run on 1 or 2 different compute resources as following:

1. Execute the program from a single compute resource:

cd $WORKING_DIRECTORY
make
make run


2. Execute the program from 2 compute resources:

2.1 Validate $WORKING_DIRECTORY is the same absolute path in the 2 compute resources.

2.2 On master compute resource:

cd $WORKING_DIRECTORY
echo "${master_ip_address}" > mf
echo "${slave_ip_address}" >> mf

2.3 On both compute resources:

cd $WORKING_DIRECTORY
make

2.4 On master compute resources:

cd $WORKING_DIRECTORY
make runOn2


