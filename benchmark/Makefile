CC = gcc
MPICC = mpicc
CFLAGS = -O2 -Wall -fopenmp

SRC_DIR = ../src

.PHONY: all clean

all: conway_seq conway_fases_mpi conway_fases_hibrido

conway_seq: $(SRC_DIR)/conwaySequencial.c
	$(CC) $(CFLAGS) -o $@ $<

conway_fases_mpi: $(SRC_DIR)/conway_fases_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $<

conway_fases_hibrido: $(SRC_DIR)/conway_fases_mpi_omp.c
	$(MPICC) $(CFLAGS) -o $@ $<

clean:
	rm -f conway_seq conway_fases_mpi conway_fases_hibrido
