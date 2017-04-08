# GoldsteinBB
Implementation of parallel version of an interval arithmetic branch-and-bound procedure which allows to find the minimum of a binary real function over a box of domains.

## How to install

Clone the repository :
```bash
git clone https://github.com/Asteip/GoldsteinBB.git
```
To compile the project, run this following command at the root of the directory project
```bash
make
```

To run the sequential version
```bash
./optimization-seq
```

To run the version with MPI
```bash
# this command will run the program on 2 machines
mpirun -np 2 --hostfile ./hostfile ./optimization-mpi
```

To run the version with MPI and OpenMP
```bash
# this command will run the program on 2 machines 
mpirun -np 2 --hostfile ./hostfile ./optimization-par
```