from mpi4py import MPI

communicator = MPI.COMM_WORLD
rank = communicator.Get_rank()
size = communicator.Get_size()

print(f"Hello! I am process {rank} of {size}.")