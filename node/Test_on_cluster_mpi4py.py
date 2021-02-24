from mpi4py import MPI
from os import environ
from typing import *

MPIComm = Union[MPI.Intracomm, MPI.Intercomm]


def main():
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_size == 1:
        print("You are running an MPI program with only one slot/task!")
        print("Are you using `mpirun` (or `srun` when in SLURM)?")
        print("If you are, then please use an `-n` of at least 2!")
        print("(Or, when in SLURM, use an `--ntasks` of at least 2.)")
        print("If you did all that, then your MPI setup may be bad.")
        return 1

    if mpi_size >= 1000 and mpi_rank == 0:
        print("WARNING:  Your world size {} is over 999!".format(mpi_size))
        print("The output formatting will be a little weird, but that's it.")

    if mpi_rank == 0:
        return mpi_root(mpi_comm)
    else:
        return mpi_nonroot(mpi_comm)


def mpi_root(mpi_comm):
    import random

    random_number = random.randrange(2 ** 32)
    mpi_comm.bcast(random_number)
    print("Controller @ MPI Rank   0:  Input {}".format(random_number))

    GatherResponseType = List[Tuple[str, int]]
    response_array = mpi_comm.gather(None)  # type: GatherResponseType

    mpi_size = mpi_comm.Get_size()
    if len(response_array) != mpi_size:
        print(
            "ERROR!  The MPI world has {} members, but we only gathered {}!".format(
                mpi_size, len(response_array)
            )
        )
        return 1

    for i in range(1, mpi_size):
        if len(response_array[i]) != 2:
            print(
                "WARNING!  MPI rank {} sent a mis-sized ({}) tuple!".format(
                    i, len(response_array[i])
                )
            )
            continue
        if type(response_array[i][0]) is not str:
            print(
                "WARNING!  MPI rank {} sent a tuple with a {} instead of a str!".format(
                    i, str(type(response_array[i][0]))
                )
            )
            continue
        if type(response_array[i][1]) is not int:
            print(
                "WARNING!  MPI rank {} sent a tuple with a {} instead of an int!".format(
                    i, str(type(response_array[i][1]))
                )
            )
            continue
        if random_number + i == response_array[i][1]:
            result = "OK"
        else:
            result = "BAD"

        print(
            "   Worker at MPI Rank {: >3}: Output {} is {} (from {})".format(
                i,
                response_array[i][1],
                result,
                response_array[i][0],
            )
        )

        mpi_comm.send(
            obj=0,
            dest=i,
            tag=0,
        )

    mpi_comm.barrier()

    return 0


def mpi_nonroot(mpi_comm):

    mpi_rank = mpi_comm.Get_rank()

    random_number = mpi_comm.bcast(None)

    if type(random_number) is not int:
        print(
            'ERROR in MPI rank {}: Received a non-integer "{}" from the broadcast!'.format(
                mpi_rank,
                random_number,
            )
        )
        return 1

    response_number = random_number + mpi_rank

    response = (
        MPI.Get_processor_name(),
        response_number,
    )
    mpi_comm.gather(response)

    def get_message(mpi_comm):
        message = mpi_comm.recv(
            source=0,
            tag=0,
        )  # type: int
        if type(message) is not int:
            print(
                "ERROR in MPI rank {}: Received a non-integer message!".format(
                    mpi_rank,
                )
            )
            return None
        else:
            return message

    message = get_message(mpi_comm)
    while (message is not None) and (message != 0):
        mpi_comm.send(
            obj=int(message / 2),
            dest=0,
            tag=0,
        )

        message = get_message(mpi_comm)

    if message is None:
        return 1

    mpi_comm.barrier()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
