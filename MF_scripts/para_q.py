from mpi4py import MPI
import MF_functions as MF
import numpy as np
import pickle

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

MF.U = 10
n = 1
T = 0.1
MF.beta = 1/T

Q_min, Q_max, Q_number = 0, np.pi, 100
Qr = np.linspace(Q_min, Q_max, Q_number)

if rank < Q_number:
    qy_i = rank
    
    print("Process n°", rank, "has started.")
    
    sendbuf = np.zeros(Q_number, dtype='float')
    for qx_i in range(Q_number):
        Q = (Qr[qx_i],Qr[qy_i],0)
        n_res,m, mu = MF.solve_mean_field(n,Q)
        sendbuf[qx_i] = MF.compute_free_energy('Helmholtz', n_res, m, mu, Q)
        #sendbuf[qx_i] = qx_i - rank
    
    #Say the process ended
    print("Process n°", rank, "has finished.")

else:
    sendbuf = np.zeros(Q_number, dtype='float')
    
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, Q_number], dtype='float')
comm.Gather(sendbuf, recvbuf, root=0)
if rank == 0:
    Q_grid = recvbuf.T[:Q_number,:Q_number]
    print(Q_grid)
    with open("Q_grid.pickle", "wb") as Q_file:
        pickle.dump(Q_grid, Q_file)
        print("Phase diagram saved as \"Q_grid.pickle\"")
