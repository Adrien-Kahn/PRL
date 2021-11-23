from mpi4py import MPI
import MF_functions
import numpy as np
import pickle
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


#Diagram parameters
n_min, n_max, n_number = 0.01, 1, 100
nr = np.linspace(n_min, n_max, n_number)

T_min, T_max, T_number = 0.01, 1, 100
Tr = np.linspace(T_min, T_max, T_number)

U_min, U_max, U_number = 4, 4, 1
Ur = np.linspace(U_min, U_max, U_number)

Qx_min, Qx_max, Qx_number = 0,np.pi, 1
Qy_min, Qy_max, Qy_number = 0,np.pi, 1

Qxr = np.linspace(Qx_min,Qx_max,Qx_number)
Qyr = np.linspace(Qy_min,Qy_max,Qy_number)


#Find the n,T range to compute
all_points = [(U,n,T,qx,qy) for U in Ur for n in nr for T in Tr for qx in Qxr for qy in Qyr]
my_points = [all_points[i] for i in range(U_number*T_number*n_number*Qx_number*Qy_number) if i % size == rank]
my_res_dico = dict()

print("Process n°", rank, "has started.")
prog_ind = 0.
for U,n,T,qx,qy in my_points:
    
    MF_functions.U = U
    MF_functions.compute_point(U,n,T,my_res_dico)
    start = int(time.time())
    prog_ind += 100/len(my_points)
    if(MPI.is_master_node() and int(prog_ind) % 4 == 0):
        print("Process n°", rank, " has computed ", int(prog_ind), "% of its points.")
        print("Last point in", int(time.time()) - start, "s.")

#Say the process ended
print("Process n°", rank, "has finished.")

#Sed your part or agregate them
if rank == 0:
    # Process 0 receives all dictionary and aggregates them
    res_dict = dict()
    res_dict.update(my_res_dico)
    
    for p in range(1, size):
        transf_dico = comm.recv(source=p,tag=11)
        res_dict.update(transf_dico)

    #Save the final dictionary
    #with open("nT_grid_triangle_u" + str(MF_functions.U) +  ".pickle", "wb") as phase_file:
    archive_name = "nT_grid_diag_u%s.pickle"%str(MF_function.U)
    with open(archive_name, "wb") as phase_file:
        pickle.dump(res_dict, phase_file)
        #print("Phase diagram saved as \"nT_grid_triangle_u" + str(MF_functions.U) +  ".pickle\".")
        print("Phase diagram saved as \"%s\"."%archive_name)
else:
    # All other processes just send their dictionary to process 0
    comm.send(my_res_dico, dest=0,tag=11)



