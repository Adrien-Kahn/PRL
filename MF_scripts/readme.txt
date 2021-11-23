//---- MF scripts ----//

Contact: liam.rampon@gmail.com, or Slack if you happen to have been added


//--- Dependencies ---//
- numpy
- scipy
- mpi4py



//--- File contents ---//


//-- MF_function.py --//
- functions to compute (F, m, µ) given a set of paramters (U,n,T,Q), using "solve_mean_field()" and "compute_free_energy()". This last function needs an argument, "Gibbs" or "Helmholtz". To get the free energy associated to the canonical ensemble, which is the one that needs to be minimized, use "Helmholtz". "Gibbs" returns the grand canonical thermodynamical potential.
- the following functions are named "opti_XXX()", and used when called by an optimization function, 
- "compute_point_XX" functions take a set of parameters (U,n,T) and a dictionnary, compute the free-energy minimizing Q vector along with the associated (F,m,µ), and register them in the dictionnary. The difference between these functions is under what key the results are stored ( (n,T), (U,T), ...) or whether the lattice's dimension is 2 or 3.
--> the most useful ones are compute_point() and compute_point_3D()
--> Q is minimized along a high symmetry axis in the FBZ, here is a reminder of the terminology for square/cubic lattices: Г = (0,0,0) is the center, X = (0,0,π) is in the middle of a face, M = (0,π,π) in the middle of an edge and R = (π,π,π) on a vertex. A segment between two vertices is named after the parametrization of its coordinates: ГR = qqq, XM = 0qπ, etc.

//-- parallelized_phase_diagram.py --//
- this code runs the "compute_point" function of your choice in parallel to fill a phase diagram whose boundaries are defined at the beginning of the script. For a 100*100 points phase diagram, it should take ~15 minutes to run on the CDF cluster.
- though it looks like you can give values to U,n,T and Q, the visualization script I coded can only handle two dimensionnal diagrams, so please stick to (n,T) when getting to know the code.
- to change the lattice dimension / diagram axis, replace the compute_point() function in this script by another one
- the result is written in a .pickle file whose name is defined at the end of the script. These quickly turn out to be unmanageable, so replacing by a more ergonomic format (such as .h5 if you plan on using TRIQS) may be a good idea.

//-- para_q.py --//
This one was used when developping the previous script, there is no real use to it.

//-- mean_field_notebook.ipynb --//
The first notebook where Michel wrote the MF equations and did some tests. Most of its functions were moved to MF_functions.py.

//-- Phase diagram visualizer.ipynb --//
Run the cells until the "Show the plot" cell is reached. In this one, write the name of the pickle file were the phase diagram is stored, specify the lattice dimension and voilà, the diagram is plotted along with phase separation. Phases are represented by mapping the predomining Q vector (Qx,Qy,Qz) to (R,G,B). 
Following cells were used to develop the phase separation code, but it is now completely integrated in the "plot_phase_diagram()" function.


//-- Mean Field and phase separation.ipynb --//
Demonstration notebook to benchmark the technique I used for DMFT, using MF data. Nothing much to say except that it works with MF.




