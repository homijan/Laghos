#!/bin/bash
mpirun -np 8 laghos -p 0 -m data/square01_quad.mesh -rs 3 -tf 0.5 -vis
mpirun -np 8 laghos -p 0 -m data/cube01_hex.mesh -rs 1 -cfl 0.1 -tf 0.25 -vis
mpirun -np 8 laghos -p 1 -m data/square01_quad.mesh -rs 3 -tf 0.8 -vis
mpirun -np 8 laghos -p 1 -m data/cube01_hex.mesh -rs 2 -tf 0.6 -vis
mpirun -np 8 laghos -p 2 -m data/square01_quad.mesh -rs 3 -tf 0.2 -vis
mpirun -np 8 laghos -p 2 -m data/cube01_hex.mesh -rs 2 -tf 0.2 -vis
mpirun -np 8 laghos -p 3 -m data/rectangle01_quad.mesh -rs 2 -tf 2.5 -cfl 0.025 -vis
mpirun -np 8 laghos -p 3 -m data/box01_hex.mesh -rs 1 -tf 2.5 -cfl 0.05 -vis 
mpirun -np 8 laghos -p 2 -m data/segment01.mesh -rs 3 -tf 0.2 -ot 3 -ov 4 -vis

