#! /bin/bash
mpirun -np 8 laghos -p 1 -m data/segment01.mesh -rs 4 -tf 0.6 -ok 3 -ot 2 -vis -pa | tee M1Sedov1D.out
mpirun -np 8 laghos -p 1 -m data/square01_quad.mesh -rs 3 -tf 0.4 -ok 3 -ot 2 -vis -pa | tee M1Sedov2D.out
mpirun -np 8 laghos -p 1 -m data/cube01_hex.mesh -rs 2 -tf 0.4 -ok 3 -ot 2 -vis -pa | tee M1Sedov3D.out
