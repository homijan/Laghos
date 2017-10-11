#!/bin/bash
mpirun -np 8 laghos -p 0 -m data/square01_quad.mesh -rs 3 -tf 0.75 -vis -pa -nl
mpirun -np 8 laghos -p 0 -m data/cube01_hex.mesh -rs 1 -tf 0.75 -vis -pa -nl
mpirun -np 8 laghos -p 1 -m data/square01_quad.mesh -rs 3 -tf 0.8 -vis -pa -nl
mpirun -np 8 laghos -p 1 -m data/cube01_hex.mesh -rs 2 -tf 0.6 -vis -pa -nl
mpirun -np 8 laghos -p 2 -m data/segment01.mesh -rs 5 -tf 0.2 -vis -fa -nl
mpirun -np 8 laghos -p 3 -m data/rectangle01_quad.mesh -rs 2 -tf 2.5 -vis -pa -nl
mpirun -np 8 laghos -p 3 -m data/box01_hex.mesh -rs 1 -tf 2.5 -vis -pa -nl
# New NTH-Sedov test
mpirun -np 8 laghos -p 1 -m data/cube01_hex.mesh -rs 2 -tf 0.6 -vis -pa -nl -op 2 -oo 3 -vs 100 -print
# GLVIS graphical output
#../glvis/glvis -np 8 -m results/Laghos_561_mesh -g results/Laghos_561_q -k "mAppppppppppppppppp444444444444444444444447777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777888"
#../glvis/glvis -np 8 -m results/Laghos_561_mesh -g results/Laghos_561_rho -k "mAppppppppppppppppp444444444444444444444447777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777888"
