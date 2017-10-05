#ifndef MFEM_IC
#define MFEM_IC

using namespace std;

#include "mfem.hpp"

namespace mfem
{

namespace nth
{

// Mesh bounding box
extern Vector bb_min, bb_max;

// Attenuation coefficient kappa.
extern double KAPPA, SIGMA, EFIELD;

// Electric field coefficient
void Efield_function(const Vector &x, Vector &v);

// BGK collision operator functions
double kappa_function(const Vector &x);
double isosigma_function(const Vector &x);
double kappaisosigma_function(const Vector &x);
double source_function_cos(const Vector &x);

} // namespace nth

} // namespace mfem


#endif
