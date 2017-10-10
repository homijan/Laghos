#include "nth_coeffs.hpp"

namespace mfem
{

namespace nth
{

// Domain (bounding box) coordinates.
Vector bb_min, bb_max;
// Default values used in the triple-point test.
double KAPPA = 0.4285;
double SIGMA = 2.0*KAPPA;
double EFIELD = 0.75*KAPPA;

// Electric field coefficient
void Efield_function(const Vector &x, Vector &E)
{
   // test the BGK effect
   if (x.Size()==1) { E(0) = 1.0; }
   else if (x.Size()==2)
   {
      E(0) = 1.;
      E(1) = 1.;
   }
   else if (x.Size()==3)
   {
      E(0) = 1.;
      E(1) = 1.;
      E(2) = 1.;
   }
   E *= 1/E.Norml2();
   E *= EFIELD;
   return;
}

// Attenuation coefficient kappa of the BGK collision operator kappa*(S - u)
double kappa_function(const Vector &x)
{
   return KAPPA;
}
// Isotropization coefficient sigma of the BGK collision operator
// sigma*(mean(u) - u)
double isosigma_function(const Vector &x)
{
   return SIGMA;
}
// Attenuation+iostropization coefficient kappasigma of the BGK collision
// operator kappa*(S - u) + sigma(mean(u) - u)
double kappaisosigma_function(const Vector &x)
{
   return kappa_function(x) + isosigma_function(x);
}

double source_function_cos(const Vector &x_coord)
{
   double pi = 3.14159265359;
   double kappa, S;
   if (x_coord.Size()==1)
   {
      double x = x_coord[0], y = x_coord[1];
      double a0 = 1.;
      kappa = kappa_function(x_coord);
      S = a0*(1. - cos(2.*pi*x/(bb_max(0) - bb_min(0))));
   }
   else if (x_coord.Size()==2)
   {
      double x = x_coord[0], y = x_coord[1];
      double a0 = 1.;
      kappa = kappa_function(x_coord);
      S = a0*(1. - cos(2.*pi*x/(bb_max(0) - bb_min(0))))*
         (1. - cos(2.*pi*y/(bb_max(1) - bb_min(1))));
   }
   else if (x_coord.Size()==3)
   {
      double x = x_coord[0], y = x_coord[1], z = x_coord[2];
      double a0 = 1.;
      kappa = kappa_function(x_coord);
      S = a0*(1. - cos(2.*pi*x/(bb_max(0) - bb_min(0))))*
         (1. - cos(2.*pi*y/(bb_max(1) - bb_min(1))))*
	     (1. - cos(2.*pi*z/(bb_max(2) - bb_min(2))));
   }
   else { MFEM_ABORT("Unsupported source dimension"); }

   return kappa*S;
}

double source_function_const(const Vector &x_coord)
{
   double kappa, S;

   kappa = kappa_function(x_coord);
   S = 1.;

   return kappa*S;
}

} // namespace nth

} //namespace mfem
