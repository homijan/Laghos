#ifndef MFEM_NTH_COEFFS
#define MFEM_NTH_COEFFS

using namespace std;

#include "mfem.hpp"

namespace mfem
{

namespace nth
{

// Generic hydro-nth coefficient.
class HydroCoefficient : public Coefficient
{
protected:
   double a0, a1, a2;
   ParGridFunction *gf1, *gf2;
public:
   HydroCoefficient(double _a0, ParGridFunction *_gf1, ParGridFunction *_gf2,
      double _a1, double _a2) : gf1(_gf1), gf2(_gf2)
      { a0 = _a0; a1 = _a1; a2 = _a2; }
   virtual double Eval(ElementTransformation &T,
      const IntegrationPoint &ip) = 0;

   virtual ~HydroCoefficient() {};
};

// A first realistic inverse-mean-free-path coefficient.
class RealisticInverseMFPCoefficient : public HydroCoefficient
{
protected:
public:
   RealisticInverseMFPCoefficient(double _a0, ParGridFunction *_gf1=NULL,
      ParGridFunction *_gf2=NULL, double _a1=1.0, double _a2=1.0 )
      : HydroCoefficient(_a0, _gf1, _gf2, _a1, _a2) {}
   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
      { return a0 * pow(gf1->GetValue(T.ElementNo, ip), a1) *
	     pow(gf2->GetValue(T.ElementNo, ip), a2); }

   ~RealisticInverseMFPCoefficient() {}
};

// A first realistic source coefficient.
class RealisticSourceCoefficient : public HydroCoefficient
{
   RealisticInverseMFPCoefficient *InvMFP_cf;
protected:
public:
   RealisticSourceCoefficient(double _a0, RealisticInverseMFPCoefficient *_cf,
      ParGridFunction *_gf1=NULL, ParGridFunction *_gf2=NULL, double _a1=1.0,
      double _a2=1.0 ) : HydroCoefficient(_a0, _gf1, _gf2, _a1, _a2),
      InvMFP_cf(_cf) { }
   double Eval(ElementTransformation &T, const IntegrationPoint &ip)
      { return a0 * InvMFP_cf->Eval(T, ip) *
	     pow(gf1->GetValue(T.ElementNo, ip), a1) *
	     pow(gf2->GetValue(T.ElementNo, ip), a2); }

   ~RealisticSourceCoefficient() {}
};

// Electric field coefficient
void Efield_function(const Vector &x, Vector &v);

// BGK collision operator functions
double kappa_function(const Vector &x);
double isosigma_function(const Vector &x);
double kappaisosigma_function(const Vector &x);
double source_function_cos(const Vector &x);
double source_function_const(const Vector &x);

// Mesh bounding box
extern Vector bb_min, bb_max;
// Attenuation coefficient kappa.
extern double KAPPA, SIGMA, EFIELD;

} // namespace nth

} // namespace mfem


#endif
