// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#ifndef MFEM_M1_SOLVER
#define MFEM_M1_SOLVER

#include "mfem.hpp"
#include "laghos_solver.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{

namespace hydrodynamics
{

class M1HydroCoefficient;

// Given a solutions state (I0, I1), this class performs all necessary
// computations to evaluate the new slopes (dI0_dt, dI1_dt).
class M1Operator : public LagrangianHydroOperator
{
protected:
   // TODO M1_dvmin does not work, because of its local nature. 
   // M1_dvmax does not seem to have an important effect.
   double M1_dvmin, M1_dvmax;
   // The grid function is necessary for velocity step estimation. 
   ParGridFunction &x_gf;
   // Velocity dependent coefficients providing physics.
   M1HydroCoefficient *mspInv_pcf, *sourceI0_pcf;

   void UpdateQuadratureData(double velocity, const Vector &S) const;

public:
   M1Operator(int size, ParFiniteElementSpace &h1_fes,
              ParFiniteElementSpace &l2_fes, Array<int> &essential_tdofs,
              ParGridFunction &rho0, double cfl_, M1HydroCoefficient *mspInv_,
              M1HydroCoefficient *sourceI0_, ParGridFunction &x_gf_, 
              ParGridFunction &T_gf_, bool pa, double cgt, int cgiter)
      : LagrangianHydroOperator(size, h1_fes, l2_fes, essential_tdofs, rho0,
                                0, cfl_, NULL, false, pa, cgt, cgiter), 
         mspInv_pcf(mspInv_), sourceI0_pcf(sourceI0_), x_gf(x_gf_) {}

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const Vector &S, Vector &dS_dt) const;

   // Calls UpdateQuadratureData to compute the new quad_data.dt_est.
   double GetVelocityStepEstimate(const Vector &S) const;
   void ResetVelocityStepEstimate() const;
   void ResetQuadratureData() const { quad_data_is_current = false; }

   ~M1Operator() {}
};

// Generic hydro equation of state (EOS) class,
// providing any physics related evaluation needed in NTH.
class EOS
{
protected:
   // Fundamental constants of nature.
   double kB, c, hbar, G;
   // Corresponding masses of electron and proton.
   double me;
public:
   EOS(double kB_ = 1.0, double me_ = 1.0, double mi_ = 1.0, double c_ = 1.0, 
       double hbar_ = 1.0, double G_ = 1.0)
      { kB = kB_, c = c_, hbar = hbar_, G = G_; me = me_; }
   double vTe(double Te) { return sqrt(kB * Te / me); }
};

// Generic hydro M1 coefficient.
class HydroCoefficient : public Coefficient
{
protected:
   // Fluid quantities used in calculations of physics.
   ParGridFunction &rho_gf, &Te_gf, &v_gf;
   // Space dependent material coefficient.
   Coefficient *material_pcf;
   // General equation of state.
   EOS *eos;
public:
   HydroCoefficient(ParGridFunction &rho_, ParGridFunction &Te_,
                    ParGridFunction &v_, Coefficient *material_, EOS *eos_)
      : rho_gf(rho_), Te_gf(Te_), v_gf(v_), material_pcf(material_), eos(eos_) 
	  {}
   virtual double Eval(ElementTransformation &T,
      const IntegrationPoint &ip) = 0;

   virtual ~HydroCoefficient() {};
};

// M1 hydro coefficient.
class M1HydroCoefficient : public HydroCoefficient
{
   void SetVelocityScale(double alpha_, double Tmax)
      { alphavT = alpha * eos->vTe(Tmax); }
protected:
   // Velocity is always scaled wit respect to maximum thermal velocity 
   // (its multiple) so it is in (0, 1)
   double alpha, Tmax, alphavT;
   // Current particle velocity from the velocity spectra. 
   double velocity;
public:
   M1HydroCoefficient(ParGridFunction &rho_, ParGridFunction &T_,
                      ParGridFunction &v_, Coefficient *material_, EOS *eos_)
      : HydroCoefficient(rho_, T_, v_, material_, eos_)
	  { alpha = 1.0; Tmax = 1.0; SetVelocityScale(alpha, Tmax); }
   virtual double Eval(ElementTransformation &T,
      const IntegrationPoint &ip) = 0;
   void SetVelocity(double v_) { velocity = v_; }
   void SetThermalVelocityMultiple(double alpha_)
      { alpha = alpha_; SetVelocityScale(alpha, Tmax); }
   void SetTmax(double Tmax_)
      { Tmax = Tmax_; SetVelocityScale(alpha, Tmax); }
   double GetVelocityScale() { return alphavT; }
};

// M1 mean-stopping-power coefficient.
class M1MeanStoppingPowerInverse : public M1HydroCoefficient
{
protected:
public:
   M1MeanStoppingPowerInverse(ParGridFunction &rho_, ParGridFunction &Te_,
                       ParGridFunction &v_, Coefficient *material_, EOS *eos_)
      : M1HydroCoefficient(rho_, Te_, v_, material_, eos_) {}
   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

// M1 source coefficient.
class M1I0Source : public M1HydroCoefficient
{
protected:
public:
   M1I0Source(ParGridFunction &rho_, ParGridFunction &Te_, ParGridFunction &v_, 
              Coefficient *material_, EOS *eos_)
      : M1HydroCoefficient(rho_, Te_, v_, material_, eos_) {}
   double Eval(ElementTransformation &T, const IntegrationPoint &ip);
};

extern double a0;

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_M1
