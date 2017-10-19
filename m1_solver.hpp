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

// Given a solutions state (I0, I1), this class performs all necessary
// computations to evaluate the new slopes (dI0_dt, dI1_dt).
class M1Operator : public LagrangianHydroOperator
{
protected:
   ParGridFunction &x_gf, &T_gf;

   void ComputeMaterialProperties(int nvalues, double v, const double gamma[],
                                  const double rho[], const double T[],
                                  double mfp[], double S[]) const
   {
	  for (int i = 0; i < nvalues; i++)
      {
         // The quantity mfp = 1/nu ~ dx/dv ~ 1/dedx,
         // the inverse of stopping power.
         // The collision frequency dependens on density and
         // from the classical binary collisions nu ~ rho*v^-3
         // Scaling valid for Sedov.
		 double coeff = 1e-1;
		 mfp[i] = coeff*(v * v * v)/rho[i];
         //S[i]   = ComputeSource(rho[i], T[i], v);
         // Compatible with the constant mass matrices.
         mfp[i] *= rho[i];
         S[i] *= rho[i];
      }
   }

   double ComputeSource(double rho, double T, double v) const
   {
      // In the case of AWBS we consider dfM/dv
	  return v*rho*exp(-v*v/T);
   }

   void UpdateQuadratureData(double velocity, const Vector &S) const;

public:
   M1Operator(int size, ParFiniteElementSpace &h1_fes,
              ParFiniteElementSpace &l2_fes, Array<int> &essential_tdofs,
              ParGridFunction &rho0, double cfl_, Coefficient *material_,
              ParGridFunction &x_gf_, ParGridFunction &T_gf_, bool pa,
              double cgt, int cgiter)
      : LagrangianHydroOperator(size, h1_fes, l2_fes, essential_tdofs, rho0,
        0, cfl_, material_, false, pa, cgt, cgiter), x_gf(x_gf_), T_gf(T_gf_) {}

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const Vector &S, Vector &dS_dt) const;

   // Calls UpdateQuadratureData to compute the new quad_data.dt_est.
   double GetVelocityStepEstimate(const Vector &S) const;
   void ResetVelocityStepEstimate() const;
   void ResetQuadratureData() const { quad_data_is_current = false; }

   ~M1Operator() {}
};

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_M1
