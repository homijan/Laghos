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
#include "m1_assembly.hpp"

#ifdef MFEM_USE_MPI

#include <memory>
#include <iostream>
#include <fstream>

namespace mfem
{

namespace nth
{

/// Visualize the given parallel grid function, using a GLVis server on the
/// specified host and port. Set the visualization window title, and optionally,
/// its geometry.
void VisualizeField(socketstream &sock, const char *vishost, int visport,
                    ParGridFunction &gf, const char *title,
                    int x = 0, int y = 0, int w = 400, int h = 400,
                    bool vec = false);

struct TimingData
{
   // Total times for all major computations:
   // CG solves (H1 and L2) / force RHS assemblies / quadrature computations.
   StopWatch sw_cgH1, sw_cgL2, sw_force, sw_qdata;

   // These accumulate the total processed dofs or quad points:
   // #dofs  * #(CG iterations) for the CG solves (H1 and L2).
   // #dofs  * #(RK sub steps) for the Force application and assembly.
   // #quads * #(RK sub steps) for the quadrature data computations.
   long long int H1dof_iter, L2dof_iter, dof_tstep, quad_tstep;

   TimingData()
      : H1dof_iter(0), L2dof_iter(0), dof_tstep(0), quad_tstep(0) { }
};

class M1HydroCoefficient;

// Given a solutions state (f0, f1), this class performs all necessary
// computations to evaluate the new slopes (df0_dt, df1_dt).
class M1Operator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &H1FESpace;
   ParFiniteElementSpace &L2FESpace;
   mutable ParFiniteElementSpace H1compFESpace;

   Array<int> &ess_tdofs;

   const int dim, nzones, l2dofs_cnt, h1dofs_cnt;
   const double cfl;
   const bool p_assembly;
   const double cg_rel_tol;
   const int cg_max_iter;

   // Velocity mass matrix and local inverses of the energy mass matrices. These
   // are constant in time, due to the pointwise mass conservation property.
   mutable ParBilinearForm Mv;
   DenseTensor Me, Me_inv;

   // Integration rule for all assemblies.
   const IntegrationRule &integ_rule;

   // Data associated with each quadrature point in the mesh. These values are
   // recomputed at each time step.
   mutable QuadratureData quad_data;
   mutable bool quad_data_is_current;

   // Force matrix that combines the kinematic and thermodynamic spaces. It is
   // assembled in each time step and then it's used to compute the final
   // right-hand sides for momentum and specific internal energy.
   mutable MixedBilinearForm vForce, tForce;

   // Same as above, but done through partial assembly.
   ForcePAOperator ForcePA;

   // Mass matrices done through partial assembly:
   // velocity (coupled H1 assembly) and energy (local L2 assemblies).
   mutable MassPAOperator VMassPA;
   mutable LocalMassPAOperator locEMassPA;

   // Linear solver for energy.
   CGSolver locCG;

   mutable TimingData timer;

   void UpdateQuadratureData(double velocity, const Vector &S) const;

   // TODO M1_dvmin does not work, because of its local nature. 
   // M1_dvmax does not seem to have an important effect.
   double M1_dvmin, M1_dvmax;
   // The grid function is necessary for velocity step estimation. 
   ParGridFunction &x_gf;
   // Velocity dependent coefficients providing physics.
   M1HydroCoefficient *mspInv_pcf, *sourceI0_pcf;

public:
   M1Operator(int size, ParFiniteElementSpace &h1_fes,
              ParFiniteElementSpace &l2_fes, Array<int> &essential_tdofs,
              ParGridFunction &rho0, double cfl_, M1HydroCoefficient *mspInv_,
              M1HydroCoefficient *sourceI0_, ParGridFunction &x_gf_, 
              ParGridFunction &T_gf_, bool pa, double cgt, int cgiter);

   // Solve for dx_dt, dv_dt and de_dt.
   virtual void Mult(const Vector &S, Vector &dS_dt) const;

   // Calls UpdateQuadratureData to compute the new quad_data.dt_est.
   double GetVelocityStepEstimate(const Vector &S) const;
   void ResetVelocityStepEstimate() const;
   void ResetQuadratureData() const { quad_data_is_current = false; }

   // The density values, which are stored only at some quadrature points, are
   // projected as a ParGridFunction.
   void ComputeDensity(ParGridFunction &rho);

   void PrintTimingData(bool IamRoot, int steps);

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

} // namespace nth

} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_M1_SOLVER
