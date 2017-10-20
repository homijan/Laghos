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

#include "m1_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace hydrodynamics
{

void M1Operator::Mult(const Vector &S, Vector &dS_dt) const
{
   dS_dt = 0.0;

   const double velocity = GetTime(); 

   UpdateQuadratureData(velocity, S);

   sourceI0_pcf->SetVelocity(velocity);
   ParGridFunction I0source(&L2FESpace);
   I0source.ProjectCoefficient(*sourceI0_pcf);

   // The monolithic BlockVector stores the unknown fields as follows:
   // - isotropic I0 (energy density)
   // - anisotropic I1 (flux density)

   const int VsizeL2 = L2FESpace.GetVSize();
   const int VsizeH1 = H1FESpace.GetVSize();

   ParGridFunction I0, I1;
   Vector* sptr = (Vector*) &S;
   I0.MakeRef(&L2FESpace, *sptr, 0);
   I1.MakeRef(&H1FESpace, *sptr, VsizeL2);

   ParGridFunction dI0, dI1;
   dI0.MakeRef(&L2FESpace, dS_dt, 0);
   dI1.MakeRef(&H1FESpace, dS_dt, VsizeL2);

   if (!p_assembly)
   {
      Force = 0.0;
      timer.sw_force.Start();
      Force.Assemble();
      timer.sw_force.Stop();
   }

   // Solve for velocity.
   Vector rhs(VsizeH1), B, X;
   if (p_assembly)
   {
      timer.sw_force.Start();
      ForcePA.Mult(I0, rhs);
      timer.sw_force.Stop();
      timer.dof_tstep += H1FESpace.GlobalTrueVSize();
      rhs.Neg();

      // Partial assembly solve for each velocity component.
      const int size = H1compFESpace.GetVSize();
      for (int c = 0; c < dim; c++)
      {
         Vector rhs_c(rhs.GetData() + c*size, size),
                dI1_c(dI1.GetData() + c*size, size);

         Array<int> c_tdofs;
         Array<int> ess_bdr(H1FESpace.GetParMesh()->bdr_attributes.Max());
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e.,
         // we must enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[c] = 1;
         // Essential true dofs as if there's only one component.
         H1compFESpace.GetEssentialTrueDofs(ess_bdr, c_tdofs);

         dI1_c = 0.0;
         Vector B(H1compFESpace.TrueVSize()), X(H1compFESpace.TrueVSize());
         H1compFESpace.Dof_TrueDof_Matrix()->MultTranspose(rhs_c, B);
         H1compFESpace.GetRestrictionMatrix()->Mult(dI1_c, X);

         VMassPA.EliminateRHS(c_tdofs, B);

         CGSolver cg(H1FESpace.GetParMesh()->GetComm());
         cg.SetOperator(VMassPA);
         cg.SetRelTol(cg_rel_tol);
         cg.SetAbsTol(0.0);
         cg.SetMaxIter(cg_max_iter);
         cg.SetPrintLevel(-1);
         timer.sw_cgH1.Start();
         cg.Mult(B, X);
         timer.sw_cgH1.Stop();
         timer.H1dof_iter += cg.GetNumIterations() *
                             H1compFESpace.GlobalTrueVSize();
         H1compFESpace.Dof_TrueDof_Matrix()->Mult(X, dI1_c);
      }
   }
   else
   {
      timer.sw_force.Start();
      Force.Mult(I0, rhs);
      timer.sw_force.Stop();
      timer.dof_tstep += H1FESpace.GlobalTrueVSize();
      rhs.Neg();
      HypreParMatrix A;
      dI1 = 0.0;
      Mv.FormLinearSystem(ess_tdofs, dI1, rhs, A, X, B);
      CGSolver cg(H1FESpace.GetParMesh()->GetComm());
      cg.SetOperator(A);
      cg.SetRelTol(1e-8); cg.SetAbsTol(0.0);
      cg.SetMaxIter(200);
      cg.SetPrintLevel(0);
      timer.sw_cgH1.Start();
      cg.Mult(B, X);
      timer.sw_cgH1.Stop();
      timer.H1dof_iter += cg.GetNumIterations() *
                          H1compFESpace.GlobalTrueVSize();
      Mv.RecoverFEMSolution(X, rhs, dI1);
   }

   // Solve for energy, assemble the energy source if such exists.
   //LinearForm *I0_source = NULL;
   // TODO I0_source should be evaluated based on precomputed quadrature points.
   //if (source_type == 1) // 2D Taylor-Green.
   //{
   //   e_source = new LinearForm(&L2FESpace);
   //   TaylorCoefficient coeff;
   //   DomainLFIntegrator *d = new DomainLFIntegrator(coeff, &integ_rule);
   //   e_source->AddDomainIntegrator(d);
   //   e_source->Assemble();
   //}
   Array<int> l2dofs;
   Vector I0_rhs(VsizeL2), loc_rhs(l2dofs_cnt), loc_I0source(l2dofs_cnt),
          loc_MeMultI0source(l2dofs_cnt), loc_dI0(l2dofs_cnt);
   if (p_assembly)
   {
      timer.sw_force.Start();
      ForcePA.MultTranspose(I1, I0_rhs);
      timer.sw_force.Stop();
      timer.dof_tstep += L2FESpace.GlobalTrueVSize();

      //if (I0_source) { I0_rhs += *I0_source; }
      for (int z = 0; z < nzones; z++)
      {
         L2FESpace.GetElementDofs(z, l2dofs);
         I0_rhs.GetSubVector(l2dofs, loc_rhs);
         locEMassPA.SetZoneId(z);
         //
         I0source.GetSubVector(l2dofs, loc_I0source);
         locEMassPA.Mult(loc_I0source, loc_MeMultI0source);
         loc_rhs += loc_MeMultI0source;
         //
         timer.sw_cgL2.Start();
         locCG.Mult(loc_rhs, loc_dI0);
         timer.sw_cgL2.Stop();
         timer.L2dof_iter += locCG.GetNumIterations() * l2dofs_cnt;
         dI0.SetSubVector(l2dofs, loc_dI0);
      }
   }
   else
   {
      timer.sw_force.Start();
      Force.MultTranspose(I1, I0_rhs);
      timer.sw_force.Stop();
      timer.dof_tstep += L2FESpace.GlobalTrueVSize();
      if (I0_source) { I0_rhs += *I0_source; }
      for (int z = 0; z < nzones; z++)
      {
         L2FESpace.GetElementDofs(z, l2dofs);
         I0_rhs.GetSubVector(l2dofs, loc_rhs);
         timer.sw_cgL2.Start();
         Me_inv(z).Mult(loc_rhs, loc_dI0);
         timer.sw_cgL2.Stop();
         timer.L2dof_iter += l2dofs_cnt;
         dI0.SetSubVector(l2dofs, loc_dI0);
      }
   }
   delete I0_source;

   quad_data_is_current = false;
}

double M1Operator::GetVelocityStepEstimate(const Vector &S) const
{
   const double velocity = GetTime();
   UpdateQuadratureData(velocity, S);

   double glob_dt_est;
   MPI_Allreduce(&quad_data.dt_est, &glob_dt_est, 1, MPI_DOUBLE, MPI_MIN,
                 H1FESpace.GetParMesh()->GetComm());
   return glob_dt_est;
}

void M1Operator::ResetVelocityStepEstimate() const
{
   quad_data.dt_est = numeric_limits<double>::infinity();
}

void M1Operator::UpdateQuadratureData(double velocity, const Vector &S) const
{
   if (quad_data_is_current) { return; }
   timer.sw_qdata.Start();

   mspInv_pcf->SetVelocity(velocity);
   const int nqp = integ_rule.GetNPoints();

   ParGridFunction I0, I1;
   Vector* sptr = (Vector*) &S;
   I0.MakeRef(&L2FESpace, *sptr, 0);
   I1.MakeRef(&H1FESpace, *sptr, L2FESpace.GetVSize());

   Vector vector_vals(h1dofs_cnt * dim);
   DenseMatrix Jpi(dim), Jinv(dim), I0stress(dim), I0stressJiT(dim),
               I1stress(dim), I1stressJiT(dim),
               vecvalMat(vector_vals.GetData(), h1dofs_cnt, dim);
   Array<int> L2dofs, H1dofs;

   // Batched computations are needed, because hydrodynamic codes usually
   // involve expensive computations of material properties. Although this
   // miniapp uses simple EOS equations, we still want to represent the batched
   // cycle structure.
   int nzones_batch = 3;
   const int nbatches =  nzones / nzones_batch + 1; // +1 for the remainder.
   int nqp_batch = nqp * nzones_batch;
   double *mspInv_b = new double[nqp_batch];
   // Jacobians of reference->physical transformations for all quadrature
   // points in the batch.
   DenseTensor *Jpr_b = new DenseTensor[nqp_batch];
   for (int b = 0; b < nbatches; b++)
   {
      int z_id = b * nzones_batch; // Global index over zones.
      // The last batch might not be full.
      if (z_id == nzones) { break; }
      else if (z_id + nzones_batch > nzones)
      {
         nzones_batch = nzones - z_id;
         nqp_batch    = nqp * nzones_batch;
      }

      for (int z = 0; z < nzones_batch; z++)
      {
         ElementTransformation *T = H1FESpace.GetElementTransformation(z_id);
         Jpr_b[z].SetSize(dim, dim, nqp);

         if (p_assembly)
         {
            // All reference->physical Jacobians at the quadrature points.
            H1FESpace.GetElementVDofs(z_id, H1dofs);
            x_gf.GetSubVector(H1dofs, vector_vals);
            evaluator->GetVectorGrad(vecvalMat, Jpr_b[z]);
         }
         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
            T->SetIntPoint(&ip);
            if (!p_assembly) { Jpr_b[z](q) = T->Jacobian(); }

            const int idx = z * nqp + q;
            mspInv_b[idx] = mspInv_pcf->Eval(*T, ip);
         }
         ++z_id;
      }

      z_id -= nzones_batch;
      for (int z = 0; z < nzones_batch; z++)
      {
/*
         ElementTransformation *T = H1FESpace.GetElementTransformation(z_id);
*/
         for (int q = 0; q < nqp; q++)
         {
/*
            const IntegrationPoint &ip = integ_rule.IntPoint(q);
            T->SetIntPoint(&ip);
*/
            // Note that the Jacobian was already computed above. We've chosen
            // not to store the Jacobians for all batched quadrature points.
            const DenseMatrix &Jpr = Jpr_b[z](q);
            CalcInverse(Jpr, Jinv);
            const double detJ = Jpr.Det();
            double mspInv = mspInv_b[z*nqp + q];

            // Time step estimate at the point. Here the more relevant length
            // scale is related to the actual mesh deformation; we use the min
            // singular value of the ref->physical Jacobian. In addition, the
            // time step estimate should be aware of the presence of shocks.
            const double h_min =
               Jpr.CalcSingularvalue(dim-1) / (double) H1FESpace.GetOrder(0);
            double inv_dt = mspInv / h_min;
            //if (M1_dvmax * inv_dt / cfl < 1.0) { inv_dt = cfl / M1_dvmax; }
            //if (M1_dvmin * inv_dt > 1.0)
            //{
            //   inv_dt = 1.0 / M1_dvmin;
            //   mspInv = inv_dt * h_min;
            //}
            quad_data.dt_est = min(quad_data.dt_est, cfl * (1.0 / inv_dt) );

            I0stress = 0.0;
            I1stress = 0.0;
			for (int d = 0; d < dim; d++)
            {
               I0stress(d, d) = mspInv;
               I1stress(d, d) = mspInv/3.0; // P1 closure.
            }

/*
               I1stress.Add(visc_coeff, sgrad_v);
*/

            // Quadrature data for partial assembly of the force operator.
            MultABt(I0stress, Jinv, I0stressJiT);
            I0stressJiT *= integ_rule.IntPoint(q).weight * detJ;
            MultABt(I1stress, Jinv, I1stressJiT);
            I1stressJiT *= integ_rule.IntPoint(q).weight * detJ;
            for (int vd = 0 ; vd < dim; vd++)
            {
               for (int gd = 0; gd < dim; gd++)
               {
                  quad_data.vstressJinvT(vd)(z_id*nqp + q, gd) =
                     I1stressJiT(vd, gd);
                  quad_data.tstressJinvT(vd)(z_id*nqp + q, gd) =
                     I0stressJiT(vd, gd);
               }
            }
         }
         ++z_id;
      }
   }
   delete [] mspInv_b;
   delete [] Jpr_b;
   quad_data_is_current = true;

   timer.sw_qdata.Stop();
   timer.quad_tstep += nzones * nqp;
}

double a0 = 5e3;

double M1MeanStoppingPowerInverse::Eval(ElementTransformation &T,
                                        const IntegrationPoint &ip)
{
   double rho = rho_gf.GetValue(T.ElementNo, ip);
   double Te = Te_gf.GetValue(T.ElementNo, ip);
   double a = a0 * (Tmax * Tmax); //1e8; // The plasma collision model.
   double nu_scaled = a * rho / pow(alphavT, 4.0) / pow(velocity, 3.0);
   // M1 requires the inverse of the mean stopping power,
   // further multiplied by rho (compensates constant mass matrices).
   return rho / nu_scaled;
}

double M1I0Source::Eval(ElementTransformation &T, const IntegrationPoint &ip)
{
   double rho = rho_gf.GetValue(T.ElementNo, ip);
   double Te = max(1e-6, Te_gf.GetValue(T.ElementNo, ip));

   return -1e-0 * rho * pow(alphavT, 3.0) * (2.0 * velocity / alphavT -
      alphavT * pow(velocity, 3.0) / pow(eos->vTe(Te), 2.0)) *
      exp(- pow(alphavT, 2.0) / 2.0 / pow(eos->vTe(Te), 2.0) *
      pow(velocity, 2.0));
}

} // namespace hydrodynamics

} // namespace mfem

#endif // MFEM_USE_MPI
