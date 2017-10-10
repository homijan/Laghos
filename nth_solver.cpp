#include "nth_solver.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace nth
{

EIIO_BGK_solver::EIIO_BGK_solver(HypreParMatrix *_IinvK, HypreParMatrix *_IG, 
   Vector *_Ib, HypreParMatrix *_IS, HypreParMatrix *_TM, 
   HypreParMatrix *_TNGrad, Vector *_Tb): IinvK(_IinvK), IG(_IG), Ib(_Ib), 
   IS(_IS), TM(_TM), TNGrad(_TNGrad), Tb(_Tb), NGradInvK(NULL), 
   NGradInvKS(NULL), pT_matrix(NULL), zg(_IinvK->Height()), 
   zs(_IinvK->Height()), z(_TM->Height()), pT_solver(_IinvK->GetComm())
{
   // The temperature T system matrix inversion/solver.
   const double rel_tol = 1e-8;
   pT_solver.iterative_mode = false;
   pT_solver.SetRelTol(rel_tol); 
   pT_solver.SetAbsTol(0.0);
   pT_solver.SetMaxIter(30);
   pT_solver.SetPrintLevel(0);
   pT_prec.SetType(HypreSmoother::Jacobi);
   pT_solver.SetPreconditioner(pT_prec);   
}	 

// Initialization of the system matrix.
void EIIO_BGK_solver::Init(const double dt)
{
   // I = (nGrad + kM - GammaIm)^{-1} (GammaEx I + b(T))
   // IinvK = (nGrad + kM - GammaIm)^{-1}
   // I = IinvK*(IS*T1 + IG*I + Ib)) 

   // M*(T1-T0)/dt + nGrad I(T1) = S_external
   // 1/dt*TM*T1 + TNGrad*IinvK*IS*T1 = 1/dt*TM*T0 - TNGrad*IinvK*(IG*I + Ib) 
   // + Tb
   // pT_matrix = 1/dt*TM + TNGrad*IinvK*IS
/*
   NGradInvK = ParMult(TNGrad, IinvK);
   cout << "NGradInvK-MxN: " << NGradInvK->M() << ", " << NGradInvK->N() <<
      endl << flush;
   NGradInvKS = ParMult(NGradInvK, IS);
   cout << "NGradInvKS-MxN: " << NGradInvKS->M() << ", " << NGradInvKS->N() <<
      endl << flush;
   *TM *= 1/dt;
   cout << "TM-MxN: " << TM->M() << ", " << TM->N() << endl << flush;
   pT_matrix = ParAdd(TM, NGradInvKS);
   pT_solver.SetOperator(*pT_matrix);
   //pT_solver.SetOperator(TM);
*/
}

// Implementation of class EIIO_BGK_solver
void EIIO_BGK_solver::Step(Vector &I) const
{
   // y = (nGrad + kM - GammaIm)^{-1} (GammaEx y + b) 
   // K = nGrad + kM - GammaIm
   IG->Mult(I, zg);
   zg += *Ib;
   IinvK->Mult(zg, I);
}

void EIIO_BGK_solver::Step(const double dt, Vector &T0, Vector &T, Vector &I) 
   const
{
   // y = (nGrad + kM - GammaIm)^{-1} (GammaEx y + b(T))
   // IinvK = (nGrad + kM - GammaIm)^{-1}
   // I = IinvK*(IS*T1 + IG*I + Ib)) 
   IG->Mult(I, zg);
   IS->Mult(T, zs);
   zg += zs;
   IinvK->Mult(zg, I);

   // M*(T1-T0)/dt + nGrad I(T1) = S_external
   // 1/dt*TM*T1 + TNGrad*IinvK*IS*T1 = 1/dt*TM*T0 - TNGrad*IinvK*(IG*I + Ib) 
   // + Tb
   // z = 1/dt*TM*T0 + Tb - TNGrad*IinvK*(IG*I + Ib)
/*
   *Tb = 0.;
   TM->Mult(T0, z); // TM has been already multiplied by 1/dt in intialization.
   *Tb += z;
   IG->Mult(I, zg);
   zg += *Ib;
   NGradInvK->Mult(zg, z);
   z *= -1.;
   *Tb += z;
   // invert vector mass matrix	   
   pT_solver.Mult(*Tb, T);  
*/
}

ParNonlocalOperator::ParNonlocalOperator(ParMesh *_pmesh, 
   ParFiniteElementSpace *_pT_fes, int _order_I_x, int _order_T_x, 
   int _order_I_phi, int _order_I_theta) 
   : Amesh_pi(NULL), Amesh_twopi(NULL), Afec_phi(NULL), Afec_theta(NULL),
   Afes(NULL), pmesh(NULL), Xfec(NULL), pI_fes(NULL), pT_fes(NULL), pIK(NULL),
   pIG(NULL), pIb(NULL), pIS(NULL), pTNGrad(NULL), pTM(NULL), pTb(NULL),
   pI_pgf(NULL), pMoments_pcf(NULL)
{
   // Assign ParMesh and temperature finite element space.
   pmesh = _pmesh;
   pT_fes = _pT_fes;          
   // Construct the angular finite element space.
   Afec_phi = new L2_FECollection(_order_I_phi, 1, BasisType::Positive);
   Afec_theta = new L2_FECollection(_order_I_theta, 1, BasisType::Positive);
   if (pmesh->Dimension() < 3)
   {
      const char *pi_mesh_file = "data/pi_segment.mesh";
      Amesh_pi = new Mesh(pi_mesh_file, 1, 1);
      Afes = new AngularFiniteElementSpace(Amesh_pi, Afec_phi, Amesh_pi,
         Afec_theta);
   }
   else
   {
      const char *pi_mesh_file = "data/pi_segment.mesh";
      const char *twopi_mesh_file = "data/twopi_segment.mesh";
      Amesh_pi = new Mesh(pi_mesh_file, 1, 1);
      Amesh_twopi = new Mesh(twopi_mesh_file, 1, 1);
      Afes = new AngularFiniteElementSpace(Amesh_pi, Afec_phi, Amesh_twopi,
         Afec_theta);
   }
   int dim = pmesh->Dimension();
   int ndof_afes = Afes->GetAfesNDofs();
   //if (_order_I_phi==0)
   //{
   //   Afes->SetUniqueIntPointPhi(0.001);
   //}
   // DG Finite element space (same order as hydro velocity).
   // DG_FECollection is typedefed L2_FECollection
   int basis_type = BasisType::Positive;
   //int basis_type = BasisType::GaussLegendre;
   //int basis_type = BasisType::GaussLobatto;
   Xfec = new DG_FECollection(_order_I_x, dim, basis_type);
   pI_fes = new ParFiniteElementSpace(pmesh, Xfec, ndof_afes);
   // Create the intensity grid function.
   pI_pgf = GetIntensityGridFunction();
}

ParGridFunction* ParNonlocalOperator::GetIntensityGridFunction()
{
   if (pI_pgf==NULL)
   {
      pI_pgf = new ParGridFunction(pI_fes);
      int dim  = pmesh->Dimension();
      pMoments_pcf = new MomentsCoefficient(dim, Afes, pI_pgf);
      // TMP components.
      pZeroMoment_pcf = new MomentsCoefficient(dim, Afes, pI_pgf);
      pZeroMoment_pcf->SetComponent(0);
      pFirstMomentz_pcf = new MomentsCoefficient(dim, Afes, pI_pgf);
      pFirstMomentz_pcf->SetComponent(1);
      pFirstMomentx_pcf = new MomentsCoefficient(dim, Afes, pI_pgf);
      pFirstMomentx_pcf->SetComponent(2);
      pFirstMomenty_pcf = new MomentsCoefficient(dim, Afes, pI_pgf);
      pFirstMomenty_pcf->SetComponent(3);
      pFirstMomentMagnitude_pcf = new MomentsCoefficient(dim, Afes, pI_pgf);
      pFirstMomentMagnitude_pcf->SetComponent(-1);	 
   }
   
   return pI_pgf;
}

void ParNonlocalOperator::ModelAlgebraicTranslation(Coefficient *Cv, 
   Coefficient *kappa, Coefficient *isosigma, Coefficient *sourceb, 
   Coefficient *sourceCoeffT, Coefficient *sourceT, 
   VectorCoefficient *Efield)
{
   // Intensity related forms.
   pIK = new ParInvLocBilinearForm(pI_fes);
   pIK->AddDomainIntegrator(new KineticCollisionIntegrator(Afes, 1., kappa, 
      isosigma, Efield));
   pIK->AddInteriorFaceIntegrator(new KineticGammaImIntegrator(Afes, -1));
   pIK->AddBdrFaceIntegrator(new KineticGammaImIntegrator(Afes, -1.));  
   pIG = new ParBilinearForm(pI_fes);
   pIG->AddInteriorFaceIntegrator(new KineticGammaExIntegrator(Afes)); 
   pIG->AddBdrFaceIntegrator(new KineticGammaExIntegrator(Afes));
   pIb = new ParLinearForm(pI_fes);
   pIb->AddDomainIntegrator(new KineticDomainLFIntegrator(Afes, sourceb));
   // Intensity-Temperature coupling related forms.
   pIS = new ParMixedBilinearForm(pT_fes, pI_fes);
   pIS->AddDomainIntegrator(new KineticSourceIntegrator(Afes, 1., 
      sourceCoeffT));
   pTNGrad = new ParMixedBilinearForm(pI_fes, pT_fes);
   pTNGrad->AddDomainIntegrator(new KineticNGradIntegrator(Afes));
   // Temperature related forms.
   pTM = new ParInvLocBilinearForm(pT_fes);
   pTM->AddDomainIntegrator(new InvMassLocalIntegrator(Cv));
   pTb = new ParLinearForm(pT_fes);
   pTb->AddDomainIntegrator(new DomainLFIntegrator(*sourceT));
}

void ParNonlocalOperator::Compute(const double dt, const double tol, 
   double &Umax, double &dUmax, int &nti, ParGridFunction *u, 
   ParGridFunction *T)
{
   // Algebraic assemble.
   //cout << "update starts..." << endl << flush;
   pIK->Update();
   pIG->Update(); 
   pIb->Update();
   if (T!=NULL)
   {
      pIS->Update();
      pTNGrad->Update();
      pTM->Update();
   }
   //cout << "update finished." << endl << flush;
   int skip_zeros = 0;
   //cout << "k-assemble starts..." << endl << flush;
   pIK->Assemble(skip_zeros);
   //cout << "k-assemble finished." << endl << flush;
   //cout << "k-local inverse starts..." << endl << flush;
   pIK->LocalInverse(); 
   //cout << "k-local inverse finished." << endl << flush;
   //cout << "k-finelize starts..." << endl << flush;
   pIK->Finalize(skip_zeros);
   //cout << "k-assemble finalize finished." << endl << flush;
   //cout << "g-assemble starts..." << endl << flush;
   pIG->Assemble(skip_zeros);
   //cout << "g-assemble finished." << endl << flush;
   //cout << "g-finalize starts..." << endl << flush;
   pIG->Finalize(skip_zeros);
   //cout << "g-finalize finished." << endl << flush;
   //cout << "b-assemble starts..." << endl << flush;
   pIb->Assemble();
   //cout << "b-assemble finished." << endl << flush;
   if (T!=NULL)
   {
      //cout << "s-assemble starts..." << endl << flush;
      pIS->Assemble(skip_zeros);
      //cout << "s-assemble finished." << endl << flush;
      //cout << "s-finalize starts..." << endl << flush;
      pIS->Finalize(skip_zeros);
      //cout << "s-finalize finished." << endl << flush;
      //cout << "m-assemble starts..." << endl << flush;
      pTM->Assemble(skip_zeros);
      //cout << "m-assemble finished." << endl << flush;
      //cout << "m-finalize starts..." << endl << flush;
      pTM->Finalize(skip_zeros);
      //cout << "m-finalize finished." << endl << flush;
      //cout << "ngrad-assemble starts..." << endl << flush;
      pTNGrad->Assemble(skip_zeros);
      //cout << "ngrad-assemble finished." << endl << flush;
      //cout << "ngrad-finalize starts..." << endl << flush;
      pTNGrad->Finalize(skip_zeros);
      //cout << "ngrad-finalize finished." << endl << flush;     
      //cout << "b-assemble starts..." << endl << flush;
      pTb->Assemble();
      //cout << "b-assemble finished." << endl << flush;
   }
   // Hypre representation for parallel computation
   *u = 0.;
   //cout << "parallel-assemble starts..." << endl << flush;
   HypreParVector *vU = u->GetTrueDofs();
   HypreParVector *vT = NULL, *vT0 = NULL;
   if (T!=NULL)
   {
      vT = T->GetTrueDofs();
      vT0 = T->GetTrueDofs();
   }
   HypreParMatrix *IK = pIK->ParallelAssemble();
   HypreParMatrix *IG = pIG->ParallelAssemble(); 
   HypreParVector *Ib = pIb->ParallelAssemble();
   HypreParMatrix *IS = NULL;
   HypreParMatrix *TM = NULL;
   HypreParMatrix *TNGrad = NULL;
   HypreParVector *Tb = NULL;
   if (T!=NULL)
   {
      IS = pIS->ParallelAssemble();
      TM = pTM->ParallelAssemble();
      TNGrad = pTNGrad->ParallelAssemble();
      Tb = pTb->ParallelAssemble();
   }
   //cout << "parallel-assemble finished." << endl << flush;
   // The BGK solver
   EIIO_BGK_solver *nonlocalBGK;
   if (T!=NULL)
   {	  
      nonlocalBGK = new EIIO_BGK_solver(IK, IG, Ib, IS, TM, TNGrad, Tb);
      nonlocalBGK->Init(dt);
   }
   else
   {
      nonlocalBGK = new EIIO_BGK_solver(IK, IG, Ib);
   }
   // Proper iteration of the solver
   //cout << "iteration starts..." << endl << flush;
   Umax = -1e32;
   dUmax = 1e32;
   bool done = false;
   nti = 0;
   for (nti = 0; !done; )
   {
      if (T!=NULL)
      {
         nonlocalBGK->Step(dt, *vT0, *vT, *vU);
      }
      else
      {
         nonlocalBGK->Step(*vU); 
      }
      nti++;

      double Umax_local = vU->Max();
      double Umax_global;
      MPI_Barrier(pmesh->GetComm());
      MPI_Reduce(&Umax_local, &Umax_global, 1, MPI_DOUBLE, MPI_MAX, 0,
         pmesh->GetComm());
      MPI_Barrier(pmesh->GetComm());
      MPI_Bcast(&Umax_global, 1, MPI_DOUBLE, 0, pmesh->GetComm());
      MPI_Barrier(pmesh->GetComm()); 
      dUmax = abs((Umax_global - Umax)/Umax);
      Umax = Umax_global;
      //cout << "nti, Umax, dUmax: " << nti << ", " << Umax << ", " 
      //   << dUmax << endl << flush;
      done = (dUmax <= tol);
   }  
   //cout << "iteration ended." << endl << flush;
   // Extract the parallel grid function corresponding to the finite 
   // element approximation U (the local solution on each processor).
   *u = *vU;
   if (T!=NULL)
   {
      *T = *vT;
   }
   // Delete Hypre structures.
   delete vU;
   delete vT;
   delete IK;
   delete IG;
   delete Ib;
   delete IS;
   delete TM;
   delete TNGrad;
   delete Tb;
   delete nonlocalBGK; 
} 

double ParNonlocalOperator::MomentsCoefficient::Eval(ElementTransformation &T, 
   const IntegrationPoint &ip)
{
   int afes_ndofs = Afes->GetAfesNDofs();
   double Omega;
   Vector Ix, afes_meanshape;
   // Evaluate intensity in a spatial point given by ip for every 
   // component of the angular discretization.  
   pI->GetVectorValue(T.ElementNo, ip, Ix);
   // Get the zero moment of angular components.
   afes_meanshape.SetSize(afes_ndofs); 
   Afes->CalcBasisZeroMoment(Omega, afes_meanshape);
        
   //cout << "meanshape" << endl << flush;
   //afes_meanshape.Print();
		 
   if (component == 0) { return afes_meanshape*Ix; }

   Vector afes_zcomponent, afes_xcomponent, afes_ycomponent;
   afes_zcomponent.SetSize(afes_ndofs); 
   afes_xcomponent.SetSize(afes_ndofs);
   afes_ycomponent.SetSize(afes_ndofs);
   Afes->CalcBasisFirstMoment(Omega, afes_zcomponent, afes_xcomponent,
      afes_ycomponent);

   //cout << "z-component" << endl << flush; 
   //afes_zcomponent.Print();
   //cout << "x-component" << endl << flush;
   //afes_xcomponent.Print();
   //cout << "y-component" << endl << flush;
   //afes_ycomponent.Print();

   if (component == 1) { return afes_zcomponent*Ix; }
   else if (component == 2) { return afes_xcomponent*Ix; }
   else if (component == 3) { return afes_ycomponent*Ix; }
   else if (component == -1) 
   {
      if (dim == 1) { return (afes_zcomponent*Ix) * (afes_zcomponent*Ix); }
      else if (dim == 2)
      {
         return (afes_zcomponent*Ix) * (afes_zcomponent*Ix) + 
            (afes_xcomponent*Ix) * (afes_xcomponent*Ix);
      }
      else if (dim == 3)
      {
         return (afes_zcomponent*Ix) * (afes_zcomponent*Ix) + \
            (afes_xcomponent*Ix) * (afes_xcomponent*Ix) +
            (afes_ycomponent*Ix) * (afes_ycomponent*Ix);
      } 
      else { MFEM_ABORT("Unsupported mesh dimension"); }  
   }
}

void ParNonlocalOperator::MomentsCoefficient::Eval(Vector &flux, 
   ElementTransformation &T, const IntegrationPoint &ip) 
{
   int afes_ndofs = Afes->GetAfesNDofs();
   double Omega;
   Vector Ix, afes_zcomponent, afes_xcomponent, afes_ycomponent;
   // Evaluate intensity in a spatial point given by ip for every 
   // component of the angular discretization.  
   pI->GetVectorValue(T.ElementNo, ip, Ix);
   // Get the zero moment of angular components.
   afes_zcomponent.SetSize(afes_ndofs); 
   afes_xcomponent.SetSize(afes_ndofs);
   afes_ycomponent.SetSize(afes_ndofs);
   Afes->CalcBasisFirstMoment(Omega, afes_zcomponent, afes_xcomponent,
      afes_ycomponent);

   if (dim == 1) { flux(0) = afes_zcomponent*Ix; }
   else if (dim == 2)
   {
      flux(0) = afes_zcomponent*Ix;
      flux(1) = afes_xcomponent*Ix;
   }
   else if (dim == 3)
   {
      flux(0) = afes_zcomponent*Ix;
      flux(1) = afes_xcomponent*Ix;
      flux(2) = afes_ycomponent*Ix;
   }
}

} // namespace nth

} // namespace mfem

#endif // MFEM_USE_MPI
