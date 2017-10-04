#ifndef MFEM_PNONLOCALOPERATOR
#define MFEM_PNONLOCALOPERATOR

#include "mfem.hpp"
#include "nonlocalinteg.hpp"

#ifdef MFEM_USE_MPI

using namespace std;

namespace mfem
{

namespace nth
{

/** An Explicit-In-Implicit-Out (EI^2O) solver of the steady BGK transport 
    equation. The DG weak form of n.grad(u) = k*(S - u) is 
	K uI + k uI - GIO uI = b - GEI uE, where K, k, and GIO are the advection, 
	attenuation, and internal face Gamma (u-IO) matrices acting on 
	uI (implicit), b describes the explicit source S and flow on the boundary, 
	and GEI is the internal face flow Gamma matrix acting in uE (explicit). 
	This can be written as a local stencil inversion, 
	uI = (K + k - GIO)^{-1} (b - GEI uE). */
class EIIO_BGK_solver
{
private:
   HypreParMatrix *IinvK, *IG, *IS, *TM, *TNGrad;
   Vector *Ib, *Tb;
   HypreParMatrix *NGradInvK, *NGradInvKS;

   mutable HypreParMatrix *pT_matrix;    
   // Krylov solver for inverting the mass matrix Mv
   mutable CGSolver pT_solver;  
   // Preconditioner for the mass matrix Mv
   HypreSmoother pT_prec;

   mutable Vector zg, zs, z;

public:
   // Implementation of class EIIO_BGK_solver
   EIIO_BGK_solver(HypreParMatrix *_IinvK, HypreParMatrix *_IG, Vector *_Ib): 
      IinvK(_IinvK), IG(_IG), Ib(_Ib), IS(NULL), TM(NULL), 
	  TNGrad(NULL), Tb(NULL), NGradInvK(NULL), NGradInvKS(NULL), 
	  pT_matrix(NULL), zg(_IinvK->Height()), zs(0), 
	  z(0), pT_solver(_IinvK->GetComm()) {}
   EIIO_BGK_solver(HypreParMatrix *_IinvK, HypreParMatrix *_IG, Vector *_Ib, 
      HypreParMatrix *_IS, HypreParMatrix *_TM, HypreParMatrix *_TNGrad, 
	  Vector *_Tb): IinvK(_IinvK), IG(_IG), Ib(_Ib), IS(_IS), TM(_TM), 
	  TNGrad(_TNGrad), Tb(_Tb), NGradInvK(NULL), NGradInvKS(NULL), 
	  pT_matrix(NULL), zg(_IinvK->Height()), zs(_IinvK->Height()), 
	  z(_TM->Height()), pT_solver(_IinvK->GetComm())
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
   virtual void Init(const double dt);
   virtual void Step(Vector &I) const;
   virtual void Step(const double dt, Vector &T0, Vector &T, Vector &I) const;

   virtual ~EIIO_BGK_solver() 
   {
      delete NGradInvK;
	  delete NGradInvKS;
	  delete pT_matrix;
   }
};

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


/** After spatial discretization, the hydrodynamic model can be written as a
 *  system of ODEs:
 *     dv/dt = -Mv^{-1}*(F(v, e, x)*I)
 *     de/dt = Me^{-1}*(F(v, e, x)^T*v)
 *     dx/dt = v,
 *  where x is the vector representing the position of moving mesh, v is 
 *  the velocity field, and e is the internal energy field
 *  Mv is the mass matrix of the velocity field, Me is the mass matrix of 
 *  the internal energy field, and F(v, e, x) is the force matrix.
 *
 *  Class HydrodynamicOperator represents the right-hand side of the above
 *  system of ODEs. */
class ParNonlocalOperator
{
protected:
   class MomentsCoefficient : public Coefficient, public VectorCoefficient, 
      public MatrixCoefficient
   {
   protected:
      int vdim;
	  AngularFiniteElementSpace *Afes;
	  ParGridFunction *pI;
      int component; // TMP switch: 0=I0, 1=I1z, 2=I1x
   public:
	  MomentsCoefficient(int _vdim, AngularFiniteElementSpace *_Afes, 
	     ParGridFunction *_pI) : VectorCoefficient(_vdim), 
		 MatrixCoefficient(_vdim), pI(_pI), Afes(_Afes) 
	  { 
	     vdim = _vdim;
		 component = 0; 
	  } 
      void SetComponent(const int _component) { component = _component; }
	  virtual double Eval(ElementTransformation &T, 
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
		 
		 if (component==0)
		 {
            return afes_meanshape*Ix;
		 }

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

         if (component==1)
		 {
		    return afes_zcomponent*Ix;
		 }
         if (component==2)
		 {
		    return afes_xcomponent*Ix;
		 }
	  }
      virtual void Eval(Vector &flux, ElementTransformation &T,
	     const IntegrationPoint &ip) 
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

         if (vdim == 1)
	     {
	        flux(0) = afes_zcomponent*Ix;
		 }
		 if (vdim == 2)
	     {
	        flux(0) = afes_zcomponent*Ix;
		    flux(1) = afes_xcomponent*Ix;
		 }
         if (vdim == 3)
	     {
	        flux(0) = afes_zcomponent*Ix;
		    flux(1) = afes_xcomponent*Ix;
            flux(2) = afes_ycomponent*Ix;
		 }
	  }
      virtual void Eval(DenseMatrix &pressure, ElementTransformation &T,
	     const IntegrationPoint &ip) 
	  {
	  }
   };

   Mesh *Amesh;
   L2_FECollection *Afec_phi, *Afec_theta;
   AngularFiniteElementSpace *Afes;

   ParMesh *pmesh;
   DG_FECollection *Xfec;
   ParFiniteElementSpace *pI_fes, *pT_fes;

   ParInvLocBilinearForm *pIK, *pTM;
   ParBilinearForm *pIG;
   ParLinearForm *pIb, *pTb;
   ParMixedBilinearForm *pIS, *pTNGrad; 

   ParGridFunction *pI_pgf;
   MomentsCoefficient *pMoments_pcf;
   // TMP components.
   MomentsCoefficient *pZeroMoment_pcf, *pFirstMomentz_pcf, *pFirstMomentx_pcf;

public:
   ParNonlocalOperator(ParMesh *_pmesh, ParFiniteElementSpace *_pT_fes, 
      int _order_I_x=3, 
      int _order_T_x=2, int _order_I_phi=1, int _order_I_theta=0) 
      : Amesh(NULL), Afec_phi(NULL), Afec_theta(NULL), Afes(NULL), pmesh(NULL), 
	  Xfec(NULL), pI_fes(NULL), pT_fes(NULL), pIK(NULL), pIG(NULL), pIb(NULL), 
	  pIS(NULL), pTNGrad(NULL), pTM(NULL), pTb(NULL), pI_pgf(NULL), 
	  pMoments_pcf(NULL)
	  {
 		 // Assign ParMesh and temperature finite element space.
		 pmesh = _pmesh;
		 pT_fes = _pT_fes;          
		 // Construct the angular finite element space.
		 const char *pi_mesh_file = "data/pi_segment.mesh";
         Amesh = new Mesh(pi_mesh_file, 1, 1); 
         Afec_phi = new L2_FECollection(_order_I_phi, 1);
         Afec_theta = new L2_FECollection(_order_I_theta, 1);
         Afes = new AngularFiniteElementSpace(Amesh, Afec_phi, Amesh, 
		    Afec_theta);
         int dim = pmesh->Dimension();
		 int ndof_afes = Afes->GetAfesNDofs();
         //if (_order_I_phi==0)
         //{
         //   Afes->SetUniqueIntPointPhi(0.001);
         //}
         // DG Finite element space (same order as hydro velocity).
         // DG_FECollection is typedefed L2_FECollection
		 int basis_type = BasisType::GaussLegendre;
		 //int basis_type = BasisType::GaussLobatto;
		 Xfec = new DG_FECollection(_order_I_x, dim, basis_type);
         pI_fes = new ParFiniteElementSpace(pmesh, Xfec, ndof_afes);
		 // Create the intensity grid function.
		 pI_pgf = GetIntensityGridFunction();
	  }

   DG_FECollection* GetXfec() { return Xfec; }
   Coefficient* GetZeroMomentCoefficient() { return pMoments_pcf; }
   // TMP coefficients.
   Coefficient* GetFirstMomentZCoefficient() { return pFirstMomentz_pcf; }
   Coefficient* GetFirstMomentXCoefficient() { return pFirstMomentx_pcf; }
   VectorCoefficient* GetFirstMomentCoefficient() { return pMoments_pcf; }

   ParGridFunction* GetIntensityGridFunction()
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
	  }
	  return pI_pgf;
   }

   virtual void ModelAlgebraicTranslation(Coefficient *Cv, 
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
   /// Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &T, Vector &dT_dt) const {}
   void Compute(const double dt, const double tol, double &Umax, double &dUmax, 
      int &nti, ParGridFunction *u, ParGridFunction *T=NULL)
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

   virtual ~ParNonlocalOperator() 
   {
      delete Amesh;
	  delete Afec_phi;
	  delete Afec_theta;
	  delete Afes;
	  delete Xfec;
	  delete pI_fes;
	  delete pI_pgf;
	  delete pIK;
	  delete pIG;
	  delete pIb;
	  delete pIS;
	  delete pTNGrad;
	  delete pTM;
	  delete pTb;
	  delete pMoments_pcf;
      // TMP components.
      delete pZeroMoment_pcf; 
	  delete pFirstMomentz_pcf;
	  delete pFirstMomentx_pcf;  
   }
};

} // namespace nth

} // namespace mfem

#endif // MFEM_USE_MPI

#endif
