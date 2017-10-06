#ifndef MFEM_NTH_SOLVER
#define MFEM_NTH_SOLVER

#include "mfem.hpp"
#include "nth_integ.hpp"
#include "nth_coeffs.hpp"

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
	  Vector *_Tb); 

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
      int dim, vdim;
	  AngularFiniteElementSpace *Afes;
	  ParGridFunction *pI;
      int component; // TMP switch: 0=I0, 1=I1z, 2=I1x
   public:
	  MomentsCoefficient(int _vdim, AngularFiniteElementSpace *_Afes, 
	     ParGridFunction *_pI) : VectorCoefficient(_vdim), 
		 MatrixCoefficient(_vdim), pI(_pI), Afes(_Afes) 
	  { 
	     dim = pI->FESpace()->GetMesh()->Dimension();
		 vdim = _vdim;
		 component = 0; 
	  } 
      void SetComponent(const int _component) { component = _component; }
	  virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
      virtual void Eval(Vector &flux, ElementTransformation &T,
	     const IntegrationPoint &ip); 
      virtual void Eval(DenseMatrix &pressure, ElementTransformation &T,
	     const IntegrationPoint &ip) {}
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
   MomentsCoefficient *pZeroMoment_pcf, *pFirstMomentz_pcf, *pFirstMomentx_pcf,
      *pFirstMomenty_pcf, *pFirstMomentMagnitude_pcf;

public:
   ParNonlocalOperator(ParMesh *_pmesh, ParFiniteElementSpace *_pT_fes, 
      int _order_I_x=3, int _order_T_x=2, int _order_I_phi=1, 
      int _order_I_theta=0);

   DG_FECollection* GetXfec() { return Xfec; }
   Coefficient* GetZeroMomentCoefficient() { return pMoments_pcf; }
   // TMP coefficients.
   Coefficient* GetFirstMomentZCoefficient() { return pFirstMomentz_pcf; }
   Coefficient* GetFirstMomentXCoefficient() { return pFirstMomentx_pcf; }
   Coefficient* GetFirstMomentYCoefficient() { return pFirstMomenty_pcf; }
   Coefficient* GetFirstMomentMagnitudeCoefficient() 
      { return pFirstMomentMagnitude_pcf; }
   VectorCoefficient* GetFirstMomentCoefficient() { return pMoments_pcf; }
   ParGridFunction* GetIntensityGridFunction();
   virtual void ModelAlgebraicTranslation(Coefficient *Cv, 
      Coefficient *kappa, Coefficient *isosigma, Coefficient *sourceb, 
	  Coefficient *sourceCoeffT, Coefficient *sourceT, 
	  VectorCoefficient *Efield);
   /// Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &T, Vector &dT_dt) const {}
   void Compute(const double dt, const double tol, double &Umax, double &dUmax, 
      int &nti, ParGridFunction *u, ParGridFunction *T=NULL);

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
      delete pZeroMoment_pcf;
	  delete pFirstMomentz_pcf;
	  delete pFirstMomentx_pcf;
	  delete pFirstMomenty_pcf;
      delete pFirstMomentMagnitude_pcf;
   }
};

} // namespace nth

} // namespace mfem

#endif // MFEM_USE_MPI

#endif
