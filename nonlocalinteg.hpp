#ifndef MFEM_NONLOCALINTEG
#define MFEM_NONLOCALINTEG

#include "mfem.hpp"

using namespace std;

namespace mfem
{

namespace nth
{

// General bilinear form to be placed to mfem directly.
/// Class for parallel bilinear form with local inversion.
class ParInvLocBilinearForm : public ParBilinearForm
{
public:
   ParInvLocBilinearForm(ParFiniteElementSpace *pf) : ParBilinearForm(pf) { } 	
   
   void LocalInverse();
   
   ~ParInvLocBilinearForm() { }
};

/** Class for local element inverted mass matrix assembling a(u,v) := (Q u, v) */
class InvMassLocalIntegrator: public BilinearFormIntegrator
{
protected:
#ifndef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   Coefficient *Q;

public:
   InvMassLocalIntegrator(const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir) { Q = NULL; }
   /// Construct a mass integrator with coefficient q
   InvMassLocalIntegrator(Coefficient &q, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), Q(&q) { }
   InvMassLocalIntegrator(Coefficient *q, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), Q(q) { }

   /** Given a particular Finite Element
       computes the element mass matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};


// General finite element space to be placed to mfem directly.
/// Class for angular finite element space.
class AngularFiniteElementSpace
{
private:
   FiniteElementSpace fes_phi, fes_theta;
   const FiniteElement *fe_phi, *fe_theta;
   ElementTransformation *Trans_phi, *Trans_theta;
   const IntegrationRule *ir_phi, *ir_theta;
   IntegrationRule *ir_phi_unique, *ir_theta_unique;

   int afes_ndofs;

public:
   AngularFiniteElementSpace(Mesh *mesh_phi,  
      const FiniteElementCollection *fec_phi, Mesh *mesh_theta, 
	  const FiniteElementCollection *fec_theta); 

   int GetAfesNDofs() { return afes_ndofs; }
   const IntegrationRule * GetIntegrationRulePhi() { return ir_phi; }
   const IntegrationRule * GetIntegrationRuleTheta() { return ir_theta; }
   ElementTransformation * GetTransPhi() { return Trans_phi; }
   ElementTransformation * GetTransTheta() { return Trans_theta; }
   virtual void CalcShape(const IntegrationPoint &ip_phi, 
      const IntegrationPoint &ip_theta, Vector &shape, double &w);
   void CalcBasisZeroMoment(double &Omega, Vector &I0);
   void CalcBasisFirstMoment(double &Omega, Vector &I1z, Vector &I1x, 
      Vector &I1y);
   void SetUniqueIntPointPhi(double x1);
   void SetUniqueIntPointTheta(double x1);
   void GetCartesianDirection(const IntegrationPoint &ip_phi, 
      const IntegrationPoint &ip_theta, Vector &n);

   ~AngularFiniteElementSpace() 
      { 
	     if (ir_phi_unique!=NULL) delete ir_phi_unique;
         if (ir_theta_unique!=NULL) delete ir_theta_unique;
	  }
};

// General linear form integrators to be placed to mfem directly.

/** Class for local mass-collision matrix assembling 
 * a(u,v) := (Qmom mean(u) - (Qen + Qmom) u, v) */
class KineticCollisionIntegrator: public BilinearFormIntegrator
{
protected:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix tr_dshape, adjJ;
   Vector tr_shape, te_shape, afes_shape, afes_meanshape, n_trans, Jn_trans,
      tr_nidFidxT;
   Vector full_tr_shape, full_tr_meanshape, full_tr_nidFidxT, full_te_shape;
#endif
   AngularFiniteElementSpace *Afes;
   Coefficient *Qen_scalar, *Qmom_scalar;
   VectorCoefficient *Q_vector;
   double c;

public:
   /// Construct a mass integrator with coefficient q
   KineticCollisionIntegrator(AngularFiniteElementSpace *afes, double _c=1., 
      Coefficient *qen_scalar=NULL, Coefficient *qmom_scalar=NULL,
	  VectorCoefficient *q_vector=NULL, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), Afes(afes), Qen_scalar(qen_scalar), 
	  Qmom_scalar(qmom_scalar), Q_vector(q_vector) { c = _c; }

   /** Given a particular Finite Element
       computes the element mass matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/** Integrator for the DG form:
    alpha < (n.n_Gamma) {v}, w > + beta < |n.n_Gamma| [v], w >,
    where v and w are the trial and test variables, respectively, n and n_Gamma
	are directions vectors of the transport and face, respectively. */
class KineticGammaImIntegrator : public BilinearFormIntegrator
{
private:
   AngularFiniteElementSpace *Afes;
   double c;
#ifndef MFEM_THREAD_SAFE
   Vector afes_shape, shape1, shape2, full_shape1, full_shape2;
#endif

public:
   /// Construct integrator with rho = 1.
   KineticGammaImIntegrator(AngularFiniteElementSpace *afes, double _c=1) 
      : Afes(afes) { c = _c; }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Integrator for the DG form:
    alpha < (n.n_Gamma) {v}, w > + beta < |n.n_Gamma| [v], w >,
    where v and w are the trial and test variables, respectively, n and n_Gamma
	are directions vectors of the transport and face, respectively. */
class KineticGammaExIntegrator : public BilinearFormIntegrator
{
private:
   AngularFiniteElementSpace *Afes;
   double c;
#ifndef MFEM_THREAD_SAFE
   Vector afes_shape, shape1, shape2, full_shape1, full_shape2;
#endif

public:
   /// Construct integrator with rho = 1.
   KineticGammaExIntegrator(AngularFiniteElementSpace *afes, double _c=1) 
      : Afes(afes) { c = _c; }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/// Class for domain integration L(v) := (f, v)
class KineticDomainLFIntegrator : public LinearFormIntegrator
{
   Vector te_shape, full_te_shape;
   AngularFiniteElementSpace *Afes;
   Coefficient *Q;
   int oa, ob;
public:
   /// Constructs a domain integrator with a given Coefficient
   KineticDomainLFIntegrator(AngularFiniteElementSpace *afes, 
      Coefficient *q=NULL, int a = 2, int b = 0)
   // the old default was a = 1, b = 1
   // for simple elliptic problems a = 2, b = -2 is ok
      : Afes(afes), Q(q), oa(a), ob(b) { }

   /// Constructs a domain integrator with a given Coefficient
   KineticDomainLFIntegrator(AngularFiniteElementSpace *afes, Coefficient *q, 
      const IntegrationRule *ir)
      : LinearFormIntegrator(ir), Afes(afes), Q(q), oa(1), ob(1) { }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Trans,
                                       Vector &elvect);
};

/** Class for local mass matrix assembling a(u,v) := (Q u, v) */
class KineticSourceIntegrator: public BilinearFormIntegrator
{
protected:
#ifndef MFEM_THREAD_SAFE
   Vector tr_shape, te_shape, afes_shape, full_te_shape;
#endif
   AngularFiniteElementSpace *Afes;
   Coefficient *Qscalar;
   double c;

public:
   /// Construct a mass integrator with coefficient q
   KineticSourceIntegrator(AngularFiniteElementSpace *afes, double _c=1., 
      Coefficient *qscalar=NULL, const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), Afes(afes), Qscalar(qscalar) 
	  { c = _c; }

   /** Given a particular Finite Element
       computes the element mass matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/// (n . grad u(n), v), where u contains the angular finite element space, 
// but v does not. 
class KineticNGradIntegrator : public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix tr_dshape, adjJ;
   Vector te_shape, afes_shape, n_trans, Jn_trans, tr_nidFidxT;
   Vector full_tr_nidFidxT;
#endif
   AngularFiniteElementSpace *Afes;
   double c;

public:
   KineticNGradIntegrator(AngularFiniteElementSpace *afes, double _c=1.0)
      : Afes(afes) { c = _c; }
   virtual void AssembleElementMatrix(const FiniteElement &fe,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/** Class for local mass matrix assembling a(u,v) := (Q u, v) */
class AdvectionBGKMassIntegrator: public BilinearFormIntegrator
{
protected:
#ifndef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   Coefficient *Q;
   double c;

public:
   /// Construct a mass integrator with coefficient q
   AdvectionBGKMassIntegrator(Coefficient *q=NULL, double _c=1,
      const IntegrationRule *ir = NULL)
      : BilinearFormIntegrator(ir), Q(q) { c = _c; }

   /** Given a particular Finite Element
       computes the element mass matrix elmat. */
   virtual void AssembleElementMatrix(const FiniteElement &el,
                                      ElementTransformation &Trans,
                                      DenseMatrix &elmat);
   virtual void AssembleElementMatrix2(const FiniteElement &trial_fe,
                                       const FiniteElement &test_fe,
                                       ElementTransformation &Trans,
                                       DenseMatrix &elmat);
};

/// alpha (n . grad u, v)
class AdvectionBGKNGradIntegrator : public BilinearFormIntegrator
{
private:
#ifndef MFEM_THREAD_SAFE
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec1, vec2, BdFidxT;
#endif
   VectorCoefficient *n;
   double c;

public:
   AdvectionBGKNGradIntegrator(VectorCoefficient *_n, double _c=1.0)
      : n(_n) { c = _c; }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};

/** Integrator for the DG form:
    alpha < (n.n_Gamma) {v}, w > + beta < |n.n_Gamma| [v], w >,
    where v and w are the trial and test variables, respectively, n and n_Gamma
	are directions vectors of the transport and face, respectively. */
class AdvectionBGKGammaImIntegrator : public BilinearFormIntegrator
{
private:
   VectorCoefficient *n;
   double c;

   Vector shape1, shape2;

public:
   /// Construct integrator with rho = 1.
   AdvectionBGKGammaImIntegrator(VectorCoefficient *_n, double _c=1) : 
      n(_n) { c = _c; }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/** Integrator for the DG form:
    alpha < (n.n_Gamma) {v}, w > + beta < |n.n_Gamma| [v], w >,
    where v and w are the trial and test variables, respectively, n and n_Gamma
	are directions vectors of the transport and face, respectively. */
class AdvectionBGKGammaExIntegrator : public BilinearFormIntegrator
{
private:
   VectorCoefficient *n;
   double c;

   Vector shape1, shape2;

public:
   /// Construct integrator with rho = 1.
   AdvectionBGKGammaExIntegrator(VectorCoefficient *_n, double _c=1) : 
      n(_n) { c = _c; }

   using BilinearFormIntegrator::AssembleFaceMatrix;
   virtual void AssembleFaceMatrix(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Trans,
                                   DenseMatrix &elmat);
};

/// Class for domain integration L(v) := (f, v)
class AdvectionBGKDomainLFIntegrator : public LinearFormIntegrator
{
   Vector shape;
   Coefficient *Q;
   double c;
   int oa, ob;
public:
   /// Constructs a domain integrator with a given Coefficient
   AdvectionBGKDomainLFIntegrator(Coefficient *q=NULL, double _c=1,
      int a = 2, int b = 0)
   // the old default was a = 1, b = 1
   // for simple elliptic problems a = 2, b = -2 is ok
      : Q(q), oa(a), ob(b) { c = _c; }

   /// Constructs a domain integrator with a given Coefficient
   AdvectionBGKDomainLFIntegrator(Coefficient *q, 
      const IntegrationRule *ir)
      : LinearFormIntegrator(ir), Q(q), oa(1), ob(1) { c = 1; }

   /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
};

} // namespace nth

} // namespace mfem


#endif
