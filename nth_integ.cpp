#include "nth_integ.hpp"

namespace mfem
{

namespace nth
{

double sine(const Vector &x_coords)
{
   double x = x_coords(0);
   return sin(x);
}

double cosine(const Vector &x_coords)
{
   double x = x_coords(0);
   return cos(x);
}

FunctionCoefficient sin(sine);
FunctionCoefficient cos(cosine);

void OuterMult(const Vector &a, const Vector &b, Vector &c)
{
   int sb = b.Size();
   int sa = a.Size();
   //c.SetSize(sa*sb);
   for (int i = 0; i < sb; i++)
      for (int j = 0; j < sa; j++)
	  {
	     c(i*sa+j) = b(i)*a(j);
	  }      	
}

void ParInvLocBilinearForm::LocalInverse()
{
   DenseMatrix elmat;

   for (int i = 0; i < pfes -> GetNE(); i++)
   {
      // For each element get its vdofs array.
	  pfes->GetElementVDofs(i, vdofs);
      elmat.SetSize(vdofs.Size());
      elmat = 0.;
	  int skip_zeros = 1;
	  // And get the corresponding element local dofs matrix.
	  mat->GetSubMatrix(vdofs, vdofs, elmat);
      // Invert the local element matrix.
	  elmat.Invert();
	  // Set inverted local matrix instead of the original local matrix.
      mat->SetSubMatrix(vdofs, vdofs, elmat, skip_zeros);
   }	   
}

void InvMassLocalIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   AssembleElementMatrix2(el, el, Trans, elmat);
}

void InvMassLocalIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   // int dim = trial_fe.GetDim();
   double w;

#ifdef MFEM_THREAD_SAFE
   Vector shape, te_shape;
#endif
   elmat.SetSize(te_nd, tr_nd);
   shape.SetSize(tr_nd);
   te_shape.SetSize(te_nd);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();

      ir = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      trial_fe.CalcShape(ip, shape);
      test_fe.CalcShape(ip, te_shape);

      Trans.SetIntPoint (&ip);
      w = Trans.Weight() * ip.weight;
      if (Q)
      {
         w *= Q -> Eval(Trans, ip);
      }

      te_shape *= w;
      AddMultVWt(te_shape, shape, elmat);
   }
   // Finally, return an inverted local matrix
   elmat.Invert();
}


AngularFiniteElementSpace::AngularFiniteElementSpace(Mesh *mesh_phi, 
   const FiniteElementCollection *fec_phi, Mesh *mesh_theta, 
   const FiniteElementCollection *fec_theta) : 
   //FiniteElementSpace(mesh_phi, fec_phi), 
   fes_phi(mesh_phi, fec_phi), fes_theta(mesh_theta, fec_theta),
   fe_phi(NULL), fe_theta(NULL), Trans_phi(NULL), Trans_theta(NULL),
   ir_phi(NULL), ir_theta(NULL), ir_phi_unique(NULL), ir_theta_unique(NULL)
{
   cout << "AS constructor..." << endl << flush;

   // Entire angular finite element space has ndof_phi*ndof_theta
   afes_ndofs = fes_phi.GetNDofs()*fes_theta.GetNDofs();

   // Because of implicit calculation, we use just one element mesh for both
   // phi and theta dimensions.
   fe_phi = fes_phi.GetFE(0); 
   fe_theta = fes_theta.GetFE(0);
   Trans_phi = fes_phi.GetElementTransformation(0);
   Trans_theta = fes_theta.GetElementTransformation(0);
   // Integration rules for phi and theta dimension.
   int multiple = 2;
   if (ir_phi == NULL)
   {
      int order = multiple*fe_phi->GetOrder(); // + Trans->OrderW();
      //cout << "order: " << order << endl << flush;
      ir_phi = &IntRules.Get(fe_phi->GetGeomType(), order);
      for (int iphi = 0; iphi < ir_phi->GetNPoints(); iphi++)
      {
         const IntegrationPoint &ip_phi = ir_phi->IntPoint(iphi);
	  	 cout << "ip_phi w, x: " << ip_phi.x << ", " << 
	  	 ip_phi.weight << endl << flush;
	  }
   }
   if (ir_theta == NULL)
   {
      int order = multiple*fe_theta->GetOrder();
      ir_theta = &IntRules.Get(fe_theta->GetGeomType(), order);
      for (int itheta = 0; itheta < ir_theta->GetNPoints(); itheta++)
      {
         const IntegrationPoint &ip_theta = ir_theta->IntPoint(itheta);
	  	 cout << "ip_theta w, x: " << ip_theta.x << ", " << 
	  	 ip_theta.weight << endl << flush;
	  }
   }
}

void AngularFiniteElementSpace::GetCartesianDirection(
   const IntegrationPoint &ip_phi, const IntegrationPoint &ip_theta, Vector &n)
{ 
   // Angular projections
   double cosphi = cos.Eval(*Trans_phi, ip_phi);
   double sinphi = sin.Eval(*Trans_phi, ip_phi); 
   double costheta = cos.Eval(*Trans_theta, ip_theta);
   double sintheta = sin.Eval(*Trans_theta, ip_theta);
   // Set Cartesian coordinates of the direction vector n.
   // the order of axis is z, x, y
   n(0) = cosphi;
   if (n.Size()>1)
   {
      n(1) = costheta*sinphi;
   }
   if (n.Size()>2)
   {
      n(2) = sintheta*sinphi;
   }
}

void AngularFiniteElementSpace::CalcShape(const IntegrationPoint &ip_phi, 
   const IntegrationPoint &ip_theta, Vector &shape, double &w)
{ 
   int ndof_phi = fe_phi->GetDof();
   int ndof_theta = fe_theta->GetDof();
   Vector shape_phi, shape_theta;
   shape_phi.SetSize(ndof_phi);
   shape_theta.SetSize(ndof_theta); 
   // Calculate basis functions for given quadrature point.       
   fe_phi->CalcShape(ip_phi, shape_phi);
   fe_theta->CalcShape(ip_theta, shape_theta);
   // Eval the outer tensor product.
   OuterMult(shape_phi, shape_theta, shape);
   // Calculate weight of the quadrature including |J| of spherical coordinates.
   Trans_phi->SetIntPoint (&ip_phi);
   Trans_theta->SetIntPoint (&ip_theta);
   w = Trans_phi->Weight() * ip_phi.weight; 
   w *= Trans_theta->Weight() * ip_theta.weight; 
   // if symmetric f(theta) = f(-theta)
   w *= 2.;
   // Spherical integration factor sin(phi)
   w *= sin.Eval(*Trans_phi, ip_phi);  
}

void AngularFiniteElementSpace::CalcBasisZeroMoment(double &Omega, Vector &I0)
{
   double w;
   Vector shape;
   shape.SetSize(afes_ndofs);

   Omega = 0.;
   I0 = 0.;
   for (int itheta = 0; itheta < ir_theta->GetNPoints(); itheta++)
   {
      const IntegrationPoint &ip_theta = ir_theta->IntPoint(itheta);
	  for (int iphi = 0; iphi < ir_phi->GetNPoints(); iphi++)
      {
         const IntegrationPoint &ip_phi = ir_phi->IntPoint(iphi);
		 // Calculate angular finite element space basis shape and weight.
		 CalcShape(ip_phi, ip_theta, shape, w);
		 Omega += w;
		 add(I0, w, shape, I0);
	  }
   }
}

void AngularFiniteElementSpace::CalcBasisFirstMoment(double &Omega, 
   Vector &I1z, Vector &I1x, Vector &I1y)
{
   double w;
   int dim = 3;
   Vector shape, n_trans;
   shape.SetSize(afes_ndofs);
   n_trans.SetSize(dim);

   Omega = 0.;
   I1z = 0.;
   I1x = 0.;
   I1y = 0.;
   for (int itheta = 0; itheta < ir_theta->GetNPoints(); itheta++)
   {
      const IntegrationPoint &ip_theta = ir_theta->IntPoint(itheta);
	  for (int iphi = 0; iphi < ir_phi->GetNPoints(); iphi++)
      {
         const IntegrationPoint &ip_phi = ir_phi->IntPoint(iphi);
		 // Calculate Cartesian coordinates of transport direction vector.
		 GetCartesianDirection(ip_phi, ip_theta, n_trans);
		 // Calculate angular finite element space basis shape and weight.
		 CalcShape(ip_phi, ip_theta, shape, w);
		 Omega += w;
		 add(I1z, w*n_trans(0), shape, I1z);
         add(I1x, w*n_trans(1), shape, I1x);
         add(I1y, w*n_trans(2), shape, I1y);
	  }
   }
}

void AngularFiniteElementSpace::SetUniqueIntPointPhi(double x1)
{
   int order = 0;
   if (ir_phi_unique==NULL)
   {	   
      ir_phi_unique = new IntegrationRule();
   }
   ir_phi_unique->SetSize(1);
   double w = 1.;
   IntegrationPoint &ip = ir_phi_unique->IntPoint(0);
   ip.Set1w(x1, w);
   // Set the int rule to be used in global calculations.
   ir_phi = ir_phi_unique;
}

void AngularFiniteElementSpace::SetUniqueIntPointTheta(double x1)
{
   int order = 0;
   if (ir_theta_unique==NULL)
   {
      ir_theta_unique = new IntegrationRule();
   }
   ir_theta_unique->SetSize(1);
   double w = 1.;
   IntegrationPoint &ip = ir_theta_unique->IntPoint(0);
   ip.Set1w(x1, w);
   // Set the int rule to be used in global calculations.
   ir_theta = ir_theta_unique;
}

void KineticCollisionIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   AssembleElementMatrix2(el, el, Trans, elmat);
}

void KineticCollisionIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int dim = trial_fe.GetDim();
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   int afes_nd = Afes->GetAfesNDofs();
   int full_tr_nd = afes_nd*tr_nd;
   int full_te_nd = afes_nd*te_nd;;
   double w_x;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix tr_dshape, adjJ;
   Vector tr_shape, te_shape, afes_shape, afes_meanshape, n_trans, Jn_trans,
      tr_nidFidxT;
   Vector full_tr_shape, full_tr_meanshape, full_tr_nidFidxT, full_te_shape;
#endif
   afes_shape.SetSize(afes_nd);
   afes_meanshape.SetSize(afes_nd);
   tr_shape.SetSize(tr_nd);
   n_trans.SetSize(dim);
   Jn_trans.SetSize(dim);
   adjJ.SetSize(dim);
   tr_dshape.SetSize(tr_nd, dim);
   tr_nidFidxT.SetSize(tr_nd); 
   full_tr_shape.SetSize(full_tr_nd);
   full_tr_meanshape.SetSize(full_tr_nd);
   full_tr_nidFidxT.SetSize(full_tr_nd);
   te_shape.SetSize(te_nd);
   full_te_shape.SetSize(full_te_nd);
 
   elmat.SetSize(full_te_nd, full_tr_nd);

   const IntegrationRule *irx = IntRule;
   if (irx == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
      irx = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;

   const IntegrationRule *ir_phi = Afes->GetIntegrationRulePhi();
   const IntegrationRule *ir_theta = Afes->GetIntegrationRuleTheta();
   ElementTransformation *Trans_phi = Afes->GetTransPhi();
   ElementTransformation *Trans_theta = Afes->GetTransTheta();
   double w_afes;
   afes_meanshape = 0.;
   if (Qmom_scalar)
   {
      double Int_Omega;
	  Afes->CalcBasisZeroMoment(Int_Omega, afes_meanshape);
	  // Get the angular mean value.
	  afes_meanshape *= 1./Int_Omega;
   }
   // Fill the algebraic linear and bilinear forms.
   for (int ix = 0; ix < irx->GetNPoints(); ix++)
   {
      const IntegrationPoint &ip = irx->IntPoint(ix);
      // Calculate spatial finite element space trial basis shape.
	  trial_fe.CalcShape(ip, tr_shape);
      // Calculate spatial finite element space trial basis grad shape.
	  trial_fe.CalcDShape(ip, tr_dshape);
      // Calculate spatial finite element space test basis shape.
	  test_fe.CalcShape(ip, te_shape);
      // Apply the n*grad operator.
      Trans.SetIntPoint (&ip);
      // Pure spatial gradient calculation.
	  CalcAdjugate(Trans.Jacobian(), adjJ);
	  w_x = ip.weight;
      double qen = 1.;
	  double qmom = 0.;
	  double qvec = 0.;
	  // Apply scalar coefficients.
	  if (Qen_scalar)
      {
         qen = Qen_scalar -> Eval(Trans, ip); 
      }
	  if (Qmom_scalar)
      {
         qmom = Qmom_scalar -> Eval(Trans, ip);
      }
	  Vector qvector;
	  if (Q_vector)
      { 
		 Q_vector -> Eval(qvector, Trans, ip);
      }
	  // Impose the transformation weight (already included in nidFidxT)
	  qen *= Trans.Weight();
      qmom *= Trans.Weight();
	  qvector *= Trans.Weight();
      // Angular contribution. 
      for (int itheta = 0; itheta < ir_theta->GetNPoints(); itheta++)
      {
         const IntegrationPoint &ip_theta = ir_theta->IntPoint(itheta);
	     for (int iphi = 0; iphi < ir_phi->GetNPoints(); iphi++)
         {
            const IntegrationPoint &ip_phi = ir_phi->IntPoint(iphi);
		    // Calculate Cartesian coordinates of transport direction vector.
		    Afes->GetCartesianDirection(ip_phi, ip_theta, n_trans);
		    // Calculate angular finite element space basis shape and weight.
		    Afes->CalcShape(ip_phi, ip_theta, afes_shape, w_afes);
            // Transport direction vector projection. Contains the |J|.
			adjJ.Mult(n_trans, Jn_trans);
            tr_dshape.Mult(Jn_trans, tr_nidFidxT);
            // Get the full trial and basis shape.
			OuterMult(tr_shape, afes_shape, full_tr_shape);
			OuterMult(tr_shape, afes_meanshape, full_tr_meanshape);
			OuterMult(tr_nidFidxT, afes_shape, full_tr_nidFidxT);
			// Get the full test basis shape.
			OuterMult(te_shape, afes_shape, full_te_shape);
            // Evaluate and apply the vector coefficient. 
			if (Q_vector)
            {
			   qvec = n_trans*qvector;
            }
			// Fill the algebraic mass bilinear forms.
            for (int i = 0; i < full_te_nd; i++)
               for (int j = 0; j < full_tr_nd; j++)
               {
                  elmat(i, j) += w_afes * w_x * full_te_shape(i) 
				     * (full_tr_nidFidxT(j) - qmom*full_tr_meanshape(j) 
					 + (qen + qmom + qvec)*full_tr_shape(j));
               }
            //te_shape *= w;
            //AddMultVWt(te_shape, shape, elmat);
         }
      }
   }
   // Provide sign of the operator.
   elmat *= c;
}

void KineticGammaImIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim, afes_nd, ndof1, ndof2, full_ndof1, full_ndof2;

   double n_trans_dot_n_gamma, a, b, w_x, w_afes;

   dim = el1.GetDim();
   afes_nd = Afes->GetAfesNDofs();
   ndof1 = el1.GetDof();
   Vector n_trans(dim), n_gamma(dim);

   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
   }
   else
   {
      ndof2 = 0;
   }

   full_ndof1 = afes_nd*ndof1; 
   full_ndof2 = afes_nd*ndof2;

#ifdef MFEM_THREAD_SAFE
   Vector afes_shape, shape1, shape2, full_shape1, full_shape2;
#endif
   afes_shape.SetSize(afes_nd);
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);
   full_shape1.SetSize(full_ndof1);
   full_shape2.SetSize(full_ndof2);
   elmat.SetSize(full_ndof1 + full_ndof2);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   const IntegrationRule *ir_phi = Afes->GetIntegrationRulePhi();
   const IntegrationRule *ir_theta = Afes->GetIntegrationRuleTheta();
   ElementTransformation *Trans_phi = Afes->GetTransPhi();
   ElementTransformation *Trans_theta = Afes->GetTransTheta();
   // Fill the algebraic linear and bilinear forms.
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
         el2.CalcShape(eip2, shape2);
      }
      el1.CalcShape(eip1, shape1);

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip1);

      if (dim == 1)
      {
         n_gamma(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), n_gamma);
      }
      // Angular contribution.
	  for (int itheta = 0; itheta < ir_theta->GetNPoints(); itheta++)
      {
         const IntegrationPoint &ip_theta = ir_theta->IntPoint(itheta);
	     for (int iphi = 0; iphi < ir_phi->GetNPoints(); iphi++)
         {
            const IntegrationPoint &ip_phi = ir_phi->IntPoint(iphi);
            // Calculate Cartesian coordinates of transport direction vector.
		    Afes->GetCartesianDirection(ip_phi, ip_theta, n_trans);
		    // Calculate angular finite element space basis shape and weight.
		    Afes->CalcShape(ip_phi, ip_theta, afes_shape, w_afes);
            // Get the full element1 basis shape.	
            OuterMult(shape1, afes_shape, full_shape1);
			// Calculate transport to face projection.
            n_trans_dot_n_gamma = n_trans * n_gamma;
            a = 0.5 * n_trans_dot_n_gamma;
            b = -0.5 * fabs(n_trans_dot_n_gamma);
	        //a = 0.5 * alpha * un;
            //b = beta * fabs(un);
            // note: if |alpha/2|==|beta| then |a|==|b|, 
			//       i.e. (a==b) or (a==-b)and therefore two blocks in 
			//       the element matrix contribution
            //       (from the current quadrature point) are 0

            w_x = ip.weight * (a+b); // ndotnGamma1 < 0
            if (w_x != 0.0)
            {
               for (int i = 0; i < full_ndof1; i++)
		          for (int j = 0; j < full_ndof1; j++) 
                  {
                     elmat(i, j) += w_x * w_afes * full_shape1(j) 
					    * full_shape1(i);
                  }
            }

            if (ndof2)
            {
			   // Get the full element2 basis shape.	
               OuterMult(shape2, afes_shape, full_shape2);
               
			   w_x = ip.weight * (b-a); // ndotnGamma2 < 0 (ndotnGamma1 > 0) 
               if (w_x != 0.0)
               {
                  for (int i = 0; i < full_ndof2; i++)
			         for (int j = 0; j < full_ndof2; j++) 
                     {
                        elmat(full_ndof1+i, full_ndof1+j) += w_x * w_afes * 
				           full_shape2(j) * full_shape2(i);
                     }
               }
            }
         }
      }
   }
   // Provide sign of the operator.
   elmat *= c;
}

void KineticGammaExIntegrator::AssembleFaceMatrix(
   const FiniteElement &el1, const FiniteElement &el2,
   FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim, afes_nd, ndof1, ndof2, full_ndof1, full_ndof2;

   double n_trans_dot_n_gamma, a, b, w_x, w_afes;

   dim = el1.GetDim();
   afes_nd = Afes->GetAfesNDofs();
   ndof1 = el1.GetDof();
   Vector n_trans(dim), n_gamma(dim);

   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
   }
   else
   {
      ndof2 = 0;
   }

   full_ndof1 = afes_nd*ndof1; 
   full_ndof2 = afes_nd*ndof2;

#ifdef MFEM_THREAD_SAFE
   Vector afes_shape, shape1, shape2, full_shape1, full_shape2;
#endif
   afes_shape.SetSize(afes_nd);
   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);
   full_shape1.SetSize(full_ndof1);
   full_shape2.SetSize(full_ndof2);
   elmat.SetSize(full_ndof1 + full_ndof2);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   const IntegrationRule *ir_phi = Afes->GetIntegrationRulePhi();
   const IntegrationRule *ir_theta = Afes->GetIntegrationRuleTheta();
   ElementTransformation *Trans_phi = Afes->GetTransPhi();
   ElementTransformation *Trans_theta = Afes->GetTransTheta();
   // Fill the algebraic linear and bilinear forms.
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
         el2.CalcShape(eip2, shape2);
      }
      el1.CalcShape(eip1, shape1);

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip1);

      if (dim == 1)
      {
         n_gamma(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), n_gamma);
      }
      // Angular contribution.
	  for (int itheta = 0; itheta < ir_theta->GetNPoints(); itheta++)
      {
         const IntegrationPoint &ip_theta = ir_theta->IntPoint(itheta);
	     for (int iphi = 0; iphi < ir_phi->GetNPoints(); iphi++)
         {
            const IntegrationPoint &ip_phi = ir_phi->IntPoint(iphi);
            // Calculate Cartesian coordinates of transport direction vector.
		    Afes->GetCartesianDirection(ip_phi, ip_theta, n_trans);
		    // Calculate angular finite element space basis shape and weight.
		    Afes->CalcShape(ip_phi, ip_theta, afes_shape, w_afes);
			// Get the full element1 basis shape.	
            OuterMult(shape1, afes_shape, full_shape1);
            // Calculate transport to face projection.
			n_trans_dot_n_gamma = n_trans * n_gamma;
            a = 0.5 * n_trans_dot_n_gamma;
            b = -0.5 * fabs(n_trans_dot_n_gamma);
	        //a = 0.5 * alpha * un;
            //b = beta * fabs(un);
            // note: if |alpha/2|==|beta| then |a|==|b|, 
			//       i.e. (a==b) or (a==-b)and therefore two blocks in 
			//       the element matrix contribution
            //       (from the current quadrature point) are 0

            w_x = ip.weight * (a+b); // ndotnGamma2 > 0 (ndotnGamma1 < 0)
            if (ndof2)
            {
               //el2.CalcShape(eip2, shape2);
		       // Get the full element2 basis shape.	
               OuterMult(shape2, afes_shape, full_shape2);
         
		       if (w_x != 0.0)
                  for (int i = 0; i < full_ndof1; i++)
			         for (int j = 0; j < full_ndof2; j++) 
                     {
                        elmat(i, full_ndof1+j) -= w_x * w_afes * 
				           full_shape2(j) * full_shape1(i);
                     }

               w_x = ip.weight * (b-a); // ndotnGamma1 > 0
               if (w_x != 0.0)
               {
                  for (int i = 0; i < full_ndof2; i++)
			         for (int j = 0; j < full_ndof1; j++) 
                     {
                        elmat(full_ndof1+i, j) -= w_x * w_afes * full_shape1(j) 
				           * full_shape2(i);
                     }
               }
            }
         }
      }
   }
   // Provide sign of the operator.
   elmat *= c;
}

void KineticDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Trans, Vector &elvect)
{
   int afes_nd = Afes->GetAfesNDofs();
   int ndof = el.GetDof();
   int full_ndof = afes_nd*ndof;
   double w_x, w_afes;

   Vector afes_shape;
   afes_shape.SetSize(afes_nd);
   te_shape.SetSize(ndof);       // vector of size ndof
   full_te_shape.SetSize(full_ndof);
   elvect.SetSize(full_ndof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // ir = &IntRules.Get(el.GetGeomType(),
      //                    oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }

   const IntegrationRule *ir_phi = Afes->GetIntegrationRulePhi();
   const IntegrationRule *ir_theta = Afes->GetIntegrationRuleTheta();
   ElementTransformation *Trans_phi = Afes->GetTransPhi();
   ElementTransformation *Trans_theta = Afes->GetTransTheta(); 
   // Fill the algebraic linear form.
   for (int ix = 0; ix < ir->GetNPoints(); ix++)
   {
      const IntegrationPoint &ip = ir->IntPoint(ix);

      Trans.SetIntPoint (&ip);
      w_x = Trans.Weight() * ip.weight;
      if (Q)
      {
         w_x *= Q -> Eval(Trans, ip);
      }
      el.CalcShape(ip, te_shape);
      // Angular contribution.
      for (int itheta = 0; itheta < ir_theta->GetNPoints(); itheta++)
      {
         const IntegrationPoint &ip_theta = ir_theta->IntPoint(itheta);
	     for (int iphi = 0; iphi < ir_phi->GetNPoints(); iphi++)
         {
            const IntegrationPoint &ip_phi = ir_phi->IntPoint(iphi);
		    // Calculate angular finite element space basis shape and weight.
		    Afes->CalcShape(ip_phi, ip_theta, afes_shape, w_afes);
			// Get the full element1 basis shape.	
            OuterMult(te_shape, afes_shape, full_te_shape);
            
			for (int i = 0; i < full_ndof; i++)
            {
                  elvect(i) += w_x * w_afes * full_te_shape(i);
            }

            //add(elvect, w, shape, elvect);  
            //add(elvect, ip.weight * val, shape, elvect);
         }
      }
   }
}

void KineticSourceIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   AssembleElementMatrix2(el, el, Trans, elmat);
}

void KineticSourceIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   int afes_nd = Afes->GetAfesNDofs();
   int full_te_nd = afes_nd*te_nd;;
   double w_x;

#ifdef MFEM_THREAD_SAFE
   Vector tr_shape, te_shape, afes_shape, full_te_shape;
#endif
   tr_shape.SetSize(tr_nd);
   te_shape.SetSize(te_nd);
   afes_shape.SetSize(afes_nd); 
   full_te_shape.SetSize(full_te_nd);
   // Element matrix is based on full test basis but only spatial trial basis.
   elmat.SetSize(full_te_nd, tr_nd);

   const IntegrationRule *irx = IntRule;
   if (irx == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
      irx = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;

   const IntegrationRule *ir_phi = Afes->GetIntegrationRulePhi();
   const IntegrationRule *ir_theta = Afes->GetIntegrationRuleTheta();
   ElementTransformation *Trans_phi = Afes->GetTransPhi();
   ElementTransformation *Trans_theta = Afes->GetTransTheta();
   double w_afes;
   for (int itheta = 0; itheta < ir_theta->GetNPoints(); itheta++)
   {
      const IntegrationPoint &ip_theta = ir_theta->IntPoint(itheta);
	  for (int iphi = 0; iphi < ir_phi->GetNPoints(); iphi++)
      {
         const IntegrationPoint &ip_phi = ir_phi->IntPoint(iphi);
		 // Calculate angular finite element space basis shape and weight.
		 Afes->CalcShape(ip_phi, ip_theta, afes_shape, w_afes);
         // Fill the algebraic linear and bilinear forms.
		 for (int ix = 0; ix < irx->GetNPoints(); ix++)
         {
            const IntegrationPoint &ip = irx->IntPoint(ix);
            // Calculate spatial finite element space trial basis shape.
			trial_fe.CalcShape(ip, tr_shape);
            // Calculate spatial finite element space test basis shape.
			test_fe.CalcShape(ip, te_shape);
			// Get the full test basis shape.
			OuterMult(te_shape, afes_shape, full_te_shape);

            Trans.SetIntPoint (&ip);
            w_x = Trans.Weight() * ip.weight;
			if (Qscalar)
            {
               w_x *= Qscalar -> Eval(Trans, ip);
            }
			// Fill the algebraic mass bilinear forms.
            for (int i = 0; i < full_te_nd; i++)
               for (int j = 0; j < tr_nd; j++)
               {
                  elmat(i, j) += w_afes * w_x * 
				     full_te_shape(i) * tr_shape(j);
               }
            //te_shape *= w;
            //AddMultVWt(te_shape, shape, elmat);
         }
      }
   }
   // Provide sign of the operator.
   elmat *= c;
}

void KineticNGradIntegrator::AssembleElementMatrix
( const FiniteElement &el, ElementTransformation &Trans,
  DenseMatrix &elmat )
{
   AssembleElementMatrix2(el, el, Trans, elmat);
}

void KineticNGradIntegrator::AssembleElementMatrix2(
   const FiniteElement &trial_fe, const FiniteElement &test_fe,
   ElementTransformation &Trans, DenseMatrix &elmat)
{
   int dim = trial_fe.GetDim();
   int tr_nd = trial_fe.GetDof();
   int te_nd = test_fe.GetDof();
   int afes_nd = Afes->GetAfesNDofs();
   int full_tr_nd = afes_nd*tr_nd;
   double w_x, w_afes;

#ifdef MFEM_THREAD_SAFE
   DenseMatrix tr_dshape, adjJ;
   Vector te_shape, afes_shape, n_trans, Jn_trans, tr_nidFidxT;
   Vector full_tr_nidFidxT;
#endif
   adjJ.SetSize(dim);
   // Transport direction vector.
   n_trans.SetSize(dim);
   Jn_trans.SetSize(dim);
   afes_shape.SetSize(afes_nd);
   te_shape.SetSize(te_nd);
   tr_dshape.SetSize(tr_nd, dim);
   tr_nidFidxT.SetSize(tr_nd);   
   full_tr_nidFidxT.SetSize(full_tr_nd); 
   elmat.SetSize(te_nd, full_tr_nd);

   const IntegrationRule *irx = IntRule;
   if (irx == NULL)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
      irx = &IntRules.Get(trial_fe.GetGeomType(), order);
   }

   elmat = 0.0;

   const IntegrationRule *ir_phi = Afes->GetIntegrationRulePhi();
   const IntegrationRule *ir_theta = Afes->GetIntegrationRuleTheta();
   ElementTransformation *Trans_phi = Afes->GetTransPhi();
   ElementTransformation *Trans_theta = Afes->GetTransTheta();
   // Fill the algebraic linear and bilinear forms.
   for (int ix = 0; ix < irx->GetNPoints(); ix++)
   {
      const IntegrationPoint &ip = irx->IntPoint(ix);
      // Calculate spatial finite element space trial basis grad shape.
	  trial_fe.CalcDShape(ip, tr_dshape);
      // Calculate spatial finite element space test basis shape.
	  test_fe.CalcShape(ip, te_shape);
      // Apply the n*grad operator.
      Trans.SetIntPoint (&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      w_x = ip.weight; // The trans weight is contained in nidFidxT
      // Angular contribution.
	  for (int itheta = 0; itheta < ir_theta->GetNPoints(); itheta++)
      {
         const IntegrationPoint &ip_theta = ir_theta->IntPoint(itheta);
	     for (int iphi = 0; iphi < ir_phi->GetNPoints(); iphi++)
         {
            const IntegrationPoint &ip_phi = ir_phi->IntPoint(iphi);
            // Calculate Cartesian coordinates of transport direction vector.
		    Afes->GetCartesianDirection(ip_phi, ip_theta, n_trans);
		    // Calculate angular finite element space basis shape and weight.
		    Afes->CalcShape(ip_phi, ip_theta, afes_shape, w_afes); 
            // Transport direction vector projection. Contains the |J|.
			adjJ.Mult(n_trans, Jn_trans);
            tr_dshape.Mult(Jn_trans, tr_nidFidxT);
            // Get the full trial basis shape.	
			OuterMult(tr_nidFidxT, afes_shape, full_tr_nidFidxT);
            // Fill the algebraic mass bilinear forms.
            for (int i = 0; i < te_nd; i++)
               for (int j = 0; j < full_tr_nd; j++)
               {
                  elmat(i, j) += w_afes * w_x * 
				     te_shape(i) * full_tr_nidFidxT(j);
               }
            //te_shape *= w;
            //AddMultVWt(te_shape, shape, elmat);
         }
      }
   }
   // Provide sign of the operator.
   elmat *= c;
}

} // namespace nth

} // namespace mfem
