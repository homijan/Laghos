//                       MFEM Example 10 - Parallel Version
//
// Compile with: make ex10p
//
// Sample runs:
//   ./ex655 -m ../data/Sod8.mesh -o 9 -dt 0.0005 
//
// Description:  This examples solves a time dependent nonlinear elasticity
//               problem of the form dv/dt = H(x) + S v, dx/dt = v, where H is a
//               hyperelastic model and S is a viscosity operator of Laplacian
//               type. The geometry of the domain is assumed to be as follows:
//
//                                 +---------------------+
//                    boundary --->|                     |
//                    attribute 1  |                     |
//                    (fixed)      +---------------------+
//
//               The example demonstrates the use of nonlinear operators (the
//               class HydrodynamicOperator defining H(x)), as well as their
//               implicit time integration using a Newton method for solving an
//               associated reduced backward-Euler type nonlinear equation
//               (class ReducedSystemOperator). Each Newton step requires the
//               inversion of a Jacobian matrix, which is done through a
//               (preconditioned) inner solver. Note that implementing the
//               method HydrodynamicOperator::ImplicitSolve is the only
//               requirement for high-order implicit (SDIRK) time integration.
//
//               We recommend viewing examples 2 and 9 before viewing this
//               example.

#include "mfem.hpp"
#include <memory>
#include <iostream>
#include <fstream>

#include "pete/hydrointeg.hpp"
#include "pete/eos.hpp"
#include "pete/hydrocoefficients.hpp"
#include "pete/hydrooperator.hpp"
#include "pete/phydrooperator.hpp"
#include "pete/ic.hpp"
#include "pete/nonlocalinteg.hpp"
#include "pete/pnonlocaloperator.hpp"
#include "pete/hydrointeg.cpp"
#include "pete/eos.cpp"
#include "pete/hydrocoefficients.cpp"
#include "pete/hydrooperator.cpp"
#include "pete/phydrooperator.cpp"
#include "pete/ic.cpp"
#include "pete/nonlocalinteg.cpp"

using namespace std;
using namespace mfem;
//using namespace pete;

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

// Velocity coefficient
void direction_function(const Vector &x, Vector &v);
void Efield_function(const Vector &x, Vector &v);

// BGK collision operator functions
double kappa_function(const Vector &x);
double isosigma_function(const Vector &x);
double kappaisosigma_function(const Vector &x);
double source_function_cosx(const Vector &x);
double source_function_cosxcosy(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

// Attenuation coefficient kappa.
double KAPPA, SIGMA, EFIELD;

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *field, 
   const char *field_name = NULL, bool init_vis = false);


int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 2. Parse command-line options.
   //const char *mesh_file = "../data/laser_omega_half.mesh";
   const char *mesh_file = "../data/laser_omega_full.mesh";
   int ser_ref_levels = 1;
   int par_ref_levels = 0;
   int order = 3;
   int ode_solver_type = 4;
   int artificial_viscosity_type = 2;
   double dt = 0.1;
   double t_final = 140.0;
   bool visualization = true;
   double dt_vis = 1.;
   bool nonlocal = true;  
   KAPPA = 1e-2;
   SIGMA = 1e-3;
   EFIELD = 1e-4;
   double dt_nonlocal = 1.;
   
   int precision = 8;
   cout.precision(precision);
    
   int example = 5; // Omega-laser.
   const int Sod_example = 1, Saltzman_example = 2, triplepoint_example = 3,
      OmegaSedov_example = 4, OmegaLaser_example = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&example, "-x", "--example",
                  "PETE examples: 1 - Sod shock-tube, 2 - Saltzman tube,\n\t"
                  "               3 - Triple-point test,\n\t" 
				  "               4 - Omega-Sedov, 5 - Omega-Laser");
   args.AddOption(&nonlocal, "-nonl", "--nonlocal", "-no-nonl",
                  "--no-nonlocal",
                  "Enable or disable nonlocal calculation.");
   args.AddOption(&KAPPA, "-k", "--kappa",
                  "Attenuation coefficient kappa.");
   args.AddOption(&dt_nonlocal, "-dtn", "--dt-nonlocal",
                  "Attenuation coefficient kappa.");
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler, 2 - RK2,\n\t"
                  "            3 - RK3 SSP, 4 - RK4.");
   args.AddOption(&artificial_viscosity_type, "-v", "--viscosity-type",
                  "Artificial viscosity: 1 - mu1 Grad(v), 2 - mu1 epsilon,\n\t"
                  "                      3 - mu1 lambda1 s1 x s1,\n\t"
				  "                      4 - Sum muk lambdak sk x sk.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&dt_vis, "-vdt", "--visualization-timestep",
                  "Visualize after every dt_vis timestep.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
	  {
	     args.PrintUsage(cout);
      }
	  MPI_Finalize();
	  return 1;
   }
   if (myid == 0)
   {	   
      args.PrintOptions(cout);
   }


   // Default list contains no boundary attributes of a 2D rectangular mesh.
   Array<int> slip_velocity_bdr_attributes;
   slip_velocity_bdr_attributes.SetSize(0);

   switch (example)
   {   
      // Example Sod.
      case Sod_example:
	  mesh_file = "pete/meshes/Sod50.mesh";
      ser_ref_levels = 0;
      par_ref_levels = 0;
      order = 4;
      ode_solver_type = 4;
      artificial_viscosity_type = 3;
	  dt = 0.0001;
      t_final = 0.2;
      dt_vis = 0.004; 
	  KAPPA = 3e0;
	  //dt_nonlocal = 1e-3; 
	  // Full slip velocity BC.
      slip_velocity_bdr_attributes.SetSize(4);
      slip_velocity_bdr_attributes[0] = 1; // bottom
      slip_velocity_bdr_attributes[1] = 2; // right
      slip_velocity_bdr_attributes[2] = 3; // top
      slip_velocity_bdr_attributes[3] = 4; // left  
      break;
      // Example Saltzman.
      case Saltzman_example:
	  mesh_file = "pete/meshes/Sod50.mesh";
      ser_ref_levels = 0;
      par_ref_levels = 0;
      order = 3;
      ode_solver_type = 4;
      artificial_viscosity_type = 3;
      dt = 1e-5;
      t_final = 0.77;
	  //t_final = 1.0;
      dt_vis = 2e-2;  
      // Full slip velocity BC.
      slip_velocity_bdr_attributes.SetSize(4);
      slip_velocity_bdr_attributes[0] = 1; // bottom
      slip_velocity_bdr_attributes[1] = 2; // right
      slip_velocity_bdr_attributes[2] = 3; // top
      slip_velocity_bdr_attributes[3] = 4; // left     
	  break;
      // Example Triple-point.
      case triplepoint_example:
	  mesh_file = "pete/meshes/triplepoint_uniform.mesh";
      ser_ref_levels = 1;
      par_ref_levels = 0;
      order = 4;
      ode_solver_type = 4;
      artificial_viscosity_type = 4;
      dt = 0.001;
      t_final = 3.3;
	  //t_final = 10.0;
      dt_vis = 0.05; 
	  //dt_nonlocal = 1e-2;
	  //KAPPA = 0.4285;
	  //SIGMA = 0.0*KAPPA;
	  //EFIELD = 0.0*KAPPA;
	  //KAPPA = 0.4285;
	  //SIGMA = 2.0*KAPPA;
	  //EFIELD = 0.5*KAPPA; 
      // Full slip velocity BC.
      slip_velocity_bdr_attributes.SetSize(4);
      slip_velocity_bdr_attributes[0] = 1; // bottom
      slip_velocity_bdr_attributes[1] = 2; // right
      slip_velocity_bdr_attributes[2] = 3; // top
      slip_velocity_bdr_attributes[3] = 4; // left     
	  break;
      // Example Omega-Sedov.
      case OmegaSedov_example:
	  mesh_file = "pete/meshes/laser_omega_full.mesh";
      ser_ref_levels = 1;
      par_ref_levels = 0;
      order = 3;
      ode_solver_type = 4;
      artificial_viscosity_type = 2;
      dt = 0.1;
      t_final = 80.0;
      //t_final = 140.0;
	  dt_vis = 1.;
	  //dt_nonlocal = 1e-1;
	  KAPPA = 7.5e-3;
      SIGMA = 2.0*KAPPA;
      EFIELD = 0.5*KAPPA;
	  // Three side slip velocity BC.
      slip_velocity_bdr_attributes.SetSize(3);
      slip_velocity_bdr_attributes[0] = 1; // bottom
      slip_velocity_bdr_attributes[1] = 3; // top
      slip_velocity_bdr_attributes[2] = 4; // left
      break;     
	  // Example Omega-Laser.
      case OmegaLaser_example:
	  mesh_file = "pete/meshes/laser_omega_full.mesh";
      ser_ref_levels = 1;
      par_ref_levels = 0;
      order = 3;
      ode_solver_type = 4;
      artificial_viscosity_type = 2;
      dt = 0.1;
      t_final = 140.0;
      dt_vis = 1.;
      // No slip velocity BC.
      slip_velocity_bdr_attributes.SetSize(0);
      break;  
	  default:
         if (myid == 0)
         {
            cout << "Unknown PETE example: " << example << '\n';
         }
         MPI_Finalize();
		 return 1;
   }
   // Parse the arguments once more, if some of default parameters are 
   // required to be set.
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
	  {
	     args.PrintUsage(cout);
      }
	  MPI_Finalize();
	  return 1;
   }
   if (myid == 0)
   {	   
      args.PrintOptions(cout);
   }

   // 3. Read the serial mesh from the given mesh file on all processors. We can
   //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
   //    with the same code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // Set the mesh to correspond to the velocity FEspace
   H1_FECollection x_fe_coll(order, dim);
   FiniteElementSpace x_fespace(mesh, &x_fe_coll, dim); 
   // Set the finite element space for the mesh to correspond to velocity.
   mesh->SetNodalFESpace(&x_fespace);

   // 4. Define the ODE solver used for time integration. Several implicit
   //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
   //    explicit Runge-Kutta methods are available.
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Explicit methods
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break; // midpoint method
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         MPI_Finalize();
		 return 1;
   }

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }

   // 7. Define the vector finite element spaces representing the mesh
   //    deformation x_gf, the velocity v_gf, and the initial configuration,
   //    x_ref. Define also the internal temperature, temperature_gf, 
   //    which is in a discontinuous higher-order space. Since v, T, and x 
   //    are integrated in time as a system, we group them together in block 
   //    vector vTx, on the unique degrees of freedom, with offsets given by 
   //    array true_offset. We also define block vector vTxn because of minimal
   //    time step control and possible step back "in time". 

   H1_FECollection &v_fe_coll = x_fe_coll;
   L2_FECollection T_fe_coll(order - 1, dim);
   ParFiniteElementSpace pv_fespace(pmesh, &v_fe_coll, dim); 
   ParFiniteElementSpace pT_fespace(pmesh, &T_fe_coll);

   HYPRE_Int v_glob_size = pv_fespace.GlobalTrueVSize();
   HYPRE_Int T_glob_size = pT_fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of velocity/coordinates and temperature unknowns: " << 
	     v_glob_size << ", " << T_glob_size << endl;
   }
  
   int v_true_size = pv_fespace.GetTrueVSize();
   int T_true_size = pT_fespace.GetTrueVSize();
   //int v_true_size = v_fespace.GetTrueVSize();
   //int T_true_size = T_fespace.GetTrueVSize();
   Array<int> true_offset(4);
   true_offset[0] = 0;
   true_offset[1] = v_true_size;
   true_offset[2] = v_true_size + T_true_size;
   true_offset[3] = v_true_size + T_true_size + v_true_size;

   BlockVector vTxn(true_offset);

   // Prepare auxiliary grid functions for vTxn vector initialization.
   ParGridFunction v(&pv_fespace), T(&pT_fespace), x(&pv_fespace);

   // 7. Initialize the hydrodynamic operator based on mixed finite element 
   // spaces and initial density and material coefficient and 
   // the GLVis visualization.
   VectorFunctionCoefficient *velocity = NULL; 
   FunctionCoefficient *material = NULL;
   FunctionCoefficient *density = NULL;  
   FunctionCoefficient *temperature = NULL; 
   FunctionCoefficient *Tmelt_portion = NULL; 
   FunctionCoefficient *sigma_out = NULL;
   
   // Set given example initial conditions.
   switch (example)
   {
      // Example Sod.
      case Sod_example:
	  velocity = new VectorFunctionCoefficient(dim, 
	     pete::InitialVelocity_zero); 
      material = new FunctionCoefficient(pete::InitialMaterial_Sod);
      density = new FunctionCoefficient(pete::InitialDensity_Sod);  
      temperature = new FunctionCoefficient(pete::InitialInternalEnergy_Sod); 
      sigma_out = new FunctionCoefficient(pete::SigmaOut_zero);
	  break;
      // Example Saltzman.
      case Saltzman_example:
	  velocity = new VectorFunctionCoefficient(dim, 
         pete::InitialVelocity_Saltzman); 
      material = new FunctionCoefficient(pete::InitialMaterial_Saltzman);
      density = new FunctionCoefficient(pete::InitialDensity_Saltzman);  
      temperature = new FunctionCoefficient(
	     pete::InitialInternalEnergy_Saltzman); 
      sigma_out = new FunctionCoefficient(pete::SigmaOut_Saltzman);
	  break;
      // Example triplepoint.
      case triplepoint_example:
	  velocity = new VectorFunctionCoefficient(dim, 
	     pete::InitialVelocity_zero); 
      material = new FunctionCoefficient(pete::InitialMaterial_triplepoint);
      density = new FunctionCoefficient(pete::InitialDensity_triplepoint);  
      temperature = new FunctionCoefficient(
         pete::InitialInternalEnergy_triplepoint); 
      sigma_out = new FunctionCoefficient(pete::SigmaOut_zero);
      break;
      // Example Omega-Sedov.
      case OmegaSedov_example:
	  velocity = new VectorFunctionCoefficient(dim, 
	     pete::InitialVelocity_zero); 
      material = new FunctionCoefficient(pete::InitialMaterial_Omega);
      density = new FunctionCoefficient(pete::InitialDensity_Omega);  
      temperature = new FunctionCoefficient(pete::InitialTemperature_Omega); 
      sigma_out = new FunctionCoefficient(pete::SigmaOut_const);
      Tmelt_portion = 
         new FunctionCoefficient(pete::TmeltPortion_BdrPlusInterface_Omega);  
      break;
      // Example Omega-Laser.
      case OmegaLaser_example:
	  velocity = new VectorFunctionCoefficient(dim, 
	     pete::InitialVelocity_zero); 
      material = new FunctionCoefficient(pete::InitialMaterial_Omega);
      density = new FunctionCoefficient(pete::InitialDensity_Omega);  
      temperature = new FunctionCoefficient(pete::InitialTemperature_Omega); 
      sigma_out = new FunctionCoefficient(pete::SigmaOut_const);
      Tmelt_portion = 
         new FunctionCoefficient(pete::TmeltPortion_BdrPlusInterface_Omega);  
      break;
      default:
         if (myid == 0)
         {
            cout << "Unknown PETE example: " << example << '\n';
         }
         MPI_Finalize();
		 return 1;
   }
   // Initialize the auxiliary grid functions.
   v.ProjectCoefficient(*velocity);
   T.ProjectCoefficient(*temperature);
   pmesh->GetNodes(x);
   // Initialize the values of the vTxn vector.
   v.GetTrueDofs(vTxn.GetBlock(0));
   T.GetTrueDofs(vTxn.GetBlock(1));
   x.GetTrueDofs(vTxn.GetBlock(2));
   
   ParGridFunction pv_gf(&pv_fespace), pT_gf(&pT_fespace);
   ParMesh pmesh0(*pmesh);
   pete::EOS eos_ig;
   pete::ParHydrodynamicOperator operHydro(&pv_fespace, &pT_fespace, &pmesh0,
      density, material, &eos_ig, slip_velocity_bdr_attributes);

   // Eliminate slip velocity BC, i.e. set appropriate components of velocity 
   // to zero 
   //if (example == RT_example)
   //{	   
   //   operHydro.EliminateSlipVelocityBC(vTxn.GetBlock(0));
   //}
   // Initialize internal grid functions of velocity, temperature, and x of 
   // mesh coordinates and fill them based on true dof vTxn vector values
   operHydro.VelocityTemperatureXInitialization(&pv_gf, &pT_gf, vTxn);
   // Essential, the step of model discretization, i.e. algebraic translation
   operHydro.ModelAlgebraicTranslation(sigma_out);
   // Set the global "mesh Melting" reference temperature and spatial Melting 
   // function providing portion of Tmelt at boundary and material interfaces;
   // Example Omega.
   operHydro.GetSigmaCoefficient()->SetTref(1.0);
   operHydro.GetSigmaCoefficient()->SetTmeltPortionCoefficient(Tmelt_portion);

   // Set artificial viscosity type
   switch (artificial_viscosity_type)
   {
      case 1: operHydro.GetSigmaCoefficient()->SetArtificialViscosityType(1); 
	     break;
      case 2: operHydro.GetSigmaCoefficient()->SetArtificialViscosityType(2); 
	     break;
      case 3: operHydro.GetSigmaCoefficient()->SetArtificialViscosityType(3); 
	     break;
      case 4: operHydro.GetSigmaCoefficient()->SetArtificialViscosityType(4); 
	     break;
      default:
         if (myid == 0)
		 {
		    cout << "Unknown artificial viscosity type: " << 
		       artificial_viscosity_type << endl << flush;
         }
         MPI_Finalize();
		 return 1;
   }

   // In the case of avt=4 use less viscous coefficients
   if (artificial_viscosity_type==4)
   {
      cout << "SetViscosityCoefficients(1./4., 2./3.) as in the paper..." 
	     << endl << flush;
	  operHydro.GetSigmaCoefficient()->SetViscosityCoefficients(1./4., 2./3.);
   }  

   // HydrodynamicOperator oper does internally the evaluation of 
   // density, pressure, temperature (and of other physical quantities), 
   // which can be obtained by the following coefficients.
   Coefficient *material_hydro = operHydro.GetMaterialCoefficient();
   Coefficient *SMC_density_hydro = operHydro.GetDensityCoefficient();
   Coefficient *temperature_hydro = operHydro.GetTemperatureCoefficient();
   Coefficient *pressure_hydro = operHydro.GetPressureCoefficient(); 
   Coefficient *meshnaturalmap_hydro = operHydro.GetMeshNaturalMapCoefficient();
   // Grid functions for visualization of Lagrangian hydrodynamic 
   // variables.
   L2_FECollection vis_fe_coll(order - 1, dim); // the same order as T_fespace 
   ParFiniteElementSpace vis_fespace(pmesh, &vis_fe_coll);
   ParGridFunction material_gf(&vis_fespace), rho_gf(&vis_fespace), 
      temperature_gf(&vis_fespace), pressure_gf(&vis_fespace),
	  mesh_natural_map_gf(&vis_fespace); 
   
   // Variables update.
   material_gf.ProjectCoefficient(*material_hydro);
   rho_gf.ProjectCoefficient(*SMC_density_hydro);
   temperature_gf.ProjectCoefficient(*temperature_hydro); 
   pressure_gf.ProjectCoefficient(*pressure_hydro);
   mesh_natural_map_gf.ProjectCoefficient(*meshnaturalmap_hydro);

   /////////////////////////////////////////
   // NONLOCAL INITIALIZATION SECTION //////
   /////////////////////////////////////////
   //KAPPA = 2e-2;
   //KAPPA = 0.4285;
   //SIGMA = 2.0*KAPPA;
   //EFIELD = 0.5*KAPPA; 
   int Aorder_phi = 2;
   int Aorder_theta = 2;
   int Iorder = order;
   int Torder = order-1;
   double tol_NL = 1e-5;
   // Set nonlocal problem coefficients.
   VectorFunctionCoefficient *Efield=NULL;
   Coefficient *kappa=NULL;
   Coefficient *isosigma=NULL;
   Coefficient *sourceb=NULL;
   Coefficient *sourceT=NULL;
   GridFunction *S_NL=NULL; 
   ParGridFunction *source_NL=NULL;
   GridFunctionCoefficient *sourceCoeffT=NULL;
   Coefficient *Cv=NULL;

   switch (example)
   {
      // Example Sod.
      case Sod_example: break;
      // Example Saltzman.
      case Saltzman_example: break;
      // Example triplepoint.
      case triplepoint_example: 
	     KAPPA = 0.4285;
	     SIGMA = 2.0*KAPPA;
	     EFIELD = 0.75*KAPPA; 
		 Efield = new VectorFunctionCoefficient(dim, Efield_function);
         kappa = new FunctionCoefficient(kappa_function);
         isosigma = new FunctionCoefficient(isosigma_function);
         //sourceb = new FunctionCoefficient(source_function_cosx);
         sourceb = new FunctionCoefficient(source_function_cosxcosy);
		 //sourceT = new ConstantCoefficient(0.);
         // Set grid function to represent the source of intensity 
         // S = S_NL*source_NL, where source_NL is a system unknown 
         //S_NL = &rho_gf; 
         //sourceCoeffT = new GridFunctionCoefficient(S_NL);
         //Cv = new ConstantCoefficient(1.);  
		 break;
      // Example Omega-Sedov.
      case OmegaSedov_example:
	     KAPPA = 2e-2;
         //KAPPA = 0.4285;
         SIGMA = 2.0*KAPPA;
         EFIELD = 0.0*KAPPA;
	     Efield = new VectorFunctionCoefficient(dim, Efield_function);
         kappa = new FunctionCoefficient(kappa_function);
         isosigma = new FunctionCoefficient(isosigma_function);
         sourceb = new ConstantCoefficient(0.);
         sourceT = new ConstantCoefficient(0.);
         // Set grid function to represent the source of intensity 
         // S = S_NL*source_NL, where source_NL is a system unknown 
         source_NL = &temperature_gf; //&rho_gf;
         S_NL = &rho_gf; 
         sourceCoeffT = new GridFunctionCoefficient(S_NL);
         Cv = new ConstantCoefficient(1.);  
		 break;
      // Example Omega-Laser.
      case OmegaLaser_example: 
	     KAPPA = 2e-2;
         //KAPPA = 0.4285;
         SIGMA = 2.0*KAPPA;
         EFIELD = 0.0*KAPPA;
		 Efield = new VectorFunctionCoefficient(dim, Efield_function);
         kappa = new FunctionCoefficient(kappa_function);
         isosigma = new FunctionCoefficient(isosigma_function);
         sourceb = new ConstantCoefficient(0.);
         sourceT = new ConstantCoefficient(0.);
         // Set grid function to represent the source of intensity 
         // S = S_NL*source_NL, where source_NL is a system unknown 
         source_NL = &temperature_gf; //&rho_gf;
         S_NL = &rho_gf; 
         sourceCoeffT = new GridFunctionCoefficient(S_NL);
         Cv = new ConstantCoefficient(1e10);  break;
      default:
         if (myid == 0)
         {
            cout << "Unknown PETE example: " << example << '\n';
         }
         MPI_Finalize();
		 return 1;
   }



/*
   VectorFunctionCoefficient Efield(dim, Efield_function);
   FunctionCoefficient kappa(kappa_function);
   FunctionCoefficient isosigma(isosigma_function);
   //FunctionCoefficient sourceb(source_function_cosx);
   //FunctionCoefficient sourceb(source_function_cosxcosy);
   ConstantCoefficient sourceb(0.);
   ConstantCoefficient sourceT(0.);
   GridFunction *S_NL = &rho_gf; 
   GridFunctionCoefficient sourceCoeffT(S_NL);
   ConstantCoefficient Cv(1.);
*/
   //FunctionCoefficient inflow(inflow_function);
   // Create the operator for nonlocal calculation.
   pete::ParNonlocalOperator operNonlocal(pmesh, &pT_fespace, Iorder, Torder,
      Aorder_phi, Aorder_theta); 
   // Get the grid function representing the nonlocal intensity.
   ParGridFunction *u = operNonlocal.GetIntensityGridFunction();
   // Discretize the analytical model.
   operNonlocal.ModelAlgebraicTranslation(Cv, kappa, isosigma, sourceb,
      sourceCoeffT, sourceT, Efield); 
   
   // NonlocalOperator operNonlocal does internally the evaluation of 
   // intensity and its moments can be obtained by the following coefficients.
   Coefficient *I0_nonlocal = operNonlocal.GetZeroMomentCoefficient();
   // TMP coefficients.
   Coefficient *I1z_nonlocal = operNonlocal.GetFirstMomentZCoefficient();
   Coefficient *I1x_nonlocal = operNonlocal.GetFirstMomentXCoefficient();
   VectorCoefficient *I1_nonlocal = operNonlocal.GetFirstMomentCoefficient();
   // Grid functions for visualization of nonlocal Lagrangian hydrodynamic 
   // variables.
   ParFiniteElementSpace vis_I0_fespace(pmesh, operNonlocal.GetXfec());
   ParFiniteElementSpace vis_I1_fespace(pmesh, operNonlocal.GetXfec(), dim);
   ParGridFunction I0_gf(&vis_I0_fespace); 
   ParGridFunction I1z_gf(&vis_I0_fespace);
   ParGridFunction I1x_gf(&vis_I0_fespace);
   ParGridFunction I1_gf(&vis_I1_fespace);

   int nti_NL;
   double dt_NL = 1.;
   double Umax_NL = -1e-32;
   double dUmax_NL = 1e-32;
   operNonlocal.Compute(dt_NL, tol_NL, Umax_NL, dUmax_NL, nti_NL, u, source_NL);
    // Variables update.
   I0_gf.ProjectCoefficient(*I0_nonlocal);  
   I1z_gf.ProjectCoefficient(*I1z_nonlocal);
   I1x_gf.ProjectCoefficient(*I1x_nonlocal);
   I1_gf.ProjectCoefficient(*I1_nonlocal);
   if (myid == 0)
   {
      cout << "operNonlocal - Umax(iter, dUmax): " << Umax_NL 
	     << " ( " << nti_NL << ", " << dUmax_NL << " )"<< endl << flush;
   }
   /////////////////////////////////////////
   // NONLOCAL INITIALIZATION SECTION END //
   /////////////////////////////////////////

   socketstream vis_rho, vis_temperature, vis_pressure, vis_material, 
      vis_natural, vis_nonlocalI0, vis_nonlocalI1z, vis_nonlocalI1x;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_rho.open(vishost, visport);
	  if (vis_rho)
	  { 
         vis_rho.precision(8);
         visualize(vis_rho, pmesh, &rho_gf, "Density", true);
	  }
	  // Make sure all ranks have sent their 'e' solution before initiating
      // another set of GLVis connections (one from each rank):
	  MPI_Barrier(pmesh->GetComm());
      vis_temperature.open(vishost, visport);
      if (vis_temperature)
      {
         vis_temperature.precision(8);
         visualize(vis_temperature, pmesh, &temperature_gf, "Temperature", 
		    true);
      }
	  // Make sure all ranks have sent their 'rho' solution before initiating
      // another set of GLVis connections (one from each rank):
	  MPI_Barrier(pmesh->GetComm());
      vis_pressure.open(vishost, visport);
      if (vis_pressure)
      {
         vis_pressure.precision(8);
         visualize(vis_pressure, pmesh, &pressure_gf, "Pressure", true);
      } 
	  // Make sure all ranks have sent their 'rho' solution before initiating
      // another set of GLVis connections (one from each rank):
	  MPI_Barrier(pmesh->GetComm());
      vis_material.open(vishost, visport);
      if (vis_material)
      {
         vis_material.precision(8);
         visualize(vis_material, pmesh, &material_gf, "Material", true);
      } 
	  // Make sure all ranks have sent their 'rho' solution before initiating
      // another set of GLVis connections (one from each rank):
	  MPI_Barrier(pmesh->GetComm());
      vis_natural.open(vishost, visport);
      if (vis_natural)
      {
         vis_natural.precision(8);
         visualize(vis_natural, pmesh, &mesh_natural_map_gf, 
		 "Mesh natural dofs", true);
	  }
	  // Make sure all ranks have sent their 'rho' solution before initiating
      // another set of GLVis connections (one from each rank):
	  MPI_Barrier(pmesh->GetComm());
      vis_nonlocalI0.open(vishost, visport);
      if (vis_nonlocalI0)
      {
         vis_nonlocalI0.precision(8);
         visualize(vis_nonlocalI0, pmesh, &I0_gf, "Nonlocal I0", true);
	  } 
	  // Make sure all ranks have sent their 'rho' solution before initiating
      // another set of GLVis connections (one from each rank):
	  MPI_Barrier(pmesh->GetComm());
      vis_nonlocalI1z.open(vishost, visport);
      if (vis_nonlocalI1z)
      {
         vis_nonlocalI1z.precision(8);
         visualize(vis_nonlocalI1z, pmesh, &I1z_gf, "Nonlocal I1z", true);
	  }
	  // Make sure all ranks have sent their 'rho' solution before initiating
      // another set of GLVis connections (one from each rank):
	  MPI_Barrier(pmesh->GetComm());
      vis_nonlocalI1x.open(vishost, visport);
      if (vis_nonlocalI1x)
      {
         vis_nonlocalI1x.precision(8);
         visualize(vis_nonlocalI1x, pmesh, &I1x_gf, "Nonlocal I1x", true);
	  }
   }

   // Write output for GLVis.
   string path = "results/";
   // Set given example output directory.
   switch (example)
   {
      // Example Sod.
      case Sod_example: path = "results/Sod/"; break;
      // Example Saltzman.
      case Saltzman_example: path = "results/Saltzman/"; break;
      // Example triplepoint.
      case triplepoint_example: path = "results/triplepoint/"; break;
      // Example Omega-Sedov.
      case OmegaSedov_example: path = "results/OmegaSedov/"; break;
      // Example Omega-Laser.
      case OmegaLaser_example: path = "results/OmegaLaser/"; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown PETE example: " << example << '\n';
         }
         MPI_Finalize();
		 return 1;
   }


   int snapshot = 0;
   { 
	  ostringstream ss, mesh_name, rho_name, pressure_name, temperature_name, 
	     material_name, naturaldofs_name, nonlocalI0_name, nonlocalI1z_name, 
		 nonlocalI1x_name, nonlocalI1_name;
	  ss << snapshot;
	  string snp = ss.str()+".";
	  	
      mesh_name << path+"mesh_"+snp << setfill('0') << setw(6) << myid;
      rho_name << path+"rho_"+snp << setfill('0') << setw(6) << myid;
      pressure_name << path+"pressure_"+snp << setfill('0') << setw(6) << myid;
	  temperature_name << path+"temperature_"+snp << setfill('0') << setw(6) 
	     << myid;
	  material_name << path+"material_"+snp << setfill('0') << setw(6) << myid;
	  naturaldofs_name << path+"naturaldofs_"+snp << setfill('0') << setw(6) 
	     << myid;
	  nonlocalI0_name << path+"nonlocalI0_"+snp << setfill('0') << setw(6) 
	     << myid;
	  nonlocalI1z_name << path+"nonlocalI1z_"+snp << setfill('0') << setw(6) 
	     << myid;
	  nonlocalI1x_name << path+"nonlocalI1x_"+snp << setfill('0') << setw(6) 
	     << myid;
	  nonlocalI1_name << path+"nonlocalI1_"+snp << setfill('0') << setw(6) 
	     << myid;
	  
	  ofstream omesh(mesh_name.str().c_str());
      omesh.precision(precision);
      pmesh->Print(omesh);
      ofstream orho(rho_name.str().c_str());
      orho.precision(precision);
      rho_gf.Save(orho);
      ofstream opressure(pressure_name.str().c_str());
      opressure.precision(precision);
      pressure_gf.Save(opressure);
      ofstream otemperature(temperature_name.str().c_str());
      otemperature.precision(precision);
      temperature_gf.Save(otemperature);
	  ofstream omaterial(material_name.str().c_str());
      omaterial.precision(precision);
      material_gf.Save(omaterial);  
      ofstream onaturaldofs(naturaldofs_name.str().c_str());
      onaturaldofs.precision(precision);
      mesh_natural_map_gf.Save(onaturaldofs);
      ofstream ononlocalI0(nonlocalI0_name.str().c_str());
      ononlocalI0.precision(precision);
      I0_gf.Save(ononlocalI0);
      ofstream ononlocalI1z(nonlocalI1z_name.str().c_str());
      ononlocalI1z.precision(precision);
      I1z_gf.Save(ononlocalI1z);
	  ofstream ononlocalI1x(nonlocalI1x_name.str().c_str());
      ononlocalI1x.precision(precision);
      I1x_gf.Save(ononlocalI1x);  
      ofstream ononlocalI1(nonlocalI1_name.str().c_str());
      ononlocalI1.precision(precision);
      I1_gf.Save(ononlocalI1);
   }

   // Hydro unknown vector for allowing one step back.
   BlockVector vTx(true_offset);
   vTx = vTxn;

   double t = 0.0;
   operHydro.SetTime(t);
   ode_solver->Init(operHydro);

   // 8. Perform time-integration
   //     (looping over the time iterations, ti, with sigma calculated 
   //     time-step dt).
   bool last_step = false;
   double t_vis = 0; 

   for (int ti = 1; !last_step; ti++)
   {
	  double dt_real = min(min(dt, dt_vis), t_final - t);
      // Default values for time step evaluation.
	  double beta1 = 0.85, beta2 = 1.02, gamma = 0.8;
	  // Extend the time step artificially because of do-while loop.
	  dt_real *= 1/beta1;
	  int cti;
	  double tn;
	  // MPI Reduce strategy
	  double dt_min = 2e32;
	  //dt_real = 0.04/beta1;
	  bool iterate = true;
	  for (cti = 0; iterate; cti++)
	  {	 
	     dt_real *= beta1;
		 // Set Y^n.
		 vTxn = vTx;
		 tn = t;	  
		 //cout << "dt Step-iteration: " << dt_real << endl << flush;
		 operHydro.GetSigmaCoefficient()->SetDtEstimate(1e32);
		 ode_solver->Step(vTxn, tn, dt_real); 
		 // Get the local processor tau estimate.   
		 double dt_est = operHydro.GetSigmaCoefficient()->GetDtEstimate();
		 // Find the minimum of tau over all processors.
		 MPI_Barrier(pmesh->GetComm());
		 MPI_Reduce(&dt_est, &dt_min, 1, MPI_DOUBLE, MPI_MIN, 0,//myid==0 
            pmesh->GetComm());
	     // Distribute the minimum tau over all processors.
		 MPI_Barrier(pmesh->GetComm());
		 MPI_Bcast(&dt_min, 1, MPI_DOUBLE, 0, pmesh->GetComm());
		 // Check, if the time step used in "Step" was not too long.
		 // If so, loop and shrink the time step dt_real. 	
	     iterate = (dt_real >= dt_min);
	  }
	  // Assign new value of dt.
	  dt = dt_real;
	  // Extend the time step if convenient.
	  if (dt < gamma*dt_min)
	  {
	     dt = dt*beta2;
	  } 
	  
	  if (myid == 0)
	  {
		 cout << "Step(time): " << ti << "(" << t << "),   " << 
	        "dt(tau iter): " << dt << "(" << cti << ")" << endl << flush;
	  }
	  // The update of vTxn was successful, and so, update vTx as well.
	  // Y{n+1} update
	  vTx = vTxn;
	  t = tn;
 
      last_step = (t >= t_final - 1e-8*dt_real);

      // Nonuniform time step requires mod(t, dt_vis).
      t_vis = t - floor(t/dt_vis)*dt_vis; 

      if (last_step || t_vis <= dt_real)
      {
		 snapshot++;
		 if (myid == 0)
		 {
		    cout << "Snapshot(time): " << snapshot << "(" << t << ")" << endl 
		       << flush;
         }		 
  
         // Variables update.
         material_gf.ProjectCoefficient(*material_hydro);
         rho_gf.ProjectCoefficient(*SMC_density_hydro);
         temperature_gf.ProjectCoefficient(*temperature_hydro); 
         pressure_gf.ProjectCoefficient(*pressure_hydro);
         mesh_natural_map_gf.ProjectCoefficient(*meshnaturalmap_hydro);

         if ((example == Sod_example || example == triplepoint_example || 
            example == OmegaSedov_example || 
            example == OmegaLaser_example ) && nonlocal)
         {
         /////////////////////////////////////
         // NONLOCAL CALCULATION SECTION /////
         /////////////////////////////////////
         int nti_NL;
         double Umax_NL = -1e-32;
         double dUmax_NL = 1e-32;
         operNonlocal.Compute(dt_NL, tol_NL, Umax_NL, dUmax_NL, nti_NL, u, 
		    source_NL);
         // Nonlocal results update.
         I0_gf.ProjectCoefficient(*I0_nonlocal);
		 I1z_gf.ProjectCoefficient(*I1z_nonlocal);
		 I1x_gf.ProjectCoefficient(*I1x_nonlocal);
		 I1_gf.ProjectCoefficient(*I1_nonlocal);
		 
		 if (myid == 0)
         {
            cout << "operNonlocal - Umax(iter, dUmax): " << Umax_NL 
	           << " ( " << nti_NL << ", " << dUmax_NL << " )"<< endl << flush;
         }
		 //////////////////////////////////////
         // NONLOCAL CALCULATION SECTION END //
         //////////////////////////////////////
         }

         if (visualization)
         {
            if (vis_rho)
			{	
			   visualize(vis_rho, pmesh, &rho_gf);
			}
			if (vis_temperature)
            {
               visualize(vis_temperature, pmesh, &temperature_gf);
			}
			if (vis_pressure)
            {
               visualize(vis_pressure, pmesh, &pressure_gf);
			}
			if (vis_material)
            {
               visualize(vis_material, pmesh, &material_gf);
			}
			if (vis_natural)
            {
               visualize(vis_natural, pmesh, &mesh_natural_map_gf);
			}
			if (vis_nonlocalI0)
            {
               visualize(vis_nonlocalI0, pmesh, &I0_gf);
			}
			if (vis_nonlocalI1z)
            {
               visualize(vis_nonlocalI1z, pmesh, &I1z_gf);
			}		
			if (vis_nonlocalI1x)
            {
               visualize(vis_nonlocalI1x, pmesh, &I1x_gf);
			}		
         }
	     // Write the snapshot output for GLVis.
		 {
	        ostringstream ss, mesh_name, rho_name, pressure_name, 
			   temperature_name, material_name, naturaldofs_name, nonlocal_name;
	        ss << snapshot;
	        string snp = ss.str()+".";
	  	
            mesh_name << path+"mesh_"+snp << setfill('0') << setw(6) << myid;
            rho_name << path+"rho_"+snp << setfill('0') << setw(6) << myid;
            pressure_name << path+"pressure_"+snp << setfill('0') << setw(6) 
			   << myid;
	        temperature_name << path+"temperature_"+snp << setfill('0') 
			   << setw(6) << myid;
	        material_name << path+"material_"+snp << setfill('0') << setw(6) 
			   << myid;
	        naturaldofs_name << path+"naturaldofs_"+snp << setfill('0') 
			   << setw(6) << myid;
	        nonlocal_name << path+"nonlocal_"+snp << setfill('0') << setw(6) 
			   << myid;
	  
	        ofstream omesh(mesh_name.str().c_str());
            omesh.precision(precision);
            pmesh->Print(omesh);
            ofstream orho(rho_name.str().c_str());
            orho.precision(precision);
            rho_gf.Save(orho);
            ofstream opressure(pressure_name.str().c_str());
            opressure.precision(precision);
            pressure_gf.Save(opressure);
            ofstream otemperature(temperature_name.str().c_str());
            otemperature.precision(precision);
            temperature_gf.Save(otemperature);
	        ofstream omaterial(material_name.str().c_str());
            omaterial.precision(precision);
            material_gf.Save(omaterial);  
            ofstream onaturaldofs(naturaldofs_name.str().c_str());
            onaturaldofs.precision(precision);
            mesh_natural_map_gf.Save(onaturaldofs);
            ofstream ononlocal(nonlocal_name.str().c_str());
            ononlocal.precision(precision);
            u->Save(ononlocal);
         }	 
      }
   }

   // 9. Free the used memory.
   delete ode_solver;
   delete velocity; 
   delete material;
   delete density;  
   delete temperature; 
   delete Tmelt_portion; 
   delete sigma_out;  
   delete pmesh;
   delete meshnaturalmap_hydro;
   delete Efield;
   delete kappa;
   delete isosigma;
   delete sourceb;
   delete sourceT;
   delete sourceCoeffT;
   delete Cv;
   /*
   delete u; 
   delete k;
   //delete kn; 
   delete g;
   //delete gn; 
   delete b;
   //delete bn;
   delete fes;
   delete Afes;
   delete Amesh_phi;
   delete Amesh_theta;
   */

   MPI_Finalize();
   return 0;
}

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *field, 
   const char *field_name, bool init_vis)
{
   if (!out)
   {
      return;
   }

   out << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
   out << "solution\n" << *mesh << *field;

   if (init_vis)
   {
      out << "window_size 800 800\n";
      out << "window_title '" << field_name << "'\n";
      if (mesh->SpaceDimension() == 2)
      {
         out << "view 0 0\n"; // view from top
         out << "keys jl\n";  // turn off perspective and light
      }
      //out << "keys aRR\n";// rotate view
	  //out << "keys **********\n";// zoom in
	  out << "keys cm\n";         // show colorbar and mesh
      out << "autoscale value\n"; // update value-range; keep mesh-extents fixed
      out << "pause\n";
   }
   out << "keys S\n"; // save a snapshot
   out << flush;
}

// Velocity coefficient
void Efield_function(const Vector &x, Vector &v)
{
   // test the BGK effect
   v(0) = 1.;
   v(1) = 1.;
   v *= 1/v.Norml2();
   v *= EFIELD;
   return;
}

// Attenuation coefficient kappa of the BGK collision operator kappa*(S - u)
double kappa_function(const Vector &x)
{
   return KAPPA;
}
// Isotropization coefficient sigma of 
// the BGK collision operator sigma*(mean(u) - u)
double isosigma_function(const Vector &x)
{
   return SIGMA;
}
// Attenuation+iostropization coefficient kappasigma of 
// the BGK collision operator kappa*(S - u) + sigma(mean(u) - u)
double kappaisosigma_function(const Vector &x)
{
   return kappa_function(x) + isosigma_function(x);
}

double source_function_cosx(const Vector &x_coord)
{
   double pi = 3.14159265359;
   double x = x_coord[0], y = x_coord[1];
   double a0 = 1.;
   double kappa = kappa_function(x_coord);
   double S = a0*(1. - cos(2.*pi*x/(bb_max(0) - bb_min(0))));
   
   return kappa*S;
}

double source_function_cosxcosy(const Vector &x_coord)
{
   double pi = 3.14159265359;
   double x = x_coord[0], y = x_coord[1];
   double a0 = 1.;
   double kappa = kappa_function(x_coord);
   double S = a0*(1. - cos(2.*pi*x/(bb_max(0) - bb_min(0))))*
      (1. - cos(2.*pi*y/(bb_max(1) - bb_min(1))));
   
   return kappa*S;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
   return 0.0;
}
