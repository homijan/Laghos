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
//
//                     __                __
//                    / /   ____  ____  / /_  ____  _____
//                   / /   / __ `/ __ `/ __ \/ __ \/ ___/
//                  / /___/ /_/ / /_/ / / / / /_/ (__  )
//                 /_____/\__,_/\__, /_/ /_/\____/____/
//                             /____/
//
//             High-order Lagrangian Hydrodynamics Miniapp
//
// Laghos(LAGrangian High-Order Solver) is a miniapp that solves the
// time-dependent Euler equation of compressible gas dynamics in a moving
// Lagrangian frame using unstructured high-order finite element spatial
// discretization and explicit high-order time-stepping. Laghos is based on the
// numerical algorithm described in the following article:
//
//    V. Dobrev, Tz. Kolev and R. Rieben, "High-order curvilinear finite element
//    methods for Lagrangian hydrodynamics", SIAM Journal on Scientific
//    Computing, (34) 2012, pp.B606–B641, https://doi.org/10.1137/120864672.
//
// Sample runs:
//    mpirun -np 8 laghos -p 0 -m data/square01_quad.mesh -rs 3 -tf 0.5
//    mpirun -np 8 laghos -p 0 -m data/square01_tri.mesh  -rs 1 -tf 0.5
//    mpirun -np 8 laghos -p 0 -m data/cube01_hex.mesh    -rs 1 -cfl 0.1 -tf 0.5
//    mpirun -np 8 laghos -p 1 -m data/square01_quad.mesh -rs 3 -tf 0.8
//    mpirun -np 8 laghos -p 1 -m data/cube01_hex.mesh    -rs 2 -tf 0.6
//
// Test problems:
//    p = 0  --> Taylor-Green vortex (smooth problem).
//    p = 1  --> Sedov blast.


#include "laghos_solver.hpp"
#include "pnonlocaloperator.hpp"
#include <memory>
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;
using namespace mfem::hydrodynamics;

// Choice for the problem setup.
int problem;

void display_banner(ostream & os);

int main(int argc, char *argv[])
{
   // Initialize MPI.
   MPI_Session mpi(argc, argv);
   int myid = mpi.WorldRank();

   // Print the banner.
   if (mpi.Root()) { display_banner(cout); }

   // Parse command-line options.
   const char *mesh_file = "data/square01_quad.mesh";
   int rs_levels = 0;
   int rp_levels = 0;
   int order_v = 2;
   int order_e = 1;
   int ode_solver_type = 4;
   double t_final = 0.5;
   double cfl = 0.1;
   bool p_assembly = true;
   bool visualization = false;
   bool visit = false;
   int vis_steps = 5;
   int gfprint = 0;
   const char *basename = "Laghos";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&order_v, "-ov", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&p_assembly, "-pa", "--partial-assembly", "-fa",
                  "--full-assembly",
                  "Activate 1D tensor-based assembly (partial assembly).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.Parse();
   if (!args.Good())
   {
      if (mpi.Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (mpi.Root()) { args.PrintOptions(cout); }

   // Read the serial mesh from the given mesh file on all processors.
   // Refine the mesh in serial to increase the resolution.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   const int dim = mesh->Dimension();
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }

   if (p_assembly && dim == 1)
   {
      p_assembly = false;
      cout << "Full assembly will be used (equivalent to PA in 1D)." << endl;
   }

   // Parallel partitioning of the mesh.
   ParMesh *pmesh = NULL;
   const int num_tasks = mpi.WorldSize();
   const int partitions = floor(pow(num_tasks, 1.0 / dim) + 1e-2);
   int *nxyz = new int[dim];
   int product = 1;
   for (int d = 0; d < dim; d++)
   {
      nxyz[d] = partitions;
      product *= partitions;
   }
   if (product == num_tasks)
   {
      int *partitioning = mesh->CartesianPartitioning(nxyz);
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
      delete partitioning;
   }
   else
   {
      if (myid == 0)
      {
         cout << "Non-Cartesian partitioning through METIS will be used.\n";
#ifndef MFEM_USE_METIS
         cout << "MFEM was built without METIS. "
              << "Adjust the number of tasks to use a Cartesian split." << endl;
#endif
      }
#ifndef MFEM_USE_METIS
      return 1;
#endif
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   }
   delete [] nxyz;
   delete mesh;

   // Refine the mesh further in parallel to increase the resolution.
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());
   // Create the operator for nonlocal calculation.
   int Aorder_phi = 2;
   int Aorder_theta = 2;
   int Iorder = order_e + 1;
   int Torder = order_e;
   nth::ParNonlocalOperator operNonlocal(pmesh, &L2FESpace, Iorder, Torder,
      Aorder_phi, Aorder_theta); 
   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> ess_tdofs;
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max()), tdofs1d;
      for (int d = 0; d < pmesh->Dimension(); d++)
      {
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries, i.e., we must
         // enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialTrueDofs(ess_bdr, tdofs1d, d);
         ess_tdofs.Append(tdofs1d);
      }
   }

   // Define the explicit ODE solver used for time integration.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete pmesh;
         MPI_Finalize();
         return 3;
   }

   HYPRE_Int glob_size_l2 = L2FESpace.GlobalTrueVSize();
   HYPRE_Int glob_size_h1 = H1FESpace.GlobalTrueVSize();

   if (mpi.Root())
   {
      cout << "Number of kinematic (position, velocity) dofs: "
           << glob_size_h1 << endl;
      cout << "Number of specific internal energy dofs: "
           << glob_size_l2 << endl;
   }

   int Vsize_l2 = L2FESpace.GetVSize();
   int Vsize_h1 = H1FESpace.GetVSize();

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy

   Array<int> true_offset(4);
   true_offset[0] = 0;
   true_offset[1] = true_offset[0] + Vsize_h1;
   true_offset[2] = true_offset[1] + Vsize_h1;
   true_offset[3] = true_offset[2] + Vsize_l2;
   BlockVector S(true_offset);

   // Define GridFunction objects for the position, velocity and specific
   // internal energy.  There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   ParGridFunction x_gf, v_gf, e_gf;
   x_gf.MakeRef(&H1FESpace, S, true_offset[0]);
   v_gf.MakeRef(&H1FESpace, S, true_offset[1]);
   e_gf.MakeRef(&L2FESpace, S, true_offset[2]);

   // Initialize x_gf using the starting mesh coordinates. This also links the
   // mesh positions to the values in x_gf.
   pmesh->SetNodalGridFunction(&x_gf);

   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);

   // Initialize density and  specific internal energy values. We interpolate
   // in a non-positive basis to get the correct values at the dofs.
   // Then we do an L2 projection to the positive basis in which we actually
   // compute. The goal of all this is to get a high-order representation of
   // the initial condition. Note that this density is a temporary function
   // and it will not be updated during the time evolution.
   ParGridFunction rho(&L2FESpace);
   FunctionCoefficient rho_coeff(hydrodynamics::rho0);
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction l2_rho(&l2_fes), l2_e(&l2_fes);
   l2_rho.ProjectCoefficient(rho_coeff);
   rho.ProjectGridFunction(l2_rho);
   if (problem == 1)
   {
      // For the Sedov test, we use a delta function at the origin.
      DeltaCoefficient e_coeff(0, 0, 0.25);
      l2_e.ProjectCoefficient(e_coeff);
   }
   else
   {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
   }
   e_gf.ProjectGridFunction(l2_e);

   // Space-dependent ideal gas coefficient over the Lagrangian mesh.
   Coefficient *material_pcf = new FunctionCoefficient(hydrodynamics::gamma);

   // Additional details, depending on the problem.
   int source = 0; bool visc;
   switch (problem)
   {
      case 0: if (pmesh->Dimension() == 2) { source = 1; }
         visc = false; break;
      case 1: visc = true; break;
      case 2: visc = true; break;
      case 3: visc = true; break;
      default: MFEM_ABORT("Wrong problem specification!");
   }

   LagrangianHydroOperator oper(S.Size(), H1FESpace, L2FESpace,
                                ess_tdofs, rho, source, cfl, material_pcf,
                                visc, p_assembly);

   socketstream vis_rho, vis_v, vis_e;
   char vishost[] = "localhost";
   int  visport   = 19916;

   ParGridFunction rho_gf;
   if (visualization || visit) { oper.ComputeDensity(rho_gf); }

   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());

      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);

      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      int offx = Ww+10; // window offsets

      miniapps::VisualizeField(vis_rho, vishost, visport, rho_gf,
                               "Density", Wx, Wy, Ww, Wh);
      Wx += offx;
      miniapps::VisualizeField(vis_v, vishost, visport, v_gf,
                               "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      miniapps::VisualizeField(vis_e, vishost, visport, e_gf,
                               "Specific Internal Energy", Wx, Wy, Ww, Wh);
   }

   // Save data for VisIt visualization
   VisItDataCollection visit_dc(basename, pmesh);
   if (visit)
   {
      visit_dc.RegisterField("Density",  &rho_gf);
      visit_dc.RegisterField("Velocity", &v_gf);
      visit_dc.RegisterField("Specific Internal Energy", &e_gf);
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();
   }

   // Perform time-integration (looping over the time iterations, ti, with a
   // time-step dt). The object oper is of type LagrangianHydroOperator that
   // defines the Mult() method that used by the time integrators.
   ode_solver->Init(oper);
   oper.ResetTimeStepEstimate();
   double t = 0.0, dt = oper.GetTimeStepEstimate(S), t_old;
   bool last_step = false;
   BlockVector S_old(S);
   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }

      S_old = S;
      t_old = t;
      oper.ResetTimeStepEstimate();

      // S is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Step(S, t, dt);

      // Adaptive time step control.
      const double dt_est = oper.GetTimeStepEstimate(S);
      if (dt_est < dt)
      {
         // Repeat (solve again) with a decreased time step - decrease of the
         // time estimate suggests appearance of oscillations.
         dt *= 0.85;
         t = t_old;
         S = S_old;
         oper.ResetQuadratureData();
         if (mpi.Root()) { cout << "Repeating step " << ti << endl; }
         ti--; continue;
      }
      else if (dt_est > 1.25 * dt) { dt *= 1.02; }

      // Make sure that the mesh corresponds to the new solution state.
      pmesh->NewNodes(x_gf, false);

      if (gfprint == 1)
      {
         ostringstream v_name, e_name, mesh_name;
         v_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "v." << setfill('0') << setw(6) << myid;
         e_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                << "e." << setfill('0') << setw(6) << myid;
         mesh_name << basename << "_" << setfill('0') << setw(6) << t << "_"
                   << "mesh." << setfill('0') << setw(6) << myid;

         ofstream mesh_ofs(mesh_name.str().c_str());
         mesh_ofs.precision(8);
         pmesh->Print(mesh_ofs);
         mesh_ofs.close();

         ofstream v_ofs(v_name.str().c_str());
         v_ofs.precision(8);
         v_gf.Save(v_ofs);
         v_ofs.close();

         ofstream e_ofs(e_name.str().c_str());
         e_ofs.precision(8);
         e_gf.Save(e_ofs);
         e_ofs.close();
      }

      if (last_step || (ti % vis_steps) == 0)
      {
         double loc_norm = e_gf * e_gf, tot_norm;
         MPI_Allreduce(&loc_norm, &tot_norm, 1, MPI_DOUBLE, MPI_SUM,
                       pmesh->GetComm());
         if (mpi.Root())
         {
            cout << fixed;
            cout << "step " << setw(5) << ti
                 << ",\tt = " << setw(5) << setprecision(4) << t
                 << ",\tdt = " << setw(5) << setprecision(6) << dt
                 << ",\t|e| = " << setprecision(10)
                 << sqrt(tot_norm) << endl;
         }

         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization || visit) { oper.ComputeDensity(rho_gf); }
         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets

            miniapps::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                     "Density", Wx, Wy, Ww, Wh);
            Wx += offx;
            miniapps::VisualizeField(vis_v, vishost, visport,
                                     v_gf, "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            miniapps::VisualizeField(vis_e, vishost, visport, e_gf,
                                     "Specific Internal Energy", Wx, Wy, Ww,Wh);
            Wx += offx;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();
         }
      }
   }
   double rho_max, rho_min;
   if (visualization || visit)
   {
      double rho_locmax = rho_gf.Max(), rho_locmin = rho_gf.Min();
      MPI_Allreduce(&rho_locmin, &rho_min, 1, MPI_DOUBLE, MPI_MIN,
         pmesh->GetComm());
      MPI_Allreduce(&rho_locmax, &rho_max, 1, MPI_DOUBLE, MPI_MAX,
         pmesh->GetComm());
   }
   double e_locmax = e_gf.Max(), e_locmin = e_gf.Min(), e_max, e_min;
   MPI_Allreduce(&e_locmin, &e_min, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
   MPI_Allreduce(&e_locmax, &e_max, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
   double v_locmax = v_gf.Max(), v_locmin = v_gf.Min(), v_max, v_min;
   MPI_Allreduce(&v_locmin, &v_min, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());
   MPI_Allreduce(&v_locmax, &v_max, 1, MPI_DOUBLE, MPI_MAX, pmesh->GetComm());
   if (mpi.Root())
   {
      cout << fixed;
      if (visualization || visit)
      {
         cout << "min(rho) = " << setprecision(10) << rho_min
            << ",\tmax(rho) = " << setprecision(10) << rho_max
            << endl;
      }
      cout << "min(v)   = " << setprecision(10) << v_min
         << ",\tmax(v)   = " << setprecision(10) << v_max
         << endl;
      cout << "min(e)   = " << setprecision(10) << e_min
         << ",\tmax(e)   = " << setprecision(10) << e_max
         << endl;
   }
   if (visualization)
   {
      vis_v.close();
      vis_e.close();
   }

   // Free the used memory.
   delete ode_solver;
   delete pmesh;
   delete material_pcf;

   return 0;
}

namespace mfem
{

namespace hydrodynamics
{

double rho0(const Vector &x)
{
   switch (problem)
   {
      case 0: return 1.0;
      case 1: return 1.0;
      case 2: if (x(0) < 0.5) { return 1.0; }
         else return 0.1;
      case 3: if (x(0) > 1.0 && x(1) <= 1.5) { return 1.0; }
         else return 0.125;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

double gamma(const Vector &x)
{
   switch (problem)
   {
      case 0: return 5./3.;
      case 1: return 1.4;
      case 2: return 1.4;
      case 3: if (x(0) > 1.0 && x(1) <= 1.5) { return 1.4; }
         else return 1.5;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

void v0(const Vector &x, Vector &v)
{
   switch (problem)
   {
      case 0:
         v(0) =  sin(M_PI*x(0)) * cos(M_PI*x(1));
         v(1) = -cos(M_PI*x(0)) * sin(M_PI*x(1));
         if (x.Size() == 3)
         {
            v(0) *= cos(M_PI*x(2));
            v(1) *= cos(M_PI*x(2));
            v(2) = 0.0;
         }
         break;
      case 1: v = 0.0; break;
      case 2: v = 0.0; break;
      case 3: v = 0.0; break;
      default: MFEM_ABORT("Bad number given for problem id!");
   }
}

double e0(const Vector &x)
{
   switch (problem)
   {
      case 0:
      {
         const double denom = 2.0 / 3.0;  // (5/3 - 1) * density.
         double val;
         if (x.Size() == 2)
         {
            val = 1.0 + (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) / 4.0;
         }
         else
         {
            val = 100.0 + ((cos(2*M_PI*x(2)) + 2) *
                           (cos(2*M_PI*x(0)) + cos(2*M_PI*x(1))) - 2) / 16.0;
         }
         return val/denom;
      }
      case 1: return 0.0; // This case in initialized in main().
      case 2: if (x(0) < 0.5) { return 1.0 / rho0(x) / (gamma(x) - 1.0); }
         else return 0.1 / rho0(x) / (gamma(x) - 1.0);
      case 3: if (x(0) > 1.0) { return 0.1 / rho0(x) / (gamma(x) - 1.0); }
         else return 1.0 / rho0(x) / (gamma(x) - 1.0);
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

} // namespace hydrodynamics

} // namespace mfem

void display_banner(ostream & os)
{
os << endl
<< "    __   __ _______ __   __    __                __                "<< endl
<< "   /    / //__  __// /  / /   / /   ____  ____  / /_  ____  _____  "<< endl
<< "  / /| / /   / /  / /__/ /   / /   / __ `/ __ `/ __ \\/ __ \\/ ___/"<< endl
<< " / / |  /   / /  / ___  /   / /___/ /_/ / /_/ / / / / /_/ (__  )   "<< endl
<< "/_/  |_/   /_/  /_/  /_/   /_____/\\__,_/\\__, /_/ /_/\\____/____/ "<< endl
<< "                                       /____/                      "<< endl
<< endl;
}
