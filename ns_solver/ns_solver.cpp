/*
 * 2D Lid-Driven Cavity: Incompressible Navier-Stokes
 * ====================================================
 * Exact C++ port of Barba's "12 Steps to NS" (Step 12).
 * Uses:
 *   - Backward difference (upwind) for advection
 *   - Central difference for diffusion
 *   - Fixed-iteration Poisson for pressure
 *   - Neumann BC for pressure, pinned at corner
 *
 * Build:  g++ -O3 -o ns_solver/ns_solver.exe ns_solver/ns_solver.cpp
 * (or with OpenMP): g++ -O3 -fopenmp -o ns_solver/ns_solver.exe ns_solver/ns_solver.cpp
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>

struct F2D {
    int nx, ny;
    std::vector<double> d;
    F2D(){}
    F2D(int r, int c) : nx(c), ny(r), d(r*c, 0.0) {}
    double& at(int i, int j) { return d[i*nx+j]; }
    double  at(int i, int j) const { return d[i*nx+j]; }
    void copyFrom(const F2D& o) { d = o.d; }
};

class CavitySolver {
    int NX, NY;
    double Ulid, dx, dy, dt, rho, nu;
    int nit;
    F2D u, v, p, b;
    F2D un, vn, pn;

    void vel_bc() {
        for(int j=0;j<NX;j++){u.at(0,j)=0; v.at(0,j)=0; u.at(NY-1,j)=Ulid; v.at(NY-1,j)=0;}
        for(int i=0;i<NY;i++){u.at(i,0)=0; v.at(i,0)=0; u.at(i,NX-1)=0; v.at(i,NX-1)=0;}
    }
    void p_bc() {
        for(int j=0;j<NX;j++){p.at(NY-1,j)=p.at(NY-2,j); p.at(0,j)=p.at(1,j);}
        for(int i=0;i<NY;i++){p.at(i,0)=p.at(i,1); p.at(i,NX-1)=p.at(i,NX-2);}
    }

public:
    CavitySolver(int n, double re, double ulid, int poisson_nit=50)
        : NX(n), NY(n), Ulid(ulid), nit(poisson_nit)
    {
        dx = dy = 1.0/(n-1);  // domain [0,1]x[0,1]
        rho = 1.0;
        nu = 1.0/re;
        // Stable CFL
        double dt_a = 0.25*dx/std::max(fabs(ulid),0.01);
        double dt_d = 0.2*dx*dx/nu;
        dt = std::min({dt_a, dt_d, 0.001});  // Cap at 0.001 (Barba reference)
        
        u=F2D(NY,NX); v=F2D(NY,NX); p=F2D(NY,NX); b=F2D(NY,NX);
        un=F2D(NY,NX); vn=F2D(NY,NX); pn=F2D(NY,NX);
        for(int j=0;j<NX;j++) u.at(NY-1,j) = ulid;
    }

    void build_rhs() {
        for(int i=1;i<NY-1;i++) for(int j=1;j<NX-1;j++){
            double dudx = (u.at(i,j+1)-u.at(i,j-1))/(2*dx);
            double dvdy = (v.at(i+1,j)-v.at(i-1,j))/(2*dy);
            double dudy = (u.at(i+1,j)-u.at(i-1,j))/(2*dy);
            double dvdx = (v.at(i,j+1)-v.at(i,j-1))/(2*dx);
            
            b.at(i,j) = rho * (1.0/dt * (dudx + dvdy)
                              - dudx*dudx
                              - 2.0*dudy*dvdx
                              - dvdy*dvdy);
        }
    }

    void pressure_poisson() {
        double dx2=dx*dx, dy2=dy*dy;
        for(int it=0;it<nit;it++){
            pn.copyFrom(p);
            for(int i=1;i<NY-1;i++) for(int j=1;j<NX-1;j++){
                p.at(i,j) = ((pn.at(i,j+1)+pn.at(i,j-1))*dy2
                            +(pn.at(i+1,j)+pn.at(i-1,j))*dx2
                            - b.at(i,j)*dx2*dy2)
                           / (2.0*(dx2+dy2));
            }
            p_bc();
        }
    }

    // Velocity update: EXACT Barba formulation with backward (upwind) advection
    void cavity_step() {
        un.copyFrom(u); vn.copyFrom(v);
        
        for(int i=1;i<NY-1;i++) for(int j=1;j<NX-1;j++){
            // Backward diff advection (upwind)
            double adv_u = un.at(i,j)*(un.at(i,j)-un.at(i,j-1))/dx
                         + vn.at(i,j)*(un.at(i,j)-un.at(i-1,j))/dy;
            // Diffusion (central)
            double diff_u = nu*((un.at(i,j+1)-2*un.at(i,j)+un.at(i,j-1))/(dx*dx)
                               +(un.at(i+1,j)-2*un.at(i,j)+un.at(i-1,j))/(dy*dy));
            // Pressure gradient
            double dpdx = (p.at(i,j+1)-p.at(i,j-1))/(2*dx*rho);
            
            u.at(i,j) = un.at(i,j) + dt*(-adv_u - dpdx + diff_u);
            
            // v-velocity
            double adv_v = un.at(i,j)*(vn.at(i,j)-vn.at(i,j-1))/dx
                         + vn.at(i,j)*(vn.at(i,j)-vn.at(i-1,j))/dy;
            double diff_v = nu*((vn.at(i,j+1)-2*vn.at(i,j)+vn.at(i,j-1))/(dx*dx)
                               +(vn.at(i+1,j)-2*vn.at(i,j)+vn.at(i-1,j))/(dy*dy));
            double dpdy = (p.at(i+1,j)-p.at(i-1,j))/(2*dy*rho);
            
            v.at(i,j) = vn.at(i,j) + dt*(-adv_v - dpdy + diff_v);
        }
        vel_bc();
    }

    void step() {
        build_rhs();
        pressure_poisson();
        cavity_step();
    }

    double div_max() {
        double mx=0;
        for(int i=1;i<NY-1;i++) for(int j=1;j<NX-1;j++){
            double d = (u.at(i,j+1)-u.at(i,j-1))/(2*dx)
                      +(v.at(i+1,j)-v.at(i-1,j))/(2*dy);
            mx = std::max(mx, fabs(d));
        }
        return mx;
    }

    double KE() {
        double s=0;
        for(int i=0;i<NY;i++) for(int j=0;j<NX;j++)
            s += u.at(i,j)*u.at(i,j)+v.at(i,j)*v.at(i,j);
        return 0.5*s*dx*dy;
    }

    double u_center_y() {
        // u-velocity at center column (like Ghia benchmark)
        int j=NX/2;
        double vmax=0;
        for(int i=0;i<NY;i++) vmax=std::max(vmax,fabs(u.at(i,j)));
        return vmax;
    }

    void run(int nsteps, int prt=100) {
        for(int s=0;s<nsteps;s++){
            step();
            if(prt>0 && (s+1)%prt==0)
                printf("  Step %4d: div=%.2e KE=%.6f\n", s+1, div_max(), KE());
        }
    }

    void save(const char* fn) {
        FILE* f=fopen(fn,"wb"); if(!f) return;
        fwrite(&NY,4,1,f); fwrite(&NX,4,1,f);
        fwrite(u.d.data(),8,NY*NX,f);
        fwrite(v.d.data(),8,NY*NX,f);
        fwrite(p.d.data(),8,NY*NX,f);
        fclose(f);
    }

    double getDt() const { return dt; }
    double getDx() const { return dx; }
};

int main(int argc, char* argv[]) {
    printf("2D Lid-Driven Cavity (Navier-Stokes)\n");

    if(argc == 1) {
        int N=41, nsteps=1000, nit=200;
        printf("\n=== VALIDATION: Re=100, %dx%d, %d steps, %d Poisson/step ===\n",N,N,nsteps,nit);
        CavitySolver sol(N,100.0,1.0,nit);
        printf("dt=%.6f dx=%.6f\n\n",sol.getDt(),sol.getDx());
        
        clock_t t0=clock();
        sol.run(nsteps,100);
        double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
        
        printf("\n--- RESULT ---\n");
        printf("  Time: %.2fs\n",el);
        printf("  div=%.2e  KE=%.6f\n",sol.div_max(),sol.KE());
        printf("  u_center=%.4f\n",sol.u_center_y());
        return 0;
    }

    if(strcmp(argv[1],"sweep")==0 && argc>=6){
        double v0=atof(argv[2]),v1=atof(argv[3]);
        int nc=atoi(argv[4]),ns=atoi(argv[5]);
        int N=argc>=7?atoi(argv[6]):41;
        int nit=argc>=8?atoi(argv[7]):50;
        
        printf("\nSweep: v=[%.3f,%.3f] %d cases, %dx%d, %d steps, %d nit\n",
               v0,v1,nc,N,N,ns,nit);
        
        #ifdef _WIN32
        system("mkdir ns_solver\\data 2>nul");
        #else
        system("mkdir -p ns_solver/data");
        #endif
        
        double tt=0, wd=0;
        for(int c=0;c<nc;c++){
            double lv=nc==1?v0:v0+c*(v1-v0)/(nc-1);
            CavitySolver sol(N,100.0,lv,nit);
            clock_t t0=clock();
            sol.run(ns,ns+1);
            double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
            double dv=sol.div_max();
            tt+=el; wd=std::max(wd,dv);
            printf("  %3d/%d v=%.3f div=%.2e KE=%.6f t=%.2fs\n",c+1,nc,lv,dv,sol.KE(),el);
            char fn[256]; snprintf(fn,256,"ns_solver/data/state_v%.3f.bin",lv);
            sol.save(fn);
        }
        printf("\nTotal=%.2fs WorstDiv=%.2e\n",tt,wd);
        return 0;
    }

    printf("Usage: ns_solver | ns_solver sweep V0 V1 N STEPS [NX] [NIT]\n");
    return 1;
}
