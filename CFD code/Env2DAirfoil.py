import numpy as np
from fenics import *
from mshr import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import *


class Env2DAirfoil:
    def naca0012(self, x, chord=self.D):
        return 0.6 * (-0.1015 * x ** 4 + 0.2843 * x ** 3 - 0.3576 * x ** 2 - 0.1221 * x + 0.2969 * np.sqrt(x))

    def naca0015(self, x, chord=sefl.D):
        return 0.6 * (-0.0644 * x ** 4 + 0.2726 * x ** 3 - 0.3576 * x ** 2 - 0.1270 * x + 0.2983 * np.sqrt(x))

    def __init__(self, Re=2500, attack_angle=pi / 7.2, probes_mode=0, probe_locations_mode=0, save_data=1,
                 save_fre=400):

        self.xdmffile_u = XDMFFile('Airfoil_Re2500/velocity.xdmf')
        self.xdmffile_p = XDMFFile('Airfoil_Re2500/pressure.xdmf')

        self.timeseries_u = TimeSeries('Airfoil_Re2500/velocity_series')
        self.timeseries_p = TimeSeries('Airfoil_Re2500/pressure_series')

        self.mesh_for_airfoil = File('Airfoil_Re2500/cylinder.xml.gz')

        self.mesh_index = 200
        self.T = 10
        self.num_steps = 100000
        self.dt = self.T / self.num_steps
        self.D = 1.0
        self.Re = Re
        self.U_m = 0.45
        self.mu = 2 * self.U_m * self.D / (3 * self.Re)
        self.rho = 1
        self.Q0 = 0
        self.Q_total_n = 0  # the total mass flow rate of all the jets at time step n, calculated by self.update_jetBCs
        self.chord = self.D
        self.attack_angle = attack_angle
        self.num_points = 100
        self.n_outer_iterations = 3
        self.t = 0
        self.n = 0
        self.probes_mode = probes_mode
        self.probe_locations_mode = probe_locations_mode
        self.length = 16  # the number of observation points in the x direction
        self.width = 12  # the number of observation points in the y direction
        self.save_data = save_data  # if 1, save the data; if 0, do not save
        self.save_fre = save_fre  # the frequency of saving data, in terms of time steps.

        self.locations = [] #locations of the observation points
        self.jet_locations = [] #exact locations of the jets
        self.probes = []
        self.recall_step = 40
        self.probes_num = 192
        self.state_matrix = np.zeros((self.recall_step, self.probes_num * 3))

        self.jet1_location = 0.2
        self.jet2_location = 0.4
        self.jet3_location = 0.6
        self.jet_width_rate = 0.01  # Real width of jet = 2*self.jet_width_rate*self.chord

        self.jet_locations.append(Point((self.jet1_location,self.naca0012(self.jet1_location))))
        self.jet_locations.append(Point((self.jet2_location,self.naca0012(self.jet2_location))))
        self.jet_locations.append(Point((self.jet3_location,self.naca0012(self.jet3_location))))

        self.Cd_max = 0
        self.Cl_max = 0
        self.x = range(50, self.num_steps, 50)
        self.y = np.ones(int(self.num_steps / 50))
        self.z = np.ones(int(self.num_steps / 50))

        self.mem_episode = 1000
        self.mem_state = []

        self.avg_drag_len = 25
        self.drag_list = [0] * self.avg_drag_len
        self.avg_lift_len = 25
        self.lift_list = [0] * self.avg_lift_len

        self.drag_mem = [0] * self.avg_drag_len
        self.lift_mem = [0] * self.avg_lift_len

        # Create airfoil
        self.x_coords = np.linspace(0, self.chord, self.num_points)
        self.airfoil_points = [Point(x, self.naca0012(x, self.chord)) for x in self.x_coords]
        self.airfoil_points = sorted(self.airfoil_points, key=lambda p: atan2(p.y(), p.x() - 0.05))
        lower_points = [Point(p.x(), -p.y()) for p in self.airfoil_points]
        self.airfoil_points += lower_points
        self.airfoil_points = sorted(self.airfoil_points, key=lambda p: atan2(p.y(), p.x() - 0.05))

        # Rotate the airfoil according to the self.attack_angle
        rotation_matrix = np.array(
            [[cos(-self.attack_angle), -sin(-self.attack_angle)], [sin(-self.attack_angle), cos(-self.attack_angle)]])
        rotated_airfoil = [Point(np.dot(rotation_matrix, (p.x(), p.y()))) for p in self.airfoil_points]
        self.jet_locations = [Point(np.dot(rotation_matrix, (p.x(), p.y()))) for p in self.jet_locations]
        self.jet_locations = [(p.x(), p.y()) for p in self.jet_locations]
        self.airfoil0012 = Polygon(rotated_airfoil)
        self.channel = Rectangle(Point(-0.5 * self.D, -0.7 * self.D), Point(3 * self.D, 0.7 * self.D))
        self.domain = self.channel - self.airfoil0012
        mesh = generate_mesh(self.domain, self.mesh_index)

        # Refine the mesh
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        origin = Point(2 * self.D, 2 * self.D)
        for cell in cells(mesh):
            p = cell.midpoint()
            for q in rotated_airfoil:
                if p.distance(q) < 0.08:
                    cell_markers[cell] = True
        mesh = refine(mesh, cell_markers, redistribute=True)
        self.mesh = mesh
        self.mesh_for_airfoil << self.mesh

        self.V = VectorFunctionSpace(self.mesh, 'P', 2)
        self.Q = FunctionSpace(self.mesh, 'P', 1)

        # Define boundaries
        self.inflow = 'near(x[0],-0.5)'
        self.outflow = 'near(x[0],3)'
        self.walls = 'near(x[1],-0.7)||near(x[1],0.7)'
        self.airfoil = 'on_boundary && x[0]>-0.1 && x[0]<1.1 && x[1] >-0.6&& x[1] <0.3'

        # self.jet1 = 'on_boundary && x[0]>({0}*{1}) && x[0]<({0}*{1}+0.03*{1}) && x[1]>0 && x[1]<(0.5*{1})'.format(self.jet1_location,self.D)
        self.jet1 = 'on_boundary && x[0]>({0}*{1}-{2}*{1}) && x[0]<({0}*{1}+{2}*{1}) '.format(self.jet1_location,
                                                                                              self.D * cos(
                                                                                                  self.attack_angle),
                                                                                              self.jet_width_rate * cos(
                                                                                                  self.attack_angle))
        self.jet2 = 'on_boundary && x[0]>({0}*{1}-{2}*{1}) && x[0]<({0}*{1}+{2}*{1}) '.format(self.jet2_location,
                                                                                              self.D * cos(
                                                                                                  self.attack_angle),
                                                                                              self.jet_width_rate * cos(
                                                                                                  self.attack_angle))
        self.jet3 = 'on_boundary && x[0]>({0}*{1}-{2}*{1}) && x[0]<({0}*{1}+{2}*{1}) '.format(self.jet3_location,
                                                                                              self.D * cos(
                                                                                                  self.attack_angle),
                                                                                              self.jet_width_rate * cos(
                                                                                                  self.attack_angle))

        # Inflow profile
        self.inflow_profile = ('4.0*(U_m)*(0.7*D+x[1])*(0.7*D-x[1])/pow(1.4,2)', '0')
        # inflow_profile = ('1.0','0')
        self.inflow_f = Expression(self.inflow_profile, U_m=Constant(0.45), D=Constant(1), degree=2)

        # Jet profile. Jet1 is at the top of the cylinder, and Jet2 is at the bottom of the cylinder.
        # "radius" here refers to the chord length of the airfoil.

        self.jet1_f = Expression((
            'cos(atan2(-1,(0.6*(-0.406*pow(x[0]/cos(attack_angle),3)+0.8529*pow(x[0]/cos('
            'attack_angle),2)-0.7152*x[0]/cos(attack_angle)-0.1221+0.14845*pow(x[0]/cos('
            'attack_angle),-0.5))))-attack_angle)*cos((x[0]-x_center)/(width*cos('
            'attack_angle)))*Qjet*pi/(2*width*pow(radius,2))', \
            'sin(atan2(-1,(0.6*(-0.406*pow(x[0]/cos(attack_angle),3)+0.8529*pow(x[0]/cos('
            'attack_angle),2)-0.7152*x[0]/cos(attack_angle)-0.1221+0.14845*pow(x[0]/cos('
            'attack_angle),-0.5))))-attack_angle)*cos((x[0]-x_center)/(width*cos('
            'attack_angle)))*Qjet*pi/(2*width*pow(radius,2))'), \
            Qjet=0, width=0.01, attack_angle=0, x_center=0.2, radius=Constant(1), degree=2)

        self.jet2_f = Expression((
            'cos(atan2(-1,(0.6*(-0.406*pow(x[0]/cos(attack_angle),3)+0.8529*pow(x[0]/cos('
            'attack_angle),2)-0.7152*x[0]/cos(attack_angle)-0.1221+0.14845*pow(x[0]/cos('
            'attack_angle),-0.5))))-attack_angle)*cos((x[0]-x_center)/(width*cos('
            'attack_angle)))*Qjet*pi/(2*width*pow(radius,2))', \
            'sin(atan2(-1,(0.6*(-0.406*pow(x[0]/cos(attack_angle),3)+0.8529*pow(x[0]/cos('
            'attack_angle),2)-0.7152*x[0]/cos(attack_angle)-0.1221+0.14845*pow(x[0]/cos('
            'attack_angle),-0.5))))-attack_angle)*cos((x[0]-x_center)/(width*cos('
            'attack_angle)))*Qjet*pi/(2*width*pow(radius,2))'), \
            Qjet=0, width=0.01, attack_angle=0, x_center=0.4, radius=Constant(1), degree=2)

        self.jet3_f = Expression((
            'cos(atan2(-1,(0.6*(-0.406*pow(x[0]/cos(attack_angle),3)+0.8529*pow(x[0]/cos('
            'attack_angle),2)-0.7152*x[0]/cos(attack_angle)-0.1221+0.14845*pow(x[0]/cos('
            'attack_angle),-0.5))))-attack_angle)*cos((x[0]-x_center)/(width*cos('
            'attack_angle)))*Qjet*pi/(2*width*pow(radius,2))', \
            'sin(atan2(-1,(0.6*(-0.406*pow(x[0]/cos(attack_angle),3)+0.8529*pow(x[0]/cos('
            'attack_angle),2)-0.7152*x[0]/cos(attack_angle)-0.1221+0.14845*pow(x[0]/cos('
            'attack_angle),-0.5))))-attack_angle)*cos((x[0]-x_center)/(width*cos('
            'attack_angle)))*Qjet*pi/(2*width*pow(radius,2))'), \
            Qjet=0, width=0.01, attack_angle=0, x_center=0.6, radius=Constant(1), degree=2)

        # Adjust the parameters in the Expressions for the jets according to the initial settings.
        self.jet1_f.width = self.jet_width_rate
        self.jet2_f.width = self.jet_width_rate
        self.jet3_f.width = self.jet_width_rate

        self.jet1_f.attack_angle = self.attack_angle
        self.jet2_f.attack_angle = self.attack_angle
        self.jet3_f.attack_angle = self.attack_angle

        self.jet1_f.x_center = self.jet1_location
        self.jet2_f.x_center = self.jet2_location
        self.jet3_f.x_center = self.jet3_location

        # boundary conditions.
        self.bcu_inflow = DirichletBC(self.V, self.inflow_f, self.inflow)
        self.bcu_walls = DirichletBC(self.V, Constant((0, 0)), self.walls)
        self.bcu_airfoil = DirichletBC(self.V, Constant((0, 0)), self.airfoil)
        self.bcp_outflow = DirichletBC(self.Q, Constant(0), self.outflow)
        self.bcp = [self.bcp_outflow]
        self.bcu_jet1 = DirichletBC(self.V, self.jet1_f, self.jet1)
        self.bcu_jet2 = DirichletBC(self.V, self.jet2_f, self.jet2)
        self.bcu_jet3 = DirichletBC(self.V, self.jet3_f, self.jet3)

        self.bcu = [self.bcu_inflow, self.bcu_walls, self.bcu_airfoil, self.bcu_jet1, self.bcu_jet2, self.bcu_jet3]

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        self.p = TrialFunction(self.Q)
        self.q = TestFunction(self.Q)

        # Functions for solutions at previous and current time steps
        self.u_n = Function(self.V)
        self.u_ = Function(self.V)
        self.p_n = Function(self.Q)
        self.p_ = Function(self.Q)

        # Expressions used in variational forms
        self.U = 0.5 * (self.u_n + self.u)
        n = -FacetNormal(self.mesh)
        self.f = Constant((0, 0))
        f = self.f
        k = Constant(self.dt)
        mu = Constant(self.mu)

        # Variational problem for step 1
        F1 = self.rho * dot((self.u - self.u_n) / k, self.v) * dx \
             + self.rho * dot(dot(self.u_n, nabla_grad(self.u_n)), self.v) * dx \
             + inner(self.sigma(self.U, self.p_n), self.epsilon(self.v)) * dx \
             + dot(self.p_n * n, self.v) * ds - dot(mu * nabla_grad(self.U) * n, self.v) * ds \
             - dot(f, self.v) * dx
        self.a1 = lhs(F1)
        self.L1 = rhs(F1)

        # Variational problem for step 2
        self.a2 = dot(nabla_grad(self.p), nabla_grad(self.q)) * dx
        self.L2 = dot(nabla_grad(self.p_n), nabla_grad(self.q)) * dx - (1 / k) * div(self.u_) * (self.q) * dx

        # Variational problem for step 3
        self.a3 = dot(self.u, self.v) * dx
        self.L3 = dot(self.u_, self.v) * dx - k * dot(nabla_grad(self.p_ - self.p_n), self.v) * dx

        # Assemble matrices
        self.A1 = assemble(self.a1)
        self.A2 = assemble(self.a2)
        self.A3 = assemble(self.a3)

        # Apply bcs to matrices
        [bc.apply(self.A1) for bc in self.bcu]
        [bc.apply(self.A2) for bc in self.bcp]

        self.observation_locations()

    def observation_locations(self):

        if self.probe_locations_mode == 0:
            for x in np.linspace(self.D, 2.5 * self.D, self.length):
                for y in np.linspace(0.4 * self.D, -0.4 * self.D, self.width):
                    self.locations.append((x, y))
                    # self.locations.append(x)
                    # self.locations.append(y)

        # update the boundary conditons related to the jets

    def update_jetBCs(self, new_Qjet1, new_Qjet2, new_Qjet3):
        self.jet1_f.Qjet = new_Qjet1
        self.jet2_f.Qjet = new_Qjet2
        self.jet3_f.Qjet = new_Qjet3
        self.bcu_jet1 = DirichletBC(self.V, self.jet1_f, self.jet1)
        self.bcu_jet2 = DirichletBC(self.V, self.jet2_f, self.jet2)
        self.bcu_jet3 = DirichletBC(self.V, self.jet3_f, self.jet3)
        self.bcu = [self.bcu_inflow, self.bcu_walls, self.bcu_airfoil, self.bcu_jet1, self.bcu_jet2, self.bcu_jet3]

        # Apply bcs to matrices
        [bc.apply(self.A1) for bc in self.bcu]
        [bc.apply(self.A2) for bc in self.bcp]

        return (new_Qjet1 + new_Qjet2 + new_Qjet3)

    def epsilon(self, u):
        return sym(nabla_grad(u))

        # Stress tensor

    def sigma(self, u, p):
        return 2 * self.mu * self.epsilon(u) - p * Identity(len(u))

    def compute_drag_lift_coefficients(self, u, p):
        # Define normal vector along the cylinder surface
        n = FacetNormal(self.mesh)
        #     stress_tensor=sigma(u,p_n)
        stress_tensor = self.sigma(u, p)

        boundary_parts = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        boundary_parts.set_all(0)

        class CylinderBoundary(SubDomain):
            def inside(self, x, on_boundary):
                tol = 1E-14
                self.D = 1  ### needs to be modified later on
                return on_boundary and x[0] > -0.1 * self.D and x[0] < 1.6 * self.D and x[1] > -0.6 * self.D and x[
                    1] < 0.3 * self.D

        Gamma_1 = CylinderBoundary()
        Gamma_1.mark(boundary_parts, 1)

        ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_parts, subdomain_id=1)

        force = dot(stress_tensor, n)
        drag_force = assemble(force[0] * ds)
        lift_force = assemble(force[1] * ds)
        # Compute drag and lift coefficients
        drag_coefficient = abs(2 * drag_force / (self.rho * 1.0 * self.D))
        lift_coefficient = abs(2 * lift_force / (self.rho * 1.0 * self.D))

        return drag_coefficient, lift_coefficient

        # progress=Progress('Time-stepping')
        # set_log_level(PROGRESS)

    def probes_vp(self):
        self.probes = []
        for p in self.locations:
            self.probes.append(self.u_((p[0], p[1]))[0])
            self.probes.append(self.u_((p[0], p[1]))[1])
            self.probes.append(self.p_((p[0], p[1])))

        self.state_matrix = np.concatenate((self.state_matrix[1:], [self.probes]), axis=0)

        # self.probes=evals
        # self.nprobes=len(locations)

        if self.probes_mode == 0:
            self.probes_num = len(self.locations)
            return self.probes
        if self.probes_mode == 1:
            return self.state_matrix

    def get_reward(self, Cd, Cl, mode=0):  ############# more modes might be added later on
        return -Cd - 0.2 * Cl + 5  # plus the average drag without control

    def evolve(self, Q1=0, Q2=0, Q3=0,plot_p_field=0,show_observation_points=0,plot_fre=400):
        # Time-stepping

        if self.n % 200 == 0:
            print("Temp num step:", self.n)

        self.Q_total = self.update_jetBCs(Q1, Q2, Q3)

        # for outer_iter in range(self.n_outer_iterations):

        self.t += self.dt
        self.n += 1
        # update the BCs with a Qjet number
        # update_jetBCs(3000

        # 1 Tentative velocity step
        self.b1 = assemble(self.L1)
        [bc.apply(self.b1) for bc in self.bcu]
        solve(self.A1, self.u_.vector(), self.b1, 'bicgstab', 'hypre_amg')

        # 2 pressure correction step
        self.b2 = assemble(self.L2)
        [bc.apply(self.b2) for bc in self.bcp]
        solve(self.A2, self.p_.vector(), self.b2, 'bicgstab', 'hypre_amg')

        # 3 Velocity correction step
        self.b3 = assemble(self.L3)
        solve(self.A3, self.u_.vector(), self.b3, 'cg', 'sor')

        self.drag_coefficient, self.lift_coefficient = self.compute_drag_lift_coefficients(self.u_, self.p_)
        self.drag_list.pop(0)
        self.drag_list.append(self.drag_coefficient)
        self.avg_drag = np.mean(self.drag_list)
        self.lift_list.pop(0)
        self.lift_list.append(self.lift_coefficient)
        self.avg_lift = np.mean(self.lift_list)

        if plot_p_field == 1 and self.n % plot_fre ==0:
            self.plot_p_field(show_observation_points)

        if self.save_data == 1 and self.n % self.save_fre == 0:
            self.xdmffile_u.write(self.u_, self.n * self.dt)
            self.xdmffile_p.write(self.p_, self.n * self.dt)
            self.timeseries_u.store(self.u_.vector(), self.n * self.dt)
            self.timeseries_p.store(self.p_.vector(), self.n * self.dt)

        self.u_n.assign(self.u_)
        self.p_n.assign(self.p_)

        probe_results = self.probes_vp()
        # done=self.n>self.num_steps
        # return s_,r,done
        return probe_results, self.get_reward(self.avg_drag, self.avg_lift), False

    def evolve_n(self, n, Q1=0, Q2=0, Q3=0,plot_p_field=0,show_observation_points=0,plot_fre=400):
        for i in range(n):
            self.evolve(Q1, Q2, Q3,plot_p_field,show_observation_points,plot_fre)

    def plot_p_field(self,show_observation_points=0):
               
        plt.clf()
        self.p_array = self.p_.compute_vertex_values(self.mesh)
        self.p_array = self.p_array.reshape((self.mesh.num_vertices(),))
        plt.figure(figsize=(10, 4))
        plt.tripcolor(self.mesh.coordinates()[:, 0], self.mesh.coordinates()[:, 1], self.mesh.cells(), self.p_array,
                      shading="gouraud", cmap='coolwarm')
        plt.colorbar()

         if show_observation_points == 1:
            x_coords = np.array(self.locations)[:, 0]
            y_coords = np.array(self.locations)[:, 1]
            plt.scatter(x_coords, y_coords, color='black', s=5)

            x1_coords = np.array(self.jet_locations)[:, 0]
            y1_coords = np.array(self.jet_locations)[:, 1]
            plt.scatter(x1_coords, y1_coords, color='navy', s=5)
             
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('P_field')

       
            
        # plt.savefig("Airfoil_Re2500/" + str(self.n / self.num_steps).zfill(6) + "Re2500" + ".png")
        if self.n % 200 == 0:
            plt.show()

    def update_plot_p_field(self):
        self.evolve()
        self.p_array = self.p_.compute_vertex_values(self.mesh)
        self.p_array = self.p_array.reshape((self.mesh.num_vertices(),))
        plt.clf()

        plt.tripcolor(self.mesh.coordinates()[:, 0], self.mesh.coordinates()[:, 1], self.mesh.cells(), self.p_array,
                      shading="gouraud")
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('P_field')

    def update(self, i):
        if self.n < i:
            self.evolve()
        self.update_plot_p_field()

    def generate_mp4(self):
        self.fig, self.ax = plt.subplots()
        ani = FuncAnimation(self.fig, self.update, frames=range(17000, self.num_steps), interval=0.5)
        ani.save('naca0012win.gif', writer='pillow', fps=400)
        plt.close(self.fig)

    def plot_mesh(self):
        # Plot the mesh
        fig = plt.figure(figsize=(160, 60), dpi=150)
        plot(self.mesh)
        plt.show()

    def generate_gif(self):
        self.__init__()

        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()

        def animate_Cd(i):
            self.update_pressure_field(i)
            ax1.clear()
            self.plot_Cd_curve(self.drag_list)

            plt.title(f'Cd of last {len(Cd_list)} numsteps')

        plt.xlabel('Time Step')
        plt.ylabel('Drag Coefficient')

        def animate(i):
            self.update_pressure_field(i)

            ax.clear()
            self.plot_p_field()

            plt.title(f'Time Step: {i}')
            plt.xlabel('X')
            plt.ylabel('Y')

            self.plot_Cd_curve(self.drag_list)

        anim = FuncAnimation(fig, animate, frames=self.num_steps, interval=0.2)
        anim_Cd = FuncAnimation(fig1, animate_Cd, frames=self.num_steps, interval=0.2)

        file_extension = 'gif'

        if file_extension == 'gif':
            anim.save('naca_gif_Re2500_1.gif', writer='pillow', fps=1200)
        elif file_extension == 'mp4':
            anim.save('naca_gif.mp4', writer='ffmpeg', fps=400)

        anim_Cd.save('naca_Cd.gif', writer='pillow', fps=1200)

    def update_pressure_field(self, i):  # used to help generate gif
        if self.n < i:
            self.evolve()
            self.n += 1

    def plot_Cd_curve(self, Cd_list):

        x = range(len(Cd_list))
        plt.plot(x, Cd_list)


env = Env2DAirfoil()
for i in range(env.num_steps):
    env.evolve(0)
# env.plot_p_field()
