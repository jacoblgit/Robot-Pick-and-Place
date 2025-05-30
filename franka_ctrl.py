import numpy as np
from franka_simulator import Franka_Simulator
import modern_robotics as mr


class Franka_TargetController(): 
    def __init__(self, model: Franka_Simulator):
        self.sim = model

    def get_ctrl(self, qddot_des: np.ndarray) -> np.ndarray:
        """
        Computes the torques necessary to achieve desired accelerations, using dynamics.

        qddot_des: desired generalized accelerations (7 arm and 1 of gripper joints)
        tau: generalized joint torques (7 arm and 1 of gripper joints)
        """
        if len(qddot_des) != self.sim.nq_panda:
            raise ValueError(f"qddot_des of length {len(qddot_des)}, not expected {self.sim.nq_panda}")
        
        M       = self.sim.get_mass_matrix()        # M(q)
        h       = self.sim.get_h_bias()             # h(q, qdot)  
        tau     = M @ qddot_des + h                 # formula 8.1 in Lynch
        return tau
    
    def set_target_ctrl(self, target, T=None, dt=None):
        """ 
        Generates a straight line trajectory in joint space with quintic time scaling. 
        Assumes 0 start and end velocity and acceleration.
        
        target: target generalized joint position
        T: time to reach target. if None, compute optimal
        dt: time step for trajectory
        returns: robot state at each step in trajectory
        """
        if len(target) != self.sim.nq_panda: 
            raise ValueError(f"target of length {len(target)}, not expected {self.sim.nq_panda}")
        if type(T) != float and T is not None: 
            raise ValueError(f"T of type {type(T)}, not expected float")
        if type(dt) != float and dt is not None: 
            raise ValueError(f"dt of type {type(dt)}, not expected float")  
        
        if dt is None: dt = self.sim.dt
        q_start= self.sim.get_robot_joint_state()
        qdiff = target - q_start
        
        
        if T is None:
            safety_factor = 0.05

            # see frankaemika.github.io/docs/control_parameters
            qdotlimit = 2.175 * safety_factor  # rad/s
            qddotlimit = 7.5  * safety_factor  # rad/s^2

            # compute min time to reach target with quintic time scaling while respecting joint limits
            T = 0

            maxdiff = np.max(np.abs(qdiff))
            if maxdiff == 0: maxdiff = 1e-3
            # max joint velocity is 15*qdiff/(8T), occuring at t=T/2 (qddot=0)
            T = max(T, (15*maxdiff)/(8*qdotlimit))  
            # max joint accel is 10*qdiff/(sqrt(3)*T^2), occuring at t=1/6(3-sqrt(3))T (qdddot=0)
            T = max(T, np.sqrt((10 * maxdiff)/(np.sqrt(3)*qddotlimit)))  
            print(f"Optimal T: {T:.2f} sec")
                            
        t_list = np.arange(0.,T+dt,dt)
        n_steps = len(t_list)

        # precompute for efficiency
        t_list2 = t_list**2
        t_list3 = t_list**3
        t_list4 = t_list**4
        t_list5 = t_list**5

        # equations are for straight line through joint space with quintic polynomial time scaling
        # polynomial coefficients set for 0 start and end velocity/acceleration to reduce vibrations
        # numpy broadcasting used for computational efficiency

        # generalized coords form convex set so every point on line between valid points is valid
        s_values = (10*t_list3/T**3 - 15*t_list4/T**4 + 6*t_list5/T**5)
        q = s_values[:, np.newaxis] * qdiff + q_start   

        sdot_values = (30*t_list2/T**3 - 60*t_list3/T**4 + 30*t_list4/T**5)
        qdot = sdot_values[:, np.newaxis] * qdiff

        sddot_values = (60*t_list/T**3 - 180*t_list2/T**4 + 120*t_list3/T**5)
        qddot = sddot_values[:, np.newaxis] * qdiff
        
        return t_list, n_steps, q, qdot, qddot
    
    def pd_ctrl(self, q_des, qdot_des, qddot_des,Kp=100., Kd=20.):
        """
        PD Feedback Controller: Input target joint position, velocity and acceleration, outputs desired joint accelerations
        """
        
        if len(q_des) != self.sim.nq_panda or len(qdot_des) != self.sim.nq_panda or len(qddot_des) != self.sim.nq_panda:
            raise ValueError(f"q_des, qdot_des, and qddot_des must be of length {self.sim.nq_panda}")

        task_err = q_des - self.sim.get_robot_joint_state()  
        qddot_des = Kp*(q_des-self.sim.get_robot_joint_state()) + Kd*(qdot_des - self.sim.get_robot_jointvel_state()) + qddot_des
        return (qddot_des, task_err)

    def compute_IK(self, Tsd, sim):
        """ 
        Uses Newton-Raphson method for inverse kinematics to numerically compute the 
        generalized coordinates that will achieve the desired end-effector pose.

        input:  Tsd: desired pose T in SE(3) of hand wrt space frame
                sim: simulation object
        output: q_new: new joint position that achieves the desired end-effector pose
        """

        # Newton-Raphson parameters
        acceptable_error = 1e-3
        max_iter = 50
        err = True
        iter_count = 0

        # initialization
        q0 = sim.get_robot_joint_state()                    # current joint position in generalized coords
        qguess = q0                                         # best guess at a joint pose for Tsd

        # iterative step
        while err and (iter_count < max_iter):
            sim.set_robot_joint_state(qguess)               # set robot to guess and use mujoco FK

            Tsb = sim.get_robot_body_state("hand")          # pose of hand wrt space at guess q                 
            Tbd = mr.TransInv(Tsb) @ Tsd                    # desired pose in body frame
            Vb = mr.se3ToVec(mr.MatrixLog6(Tbd))            # desired body twist (vector form)
            JacS = sim.get_robot_ee_spatial_jacobian()      # end-effector spatial Jacobian  
            JacB = mr.Adjoint(mr.TransInv(Tsb)) @ JacS      # end-effector body Jacobian       
            
            qguess = qguess + (np.linalg.pinv(JacB) @ Vb)   # see section 6.2.2 in Lynch 
            iter_count += 1
            err = (np.linalg.norm(Vb) > acceptable_error)

        # reset robot 
        sim.set_robot_joint_state(q0)

        return (qguess, err)