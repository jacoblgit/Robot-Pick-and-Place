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
    
    def set_target_ctrl(self, target, T, dt=None):
        """ 
        Generates a straight line trajectory in joint space with quintic time scaling. 
        Assumes 0 start and end velocity and acceleration.
        
        target: target generalized joint position
        T: time to reach target
        dt: time step for trajectory
        returns: robot state at each step in trajectory
        """
        if dt is None: dt = self.sim.dt

        if len(target) != self.sim.nq_panda: 
            raise ValueError(f"target of length {len(target)}, not expected {self.sim.nq_panda}")

        q_start= self.sim.get_robot_joint_state()
        t_list = np.arange(0.,T+dt,dt)
        n_steps = len(t_list)

        # equations are for quintic polynomial with 0 start and end velocity and acceleration
        q      = [(target - q_start) * (10*t_list**3/T**3 - 15*t_list**4/T**4 + 6*t_list**5/T**5)[i] \
                  + q_start  for i in range(n_steps)]
        qdot   = [(target - q_start) * (30*t_list**2/T**3 - 60*t_list**3/T**4 + 30*t_list**4/T**5)[i] \
                            for i in range(n_steps)]
        qddot  = [(target - q_start) * (60*t_list/T**3 - 180*t_list**2/T**4 + 120*t_list**3/T**5)[i]  \
                            for i in range(n_steps)] 
        
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