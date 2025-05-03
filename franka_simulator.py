import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

class Franka_Simulator(): 

    def __init__(self, max_step=500, model_xml="franka_emika_panda/scene.xml", render=True,
                 init_state=[0.,0.,0.,-1.57,0.,1.57,0.,0.,0.,0.4, 0, 0.315, 0.9238795, 0., 0., 0.3826834]):
        

        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data  = mujoco.MjData(self.model)

        self.nq = self.model.nq         # num of generalized coords             (in order: 7 arm joints, 2 fingers, 3 box pos, 4 box quat
        self.nu_panda = self.model.nu   # num of generalized controls on panda  (in order: 7 arm joints, 1 gripper (fingers share actuator))
        self.nq_panda = self.nu_panda   # num of generalized coords   on panda  (in order: 7 arm joints, 1 gripper (use 1 coord for 2 fingers))
        
        # Set Sim Parameters
        self.t = 0.
        self.dt = self.model.opt.timestep
        self.stepcount = 0
        self.max_step = max_step
        self.render_step = 20

        # Set Initial State
        # print("Initial State: ", init_state)
        self.data.qpos = init_state
        mujoco.mj_forward(self.model, self.data)

        # Set up rendering
        self.render = render
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync() 

    def step(self, ctrl=None):
        if ctrl is None: ctrl = np.zeros(self.model.nu)
        
        if len(ctrl) != self.model.nu:
            raise ValueError(f"ctrl of length {len(ctrl)}, not expected {self.model.nu}")

        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        if self.render:
            if self.viewer.is_running():
                if self.stepcount % self.render_step == 0: self.viewer.sync()
            else: return
        self.stepcount += 1
        self.t += self.dt

    def close_sim(self):
        if self.render: self.viewer.close()
        print("Finished Simulation, step count: ", self.stepcount)

    # sets the robots generalized coords and recomputes FK
    def set_robot_joint_state(self, qpos=None):
        if qpos is None: qpos = np.zeros(self.nq_panda)
        
        if len(qpos) != self.nq_panda:
            raise ValueError(f"qpos of length {len(qpos)}, not expected {self.nq_panda}")

        self.data.qpos[:self.nq_panda] = qpos[:self.nq_panda]
        mujoco.mj_forward(self.model, self.data)

    def get_robot_joint_state(self): 
        return np.copy(self.data.qpos[:self.nq_panda])

    def get_robot_jointvel_state(self):
        return np.copy(self.data.qvel[:self.nq_panda])

    def get_robot_jointacc_state(self):
        return np.copy(self.data.qacc[:self.nq_panda])
    
    # body_name='hand' for end effector, 'box' for object
    # returns transformation matrix in SE(3) of body_name wrt world frame
    def get_robot_body_state(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body name {body_name} not found in model")
        Tsb = np.zeros((4,4))
        Tsb[:3, :3] = np.copy(self.data.body(body_id).xmat.reshape((3,3)))
        Tsb[:3, 3] = np.copy(self.data.body(body_id).xpos)
        Tsb[3, 3] = 1.0
        return Tsb

    def get_robot_ee_spatial_jacobian(self):
        # get the end-effector spatial Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        body_id = self.model.body('hand').id

        mujoco.mj_jac(self.model, self.data, jacp, jacr, np.zeros(3), body_id)
        Js = np.vstack((jacr, jacp))
        return Js[:, :self.nq_panda]

    def get_mass_matrix(self):
        mass_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, mass_matrix, self.data.qM)
        return mass_matrix[:self.nu_panda, :self.nu_panda]

    def get_h_bias(self): 
        return self.data.qfrc_bias[:self.nu_panda]

