import time
import mujoco
from franka_simulator import Franka_Simulator
from franka_ctrl import Franka_TargetController
import matplotlib.pyplot as plt
import numpy as np

# Initialize simulator and controller
sim = Franka_Simulator()
ctrller = Franka_TargetController(sim)

# Simulation Parameter
sim_time = 100

# Track state for plotting
state_vals = {
    "timevals": [],     # time              t over time
    "qposvals": [],     # joint states      qpos over time
    "qvelvals": [],     # joint velocities  qvel over time
    "xpos_ee": [],      # end effector      pos (x, y, z) over time
    "xpos_obj": [],     # object pos        (x, y, z) over time
    "task_err": [],     # task error        (q_des - q_actual) over time
    "torques": []       # joint torques
}

def track_state(statevals, sim, task_err):
    statevals["timevals"].append(sim.t)
    statevals["qposvals"].append(sim.get_robot_joint_state())
    statevals["qvelvals"].append(sim.get_robot_jointvel_state())
    statevals["xpos_ee"].append(sim.get_robot_body_state("hand")[:3, 3])
    statevals["xpos_obj"].append(sim.get_robot_body_state("box")[:3, 3])
    statevals["task_err"].append(task_err)
    statevals["torques"].append(np.copy(sim.data.ctrl[:8]))

# move robot to qtarget position    
def go_to_target(qtarget, motion_time=None, fast=True, griphard=False):

    _, n_steps, q_plan, qdot_plan, qddot_plan = \
        ctrller.set_target_ctrl(target=qtarget, T=motion_time)              # compute trajectory
    
    i = 0
    while (i < n_steps) and (sim.t < sim_time):
        qddot_des, task_err = ctrller.pd_ctrl(q_plan[i], qdot_plan[i],qddot_plan[i])  # enforce trajectory  
        torques = ctrller.get_ctrl(qddot_des)                               # compute torques  
        if (griphard): torques[7] = -50                                     # grip hard
        sim.step(torques)                                      
        i += 1

        track_state(state_vals, sim, task_err)                            
        if fast: 
            if (i % 10 == 0): time.sleep(sim.dt)                            # factor between real and sim time        
        else : time.sleep(sim.dt)


# Pause at start
time.sleep(1)
box_T0 = sim.get_robot_body_state("box")        # box position
robot_q0 = sim.get_robot_joint_state()          # robot position

# open gripper
qtarget = sim.get_robot_joint_state()
qtarget[7] = 0.04                               # open gripper
go_to_target(qtarget, motion_time=4.)

# move to box
Tsd = sim.get_robot_body_state("hand")          
Tsd[:3, 3] = box_T0[:3, 3]                      # set box position
Tsd[2, 3] = Tsd[2, 3] + 0.11                    # move up a bit
qtarget, _ = ctrller.compute_IK(Tsd, sim)
go_to_target(qtarget)

# close gripper
qtarget = sim.get_robot_joint_state()
qtarget[7] = 0                                  # close gripper
go_to_target(qtarget, motion_time=4.)

# move up
Tsd = sim.get_robot_body_state("hand")                  
Tsd[2, 3] = Tsd[2, 3] + 0.25                    # move up a bit
qtarget, _ = ctrller.compute_IK(Tsd, sim)
qtarget[7] = 0                                  # keep gripper closed
go_to_target(qtarget, griphard=True)

# move to drop location
Tsd = sim.get_robot_body_state("hand")                  
Tsd[2, 3] = Tsd[2, 3] - 0.25                    # move down a bit
Tsd[1, 3] = Tsd[1, 3] + 0.25                    # move to the left a bit
qtarget, _ = ctrller.compute_IK(Tsd, sim)
go_to_target(qtarget, griphard=True)

# release box
qtarget = sim.get_robot_joint_state()
qtarget[7] = 0.04   # open gripper
go_to_target(qtarget, motion_time=4.)

# return to home position
qtarget = robot_q0
go_to_target(qtarget)

# stay at home position
qtarget = sim.get_robot_joint_state()
go_to_target(qtarget, motion_time=10.)

# end simulations
sim.close_sim()

######################################################################
# # Plotting
# figsize = (6, 6)
# _, axs = plt.subplots(1, 6, figsize=figsize)
# plt.suptitle(f"Panda Pick and Place")

# # Plot joint trajectories
# axs[0].plot(state_vals["timevals"], state_vals["qposvals"])
# axs[0].set_title('Joint Position')
# axs[0].set_ylabel('Position (rad)')
# axs[0].legend(['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper'], loc='upper right')

# # Plot joint velocities
# axs[1].plot(state_vals["timevals"], state_vals["qvelvals"])
# axs[1].set_title('Joint Velocity')
# axs[1].set_ylabel('Velocity (rad/s)')
# axs[1].legend(['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper'], loc='upper right')

# # Plot end-effector position
# axs[2].plot(state_vals["timevals"], state_vals["xpos_ee"])
# axs[2].set_title('End-Effector Position')
# axs[2].set_ylabel('Position (m)')
# axs[2].set_xlabel('Time (s)')
# axs[2].legend(['x', 'y', 'z'], loc='upper right')

# # plot object position
# axs[3].plot(state_vals["timevals"], state_vals["xpos_obj"])
# axs[3].set_title('Object Position')
# axs[3].set_ylabel('Position (m)')
# axs[3].set_xlabel('Time (s)')
# axs[3].legend(['x', 'y', 'z'], loc='upper right')

# # plot task error
# axs[4].plot(state_vals["timevals"], state_vals["task_err"])
# axs[4].set_title('Task Error')
# axs[4].set_ylabel('Error (rad)')
# axs[4].set_xlabel('Time (s)')
# axs[4].legend(['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper'], loc='upper right')

# plot torques on new plot
# plt.figure()
# plt.plot(state_vals["timevals"], state_vals["torques"])
# plt.title('Joint Torques')
# plt.ylabel('Torque (Nm)')
# plt.xlabel('Time (s)')
# plt.legend(['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'gripper'], loc='upper right')
# plt.tight_layout()
# plt.show()
########################################################################