import numpy as np
import matplotlib.pyplot as plt

joint_angles = [1.942932368, 0.611754179, 1.291961714, -0.551500579] #INPUT YOUR 4 ANGLES HERE (radians)

def dh_transform(theta, d, a, alpha):
    #computes DH 
    c_th, s_th = np.cos(theta), np.sin(theta)
    c_al, s_al = np.cos(alpha), np.sin(alpha)
    
    return np.array([
        [c_th, -s_th * c_al,  s_th * s_al, a * c_th],
        [s_th,  c_th * c_al, -c_th * s_al, a * s_th],
        [0,     s_al,         c_al,         d     ],
        [0,     0,            0,            1     ]
    ])

def calculate_forward_kinematics(joint_angles):

    #used to calc forward kinematics
    dh_table = [
        [30.0,   0.0,   np.pi / 2],  # Joint 1
        [0.0,    195.0, 0.0      ],  # Joint 2
        [0.0,    195.0, 0.0      ],  # Joint 3
        [0.0,    94.0,  0.0      ]   # Joint 4
    ]
    
    # Base origin (Frame 0)
    positions = [np.array([0.0, 0.0, 0.0])]
    
    #matrixes multiply
    T = np.eye(4)
    
    for i in range(4):
        theta = joint_angles[i]
        d, a, alpha = dh_table[i]
        
        # Get transformation for this joint
        A_i = dh_transform(theta, d, a, alpha)
        T = T @ A_i
        
        # Extract the origin (x, y, z) translation vector from the transformation matrix
        pos = T[:3, 3]
        positions.append(pos)
        
    return positions

import matplotlib.pyplot as plt

def plot_robotic_arm(joint_positions):
    
    #create 4 plots of robotic arm with side views and forward view
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(14, 12))
    ax_list = axes.flatten()
    
    xs = [p[0] for p in joint_positions]
    ys = [p[1] for p in joint_positions]
    zs = [p[2] for p in joint_positions]
    
    # Define camera angles and titles for each of the 4 subplots
    views = [
        {"elev": 90, "azim": -90, "title": "Plot 1: X-Y Plane (Top-Down)"},
        {"elev": 0,  "azim": 0,   "title": "Plot 2: Y-Z Plane"},
        {"elev": 0,  "azim": -90, "title": "Plot 3: Z-X Plane"},
        {"elev": 30, "azim": -45, "title": "Plot 4: 45° Forward View"}
    ]
    
    for i, ax in enumerate(ax_list):
        # 1. Plot structure links (Blue lines)
        ax.plot(xs, ys, zs, '-', linewidth=5, color='royalblue', label='Links')
        
        # 2. Plot intermediate joints (Red markers)
        ax.scatter(xs[:-1], ys[:-1], zs[:-1], color='red', s=80, depthshade=False, label='Joints', zorder=5)
        
        # 3. Highlight End-Effector / TCP (Green marker)
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color='limegreen', s=150, depthshade=False, label='End-Effector (TCP)', zorder=5)
        
        # Set axis labels and specific subplot title
        ax.set_xlabel('X Axis (mm)')
        ax.set_ylabel('Y Axis (mm)')
        ax.set_zlabel('Z Axis (mm)')
        ax.set_title(views[i]["title"])
        
        # Scale axes equally (retained from original functionality)
        max_range = max(
            max(xs) - min(xs),
            max(ys) - min(ys),
            max(zs) - min(zs)
        )
        if max_range == 0:
            max_range = 100
            
        mid_x = (max(xs) + min(xs)) * 0.5
        mid_y = (max(ys) + min(ys)) * 0.5
        mid_z = (max(zs) + min(zs)) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(min(zs) - max_range * 0.2, max(zs) + max_range * 0.4)
        
        # Apply specific camera angle
        ax.view_init(elev=views[i]["elev"], azim=views[i]["azim"])
        
        # Add legend to each subplot
        ax.legend()

    plt.tight_layout()
    plt.show()
    
# Calculate joint positions
joints = calculate_forward_kinematics(joint_angles)
    
#print to verify
print("Calculated Joint Positions (X, Y, Z):")
for i, pos in enumerate(joints):
    print(f"  Joint {i}: X = {pos[0]:.4f}, Y = {pos[1]:.4f}, Z = {pos[2]:.4f}")
        
# Display plot grid
plot_robotic_arm(joints)

