import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
# import sPAM_model.vine_segment_actuator_db as feas
from . import globals as G

##############################
# HELPERS
##############################
def lazy_valid(aa, lc):
	lc = lc * G.scale / G.bound
		
	if lc < 0.06 or lc > 1:
		return False
	if aa <= 0 or aa > 3.11:
		return False
	if aa > 3.11/0.95 * lc:
		return False
	return True

def check_feas(arc):
	aa = arc.arc_angles
	lc = arc.arc_lengths
	lc = lc/1000
		
	if lc < 0.06 or lc > 10:
		return False
	if aa <= 0 or aa > 3.11:
		return False
	if aa > 3.11/3.0 * lc:
		return False
	return True

def tf_mat(pose):
	x, y, theta = pose
	T = np.array([[np.cos(theta), -np.sin(theta), 0, x],
					[np.sin(theta),  np.cos(theta), 0, y],
					[0,              0,             1, 0],
					[0,              0,             0, 1]])
	return T

def wrap_angle(theta):
	return (theta + np.pi) % (2 * np.pi) - np.pi

class Arc:
	def __init__(self, radii_of_curvature, arc_lengths, arc_angles, initial_poses, final_poses):
		self.radii_of_curvature = radii_of_curvature
		self.arc_lengths = arc_lengths
		self.arc_angles = arc_angles
		self.initial_poses = initial_poses
		self.final_poses = final_poses

class BiArc:
	def __init__(self, radii_of_curvature, arc_lengths, arc_angles, initial_poses, final_poses):
		self.radii_of_curvature = radii_of_curvature
		self.arc_lengths = arc_lengths
		self.arc_angles = arc_angles
		self.initial_poses = initial_poses
		self.final_poses = final_poses

##############################
#   Fitting
##############################

def fit_circular_arc(initial_pose, final_pose, max_arc_length=100, max_arc_angle=2*np.pi):

	chord_length = np.linalg.norm(final_pose[:2] - initial_pose[:2])
	chord_angle = np.arctan2(final_pose[1] - initial_pose[1], final_pose[0] - initial_pose[0])

	delta_theta = wrap_angle(final_pose[2] - chord_angle)

	# Handle straight-line case
	if abs(delta_theta) < 1e-3:
		R = np.inf
		L = chord_length
		arc_angle = 0.0
		return Arc(R, L, arc_angle, initial_pose, final_pose), True

	# Normal circular arc case
	arc_angle = 2 * delta_theta
	R = chord_length / (2 * np.sin(arc_angle / 2))
	L = abs(R * arc_angle)

	# Ensure consistent initial theta update (so initial tangency is respected)
	new_initial_theta = wrap_angle(chord_angle - np.sign(arc_angle) * (np.pi / 2))

	updated_initial_pose = np.array([initial_pose[0], initial_pose[1], new_initial_theta])

	# Feasibility check
	if abs(arc_angle) > max_arc_angle: # or L > max_arc_length:
		return None, False

	arc = Arc(abs(R), L, arc_angle, updated_initial_pose, final_pose)
	return arc#, check_feas(arc)

def fit_bi_arc(initial_pose, final_pose, max_arc_length=0.5, max_arc_angle=np.pi, p=1):
	is_arc_valid = True

	chord_vector = final_pose[0:2] - initial_pose[0:2]
	chord_angle = np.arctan2(final_pose[1] - initial_pose[1], final_pose[0] - initial_pose[0])

	# angles alpha and beta are calculated wrt the line connecting initial_pose and final_pose
	alpha = initial_pose[-1] - chord_angle
	beta = final_pose[-1] - chord_angle

	# Checks on alpha and beta that need to be met for bi-arc to be possible
	if abs(alpha) > np.pi or abs(beta) > np.pi or abs(alpha + beta) > 2 * np.pi:
		is_arc_valid = False
		return None, is_arc_valid

	elif abs(alpha) == 0.0 and abs(beta) == 0 and abs(alpha + beta) == 0.0:
		# essentially when angles are zero wrt chord we get straight lines as bi-arc

		# continue to create bi-arc
		w = (alpha + beta) / 2.0

		# convert the coordinate system to a local system where initial_pose is located at (-c, 0) and final_pose is located at (c, 0)
		c = np.linalg.norm(chord_vector) / 2.0

		# curvatures
		k1 = 0.0
		k2 = 0.0

		# radii of curvature
		R1 = 1e8 # np.inf
		R2 = 1e8 # np.inf

		# arc angles
		theta1 = 2 * np.angle(np.exp(-1j*alpha) + 1.0 / p * np.exp(-1j*w))
		theta2 = 2 * np.angle(np.exp(1j*beta) + p * np.exp(1j*w))

		# arc lengths
		L1 = c
		L2 = c

	else:
		# continue to create bi-arc
		w = (alpha + beta) / 2.0

		# convert the coordinate system to a local system where initial_pose is located at (-c, 0) and final_pose is located at (c, 0)
		c = np.linalg.norm(chord_vector) / 2.0

		# curvatures
		if abs(c) < 1e-4:
			k1 = 10000
			k2 = 10000
		else:
			k1 = -1.0 / c * (np.sin(alpha) + 1.0 / p * np.sin(w))
			k2 = 1.0 / c * (np.sin(beta) + p * np.sin(w))

		# radii of curvature
		R1 = np.abs(1.0 / k1)
		R2 = np.abs(1.0 / k2)

		# arc angles
		theta1 = 2 * np.angle(np.exp(-1j*alpha) + 1.0 / p * np.exp(-1j*w))
		theta2 = 2 * np.angle(np.exp(1j*beta) + p * np.exp(1j*w))

		# arc lengths
		L1 = np.abs(theta1 / k1)
		L2 = np.abs(theta2 / k2)

	# arc join point
	gamma = (alpha - beta) / 2.0
	Xj = c * (p**2 - 1) / (p**2 + 2 * p * np.cos(gamma) + 1)
	Yj = (2 * c * p * np.sin(gamma)) / (p**2 + 2 * p * np.cos(gamma) + 1)
	join_point = np.array([Xj, Yj])

	mid_point_chord = (final_pose[0:2] + initial_pose[0:2]) / 2.0

	# joint angle wrt chord-mid-point coordinates
	joint_angle = -2 * np.arctan2(p * np.sin(alpha / 2.0) + np.sin(beta / 2.0), p * np.cos(alpha / 2.0) + np.cos(beta / 2.0))

	# join point wrt original coordinate system
	rotation_pose = np.array([mid_point_chord[0], mid_point_chord[1], chord_angle])
	joint_point_in_original_coords = tf_mat(rotation_pose) @ np.array([join_point[0], join_point[1], 0.0, 1.0])

	center_pose = np.zeros((3,))
	center_pose[0:2] = joint_point_in_original_coords[0:2]
	center_pose[-1] = joint_angle + chord_angle

	# compile results
	radii_of_curvature = np.array([R1, R2])
	arc_lengths = np.array([L1, L2])
	arc_angles = np.array([theta1, theta2])
	initial_poses = [initial_pose, center_pose]
	final_poses = [center_pose, final_pose]
	
	# check if arc angles and lengths are within feasible values
	if (np.abs(arc_angles) > max_arc_angle).any(): # or (arc_lengths > max_arc_length).any():
		is_arc_valid = False

	# arc1 = Arc(radii_of_curvature[0], arc_lengths[0], arc_angles[0], initial_poses[0], final_poses[0])
	# arc2 = Arc(radii_of_curvature[1], arc_lengths[1], arc_angles[1], initial_poses[1], final_poses[1])

	#if not check_feas(arc1) or not check_feas(arc2):
	# 	is_arc_valid = False
	#return biarc, is_arc_valid
	return BiArc(radii_of_curvature, arc_lengths, arc_angles, initial_poses, final_poses), is_arc_valid

##############################
#   Plotting
##############################
def plot_circular_arc(arc, ax, color='black', linewidth=1.0):
	initial_pose = arc.initial_poses
	final_pose = arc.final_poses
	R = arc.radii_of_curvature
	arc_angle = arc.arc_angles
	dir_vec_ini, dir_vec_final, C, e1 = find_arc_parameters(initial_pose,
														 final_pose,
														 arc_angle, R,
														 color=color,
														 linewidth=linewidth)
	ax.add_patch(e1)
	ax.grid()
	
def plot_bi_arc(biarc, ax, color=None, linewidth=None, alpha=1.0):
	# This function plots both arcs in the biarc object
	#for i in range(2):
	#    plot_arc(ax, Arc(biarc.radii_of_curvature[i], biarc.arc_lengths[i], biarc.arc_angles[i], biarc.initial_poses[i], biarc.final_poses[i]))
	if color is not None:
		colors = [color, color]
	else:
		colors = ['k', 'm']
	initial_poses = biarc.initial_poses
	final_poses = biarc.final_poses
	radii_of_curvature = biarc.radii_of_curvature
	arc_angles = biarc.arc_angles

	#fig, ax = plt.subplots(figsize=(4, 4))
	for initial_pose, final_pose, arc_angle, R, color in zip(initial_poses, final_poses, arc_angles, radii_of_curvature, colors):
		dir_vec_ini, dir_vec_final, C, e1 = find_arc_parameters(initial_pose, final_pose, arc_angle, R, color, linewidth, alpha)
		ax.add_patch(e1)
	# ax.legend()

	ax.grid()


def find_arc_parameters(initial_pose, final_pose, arc_angle, R, color='k', linewidth=1.0, alpha=1.0):
	chord_length = np.linalg.norm(final_pose[0:2] - initial_pose[0:2])
	rot_vec = tf_mat(initial_pose) @ np.array([0.2 * chord_length, 0.0, 0.0, 1.0])
	dir_vec_ini = rot_vec[0:2]

	rot_vec = tf_mat(final_pose) @ np.array([0.2 * chord_length, 0.0, 0.0, 1.0])
	dir_vec_final = rot_vec[0:2]

	chord_mid_point = (initial_pose[0:2] + final_pose[0:2]) / 2.0

	perp_bisector_angle = np.arctan2(-1 * (final_pose[0] - initial_pose[0]), (final_pose[1] - initial_pose[1]))

	d = R * np.cos(arc_angle / 2.0)

	C1 = (chord_mid_point[0] + d * np.cos(perp_bisector_angle), chord_mid_point[1] + d * np.sin(perp_bisector_angle))
	C2 = (chord_mid_point[0] - d * np.cos(perp_bisector_angle), chord_mid_point[1] - d * np.sin(perp_bisector_angle))

	if arc_angle > 0.0:
		C = C2
		theta1 = np.rad2deg(np.arctan2(initial_pose[1] - C[1], initial_pose[0] - C[0]))
		theta2 = np.rad2deg(np.arctan2(final_pose[1] - C[1], final_pose[0] - C[0]))
	else:
		C = C1
		theta1 = np.rad2deg(np.arctan2(final_pose[1] - C[1], final_pose[0] - C[0]))
		theta2 = np.rad2deg(np.arctan2(initial_pose[1] - C[1], initial_pose[0] - C[0]))
	e1 = patches.Arc(C, 2 * R, 2 * R,
				angle=0.0, theta1=theta1, theta2=theta2, fill=False, zorder=2, color=color, label='arc', linewidth=linewidth, alpha=alpha)

	return dir_vec_ini, dir_vec_final, C, e1

def steer_bi_arc(biarc):
	initial_poses = biarc.initial_poses
	final_poses = biarc.final_poses
	radii_of_curvature = biarc.radii_of_curvature
	arc_angles = biarc.arc_angles
	arc_lengths = biarc.arc_lengths
	step_size = 0.1
	# Iterate through each arc in the bi-arc
	for i in range(2):
		R = radii_of_curvature[i]
		L = arc_lengths[i]
		arc_angle = arc_angles[i]
		num_steps = int(np.ceil(L / step_size))
		delta_theta = arc_angle / num_steps
		initial_pose = initial_poses[i]
		final_pose = final_poses[i]


		chord_length = np.linalg.norm(final_pose[0:2] - initial_pose[0:2])
		rot_vec = tf_mat(initial_pose) @ np.array([0.2 * chord_length, 0.0, 0.0, 1.0])
		dir_vec_ini = rot_vec[0:2]

		rot_vec = tf_mat(final_pose) @ np.array([0.2 * chord_length, 0.0, 0.0, 1.0])
		dir_vec_final = rot_vec[0:2]

		chord_mid_point = (initial_pose[0:2] + final_pose[0:2]) / 2.0

		perp_bisector_angle = np.arctan2(-1 * (final_pose[0] - initial_pose[0]), (final_pose[1] - initial_pose[1]))

		d = R * np.cos(arc_angle / 2.0)

		C1 = (chord_mid_point[0] + d * np.cos(perp_bisector_angle), chord_mid_point[1] + d * np.sin(perp_bisector_angle))
		C2 = (chord_mid_point[0] - d * np.cos(perp_bisector_angle), chord_mid_point[1] - d * np.sin(perp_bisector_angle))

		if arc_angle > 0.0:
			C = C2
			theta1 = np.rad2deg(np.arctan2(initial_pose[1] - C[1], initial_pose[0] - C[0]))
			theta2 = np.rad2deg(np.arctan2(final_pose[1] - C[1], final_pose[0] - C[0]))
		else:
			C = C1
			theta1 = np.rad2deg(np.arctan2(final_pose[1] - C[1], final_pose[0] - C[0]))
			theta2 = np.rad2deg(np.arctan2(initial_pose[1] - C[1], initial_pose[0] - C[0]))


		# Calculate the center of the arc

		current_theta = np.arctan2(initial_pose[1] - C[1], initial_pose[0] - C[0])

		# Generate nodes along the arc
		for step in range(num_steps):
			current_theta += delta_theta
			x = C[0] + R * np.cos(current_theta)
			y = C[1] + R * np.sin(current_theta)
			theta = (initial_pose[2] + step * delta_theta) % (2 * np.pi)
			plt.plot(x, y, 'xg')

	return True


def steer_arc(arc):
	"""
	Generate and plot points along the arc while correctly handling negative arc angles
	and ensuring that angles wrap correctly in the range (-π, π].
	"""
	R = arc.radii_of_curvature
	L = arc.arc_lengths
	#arc_angle = wrap_angle(arc.arc_angles)
	arc_angle = arc.arc_angles
	initial_pose = arc.initial_poses
	final_pose = arc.final_poses
	initial_pose[2] = wrap_angle(initial_pose[2])
	final_pose[2] = wrap_angle(final_pose[2])

	step_size = 0.1
	num_steps = max(1, int(np.ceil(L / step_size)))  # Ensure at least one step
	delta_theta = arc_angle / num_steps  # Angle step
	chord_length = np.linalg.norm(final_pose[:2] - initial_pose[:2])
	# Compute perpendicular bisector angle
	chord_mid_point = ((initial_pose[0] + final_pose[0])/2, (initial_pose[1] + final_pose[1])/2)
	perp_bisector_angle = np.arctan2(-(final_pose[0] - initial_pose[0]), final_pose[1] - initial_pose[1])
	# Determine center direction (left-turn vs. right-turn)
	d = R * np.sin(abs(np.pi - arc_angle) / 2.0)  # Correct perpendicular offset

	# Compute normal to chord (perpendicular bisector angle)
	perp_bisector_angle = np.arctan2(final_pose[1] - initial_pose[1], final_pose[0] - initial_pose[0]) + np.pi / 2
	# Determine center based on turn direction
	if arc_angle > 0 and arc_angle < np.pi:
		# Left turn: center is to the left of the chord
		C = (chord_mid_point[0] + d * np.cos(perp_bisector_angle),
			chord_mid_point[1] + d * np.sin(perp_bisector_angle))
	else:
		# Right turn: center is to the right of the chord
		C = (chord_mid_point[0] - d * np.cos(perp_bisector_angle),
			chord_mid_point[1] - d * np.sin(perp_bisector_angle))

	# Determine starting angle from center
	current_theta = np.arctan2(initial_pose[1] - C[1], initial_pose[0] - C[0])

	# Generate points along the arc
	for step in range(num_steps + 1):  # Include final point

		x = C[0] + R * np.cos(current_theta)
		y = C[1] + R * np.sin(current_theta)
		theta = wrap_angle(initial_pose[2] + step * delta_theta)  # Ensure theta wraps

		plt.plot(x, y, 'xg')  # Plot arc point
		if arc_angle > 0:
			current_theta += delta_theta  # Left turn
		else:
			current_theta += delta_theta  # Right turn

	return True


if __name__ == "__main__":

	initial_pose = np.array([7.0, 5.0, np.deg2rad(180)])
	final_pose = np.array([5.0, 6.0, np.deg2rad(180)])

	## fitting a simple circular arc
	arc = fit_circular_arc(initial_pose, final_pose)
	fig, ax = plt.subplots(figsize=(4, 4))
	print(arc.initial_poses)
	print(arc.final_poses)
	print(arc.radii_of_curvature)
	print(arc.arc_angles)
	plot_circular_arc(arc, ax)
	#steer_arc(arc)
	plt.axis('equal')
	#plt.savefig('test.png')
	plt.show()
	# fitting a bi-arc
	# biarc, is_arc_valid = fit_bi_arc(initial_pose,
	# 								final_pose,
	# 								max_arc_length=1,
	# 								max_arc_angle=np.pi,
	# 								p=1)

	# print("is_arc_valid:", is_arc_valid)

	# if biarc is not None:
	# 	fig, ax = plt.subplots(figsize=(4, 4))
	# 	plot_bi_arc(biarc, ax)
	# 	steer_bi_arc(biarc)
	# 	print(biarc.initial_poses)
	# 	print(biarc.final_poses)
	# 	#print(biarc.radii_of_curvature)
	# 	print(biarc.arc_angles)
	# 	plt.axis('equal')
	# 	plt.show()
