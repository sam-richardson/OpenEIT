"""

# Copyright (c) Mindseye Biomedical LLC. All rights reserved.
# Distributed under the (new) CC BY-NC-SA 4.0 License. See LICENSE.txt for more info.

	Read in a data file and plot it using an algorithm. 

"""
from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib.pyplot as plt
import OpenEIT.dashboard
import OpenEIT.reconstruction 

from OpenEIT.reconstruction.pyeit.mesh import wrapper

def parse_line(line):
    try:
        _, data = line.split(":", 1)
    except ValueError:
        return None
    items = []
    for item in data.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            items.append(float(item))
        except ValueError:
            return None
    return np.array(items)

n_el = 16
""" Load Data """
text_file = open("test2.txt", "r")
lines 		= text_file.readlines()
print ("length lines: ",len(lines))
f0 			= parse_line(lines[1])
f1 			= ((parse_line(lines[11])+parse_line(lines[12])+parse_line(lines[13])+parse_line(lines[14]))/4)


# abdomen w gel w jacobian w 7, 8

# text_file = open("data_1.bin", "r")
# lines 		= text_file.readlines()
# f0 			= parse_line(lines[5])

mesh_obj, el_pos = wrapper.create(16, h0=0.05)


""" Set up and algorithm choice """
# initialize all parameters.
# g = OpenEIT.reconstruction.GreitReconstruction(n_el=n_el)
g = OpenEIT.reconstruction.JacReconstruction(n_el=n_el)
# g = OpenEIT.reconstruction.BpReconstruction(n_el=n_el)
# print(g.__dict__)

""" 1. problem setup """
# variables needed to set up the forward simulation of data.
mesh_obj = g.mesh_obj
el_pos = g.el_pos
ex_mat = g.ex_mat
# print('ex_mat = ', ex_mat)
pts = mesh_obj['node']
tri = mesh_obj['element']
x = pts[:, 0]
y = pts[:, 1]

# plot the mesh
fig, ax = plt.subplots(figsize=(6, 4))
ax.triplot(pts[:, 0], pts[:, 1], tri, linewidth=1)
ax.plot(pts[el_pos, 0], pts[el_pos, 1], 'ro')
ax.axis('equal')
ax.axis([-1.2, 1.2, -1.2, 1.2])
ax.set_xlabel('x')
ax.set_ylabel('y')
title_src = 'number of triangles = ' + str(np.size(tri, 0)) + ', ' + \
            'number of nodes = ' + str(np.size(pts, 0))
ax.set_title(title_src)
plt.show()

step = 1
#OVERRIDE BASELINE
# f0 = [1] * len(f1)

g.update_reference(f0)
baseline = g.eit_reconstruction(f0)

image = g.eit_reconstruction(f1)

# JAC OR BP RECONSTRUCTION SHOW #
fig, ax = plt.subplots(figsize=(6, 4))
# im = ax.tripcolor(x,y, tri, image,
#                   shading='flat', cmap=plt.cm.gnuplot)
# ax.plot(x[el_pos], y[el_pos], 'ro')
# for i, e in enumerate(el_pos):
#     ax.text(x[e], y[e], str(i+1), size=12)
#
# fig.colorbar(im)

im = ax.tripcolor(x,y, tri, image)
ax.plot(x[el_pos], y[el_pos], 'ro')
for i, e in enumerate(el_pos):
    ax.text(x[e], y[e], str(i+1), size=12)
ax.axis('equal')
fig.colorbar(im)
plt.show()


ax.axis('equal')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_title(r'$\Delta$ Conductivity')

fig, ax = plt.subplots(2,figsize=(6, 4))
im = ax[0].tripcolor(x,y, tri, image,
                  shading='flat', cmap=plt.cm.gnuplot)
ax[0].plot(x[el_pos], y[el_pos], 'ro')
for i, e in enumerate(el_pos):
    ax[0].text(x[e], y[e], str(i+1), size=12)

fig.colorbar(im,ax=ax[0])

ax[0].axis('equal')
ax[0].set_xlim([-1.2, 1.2])
ax[0].set_ylim([-1.2, 1.2])
ax[0].set_title(r'$\Delta$ Conductivity')

print(image.shape)
# print(image)
print (np.mean(image))
av = np.mean(image)
# print (image[0])
for i in range(len(image)):
    if image[i] < -500:
        image[i] = av

im2 = ax[1].tripcolor(x,y, tri, image,
                  shading='flat', cmap=plt.cm.gnuplot,vmin=0,vmax=1500)
ax[1].plot(x[el_pos], y[el_pos], 'ro')
for i, e in enumerate(el_pos):
    ax[1].text(x[e], y[e], str(i+1), size=12)
fig.colorbar(im2,ax=ax[1])
ax[1].axis('equal')
ax[1].set_xlim([-1.2, 1.2])
ax[1].set_ylim([-1.2, 1.2])
ax[1].set_title(r'$\Delta$ Conductivity')
fig.set_size_inches(6, 4)

# GREIT RECONSTRUCION IMAGE SHOW # 
# print(image.shape)
# new = image[np.logical_not(np.isnan(image))]
# flat = new.flatten()
# av = np.median(flat)
# print (av)
# total = []
# for i in range(32):
#     for j in range(32):
#         if image[i,j] < -5000: 
#             image[i,j] = av

print ('image shape',image.shape)
fig, ax = plt.subplots(figsize=(6, 4))
#rotated = np.rot90(image, 1)
im = ax.imshow(image, interpolation='none', cmap=plt.cm.rainbow)
fig.colorbar(im)
ax.axis('equal')
ax.set_title(r'$\Delta$ Conductivity Map of Lungs')
fig.set_size_inches(6, 4)
# fig.savefig('../figs/demo_greit.png', dpi=96)
plt.show()


