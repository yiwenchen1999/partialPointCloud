# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 19:27:50 2022

@author: ThinkPad
"""
import numpy as np
import copy
import open3d as o3d
from numpy import linalg as LA
import time
import os
import open3d.visualization.rendering as rendering
import math

def refine(points_r):
    norms = LA.norm(points_r, axis=1, ord=2)
    norms.reshape((10000))
    points_r = points_r.transpose()
    #get normalized
    rays = points_r/norms
    points_r = points_r.transpose()
    rays = rays.transpose()
    
    # pcdr = o3d.geometry.PointCloud()
    # pcdr.points = o3d.utility.Vector3dVector(rays)
    # o3d.visualization.draw_geometries([pcdr])

    for i in range(1000):
        # print(i)
        if LA.norm(points_r[i]) == 0:
            continue
        difference = rays - rays[i]
        super_threshold_indices = abs(difference) < 0.025
        difference[super_threshold_indices] = 0
        # print(difference[0:30,:])
        indices = np.where(~difference.any(axis=1))[0]



        # print(indices)
        for j in indices:
            if LA.norm(points_r[j]) - LA.norm(points_r[i]) > 0.05 and j != i:
                points_r[j] = [0, 0, 0]
                # elif LA.norm(points_r[i]) - LA.norm(points_r[j])>0 and j!=i:
                #     points_r[i] = [0,0,0]
        # print(points_r[12324])
    
    points_r = points_r[~np.all(points_r == 0, axis=1)]
    return points_r
    

filelist = []
path =r"D:\data\dataset_small_v1.1\03001627"
for root, dirs, files in os.walk(path):
    for file in files:
        #append the file name to the list
        if os.path.join(root,file)[-14:] == "pointcloud.npz":
            filelist.append(os.path.join(root,file))
            
print(filelist[0:20])
            
idx = 0
# path = r"D:\data\dataset_small_v1.1\03001627\175e2a8cd6e9866ab37303b6dde16342\pointcloud3_partial.npz"
# t0 = time.process_time()
# data = np.load(path)
# lst = data.files
# points = data['points_r']
# print(points.shape)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([pcd])
for idx_o in range(2000):
    idx = idx_o+500
    path = filelist[idx]
    t0= time.process_time()
    data = np.load(path)
    lst = data.files
    points = data['points']
    print(points.shape)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # R = pcd.get_rotation_matrix_from_xyz((2*np.pi / 4, np.pi / 4, 4*np.pi / 4))
    for m in range(4):
        for n in range(3):
            cam = np.array((3*math.cos(m*np.pi / 4), math.sin(((n-1))*np.pi / 4), math.sin(m*np.pi / 4)))
            # print(R)
    
            # for i in range(7):
            #     R = pcd.get_rotation_matrix_from_xyz((0*np.pi / 4, i*np.pi / 4, 0*np.pi / 4))
            #     pcd.rotate(R, center=(0, 0, 0))
            points_r = points[:10000]-cam
            
            points_r = refine(points_r)
            points_r = points_r+cam
            
            # print(points_r[0:30,:])
            
            
            
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points_r)
            # o3d.visualization.draw_geometries([pcd]) 
    
    
            np.savez(path[:-4] + str(m)+ str(n) + "_partial", points_r=points_r)
        t1 = time.process_time() - t0
    print(str(idx)+" out of "+str(len(filelist))+", remaining time:", t1*(len(filelist)-idx))
    # o3d.visualization.draw_geometries([pcd])


    

