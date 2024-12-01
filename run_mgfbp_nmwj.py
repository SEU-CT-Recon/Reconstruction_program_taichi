# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:33:40 2024

@author: xji
"""

import warnings
import time
import json
import re
import taichi as ti
import sys
import os
import numpy as np
import gc
import math
from crip.io import imwriteRaw
from crip.io import imwriteTiff
from run_mgfbp import *

PI = 3.1415926536

#This version reconstruct the image view by view to save GPU memory

def run_mgfbp_nmwj(file_path):
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=0.95)#define device memeory utilization fraction
    print('Performing FBP from MandoCT-Taichi (ver Namiweijing) ...')
    print('This version reconstruct the image view by view to save GPU memory')
    # record start time point
    start_time = time.time() 
    #Delete unnecessary warinings
    warnings.filterwarnings('ignore', category=UserWarning, \
                            message='The value of the smallest subnormal for <class \'numpy.float(32|64)\'> type is zero.')
   
    if not os.path.exists(file_path):
        print(f"ERROR: Config File {file_path} does not exist!")
        #Judge whether the config jsonc file exist
        sys.exit()
    config_dict = ReadConfigFile(file_path)#读入jsonc文件并以字典的形式存储在config_dict
    fbp = Mgfbp_nmwj(config_dict) #将config_dict数据以字典的形式送入对象
    # Ensure output directory exists; if not, create the directory
    img_recon = fbp.MainFunction()#recontructed image is returned by fbp.MainFunction()
    end_time = time.time()# record end time point
    execution_time = end_time - start_time# 计算执行时间
    if fbp.file_processed_count > 0:
        print(f"\nA total of {fbp.file_processed_count:d} file(s) are reconstructed!")
        print(f"Time cost is {execution_time:.3} sec\n")# 打印执行时间（以秒为单位
    else:
        print(f"\nWarning: Did not find files like {fbp.input_files_pattern:s} in {fbp.input_dir:s}.")
        print("No images are reconstructed!\n")
    del fbp #delete the fbp object
    gc.collect()# 手动触发垃圾回收
    ti.reset()#free gpu ram
    return img_recon


@ti.data_oriented
class Mgfbp_nmwj(Mgfbp):
    def MainFunction(self):
        #Main function for reconstruction
        self.img_recon_weight_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z,self.img_dim, self.img_dim),order='ikj')
        self.img_recon_per_view_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z,self.img_dim, self.img_dim),order='ikj')
        self.img_recon_weight = np.zeros_like(self.img_recon)
        self.img_recon_per_view = np.zeros_like(self.img_recon)
        
        self.positive_u_is_positive_y = 1
        self.InitializeArrays()#initialize arrays
        self.InitializeReconKernel()#initialize reconstruction kernel
        self.file_processed_count = 0;#record the number of files processed
        self.file_processed_count = 1
        img_sgm_filtered_total = np.zeros(shape = (24,self.dect_elem_count_vertical,self.view_num,self.dect_elem_count_horizontal))
        for source_idx in range(24):
            file = 'sgm_rebin_'+str(source_idx + 1)+'.raw'
            if self.ReadSinogram(file):
                str_1 = 'Reading and filtering sinogram files: source #%2d/%2d' % (source_idx + 1, 24)
                print('\r' + str_1, end='')
                for view_idx in range(self.view_num):
                    self.img_sgm_taichi.from_numpy(self.img_sgm[:,view_idx:view_idx+1,:])
                    self.FilterSinogram()
                    img_sgm_filtered_total[source_idx,:,view_idx:view_idx+1,:] = self.img_sgm_filtered_taichi.to_numpy()
        print('\r')
        
        #read angles and source_z_pos
        angle_array_total = np.zeros(shape = (self.view_num,24))
        z_array_total = np.zeros(shape = (self.view_num,24))
        for source_idx in range(24):
            angle_array_dict = load_jsonc(self.input_dir + '/angle_'+str(source_idx+1)+'.jsonc')
            angle_array = angle_array_dict.get("Value")
            angle_array = np.array(angle_array,dtype = np.float32) / 180.0 * PI
            angle_array_total[:,source_idx] = angle_array
            
            z_array_dict = load_jsonc( self.input_dir + '/z_'+str(source_idx+1)+'.jsonc')
            z_array = z_array_dict.get("Value")
            z_array = np.array(z_array,dtype = np.float32)
            z_array_total[:,source_idx] = z_array
        angle_min = np.min(angle_array_total[:])
        angle_max = np.max(angle_array_total[:])
        M_min = int(np.floor(angle_min/PI))
        M_max = int(np.ceil(angle_max/PI))
                       
        theta_tilde_num = 500; # can be modified
        for theta_tilde_idx in range(theta_tilde_num):
            str_1 = 'Processing theta_filde idx #%4d/%4d' % (theta_tilde_idx+1, theta_tilde_num)
            print('\r' + str_1, end='')
            self.img_recon_per_view_taichi.from_numpy(np.zeros_like(self.img_recon))
            self.img_recon_weight_taichi.from_numpy(np.zeros_like(self.img_recon))
            for source_idx in range(24):
                
                angle_array = angle_array_total[:,source_idx]
                theta_max = (angle_array[0] > angle_array[self.view_num-1]) * angle_array[0] +\
                    (angle_array[0] < angle_array[self.view_num-1]) * angle_array[self.view_num-1]
                theta_min = (angle_array[0] < angle_array[self.view_num-1]) * angle_array[0] +\
                    (angle_array[0] > angle_array[self.view_num-1]) * angle_array[self.view_num-1]
                self.array_angle_taichi.from_numpy(angle_array)
                z_array = z_array_total[:,source_idx]
                
                dis_per_rad = (z_array[-1] - z_array[0]) / (angle_array[-1] - angle_array[0])
                theta_tilde = PI / theta_tilde_num * theta_tilde_idx
                for M in range(M_min,M_max+1):
                    theta = theta_tilde + M * PI
                    view_idx = (theta - angle_array[0]) / (angle_array[1] - angle_array[0])
                    view_idx_floor = int(np.floor(view_idx))
                    view_weight = view_idx - view_idx_floor
                    if view_idx_floor >= 0 and view_idx <= (self.view_num-2):
                        self.img_sgm_filtered_taichi.from_numpy(img_sgm_filtered_total[source_idx,:,view_idx_floor:view_idx_floor+1,:])
                        z_source = view_idx_floor * (angle_array[1] - angle_array[0]) * dis_per_rad + z_array[0]
                        self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual,\
                                                       self.img_dim, self.dect_elem_count_horizontal, \
                                        self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis,\
                                            self.source_dect_dis, self.total_scan_angle,\
                                        self.array_angle_taichi, self.img_rot,self.img_sgm_filtered_taichi,self.img_recon_per_view_taichi,\
                                        self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                            self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                    self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode, \
                                                        view_idx_floor,z_source,PI / theta_tilde_num,dis_per_rad, self.img_recon_weight_taichi,
                                                        1-view_weight, theta_max,theta_min)
                        
                        self.img_sgm_filtered_taichi.from_numpy(img_sgm_filtered_total[source_idx,:,view_idx_floor+1:view_idx_floor+2,:])
                        z_source =( view_idx_floor+1) * (angle_array[1] - angle_array[0]) * dis_per_rad + z_array[0]
                        self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual,\
                                                       self.img_dim, self.dect_elem_count_horizontal, \
                                        self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis,\
                                            self.source_dect_dis, self.total_scan_angle,\
                                        self.array_angle_taichi, self.img_rot,self.img_sgm_filtered_taichi,self.img_recon_per_view_taichi,\
                                        self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                            self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                    self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode, \
                                                        view_idx_floor+1,z_source,PI / theta_tilde_num,dis_per_rad, self.img_recon_weight_taichi,\
                                                            view_weight,theta_max,theta_min)
                            
                            
                            
            self.AddDataEachView(self.img_recon_taichi,self.img_recon_per_view_taichi,self.img_recon_weight_taichi,self.img_dim_z,self.img_dim)
                                           
        self.SaveReconImg()
        del img_sgm_filtered_total
        del self.img_sgm
        return self.img_recon #函数返回重建
    
    def __init__(self,config_dict):
        super(Mgfbp_nmwj,self).__init__(config_dict)
        
        del self.img_sgm_filtered_taichi
        
        #sgm in taichi contains info in only one_view to save gpu memory
        self.img_sgm_taichi = ti.field(dtype=ti.f32, shape=(self.dect_elem_count_vertical_actual,1, self.dect_elem_count_horizontal)) 
        self.img_sgm_filtered_taichi = ti.field(dtype=ti.f32, shape=(self.dect_elem_count_vertical_actual,1,self.dect_elem_count_horizontal))
        
        if self.apply_gauss_vertical:
            self.img_sgm_filtered_intermediate_taichi = ti.field(dtype=ti.f32, shape=(self.dect_elem_count_vertical_actual,1, self.dect_elem_count_horizontal))
        else:
            self.img_sgm_filtered_intermediate_taichi = ti.field(dtype=ti.f32, shape=(1,1,1))
            #if vertical gauss filter is not applied, initialize this intermediate sgm with a small size to save GPU memory
        
        self.img_recon_taichi.from_numpy(self.img_recon) #initialize img_recon_taichi (all-zero array)
        
        print('--Warning: Total scan angle value from config file is discarded!')

        
    
    def FilterSinogram(self):
        self.ConvolveSgmAndKernel(self.dect_elem_count_vertical_actual,self.view_num,self.dect_elem_count_horizontal,\
                                  self.dect_elem_width,self.img_sgm_taichi,self.array_recon_kernel_taichi,\
                                      self.array_kernel_gauss_vertical_taichi,self.dect_elem_height, self.apply_gauss_vertical,
                                      self.img_sgm_filtered_intermediate_taichi, self.img_sgm_filtered_taichi)
    @ti.kernel
    def ConvolveSgmAndKernel(self, dect_elem_count_vertical_actual:ti.i32, view_num:ti.i32, \
                             dect_elem_count_horizontal:ti.i32, dect_elem_width:ti.f32, img_sgm_taichi:ti.template(), \
                                 array_recon_kernel_taichi:ti.template(),array_kernel_gauss_vertical_taichi:ti.template(),\
                                     dect_elem_height:ti.f32, apply_gauss_vertical:ti.i32,img_sgm_filtered_intermediate_taichi:ti.template(),\
                                         img_sgm_filtered_taichi:ti.template()):
        #apply filter along vertical direction
        for i, k in ti.ndrange(dect_elem_count_vertical_actual, dect_elem_count_horizontal):
            temp_val = 0.0
            if apply_gauss_vertical:
                # if vertical filter is applied, apply vertical filtering and 
                # save the intermediate result to img_sgm_filtered_intermediate_taichi
                for n in ti.ndrange(dect_elem_count_vertical_actual):
                    temp_val += img_sgm_taichi[n, 0, k] \
                        * array_kernel_gauss_vertical_taichi[i + (dect_elem_count_vertical_actual - 1) - n]
                img_sgm_filtered_intermediate_taichi[i, 0, k] = temp_val * dect_elem_height
            else:
                pass
                
        for i, k in ti.ndrange(dect_elem_count_vertical_actual, dect_elem_count_horizontal):
            temp_val = 0.0 
            if apply_gauss_vertical:
                # if vertical filter is applied, use img_sgm_filtered_intermediate_taichi
                # for horizontal filtering
                for m in ti.ndrange(dect_elem_count_horizontal):
                    temp_val += img_sgm_filtered_intermediate_taichi[i, 0, m] \
                        * array_recon_kernel_taichi[ k + (dect_elem_count_horizontal - 1) - m]
            else:
                # if not, use img_sgm_taichi
                for m in ti.ndrange(dect_elem_count_horizontal):
                    temp_val += img_sgm_taichi[i, 0, m] \
                            * array_recon_kernel_taichi[ k + (dect_elem_count_horizontal - 1) - m]
            img_sgm_filtered_taichi[i, 0, k] = temp_val * dect_elem_width
    
        
    @ti.kernel
    def AddDataEachView(self,img_recon_taichi:ti.template(), img_recon_per_view_taichi:ti.template(),img_recon_weight_taichi:ti.template(),img_dim_z:ti.i32,img_dim:ti.i32):
        for i_x, i_y, i_z in ti.ndrange(img_dim, img_dim, img_dim_z):
            if img_recon_weight_taichi[i_z, i_y, i_x] !=0.0:
                img_recon_taichi[i_z, i_y, i_x] += img_recon_per_view_taichi[i_z, i_y, i_x] / img_recon_weight_taichi[i_z, i_y, i_x]
        
        
    @ti.kernel
    def BackProjectionPixelDriven(self, dect_elem_count_vertical_actual:ti.i32, img_dim:ti.i32, dect_elem_count_horizontal:ti.i32, \
                                  view_num:ti.i32, dect_elem_width:ti.f32,\
                                  img_pix_size:ti.f32, source_isocenter_dis:ti.f32, source_dect_dis:ti.f32,total_scan_angle:ti.f32,\
                                      array_angle_taichi:ti.template(),img_rot:ti.f32,img_sgm_filtered_taichi:ti.template(),\
                                          img_recon_taichi:ti.template(),\
                                          array_u_taichi:ti.template(), short_scan:ti.i32,cone_beam:ti.i32,dect_elem_height:ti.f32,\
                                              array_v_taichi:ti.template(),img_dim_z:ti.i32,img_voxel_height:ti.f32, \
                                                  img_center_x:ti.f32,img_center_y:ti.f32,img_center_z:ti.f32,curved_dect:ti.i32,\
                                                      bool_apply_pmatrix:ti.i32, array_pmatrix_taichi:ti.template(), \
                                                          recon_view_mode: ti.i32, view_idx:ti.i32, z_source:ti.float32, \
                                                              delta_angle:ti.float32,dis_per_rad:ti.float32, img_recon_weight_taichi:ti.template(),\
                                                                  view_weight:ti.f32, angle_max:ti.f32, angle_min: ti.f32):
        

        
        for i_x, i_y, i_z in ti.ndrange(img_dim, img_dim, img_dim_z):
            x_after_rot = 0.0; y_after_rot = 0.0; x=0.0; y=0.0;z=0.0;
            #if recon_view_mode == 1: #axial view (from bottom to top)
            x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
            y_after_rot = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_y
            z = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_z
            # elif recon_view_mode == 2: #coronal view (from fron to back)
            #     x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
            #     z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_z
            #     y_after_rot = - (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_y
            # elif recon_view_mode == 3: #sagittal view (from left to right)
            #     z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_z
            #     y_after_rot = - img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_y
            #     x_after_rot = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_x
                
            x = + x_after_rot * ti.cos(img_rot) + y_after_rot * ti.sin(img_rot)
            y = - x_after_rot * ti.sin(img_rot) + y_after_rot * ti.cos(img_rot)
            
            
            theta = array_angle_taichi[view_idx]
            t = x*ti.sin(theta) - y * ti.cos(theta)
            
            #alpha is calculated to determine whether the data are from the extrapolated area
            alpha = theta + ti.asin(t / source_isocenter_dis)
            
            beta = ti.asin(t/source_isocenter_dis)
            l = source_isocenter_dis * ti.cos(beta) - x*ti.cos(theta) - y * ti.sin(theta)
            source_to_element_xy = source_isocenter_dis * ti.cos(beta) + ti.sqrt( (source_dect_dis - source_isocenter_dis)**2 - t **2 )
            v = source_to_element_xy / l * (z - (z_source- beta * dis_per_rad))
            cone_weight = source_to_element_xy / ti.sqrt(source_to_element_xy**2 + v**2)
            u_idx = (t - array_u_taichi[0]) / dect_elem_width
            v_idx = (v - array_v_taichi[0]) / (-dect_elem_height)
            
            u_idx_floor = int(ti.floor(u_idx))
            v_idx_floor = int(ti.floor(v_idx))
            
            Q = 0.9
            w_q = 0.0
            if u_idx_floor >=0 and u_idx_floor <= dect_elem_count_horizontal-2 \
                and v_idx_floor >=0 and v_idx_floor <= dect_elem_count_vertical_actual-2\
                    and alpha > angle_min and alpha < angle_max:
                    q = v / array_v_taichi[0] # array_v_taichi[0] is the maximum value of array_v_taichi
                    if q > Q:
                        w_q = ti.cos(3.14159/2.0 * abs(v - Q)/(1-Q))**2
                    else:
                        w_q = 1.0;
                    w_u = u_idx - u_idx_floor
                    w_v = v_idx - v_idx_floor
                    img_recon_taichi[i_z, i_y, i_x] += view_weight * w_q * cone_weight*delta_angle *\
                        (img_sgm_filtered_taichi[v_idx_floor,0,u_idx_floor] * (1-w_u) * (1-w_v)\
                                + img_sgm_filtered_taichi[v_idx_floor+1,0,u_idx_floor] * (1-w_u) * (w_v)\
                                + img_sgm_filtered_taichi[v_idx_floor,0,u_idx_floor+1] * (w_u) * (1-w_v)\
                                + img_sgm_filtered_taichi[v_idx_floor+1,0,u_idx_floor+1] * (w_u) * (w_v))
                    img_recon_weight_taichi[i_z, i_y, i_x] += view_weight *w_q
            

                

