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

def run_mgfbp_v2(file_path):
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=0.95)#define device memeory utilization fraction
    print('Performing FBP from MandoCT-Taichi (ver 2.0) ...')
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
    fbp = Mgfbp_v2(config_dict) #将config_dict数据以字典的形式送入对象
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
class Mgfbp_v2(Mgfbp):
    def MainFunction(self):
        #Main function for reconstruction
        self.InitializeArrays()#initialize arrays
        self.InitializeReconKernel()#initialize reconstruction kernel
        self.file_processed_count = 0;#record the number of files processed
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):#match the file pattern
                if self.ReadSinogram(file):
                    self.file_processed_count += 1 
                    print('\nReconstructing %s ...' % self.input_path)
                    for view_idx in range(self.view_num):
                        str = 'Processing view #%4d/%4d' % (view_idx+1, self.view_num)
                        print('\r' + str, end='')
                        self.img_sgm_taichi.from_numpy(self.img_sgm[:,view_idx:view_idx+1,:])
                        if self.bool_bh_correction:
                            self.BHCorrection(self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal,self.img_sgm_taichi,\
                                              self.array_bh_coefficients_taichi,self.bh_corr_order)#pass img_sgm directly into this function using unified memory
                        self.WeightSgm(self.dect_elem_count_vertical_actual,self.short_scan,self.curved_dect,\
                                       self.total_scan_angle,self.view_num,self.dect_elem_count_horizontal,\
                                           self.source_dect_dis,self.img_sgm_taichi,\
                                               self.array_u_taichi,self.array_v_taichi,self.array_angle_taichi,view_idx)#pass img_sgm directly into this function using unified memory
                        self.FilterSinogram()
                        self.SaveFilteredSinogram()
                        
                        self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                        self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis, self.total_scan_angle,\
                                        self.array_angle_taichi, self.img_rot,self.img_sgm_filtered_taichi,self.img_recon_taichi,\
                                        self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                            self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                    self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode, view_idx)
    
                    print('\nSaving to %s !' % self.output_path)
                    self.SetTruncatedRegionToZero(self.img_recon_taichi, self.img_dim, self.img_dim_z)
                    self.SaveReconImg()
        return self.img_recon #函数返回重建
    
    def __init__(self,config_dict):
        super(Mgfbp_v2,self).__init__(config_dict)
        
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

        
    @ti.kernel
    def BHCorrection(self, dect_elem_count_vertical_actual:ti.i32, view_num:ti.i32, dect_elem_count_horizontal:ti.i32,img_sgm_taichi:ti.template(),\
                     array_bh_coefficients_taichi:ti.template(),bh_corr_order:ti.i32):
        #对正弦图做加权，包括fan beam的cos加权和短扫面加权
        for j, s in ti.ndrange(dect_elem_count_horizontal, dect_elem_count_vertical_actual):
            temp_val = 0.0
            for t in ti.ndrange(bh_corr_order):
                temp_val = temp_val + array_bh_coefficients_taichi[t] * (img_sgm_taichi[s,0,j]**(t+1))#apply ploynomial calculation
            img_sgm_taichi[s,0,j] = temp_val

    @ti.kernel
    def WeightSgm(self, dect_elem_count_vertical_actual:ti.i32, short_scan:ti.i32, curved_dect:ti.i32, scan_angle:ti.f32,\
                  view_num:ti.i32, dect_elem_count_horizontal:ti.i32, source_dect_dis:ti.f32,img_sgm_taichi:ti.template(),\
                      array_u_taichi:ti.template(),array_v_taichi:ti.template(),array_angle_taichi:ti.template(), view_idx:ti.i32):
        #对正弦图做加权，包括fan beam的cos加权和短扫面加权
        for   j in ti.ndrange(dect_elem_count_horizontal):
            u_actual = array_u_taichi[j]
            for s in ti.ndrange(dect_elem_count_vertical_actual):
                v_actual = array_v_taichi[s]
                if curved_dect:
                    img_sgm_taichi[s,0,j] = img_sgm_taichi[s,0,j] * source_dect_dis * ti.math.cos(-u_actual/source_dect_dis) \
                        * source_dect_dis / ((source_dect_dis**2 + v_actual**2)**0.5)
                else:
                    img_sgm_taichi[s,0,j]=(img_sgm_taichi[s,0,j] * source_dect_dis * source_dect_dis ) \
                        / (( source_dect_dis **2 + u_actual**2 + v_actual **2) ** 0.5)
                if short_scan:
                    #for scans longer than 360 degrees but not muliples of 360, we also need to apply parker weighting
                    #for example, for a 600 degrees scan, we also need to apply parker weighting
                    num_rounds = ti.floor(abs(scan_angle) / (PI * 2))
                    remain_angle = abs(scan_angle) - num_rounds * PI * 2
                    #angle remains: e.g., if totalScanAngle = 600 degree, remain_angle = 240 degree
                    beta = abs(array_angle_taichi[view_idx] - array_angle_taichi[0])
                    rotation_direction =  abs(scan_angle) / (scan_angle)
                    gamma = 0.0
                    if curved_dect:
                        gamma = (-u_actual / source_dect_dis) * rotation_direction
                    else:
                        gamma = ti.atan2(-u_actual, source_dect_dis) * rotation_direction
                    gamma_max = remain_angle - PI
                    #maximum gamma defined by remain angle
                    #calculation of the parker weighting
                    weighting = 0.0
                    if remain_angle <= PI and num_rounds == 0: 
                        weighting = 1
                        # if remaining angle is less than 180 degree
                        # do not apply weighting
                    elif remain_angle <= PI and num_rounds >=1: 
                        if 0 <= beta < remain_angle:
                            weighting = beta / remain_angle
                        elif remain_angle <= beta < 2*PI*num_rounds:
                            weighting = 1
                        elif  2 * PI * num_rounds <= beta <= (2 * PI * num_rounds + remain_angle):
                            weighting = (2 * PI * num_rounds + remain_angle - beta ) / remain_angle
                    elif PI < remain_angle <= 2 * PI:
                        if 0 <= beta < (gamma_max - 2 * gamma):
                            weighting = ti.sin(PI / 2 * beta / (gamma_max - 2 * gamma))
                            weighting = weighting * weighting
                        elif (gamma_max - 2 * gamma) <= beta < (PI * (2 * num_rounds + 1) - 2 * gamma):
                            weighting = 1.0
                        elif (PI * (2 * num_rounds + 1) - 2 * gamma) <= beta <= (PI * (2 * num_rounds + 1) + gamma_max):
                            weighting = ti.sin(PI / 2 * (PI + gamma_max - (beta - PI * 2 * num_rounds)) / (gamma_max + 2 * gamma))
                            weighting = weighting * weighting
                    else:
                        weighting = 1.0
                    img_sgm_taichi[s,0,j] *= weighting
    
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
    def SetTruncatedRegionToZero(self,img_recon_taichi:ti.template(),img_dim:ti.i32,img_dim_z:ti.i32):
        for i_x, i_y, i_z in ti.ndrange(img_dim, img_dim, img_dim_z):
            if img_recon_taichi[i_z,i_x,i_y] < -5000:
                img_recon_taichi[i_z,i_x,i_y] = 0.0
        
    @ti.kernel
    def BackProjectionPixelDriven(self, dect_elem_count_vertical_actual:ti.i32, img_dim:ti.i32, dect_elem_count_horizontal:ti.i32, \
                                  view_num:ti.i32, dect_elem_width:ti.f32,\
                                  img_pix_size:ti.f32, source_isocenter_dis:ti.f32, source_dect_dis:ti.f32,total_scan_angle:ti.f32,\
                                      array_angle_taichi:ti.template(),img_rot:ti.f32,img_sgm_filtered_taichi:ti.template(),img_recon_taichi:ti.template(),\
                                          array_u_taichi:ti.template(), short_scan:ti.i32,cone_beam:ti.i32,dect_elem_height:ti.f32,\
                                              array_v_taichi:ti.template(),img_dim_z:ti.i32,img_voxel_height:ti.f32, \
                                                  img_center_x:ti.f32,img_center_y:ti.f32,img_center_z:ti.f32,curved_dect:ti.i32,\
                                                      bool_apply_pmatrix:ti.i32, array_pmatrix_taichi:ti.template(), recon_view_mode: ti.i32, view_idx:ti.i32):
        
        #计算冗余加权系数
        div_factor = 1.0
        num_rounds = float(ti.floor(abs(total_scan_angle) / (PI * 2)))
        if short_scan:
            remain_angle = abs(total_scan_angle) - num_rounds * 2 * PI
            if abs(total_scan_angle) < 2 * PI:
                div_factor = 1.0
            elif remain_angle < PI:
                div_factor = 1.0 / (num_rounds*2.0)
            elif PI < remain_angle < 2*PI:
                div_factor = 1.0 / (num_rounds*2.0 + 1.0)
        else:
            div_factor = 1.0 / (num_rounds*2.0)
        
        for i_x, i_y, i_z in ti.ndrange(img_dim, img_dim, img_dim_z):
            x_after_rot = 0.0; y_after_rot = 0.0; x=0.0; y=0.0;z=0.0;
            if recon_view_mode == 1: #axial view (from bottom to top)
                x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                y_after_rot = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_y
                z = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_z
            elif recon_view_mode == 2: #coronal view (from fron to back)
                x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_z
                y_after_rot = - (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_y
            elif recon_view_mode == 3: #sagittal view (from left to right)
                z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_z
                y_after_rot = - img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_y
                x_after_rot = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_x
                
            x = + x_after_rot * ti.cos(img_rot) + y_after_rot * ti.sin(img_rot)
            y = - x_after_rot * ti.sin(img_rot) + y_after_rot * ti.cos(img_rot)
            
            #calculate angular interval for this view
            delta_angle = 0.0
            if view_idx == view_num - 1:
                delta_angle =  abs(array_angle_taichi[view_num-1] - array_angle_taichi[0]) / (view_num-1)
            else:
                delta_angle = abs(array_angle_taichi[view_idx+1] - array_angle_taichi[view_idx])
            
            pix_to_source_parallel_dis = 0.0
            mag_factor = 0.0
            temp_u_idx_floor = 0
            pix_proj_to_dect_u = 0.0
            pix_proj_to_dect_v = 0.0
            pix_proj_to_dect_u_idx = 0.0
            pix_proj_to_dect_v_idx = 0.0
            ratio_u = 0.0
            ratio_v = 0.0
            angle_this_view_exclude_img_rot = array_angle_taichi[view_idx] - img_rot
            
            pix_to_source_parallel_dis = source_isocenter_dis - x * ti.cos(angle_this_view_exclude_img_rot) - y * ti.sin(angle_this_view_exclude_img_rot)
            if bool_apply_pmatrix == 0:
                mag_factor = source_dect_dis / pix_to_source_parallel_dis
                y_after_rotation_angle_this_view = - x*ti.sin(angle_this_view_exclude_img_rot) + y*ti.cos(angle_this_view_exclude_img_rot)
                if curved_dect:
                    pix_proj_to_dect_u = source_dect_dis * ti.atan2(y_after_rotation_angle_this_view, pix_to_source_parallel_dis)
                else:
                    pix_proj_to_dect_u = mag_factor * y_after_rotation_angle_this_view
                pix_proj_to_dect_u_idx = (pix_proj_to_dect_u - array_u_taichi[0]) / (array_u_taichi[1] - array_u_taichi[0])
            else:
                mag_factor = 1.0 / (array_pmatrix_taichi[12*view_idx + 8] * x +\
                    array_pmatrix_taichi[12*view_idx + 9] * y +\
                        array_pmatrix_taichi[12*view_idx + 10] * z +\
                            array_pmatrix_taichi[12*view_idx + 11] * 1)
                pix_proj_to_dect_u_idx = (array_pmatrix_taichi[12*view_idx + 0] * x +\
                    array_pmatrix_taichi[12*view_idx + 1] * y +\
                        array_pmatrix_taichi[12*view_idx + 2] * z +\
                            array_pmatrix_taichi[12*view_idx + 3] * 1) * mag_factor
            if pix_proj_to_dect_u_idx < 0 or  pix_proj_to_dect_u_idx + 1 > dect_elem_count_horizontal - 1:
                img_recon_taichi[i_z, i_y, i_x] = -10000 #mark the truncated region with -10000
            else:
                temp_u_idx_floor = int(ti.floor(pix_proj_to_dect_u_idx))
                ratio_u = pix_proj_to_dect_u_idx - temp_u_idx_floor
                                    
                
                distance_weight = 0.0
                if curved_dect:
                    distance_weight = 1.0 / ((pix_to_source_parallel_dis * pix_to_source_parallel_dis) + \
                                             (x * ti.sin(angle_this_view_exclude_img_rot) - y * ti.cos(angle_this_view_exclude_img_rot)) \
                                            * (x * ti.sin(angle_this_view_exclude_img_rot) - y * ti.cos(angle_this_view_exclude_img_rot)))
                else:
                    distance_weight = 1.0 / (pix_to_source_parallel_dis * pix_to_source_parallel_dis)
    
                if cone_beam == True:
                    if bool_apply_pmatrix == 0:
                        pix_proj_to_dect_v = mag_factor * z
                        pix_proj_to_dect_v_idx = (pix_proj_to_dect_v - array_v_taichi[0]) / dect_elem_height \
                            * abs(array_v_taichi[1] - array_v_taichi[0]) / (array_v_taichi[1] - array_v_taichi[0])
                            #abs(array_v_taichi[1] - array_v_taichi[0]) / (array_v_taichi[1] - array_v_taichi[0]) defines whether the first 
                            #sinogram slice corresponds to the top row
                    else:
                        pix_proj_to_dect_v_idx = (array_pmatrix_taichi[12*view_idx + 4] * x +\
                            array_pmatrix_taichi[12*view_idx + 5] * y +\
                                array_pmatrix_taichi[12*view_idx + 6] * z +\
                                    array_pmatrix_taichi[12*view_idx + 7] * 1) * mag_factor
                            
                    temp_v_idx_floor = int(ti.floor(pix_proj_to_dect_v_idx))   #mark
                    if temp_v_idx_floor < 0 or temp_v_idx_floor + 1 > dect_elem_count_vertical_actual - 1:
                        img_recon_taichi[i_z, i_y, i_x] = -10000 #mark the truncated region with -10000
                    else:
                        ratio_v = pix_proj_to_dect_v_idx - temp_v_idx_floor
                        part_0 = img_sgm_filtered_taichi[temp_v_idx_floor,0,temp_u_idx_floor] * (1 - ratio_u) + \
                            img_sgm_filtered_taichi[temp_v_idx_floor,0,temp_u_idx_floor + 1] * ratio_u
                        part_1 = img_sgm_filtered_taichi[temp_v_idx_floor + 1,0,temp_u_idx_floor] * (1 - ratio_u) +\
                              img_sgm_filtered_taichi[temp_v_idx_floor + 1,0,temp_u_idx_floor + 1] * ratio_u
                        img_recon_taichi[i_z, i_y, i_x] += (source_isocenter_dis * distance_weight) * \
                            ((1 - ratio_v) * part_0 + ratio_v * part_1) * delta_angle * div_factor
                else: 
                    val_0 = img_sgm_filtered_taichi[i_z , 0 , temp_u_idx_floor]
                    val_1 = img_sgm_filtered_taichi[i_z , 0 , temp_u_idx_floor + 1]
                    img_recon_taichi[i_z, i_y, i_x] += (source_isocenter_dis * distance_weight) * \
                        ((1 - ratio_u) * val_0 + ratio_u * val_1) * delta_angle * div_factor
            

                

