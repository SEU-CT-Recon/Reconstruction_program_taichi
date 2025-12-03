# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 13:29:12 2025

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
from run_mgfbp_v2 import *

PI = 3.1415926536

#This version reconstruct the image view by view to save GPU memory

def run_mgfbp_cprebin(file_path):
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=0.95)#define device memeory utilization fraction
    print('Performing FBP (cone parallel rebin) from MandoCT-Taichi (ver 2.0) ...')
    print('This version reconstruct the image view by view to save GPU memory')
    print('Compatible for both FDK recon and helical recon')
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
    fbp = Mgfbp_cprebin(config_dict) #将config_dict数据以字典的形式送入对象
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
class Mgfbp_cprebin(Mgfbp_v2):
    def MainFunction(self):
        #Main function for reconstruction
        self.positive_u_is_positive_y = 1
        self.InitializeSinogramBuffer() #initialize sinogram buffer
        self.InitializeArrays() #initialize arrays
        A = self.array_rho_taichi.to_numpy()
        self.InitializeReconKernel() #initialize reconstruction kernel
        A = self.array_recon_kernel_taichi.to_numpy()
        plt.plot(A)
        self.file_processed_count = 0;#record the number of files processed
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):#match the file pattern
                if self.ReadSinogram(file):
                    self.file_processed_count += 1 
                    print('Reconstructing %s ...' % self.input_path)
                    for row_idx in range(self.det_elem_count_vertical_actual):
                        str = 'Rebinning detector row #%4d/%4d' % (row_idx+1, self.det_elem_count_vertical_actual)
                        print('\r' + str, end='')
                        self.img_sgm_taichi.from_numpy(self.img_sgm[row_idx:row_idx+1,:,:])
                        self.ConeParallelRebin(self.curved_dect,self.total_scan_angle,self.view_num,self.det_elem_count_horizontal,self.det_elem_count_horizontal_rebin,self.source_isocenter_dis,\
                                               self.source_det_dis, self.img_sgm_taichi, self.array_u_taichi, self.array_v_taichi, self.array_angle_taichi,\
                                               row_idx,self.array_rho_taichi,self.img_sgm_rebin_taichi_each_row)
                        self.img_sgm_rebin[row_idx:row_idx+1,:,:] = self.img_sgm_rebin_taichi_each_row.to_numpy()
                        
                    
                    print('\n')
                    for view_idx in range(self.view_num):
                        str = 'Filtering and reconstructing view #%4d/%4d' %(view_idx+1, self.view_num)
                        print('\r' + str, end='')
                        self.img_sgm_rebin_taichi_each_view.from_numpy(self.img_sgm_rebin[:,view_idx:view_idx+1,:])
                        if self.bool_bh_correction:
                            self.BHCorrection(self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal_rebin,self.img_sgm_rebin_taichi_each_view,\
                                              self.array_bh_coefficients_taichi,self.bh_corr_order)#pass img_sgm directly into this function using unified memory
                        self.WeightSgm(self.det_elem_count_vertical_actual,self.short_scan,self.curved_dect,\
                                        self.total_scan_angle,self.view_num,self.det_elem_count_horizontal_rebin,self.source_isocenter_dis,\
                                            self.source_det_dis,self.img_sgm_rebin_taichi_each_view,\
                                                self.array_rho_taichi,self.array_v_taichi,self.array_angle_taichi,view_idx)
                        self.FilterSinogram()
                        self.BackProjectionPixelDriven(self.det_elem_count_vertical_actual, self.img_dim, self.det_elem_count_horizontal_rebin, \
                                        self.view_num, self.det_elem_width_rebin,self.img_pix_size, self.source_isocenter_dis, self.source_det_dis, self.total_scan_angle,\
                                        self.array_angle_taichi, self.img_rot,self.img_sgm_filtered_taichi,self.img_recon_taichi,\
                                        self.array_rho_taichi,self.short_scan,self.cone_beam,self.det_elem_height,\
                                            self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                    self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode, view_idx)
                    self.SaveFilteredSinogram()
                    imwriteTiff(self.img_sgm_rebin_taichi_each_view.to_numpy().transpose(1,0,2),self.output_dir+'/'+ 'sgm_rebin_' + self.output_file,dtype=np.float32)
                    print('\nSaving to %s !' % self.output_path)
                    self.SetTruncatedRegionToZero(self.img_recon_taichi, self.img_dim, self.img_dim_z)
                    self.SaveReconImg()
    
    def __init__(self,config_dict):
        super(Mgfbp_v2,self).__init__(config_dict)
        self.img_recon_taichi.from_numpy(self.img_recon) #initialize img_recon_taichi (all-zero array)
        
    @ti.kernel
    def BackProjectionPixelDriven(self, det_elem_count_vertical_actual:ti.i32, img_dim:ti.i32, det_elem_count_horizontal:ti.i32, \
                                  view_num:ti.i32, det_elem_width:ti.f32,\
                                  img_pix_size:ti.f32, source_isocenter_dis:ti.f32, source_det_dis:ti.f32,total_scan_angle:ti.f32,\
                                      array_angle_taichi:ti.template(),img_rot:ti.f32,img_sgm_filtered_taichi:ti.template(),img_recon_taichi:ti.template(),\
                                          array_u_taichi:ti.template(), short_scan:ti.i32,cone_beam:ti.i32,det_elem_height:ti.f32,\
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
            #img_recon_taichi[i_z, i_y, i_x] = 0.0 this must be comment since we do not set image recon to zero for each view
            x_after_rot = 0.0; y_after_rot = 0.0; x=0.0; y=0.0;z=0.0;
            if recon_view_mode == 1: #axial view (from bottom to top)
                x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                y_after_rot = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_y
                z = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_z
            elif recon_view_mode == 2: #coronal view (from front to back)
                x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_z
                y_after_rot = - (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_y
            elif recon_view_mode == 3: #sagittal view (from right to left)
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
            
            mag_factor = 0.0
            rho_idx_floor = 0
            pix_proj_to_det_rho = 0.0
            pix_proj_to_det_v = 0.0
            pix_proj_to_det_rho_idx = 0.0
            pix_proj_to_det_v_idx = 0.0
            v_idx_floor = 0
            ratio_rho = 0.0
            ratio_v = 0.0
            angle_this_view_exclude_img_rot = array_angle_taichi[view_idx] - img_rot
            
            pix_proj_to_det_rho = x * ti.sin(angle_this_view_exclude_img_rot) - y* ti.cos(angle_this_view_exclude_img_rot)
            pix_proj_to_det_rho_idx = (pix_proj_to_det_rho - array_u_taichi[0]) / (array_u_taichi[1] - array_u_taichi[0])
            rho_idx_floor = ti.floor(pix_proj_to_det_rho_idx)
            
            gamma_temp = ti.asin( pix_proj_to_det_rho/ source_isocenter_dis)
            source_to_det_pix_dis_in_xy = source_det_dis / ti.cos(gamma_temp) #curved detector is different
            source_to_voxel_dis_in_xy = source_isocenter_dis * ti.cos(gamma_temp) - x * ti.cos(angle_this_view_exclude_img_rot) - y * ti.sin(angle_this_view_exclude_img_rot)
            mag_factor = source_to_det_pix_dis_in_xy / source_to_voxel_dis_in_xy
            pix_proj_to_det_v = mag_factor * z
            pix_proj_to_det_v_idx = (pix_proj_to_det_v - array_v_taichi[0]) / (array_v_taichi[1] - array_v_taichi[0])
            v_idx_floor = ti.floor(pix_proj_to_det_v_idx)
            
            if rho_idx_floor >=0 and  rho_idx_floor <= det_elem_count_horizontal-2:
                if v_idx_floor >=0 and  v_idx_floor <= det_elem_count_vertical_actual-2:
                    img_recon_taichi[i_z, i_y, i_x] += img_sgm_filtered_taichi[v_idx_floor,0,rho_idx_floor] * delta_angle
            else:
                img_recon_taichi[i_z, i_y, i_x] = -10000
        
    @ti.kernel
    def WeightSgm(self, det_elem_count_vertical_actual:ti.i32, short_scan:ti.i32, curved_dect:ti.i32, scan_angle:ti.f32,\
                  view_num:ti.i32, det_elem_count_horizontal:ti.i32,source_isocenter_dis:ti.f32, source_det_dis:ti.f32,img_sgm_taichi:ti.template(),\
                      array_rho_taichi:ti.template(),array_v_taichi:ti.template(),array_angle_taichi:ti.template(), view_idx:ti.i32):
        #对正弦图做加权，包括fan beam的cos加权和短扫描加权
        for   j in ti.ndrange(det_elem_count_horizontal):
            u_actual = array_rho_taichi[j]
            for s in ti.ndrange(det_elem_count_vertical_actual):
                v_actual = array_v_taichi[s]
                if curved_dect:
                    img_sgm_taichi[s,0,j] = img_sgm_taichi[s,0,j] * source_det_dis / ((source_det_dis**2 + v_actual**2)**0.5)
                else:
                    gamma_temp = ti.asin(u_actual / source_isocenter_dis)
                    img_sgm_taichi[s,0,j]=(img_sgm_taichi[s,0,j] * source_det_dis / ti.cos(gamma_temp) ) \
                        / ((  (source_det_dis / ti.cos(gamma_temp))**2 + v_actual **2) ** 0.5)
        
    def InitializeReconKernel(self):
        self.array_recon_kernel_taichi = ti.field(dtype=ti.f32, shape=2*self.det_elem_count_horizontal_rebin-1)
        if 'HammingFilter' in self.config_dict:
            self.GenerateHammingKernel(self.det_elem_count_horizontal_rebin,self.det_elem_width_rebin,\
                                       self.kernel_param,1e10,1e10,self.array_recon_kernel_taichi,self.curved_dect, self.dbt_or_not)
            #计算hamming核存储在array_recon_kernel_taichi
            
        elif 'GaussianApodizedRamp' in self.config_dict:
            self.GenerateGassianKernel(self.det_elem_count_horizontal_rebin,self.det_elem_width_rebin,\
                                       self.kernel_param,self.array_kernel_gauss_taichi)
            #计算高斯核存储在array_kernel_gauss_taichi
            self.GenerateHammingKernel(self.det_elem_count_horizontal_rebin,self.det_elem_width_rebin,1,\
                                       1e10,1e10,self.array_kernel_ramp_taichi,self.curved_dect, self.dbt_or_not)
            #1.以hamming参数1调用一次hamming核处理运算结果存储在array_kernel_ramp_taichi
            self.ConvolveKernelAndKernel(self.det_elem_count_horizontal_rebin,self.det_elem_width_rebin,\
                                         self.array_kernel_ramp_taichi,self.array_kernel_gauss_taichi,self.array_recon_kernel_taichi)
            #2.将计算出来的高斯核array_kernel_gauss_taichi与以hamming参数1计算出来的hamming核array_kernel_ramp_taichi进行一次运算得到新的高斯核存储在array_recon_kernel_taichi
        
        self.GenerateGassianKernel(self.det_elem_count_vertical_actual,self.det_elem_height,\
                                       self.det_elem_vertical_gauss_filter_size,self.array_kernel_gauss_vertical_taichi)
    
    @ti.kernel
    def ConeParallelRebin(self, curved_dect:ti.i32, scan_angle:ti.f32,\
                  view_num:ti.i32, det_elem_count_horizontal:ti.i32, det_elem_count_horizontal_rebin:ti.i32,source_iso_dis:ti.f32, source_det_dis:ti.f32,img_sgm_taichi:ti.template(),\
                      array_u_taichi:ti.template(),array_v_taichi:ti.template(),array_angle_taichi:ti.template(), row_idx:ti.i32, array_rho_taichi:ti.template(),\
                          img_sgm_rebin_taichi_each_row:ti.template()):
        
        for view_idx, rho_idx in ti.ndrange(view_num, det_elem_count_horizontal_rebin):
            gamma_temp = ti.asin(array_rho_taichi[rho_idx]/source_iso_dis)
            beta_temp = array_angle_taichi[view_idx] - gamma_temp
            if beta_temp<0:
                beta_temp = beta_temp + 2*PI
            elif beta_temp > 2*PI:
                beta_temp = beta_temp - 2*PI
            u_temp = 0.0
            if curved_dect:
                u_temp = source_det_dis * gamma_temp
            else:
                u_temp = source_det_dis * ti.tan(gamma_temp)
            u_idx = (u_temp - array_u_taichi[0]) / ( array_u_taichi[1] -  array_u_taichi[0])
            beta_idx = (beta_temp - array_angle_taichi[0]) / ( array_angle_taichi[1] -  array_angle_taichi[0])
            u_idx_floor = ti.floor(u_idx)
            beta_idx_floor = ti.floor(beta_idx)
            ratio_u = u_idx - u_idx_floor
            ratio_beta = beta_idx - beta_idx_floor
            if u_idx_floor >=0 and u_idx_floor <= det_elem_count_horizontal - 2:
                if beta_idx_floor >=0 and beta_idx_floor <= view_num - 1:
                    #todo fine-tune the interpolation
                    img_sgm_rebin_taichi_each_row[int(0),ti.int32(view_idx),ti.int32(rho_idx)] = (1-ratio_u)*img_sgm_taichi[0,ti.int32(beta_idx_floor),ti.int32(u_idx_floor)] +\
                        (ratio_u)*img_sgm_taichi[0,ti.int32(beta_idx_floor),ti.int32(u_idx_floor+ 1) ]
                    
            
    
    def InitializeSinogramBuffer(self): #new definition of InitializeSinogramBuffer function
        #cone parallel rebin is performed row-by-row; sgm is input to gpu ram row by row
        #this is different from mgfbp_v2
        self.img_sgm =  np.zeros((self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal), dtype = np.float32)
        self.img_sgm_taichi = ti.field(dtype=ti.f32, shape=(1,self.view_num, self.det_elem_count_horizontal))
        
        self.det_elem_count_horizontal_rebin =  self.det_elem_count_horizontal *  3
        
        self.gamma_max = np.arctan((self.det_elem_count_horizontal * self.det_elem_width / 2.0 + abs(self.det_offset_horizontal)) / self.source_det_dis)
        self.det_elem_width_rebin = np.sin(self.gamma_max) * self.source_isocenter_dis * 2 / (self.det_elem_count_horizontal_rebin)
        
        self.array_rho_taichi = ti.field(dtype=ti.f32,shape = self.det_elem_count_horizontal_rebin)
        self.GenerateDectPixPosArray(self.det_elem_count_horizontal_rebin,  self.det_elem_count_horizontal_rebin,\
                                     (self.positive_u_is_positive_y) *self.det_elem_width_rebin, 0 * self.det_offset_horizontal,self.array_rho_taichi, 0)
        
        self.img_sgm_rebin_taichi_each_row = ti.field(dtype=ti.f32, shape=(1,self.view_num, self.det_elem_count_horizontal_rebin))
        self.img_sgm_rebin = np.zeros((self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal_rebin), dtype = np.float32)
        self.img_sgm_rebin_taichi_each_view = ti.field(dtype=ti.f32, shape=(self.det_elem_count_vertical_actual,1, self.det_elem_count_horizontal_rebin))
        
        
        
        self.img_sgm_filtered_taichi = ti.field(dtype=ti.f32, shape=(self.det_elem_count_vertical_actual,1,self.det_elem_count_horizontal_rebin))
        
        if self.apply_gauss_vertical:
            self.img_sgm_filtered_intermediate_taichi = ti.field(dtype=ti.f32, shape=(self.det_elem_count_vertical_actual,1, self.det_elem_count_horizontal_rebin))
        else:
            self.img_sgm_filtered_intermediate_taichi = ti.field(dtype=ti.f32, shape=(1,1,1))
            #if vertical gauss filter is not applied, initialize this intermediate sgm with a small size to save GPU memory
    
    def FilterSinogram(self):
        if self.kernel_name == 'None':
            self.img_sgm_filtered_taichi = self.img_sgm_rebin_taichi_each_view
            #non filtration is performed
        else:
            self.ConvolveSgmAndKernel(self.det_elem_count_vertical_actual,self.view_num,self.det_elem_count_horizontal_rebin, \
                                      self.det_elem_width_rebin, self.img_sgm_rebin_taichi_each_view, self.array_recon_kernel_taichi, \
                                          self.array_kernel_gauss_vertical_taichi,self.det_elem_height, self.apply_gauss_vertical,
                                          self.img_sgm_filtered_intermediate_taichi, self.img_sgm_filtered_taichi)

    

                

