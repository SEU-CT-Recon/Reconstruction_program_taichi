# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:17:27 2025

@author: xji
"""

import taichi as ti
import numpy as np
import os
from run_mgfbp_v2 import *
import gc

PI = 3.1415926536

def run_mgfbp_huashi_v2(file_path):
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=0.95)#define device memeory utilization fraction
    print('Performing FBP (HUASHI) from MandoCT-Taichi (ver 2.0) ...')
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
    fbp = Mgfbp_huashi_v2(config_dict) #将config_dict数据以字典的形式送入对象
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
class Mgfbp_huashi_v2(Mgfbp_v2):
    def MainFunction(self):
        #Main function for reconstruction
        self.InitializeSinogramBuffer() #initialize sinogram buffer
        self.InitializeArrays() #initialize arrays
        self.InitializeReconKernel() #initialize reconstruction kernel
        self.file_processed_count = 0;#record the number of files processed
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):#match the file pattern
                if self.ReadSinogram(file):
                    self.file_processed_count += 1 
                    print('Reconstructing %s ...' % self.input_path)
                    
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
                                                    self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode, view_idx, self.img_rot_y,self.det_rot)
    
                    print('\nSaving to %s !' % self.output_path)
                    self.SetTruncatedRegionToZero(self.img_recon_taichi, self.img_dim, self.img_dim_z)
                    self.SaveReconImg()
        return self.img_recon #函数返回重建
    
    def __init__(self,config_dict):
        super(Mgfbp_huashi_v2,self).__init__(config_dict) 
        if 'ImageRotationY' in config_dict: #image rotation along the Y direction, right-hand rule
            self.img_rot_y = config_dict['ImageRotationY'] / 180.0 * PI
            if not isinstance(self.img_rot_y, float) and not isinstance(self.img_rot_y, int):
                print('ERROR: ImageRotationY must be a number!')
                sys.exit()
        else:
            self.img_rot_y = 0.0
        print('Image Rotation Along Y = %.1f degrees' % (self.img_rot_y / PI * 180))
        # ImageRotationY is originally in degree; change it to rad
        # all angular variables are in rad unit
        
        if 'DetectorRotation' in config_dict: #detector rotation along the w direction, left-hand rule
            self.det_rot = config_dict['DetectorRotation'] / 180.0 * PI
            if not isinstance(self.det_rot, float) and not isinstance(self.det_rot, int):
                print('ERROR: DetectorRotation must be a number!')
                sys.exit()
        else:
            self.det_rot = 0.0
        print('Detector Rotation Along W = %.2f degrees' % (self.det_rot / PI * 180))
        # DetectorRotation is originally in degree; change it to rad
        # all angular variables are in rad unit
        
    @ti.kernel
    def BackProjectionPixelDriven(self, dect_elem_count_vertical_actual:ti.i32, img_dim:ti.i32, dect_elem_count_horizontal:ti.i32, \
                                  view_num:ti.i32, dect_elem_width:ti.f32,\
                                  img_pix_size:ti.f32, source_isocenter_dis:ti.f32, source_dect_dis:ti.f32,total_scan_angle:ti.f32,\
                                      array_angle_taichi:ti.template(),img_rot:ti.f32,img_sgm_filtered_taichi:ti.template(),img_recon_taichi:ti.template(),\
                                          array_u_taichi:ti.template(), short_scan:ti.i32,cone_beam:ti.i32,dect_elem_height:ti.f32,\
                                              array_v_taichi:ti.template(),img_dim_z:ti.i32,img_voxel_height:ti.f32, \
                                                  img_center_x:ti.f32,img_center_y:ti.f32,img_center_z:ti.f32,curved_dect:ti.i32,\
                                                      bool_apply_pmatrix:ti.i32, array_pmatrix_taichi:ti.template(), recon_view_mode: ti.i32, \
                                                          view_idx:ti.i32,img_rot_y:ti.f32,det_rot:ti.f32):
        
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
                
                #huashi
                y_hat = [0,1,0]
                x_hat = [ti.cos(img_rot_y),0,-ti.sin(img_rot_y)]
                z_hat = [ti.sin(img_rot_y),0,ti.cos(img_rot_y)]
                
                x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) * x_hat[0] + \
                    img_pix_size * (- i_y + (img_dim - 1) / 2.0)  * y_hat[0] + \
                        (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height * z_hat[0] + \
                            img_center_x
                y_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) * x_hat[1] + \
                    img_pix_size * (- i_y + (img_dim - 1) / 2.0)  * y_hat[1] + \
                        (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height * z_hat[1] + \
                            img_center_y
                z = img_pix_size * (i_x - (img_dim - 1) / 2.0) * x_hat[2] + \
                    img_pix_size * (- i_y + (img_dim - 1) / 2.0)  * y_hat[2] + \
                        (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height * z_hat[2] + \
                            img_center_z
                
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
                        
                        #huashi
                        pix_proj_to_dect_u_rot = pix_proj_to_dect_u * ti.cos(det_rot) + pix_proj_to_dect_v* ti.sin(det_rot)
                        pix_proj_to_dect_v_rot = -pix_proj_to_dect_u* ti.sin(det_rot) + pix_proj_to_dect_v* ti.cos(det_rot)
                        
                        pix_proj_to_dect_u_idx = (pix_proj_to_dect_u_rot - array_u_taichi[0]) / (array_u_taichi[1] - array_u_taichi[0])
                        
                        pix_proj_to_dect_v_idx = (pix_proj_to_dect_v_rot - array_v_taichi[0]) / dect_elem_height \
                            * abs(array_v_taichi[1] - array_v_taichi[0]) / (array_v_taichi[1] - array_v_taichi[0])
                        
                        temp_u_idx_floor = int(ti.floor(pix_proj_to_dect_u_idx))
                        ratio_u = pix_proj_to_dect_u_idx - temp_u_idx_floor
                        
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
              
    
    
    

                

