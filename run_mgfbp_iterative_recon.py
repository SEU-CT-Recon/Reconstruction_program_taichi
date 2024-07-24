# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:22:56 2024

@author: xji
"""

# %reset -f
# %clear
import taichi as ti
import numpy as np
import os
from crip.io import imwriteRaw
from run_mgfbp import *
from run_mgfpj import imaddRaw
import gc

def run_mgfbp_ir(file_path):
    start_time = time.time() 
    ti.reset()
    ti.init(arch=ti.gpu)
    print('Performing Helical Rebin and FBP from MandoCT-Taichi (ver 0.1) ...')
    # record start time point
    
    #Delete unnecessary warinings
    warnings.filterwarnings('ignore', category=UserWarning, \
                            message='The value of the smallest subnormal for <class \'numpy.float(32|64)\'> type is zero.')
   
    if not os.path.exists(file_path):
        print(f"ERROR: Config File {file_path} does not exist!")
        #Judge whether the config jsonc file exist
        sys.exit()
    config_dict = ReadConfigFile(file_path)#读入jsonc文件并以字典的形式存储在config_dict中
    
    fbp =  Mgfbp(config_dict) #将config_dict数据以字典的形式送入对象中
    if not os.path.exists(fbp.output_dir):
        os.makedirs(fbp.output_dir)
    img_recon_seed = fbp.MainFunction() #generate a seed from fbp reconstruction
    img_recon_seed = np.zeros((1,512, 512),dtype=np.float32)
    
    fbp = Mgfbp_ir(config_dict) #将config_dict数据以字典的形式送入对象中
    # Ensure output directory exists; if not, create the directory
    if not os.path.exists(fbp.output_dir):
        os.makedirs(fbp.output_dir)
    img_recon = fbp.MainFunction(img_recon_seed)#use the seed to initialize the iterative process
    gc.collect()# 手动触发垃圾回收
    ti.reset()# free gpu ram
    end_time = time.time()# record end time point
    execution_time = end_time - start_time# 计算执行时间
    if fbp.file_processed_count > 0:
        print(f"\nA total of {fbp.file_processed_count:d} file(s) are reconstructed!")
        print(f"Time cost：{execution_time:.3} sec\n")# 打印执行时间（以秒为单位）
    else:
        print(f"\nWarning: Did not find files like {fbp.input_files_pattern:s} in {fbp.input_dir:s}.")
        print("No images are reconstructed!")
    del fbp #delete the fbp object
    return img_recon

# inherit a class from Mgfbp
@ti.data_oriented
class Mgfbp_ir(Mgfbp):
    def __init__(self,config_dict):
        super(Mgfbp_ir,self).__init__(config_dict)
        if 'HelicalPitch' in config_dict:
            self.helical_scan = True
            self.helical_pitch = config_dict['HelicalPitch']
        else:
            self.helical_scan = False
            self.helical_pitch = 0.0
        
        self.img_x = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_d = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_fp_x = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_r = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_b = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        
        self.img_fp_x = np.zeros((self.dect_elem_count_vertical, self.view_num,
                                self.dect_elem_count_horizontal), dtype=np.float32)
        
        self.img_fp_d = np.zeros((self.dect_elem_count_vertical, self.view_num,
                                self.dect_elem_count_horizontal), dtype=np.float32)
        
        self.img_bp_fp_d = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
   
    def MainFunction(self,img_recon_seed):
        #Main function for reconstruction
        self.InitializeArrays()#initialize arrays
        self.file_processed_count = 0;#record the number of files processed
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):#match the file pattern
                if self.ReadSinogram(file):
                    self.file_processed_count += 1 
                    print('\nReconstructing %s ...' % self.input_path)
                    print('Back Projection ...')
                    self.img_x = img_recon_seed
                    
                    #P^T b
                    self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                    self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                    self.array_angle_taichi, self.img_rot,self.img_sgm,self.img_bp_b,\
                                    self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                        self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                            self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
                        
                    #P^T P x
                    self.ForwardProjectionBilinear(self.img_x, self.img_fp_x, self.array_u_taichi,
                                                   self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                   self.dect_elem_count_horizontal,
                                                   self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                   self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                                   self.helical_scan, self.helical_pitch, 0, 1,
                                                   self.img_center_x, self.img_center_y, self.img_center_z, self.curved_dect)
                    
                    self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                    self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                    self.array_angle_taichi, self.img_rot,self.img_fp_x,self.img_bp_fp_x,\
                                    self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                        self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                            self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
                    
                    
                    self.img_r = self.img_bp_fp_x - self.img_bp_b #r
                    self.img_d = self.img_r #d
                    for idx in range(1000):
                        if idx%10==0:
                            print('%d\n' %idx)
                        
                        #P^T P d
                        self.ForwardProjectionBilinear(self.img_d, self.img_fp_d, self.array_u_taichi,
                                                       self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                       self.dect_elem_count_horizontal,
                                                       self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                       self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                                       self.helical_scan, self.helical_pitch, 0, 1,
                                                       self.img_center_x, self.img_center_y, self.img_center_z, self.curved_dect)
                        
                        self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                        self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                        self.array_angle_taichi, self.img_rot,self.img_fp_d,self.img_bp_fp_d,\
                                        self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                            self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                    self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
                            
                        alpha = - np.sum(np.multiply(self.img_d,self.img_r)) /np.sum(np.multiply(self.img_d,self.img_bp_fp_d)) 
                        self.img_x = self.img_x + np.multiply(alpha,self.img_d) 
                        self.img_r = self.img_r + np.multiply(alpha,self.img_bp_fp_d) 
                        
                        beta = - np.sum(np.multiply(self.img_r,self.img_bp_fp_d)) /np.sum(np.multiply(self.img_d,self.img_bp_fp_d)) 
                        self.img_d = self.img_r + np.multiply(beta,self.img_d)
                        
                        if idx%5==0:
                            imaddRaw(self.img_x,self.output_path,dtype=np.float32,idx = idx)
                        
                    
                    print('Saving to %s !' % self.output_path)
                    #self.SaveReconImg()
        return self.img_recon #函数返回重建图
    
    @ti.kernel
    def ForwardProjectionBilinear(self, img_image_taichi: ti.types.ndarray(dtype=ti.f32, ndim=3), img_sgm_large_taichi: ti.types.ndarray(dtype=ti.f32, ndim=3),
                                  array_u_taichi: ti.template(), array_v_taichi: ti.template(),
                                  array_angle_taichi: ti.template(), img_dim: ti.i32, img_dim_z: ti.i32,
                                  dect_elem_count_horizontal_oversamplesize: ti.i32,
                                  dect_elem_count_vertical: ti.i32, view_num: ti.i32,
                                  img_pix_size: ti.f32, img_voxel_height: ti.f32, source_isocenter_dis: ti.f32,
                                  source_dect_dis: ti.f32, cone_beam: ti.i32, helical_scan: ti.i32, helical_pitch: ti.f32,
                                  v_idx: ti.i32, fpj_step_size: ti.f32, img_center_x: ti.f32,
                                  img_center_y: ti.f32, img_center_z: ti.f32, curved_dect: ti.i32):

        # This new version of code assumes that the gantry stays the same
        # while the image object rotates
        # this can simplify the calculation

        # define aliases
        sid = source_isocenter_dis  # alias
        sdd = source_dect_dis  # alias

        # calculate the position of the source
        source_pos_x = sid
        source_pos_y = 0.0
        source_pos_z = 0.0

        img_dimension = img_dim * img_pix_size  # image dimension for each slice
        image_dimension_z = img_dim_z * img_voxel_height

        x = y = z = 0.0  # initialize position of the voxel current step
        # initialize position of the voxel for current step after rotation
        x_rot = y_rot = z_rot = 0.0
        x_idx = y_idx = z_idx = 0  # initialize index of the voxel for current step
        # weighting factor of the voxel for linear interpolation
        x_weight = y_weight = z_weight = 0.0

        # the most upper left image pixel position of the first slice
        x_0 = - (img_dim - 1.0) / 2.0 * img_pix_size + img_center_x
        y_0 = - (img_dim - 1.0) / 2.0 * (- img_pix_size) + img_center_y
        # by default, the first slice corresponds to the bottom of the image object
        z_0 = -(img_dim_z - 1.0) / 2.0 * img_voxel_height + img_center_z

        # initialize coordinate for the detector element
        dect_elem_pos_x = dect_elem_pos_y = dect_elem_pos_z = 0.0
        source_dect_elem_dis = 0.0  # initialize detector element to source distance
        # initialize detector element to source unit vector
        unit_vec_lambda_x = unit_vec_lambda_y = unit_vec_lambda_z = 0.0
        # lower range for the line integral
        l_min = sid - (2 * img_dimension ** 2 +
                       image_dimension_z ** 2)**0.5 / 2.0
        # upper range for the line integral
        l_max = sid + (2 * img_dimension ** 2 +
                       image_dimension_z ** 2)**0.5 / 2.0
        voxel_diagonal_size = (2*(img_pix_size ** 2) +
                               (img_voxel_height ** 2))**0.5
        sgm_val_lowerslice = sgm_val_upperslice = 0.0

        z_dis_per_view = 0.0
        if self.helical_scan:
            total_scan_angle = abs(
                (array_angle_taichi[view_num - 1] - array_angle_taichi[0])) / (view_num - 1) * view_num
            num_laps = total_scan_angle / (PI * 2)
            z_dis_per_view = helical_pitch * (num_laps / view_num) * (abs(
                array_v_taichi[1] - array_v_taichi[0]) * dect_elem_count_vertical) / (sdd / sid)

        # count of steps
        count_steps = int(
            ti.floor((l_max - l_min)/(fpj_step_size * voxel_diagonal_size)))

        for u_idx, angle_idx in ti.ndrange(dect_elem_count_horizontal_oversamplesize, view_num):

            if self.curved_dect:
                gamma_prime = (array_u_taichi[u_idx]) / sdd
                dect_elem_pos_x = -sdd * ti.cos(gamma_prime) + sid
                # positive u direction is - y
                dect_elem_pos_y = -sdd * ti.sin(gamma_prime)
            else:
                dect_elem_pos_x = - (sdd - sid)
                # positive u direction is - y
                dect_elem_pos_y = - array_u_taichi[u_idx]
                
            #add this distance to z position to simulate helical scan
            dect_elem_pos_z = array_v_taichi[v_idx] + z_dis_per_view * angle_idx
            # assume that the source and the detector moves upward for a helical scan (pitch>0)
            source_pos_z = z_dis_per_view * angle_idx

            source_dect_elem_dis = ((dect_elem_pos_x - source_pos_x)**2 + (
                dect_elem_pos_y - source_pos_y)**2 + (dect_elem_pos_z - source_pos_z)**2) ** 0.5

            unit_vec_lambda_x = (dect_elem_pos_x - source_pos_x) / source_dect_elem_dis
            unit_vec_lambda_y = (dect_elem_pos_y - source_pos_y) / source_dect_elem_dis
            unit_vec_lambda_z = (dect_elem_pos_z - source_pos_z) / source_dect_elem_dis

            temp_sgm_val = 0.0

            for step_idx in ti.ndrange(count_steps):
                x = source_pos_x + unit_vec_lambda_x * \
                    (step_idx * fpj_step_size * voxel_diagonal_size + l_min)
                y = source_pos_y + unit_vec_lambda_y * \
                    (step_idx * fpj_step_size * voxel_diagonal_size + l_min)
                z = source_pos_z + unit_vec_lambda_z * \
                    (step_idx * fpj_step_size * voxel_diagonal_size + l_min)

                x_rot = x * ti.cos(array_angle_taichi[angle_idx]) - \
                    y * ti.sin(array_angle_taichi[angle_idx])
                y_rot = y * ti.cos(array_angle_taichi[angle_idx]) + \
                    x * ti.sin(array_angle_taichi[angle_idx])
                z_rot = z

                x_idx = int(ti.floor((x_rot - x_0) / img_pix_size))
                y_idx = int(ti.floor((y_rot - y_0) / (- img_pix_size)))

                if x_idx >= 0 and x_idx+1 < img_dim and y_idx >= 0 and y_idx+1 < img_dim:
                    x_weight = (
                        x_rot - (x_idx * img_pix_size + x_0)) / img_pix_size
                    y_weight = (
                        y_rot - (y_idx * (- img_pix_size) + y_0)) / (- img_pix_size)

                    if self.cone_beam:
                        z_idx = int(ti.floor((z_rot - z_0) / img_voxel_height))

                        if z_idx >= 0 and z_idx + 1 < img_dim_z:
                            z_weight = (
                                z_rot - (z_idx * img_voxel_height + z_0)) / img_voxel_height
                            sgm_val_lowerslice = (1.0 - x_weight) * (1.0 - y_weight) * img_image_taichi[z_idx, y_idx, x_idx]\
                                + x_weight * (1.0 - y_weight) * img_image_taichi[z_idx, y_idx, x_idx+1]\
                                + (1.0 - x_weight) * y_weight * img_image_taichi[z_idx, y_idx+1, x_idx]\
                                + x_weight * y_weight * \
                                img_image_taichi[z_idx, y_idx+1, x_idx + 1]
                            sgm_val_upperslice = (1.0 - x_weight) * (1.0 - y_weight) * img_image_taichi[z_idx + 1, y_idx, x_idx]\
                                + x_weight * (1.0 - y_weight) * img_image_taichi[z_idx + 1, y_idx, x_idx+1]\
                                + (1.0 - x_weight) * y_weight * img_image_taichi[z_idx + 1, y_idx+1, x_idx]\
                                + x_weight * y_weight * \
                                img_image_taichi[z_idx + 1, y_idx+1, x_idx + 1]
                            temp_sgm_val += ((1.0 - z_weight) * sgm_val_lowerslice + z_weight *
                                             sgm_val_upperslice) * fpj_step_size * voxel_diagonal_size
                        else:
                            temp_sgm_val += 0.0
                    else:
                        z_idx = v_idx
                        sgm_val = (1 - x_weight) * (1 - y_weight) * img_image_taichi[z_idx, y_idx, x_idx]\
                            + x_weight * (1 - y_weight) * img_image_taichi[z_idx, y_idx, x_idx+1]\
                            + (1 - x_weight) * y_weight * img_image_taichi[z_idx, y_idx+1, x_idx]\
                            + x_weight * y_weight * \
                            img_image_taichi[z_idx, y_idx+1, x_idx + 1]
                        temp_sgm_val += sgm_val * fpj_step_size * voxel_diagonal_size

            img_sgm_large_taichi[0, angle_idx, u_idx] = temp_sgm_val
    
    @ti.kernel
    def BackProjectionPixelDriven(self, dect_elem_count_vertical_actual:ti.i32, img_dim:ti.i32, dect_elem_count_horizontal:ti.i32, \
                                  view_num:ti.i32, dect_elem_width:ti.f32,\
                                  img_pix_size:ti.f32, source_isocenter_dis:ti.f32, source_dect_dis:ti.f32,total_scan_angle:ti.f32,\
                                      array_angle_taichi:ti.template(),img_rot:ti.f32,img_sgm_filtered_taichi:ti.types.ndarray(dtype=ti.f32, ndim=3),img_recon_taichi:ti.types.ndarray(dtype=ti.f32, ndim=3),\
                                          array_u_taichi:ti.template(), short_scan:ti.i32,cone_beam:ti.i32,dect_elem_height:ti.f32,\
                                              array_v_taichi:ti.template(),img_dim_z:ti.i32,img_voxel_height:ti.f32, \
                                                  img_center_x:ti.f32,img_center_y:ti.f32,img_center_z:ti.f32,curved_dect:ti.i32,\
                                                      bool_apply_pmatrix:ti.i32, array_pmatrix_taichi:ti.template(), recon_view_mode: ti.i32):
        
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
        
        for i_x, i_y in ti.ndrange(img_dim, img_dim):
            for i_z in ti.ndrange(img_dim_z):
                img_recon_taichi[i_z, i_y, i_x] = 0.0
                x_after_rot = 0.0; y_after_rot = 0.0; x=0.0; y=0.0;z=0.0;
                if recon_view_mode == 1: #axial view (from bottom to top)
                    x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                    y_after_rot = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_y
                    z = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_z
                elif recon_view_mode == 2: #coronal view (from front to back)
                    x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                    z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_z
                    y_after_rot = - (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_y
                elif recon_view_mode == 3: #sagittal view (from left to right)
                    z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_z
                    y_after_rot = - img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_y
                    x_after_rot = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_x
                    
                x = + x_after_rot * ti.cos(img_rot) + y_after_rot * ti.sin(img_rot)
                y = - x_after_rot * ti.sin(img_rot) + y_after_rot * ti.cos(img_rot)
                for j in ti.ndrange(view_num):
                    #calculate angular interval for this view
                    delta_angle = 0.0
                    if j == view_num - 1:
                        delta_angle =  abs(array_angle_taichi[view_num-1] - array_angle_taichi[0]) / (view_num-1)
                    else:
                        delta_angle = abs(array_angle_taichi[j+1] - array_angle_taichi[j])
                    
                    pix_to_source_parallel_dis = 0.0
                    mag_factor = 0.0
                    temp_u_idx_floor = 0
                    pix_proj_to_dect_u = 0.0
                    pix_proj_to_dect_v = 0.0
                    pix_proj_to_dect_u_idx = 0.0
                    pix_proj_to_dect_v_idx = 0.0
                    ratio_u = 0.0
                    ratio_v = 0.0
                    angle_this_view_exclude_img_rot = array_angle_taichi[j] - img_rot
                    
                    pix_to_source_parallel_dis = source_isocenter_dis - x * ti.cos(angle_this_view_exclude_img_rot) - y * ti.sin(angle_this_view_exclude_img_rot)
                    if self.bool_apply_pmatrix == 0:
                        mag_factor = source_dect_dis / pix_to_source_parallel_dis
                        if curved_dect:
                            pix_proj_to_dect_u = source_dect_dis * ti.atan2(x*ti.sin(angle_this_view_exclude_img_rot)-y*ti.cos(angle_this_view_exclude_img_rot),pix_to_source_parallel_dis)
                        else:
                            pix_proj_to_dect_u = mag_factor * (x*ti.sin(angle_this_view_exclude_img_rot)-y*ti.cos(angle_this_view_exclude_img_rot))
                        pix_proj_to_dect_u_idx = (pix_proj_to_dect_u - array_u_taichi[0]) / dect_elem_width
                    else:
                        mag_factor = 1.0 / (array_pmatrix_taichi[12*j + 8] * x +\
                            array_pmatrix_taichi[12*j + 9] * y +\
                                array_pmatrix_taichi[12*j + 10] * z +\
                                    array_pmatrix_taichi[12*j + 11] * 1)
                        pix_proj_to_dect_u_idx = (array_pmatrix_taichi[12*j + 0] * x +\
                            array_pmatrix_taichi[12*j + 1] * y +\
                                array_pmatrix_taichi[12*j + 2] * z +\
                                    array_pmatrix_taichi[12*j + 3] * 1) * mag_factor
                    if pix_proj_to_dect_u_idx < 0 or  pix_proj_to_dect_u_idx + 1 > dect_elem_count_horizontal - 1:
                        img_recon_taichi[i_z, i_y, i_x] = 0
                        break
                    temp_u_idx_floor = int(ti.floor(pix_proj_to_dect_u_idx))
                    ratio_u = pix_proj_to_dect_u_idx - temp_u_idx_floor
                                        
                    
                    distance_weight = 1.0

                    if cone_beam == True:
                        if self.bool_apply_pmatrix == 0:
                            pix_proj_to_dect_v = mag_factor * z
                            pix_proj_to_dect_v_idx = (pix_proj_to_dect_v - array_v_taichi[0]) / dect_elem_height \
                                * abs(array_v_taichi[1] - array_v_taichi[0]) / (array_v_taichi[1] - array_v_taichi[0])
                                #abs(array_v_taichi[1] - array_v_taichi[0]) / (array_v_taichi[1] - array_v_taichi[0]) defines whether the first 
                                #sinogram slice corresponds to the top row
                        else:
                            pix_proj_to_dect_v_idx = (array_pmatrix_taichi[12*j + 4] * x +\
                                array_pmatrix_taichi[12*j + 5] * y +\
                                    array_pmatrix_taichi[12*j + 6] * z +\
                                        array_pmatrix_taichi[12*j + 7] * 1) * mag_factor
                                
                        temp_v_idx_floor = int(ti.floor(pix_proj_to_dect_v_idx))   #mark
                        if temp_v_idx_floor < 0 or temp_v_idx_floor + 1 > dect_elem_count_vertical_actual - 1:
                            img_recon_taichi[i_z, i_y, i_x] = 0
                            break
                        else:
                            ratio_v = pix_proj_to_dect_v_idx - temp_v_idx_floor
                            part_0 = img_sgm_filtered_taichi[temp_v_idx_floor,j,temp_u_idx_floor] * (1 - ratio_u) + \
                                img_sgm_filtered_taichi[temp_v_idx_floor,j,temp_u_idx_floor + 1] * ratio_u
                            part_1 = img_sgm_filtered_taichi[temp_v_idx_floor + 1,j,temp_u_idx_floor] * (1 - ratio_u) +\
                                  img_sgm_filtered_taichi[temp_v_idx_floor + 1,j,temp_u_idx_floor + 1] * ratio_u
                            img_recon_taichi[i_z, i_y, i_x] += ( distance_weight) * \
                                ((1 - ratio_v) * part_0 + ratio_v * part_1) * delta_angle * div_factor
                    else: 
                        val_0 = img_sgm_filtered_taichi[i_z , j , temp_u_idx_floor]
                        val_1 = img_sgm_filtered_taichi[i_z , j , temp_u_idx_floor + 1]
                        img_recon_taichi[i_z, i_y, i_x] += ( distance_weight) * \
                            ((1 - ratio_u) * val_0 + ratio_u * val_1) * delta_angle * div_factor
                            
    def SaveReconImg(self):
        self.img_recon = self.img_x
        if self.convert_to_HU:
            self.img_recon = (self.img_recon / self.water_mu - 1)*1000
        if self.output_file_format == 'raw':
            imwriteRaw(self.img_recon,self.output_path,dtype=np.float32)
        elif self.output_file_format == 'tif' or self.output_file_format == 'tiff':
            imwriteTiff(self.img_recon, self.output_path,dtype=np.float32)
        
     
