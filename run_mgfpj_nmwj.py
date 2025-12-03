# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:30:57 2024

@author: xji
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:45:27 2024

@author: xji
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 20:06:44 2024

@author: xji
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:06:09 2024

@author: xji
"""
#this new version of mgfpj inherits from mgfbp
#add pmatrix forward projection option
#may not be compatible with mgfpj.exe
#some parameter alias are not incorporated (e.g., DetectorZElementCount, DetectorZOffcenter)

#projection for each view, rather than for each slice in the ver 2

import warnings
import time
import json
import re
import taichi as ti
import sys
import os
import numpy as np
import gc
from crip.io import imwriteRaw
from crip.io import imwriteTiff
import time
from run_mgfbp import *
from run_mgfpj_v3 import *
PI = 3.1415926536


def run_mgfpj_nmwj(file_path):
    ti.reset()
    ti.init(arch=ti.gpu)
    print('Performing FPJ from MandoCT-Taichi (ver Namiweijing) ...')
    print('This new version of run_mgfpj inherits from run_mgfpj_v3.' )
    print('Curved Detector is with the isocenter as the center of the circle.' )
    print('[Rather than the source as the center of the circle.]')
    # record start time point
    start_time = time.time()
    # Delete unnecessary warinings
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='The value of the smallest subnormal for <class \'numpy.float(32|64)\'> type is zero.')
    if not os.path.exists(file_path):
        print(f"ERROR: Config File {file_path} does not exist!")
        # Judge whether the config jsonc file exist
        sys.exit()
    config_dict = ReadConfigFile(file_path)
    fpj = Mgfpj_nmwj(config_dict)
    img_sgm = fpj.MainFunction()
    end_time = time.time()
    execution_time = end_time - start_time  
    if fpj.file_processed_count > 0:
        print(
            f"\nA total of {fpj.file_processed_count:d} files are forward projected!")
        print(f"Time cost is {execution_time:.3} sec\n")
    else:
        print(
            f"\nWarning: Did not find files like {fpj.input_files_pattern:s} in {fpj.input_dir:s}.")
        print("No images are  forward projected!\n")
    gc.collect()  # 
    ti.reset()  # free gpu ram
    return img_sgm


@ti.data_oriented
class Mgfpj_nmwj(Mgfpj_v3):
    def MainFunction(self):
        self.file_processed_count = 0
        if not self.bool_uneven_scan_angle:
            self.GenerateAngleArray(
                self.view_num, self.img_rot, self.total_scan_angle, self.array_angle_taichi)
        self.GenerateDectPixPosArrayFPJ(self.det_elem_count_vertical, -self.det_elem_height, self.det_offset_vertical, self.array_v_taichi)
        self.GenerateDectPixPosArrayFPJ(self.det_elem_count_horizontal*self.oversample_size, -self.det_elem_width/self.oversample_size,
                                     -self.det_offset_horizontal, self.array_u_taichi)
        
        self.InitializeArrays()#initialize arrays; inherit from mgfpj
        if self.bool_uneven_scan_angle:
            self.array_angle_taichi.from_numpy(self.array_angle) #overlay array_angle_taichi for uneven scan angle

        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):
                if self.ReadImage(file):
                    print('\nForward projecting %s ...' % self.input_path)
                    self.file_processed_count += 1
                    for view_idx in range(self.view_num):
                        str = 'Forward projecting view: %4d/%4d' % (view_idx+1, self.view_num)
                        print('\r' + str, end='')
                        self.ForwardProjectionBilinear(self.img_image_taichi, self.img_sgm_large_taichi, self.array_u_taichi,
                                                       self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                       self.det_elem_count_horizontal*self.oversample_size,
                                                       self.det_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                       self.source_isocenter_dis, self.source_det_dis, self.cone_beam,
                                                       self.helical_scan, self.helical_pitch, view_idx, self.fpj_step_size,
                                                       self.img_center_x, self.img_center_y, self.array_img_center_z_taichi, self.curved_dect,\
                                                       self.matrix_A_each_view_taichi,self.x_s_each_view_taichi,self.bool_apply_pmatrix,\
                                                       self.det_elem_count_vertical_actual, self.det_elem_vertical_recon_range_begin,\
                                                           self.array_source_pos_z_taichi)

                        self.BinSinogram(self.img_sgm_large_taichi, self.img_sgm_taichi,
                                         self.det_elem_count_horizontal, self.det_elem_count_vertical, self.oversample_size)
                        if self.add_possion_noise:
                            self.AddPossionNoise(self.img_sgm_taichi, self.photon_number, self.det_elem_count_horizontal, self.det_elem_count_vertical)

                        self.TransferToRAM(view_idx)

                    print('\nSaving to %s !' % self.output_path)
                    self.SaveSinogram()

        return self.img_sgm

    def __init__(self, config_dict):
        super(Mgfpj_nmwj,self).__init__(config_dict)
        
        self.array_source_pos_z_taichi =  ti.field(dtype=ti.f32, shape = self.view_num)
        if 'SourcePositionZFile' in config_dict:
            temp_dict = ReadConfigFile(config_dict['SourcePositionZFile'])
            if 'Value' in temp_dict:
                self.array_source_pos_z = temp_dict['Value']
                if not isinstance(self.array_source_pos_z, list):
                    print('ERROR: SourcePositionZFile.Value is not an array')
                    sys.exit()
                if len(self.array_source_pos_z) != self.view_num:
                    print(f'ERROR: view number is {self.view_num:d} while SourcePositionZFile has {len(self.array_source_pos_z):d} elements!')
                    sys.exit()
                self.array_source_pos_z = np.array(self.array_source_pos_z, dtype = np.float32) 
                self.array_source_pos_z_taichi.from_numpy(self.array_source_pos_z.reshape(self.view_num))
                self.bool_source_pos_z_from_file = True
                print("--Source Position Z From File")
            else:
                print("ERROR: SourcePositionZFile has no member named 'Value'!")
                sys.exit()
        else:
            self.array_source_pos_z = np.zeros(shape = (self.view_num), dtype = np.float32)
            self.array_source_pos_z_taichi.from_numpy(self.array_source_pos_z)
            self.bool_source_pos_z_from_file = False
        

    @ti.kernel
    def ForwardProjectionBilinear(self, img_image_taichi: ti.template(), img_sgm_large_taichi: ti.template(),
                                  array_u_taichi: ti.template(), array_v_taichi: ti.template(),
                                  array_angle_taichi: ti.template(), img_dim: ti.i32, img_dim_z: ti.i32,
                                  det_elem_count_horizontal_oversamplesize: ti.i32,
                                  det_elem_count_vertical: ti.i32, view_num: ti.i32,
                                  img_pix_size: ti.f32, img_voxel_height: ti.f32, source_isocenter_dis: ti.f32,
                                  source_det_dis: ti.f32, cone_beam: ti.i32, helical_scan: ti.i32, helical_pitch: ti.f32,
                                  angle_idx: ti.i32, fpj_step_size: ti.f32, img_center_x: ti.f32,
                                  img_center_y: ti.f32, array_img_center_z_taichi: ti.template(), curved_dect: ti.i32, matrix_A_each_view_taichi: ti.template(),\
                                  x_s_each_view_taichi: ti.template(), bool_apply_pmatrix: ti.i32, \
                                  det_elem_count_vertical_actual: ti.i32, det_elem_vertical_recon_range_begin:ti.i32, \
                                      array_source_pos_z_taichi:ti.template()):

        # This new version of code assumes that the gantry stays stationary
        # while the image object rotates
        # this can simplify the calculation

        # define aliases
        sid = source_isocenter_dis  # alias
        sdd = source_det_dis  # alias

        # calculate the position of the source
        source_pos_x = sid
        source_pos_y = 0.0
        source_pos_z = 0.0

        img_dimension = img_dim * img_pix_size  # image size for each slice
        image_dimension_z = img_dim_z * img_voxel_height #image size z 

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
        z_0 = -(img_dim_z - 1.0) / 2.0 * img_voxel_height + array_img_center_z_taichi[0,angle_idx]

        # initialize coordinate for the detector element
        det_elem_pos_x = det_elem_pos_y = det_elem_pos_z = 0.0
        source_det_elem_dis = 0.0  # initialize detector element to source distance
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

        #calculate the distance that the gantry moves between adjacent views
        z_dis_per_view = 0.0
        if self.helical_scan:
            total_scan_angle = abs((array_angle_taichi[view_num - 1] - array_angle_taichi[0])) / (view_num - 1) * view_num
            num_laps = total_scan_angle / (PI * 2)
            z_dis_per_view = helical_pitch * (num_laps / view_num) * (abs(
                array_v_taichi[1] - array_v_taichi[0]) * det_elem_count_vertical) / (sdd / sid)
            #here pitch is calculated from det_elem_count_vertical, rather than det_elem_count_vertical_actual

        # number of steps
        count_steps = int(
            ti.floor((l_max - l_min)/(fpj_step_size * voxel_diagonal_size)))

        for u_idx, v_idx in ti.ndrange(det_elem_count_horizontal_oversamplesize, det_elem_count_vertical_actual):
            #v range from 0 to det_elem_count_vertical_actual - 1
            #caluclate the position of the detector element

            gamma_prime = ( - array_u_taichi[u_idx]) / (sdd - sid) #conterclockwise is positive, corresponding to -y direction
            det_elem_pos_x = - (sdd - sid) * ti.cos(gamma_prime)
            # positive u direction is - y
            det_elem_pos_y = - (sdd - sid) * ti.sin(gamma_prime)#negative gamma_prime corresponds to positive y
                
            #add this distance to z position to simulate helical scan
            det_elem_pos_z = array_v_taichi[v_idx] + z_dis_per_view * angle_idx + array_source_pos_z_taichi[angle_idx]
            # assume that the source and the detector moves upward for a helical scan (pitch>0)
            source_pos_z = z_dis_per_view * angle_idx + array_source_pos_z_taichi[angle_idx]
            #distance between the source and the detector element
            source_det_elem_dis = ((det_elem_pos_x - source_pos_x)**2 + (
                det_elem_pos_y - source_pos_y)**2 + (det_elem_pos_z - source_pos_z)**2) ** 0.5
            #calculate the unit vector of \vec(x_d - x_s)
            unit_vec_lambda_x = (det_elem_pos_x - source_pos_x) / source_det_elem_dis
            unit_vec_lambda_y = (det_elem_pos_y - source_pos_y) / source_det_elem_dis
            unit_vec_lambda_z = (det_elem_pos_z - source_pos_z) / source_det_elem_dis

            temp_sgm_val = 0.0
            one_over_mag = 0.0 
            # the inverse of the magnification factor for pmatrix forward projection (s)
            for step_idx in ti.ndrange(count_steps):
                # we did not use if bool_apply_pmatrix here is because we found this slows downs the computational speed
                
                #for pmatrix case
                #[x,y,z]^T = A * s * [u,v,1]^T + x_s^T
                # one_over_mag = (step_idx * fpj_step_size * voxel_diagonal_size + l_min) / source_det_elem_dis
                # x_p = one_over_mag * (matrix_A_each_view_taichi[angle_idx*9,0] * u_idx \
                #                         + matrix_A_each_view_taichi[angle_idx*9+1,0] * v_idx\
                #                             + matrix_A_each_view_taichi[angle_idx*9+2,0] * 1) \
                #                             + x_s_each_view_taichi[angle_idx*3,0]
                # y_p = one_over_mag * (matrix_A_each_view_taichi[angle_idx*9+3,0] * u_idx \
                #                         + matrix_A_each_view_taichi[angle_idx*9+4,0] * v_idx\
                #                             + matrix_A_each_view_taichi[angle_idx*9+5,0] * 1)\
                #                             + x_s_each_view_taichi[angle_idx*3+1,0]
                # z_p = one_over_mag * (matrix_A_each_view_taichi[angle_idx*9+6,0] * u_idx \
                #                         + matrix_A_each_view_taichi[angle_idx*9+7,0] * v_idx\
                #                             + matrix_A_each_view_taichi[angle_idx*9+8,0] * 1)\
                #                             + x_s_each_view_taichi[angle_idx*3+2,0] + z_dis_per_view * angle_idx
                #                             # for helical scan, if the gantry stay stationary, the object moves downward
                #                             # z coordinate of the projected area increases if helical pitch > 0
                # x_rot_p = x_p * ti.cos(array_angle_taichi[0]) - \
                #     y_p * ti.sin(array_angle_taichi[0])
                # y_rot_p = y_p * ti.cos(array_angle_taichi[0]) + \
                #     x_p * ti.sin(array_angle_taichi[0])#incorporate the image rotation angle into pmatrix
                # z_rot_p = z_p 
                
                #for none-pmatrix case                          
                x = source_pos_x + unit_vec_lambda_x * \
                    (step_idx * fpj_step_size * voxel_diagonal_size + l_min)
                y = source_pos_y + unit_vec_lambda_y * \
                    (step_idx * fpj_step_size * voxel_diagonal_size + l_min)
                z = source_pos_z + unit_vec_lambda_z * \
                    (step_idx * fpj_step_size * voxel_diagonal_size + l_min)
                x_rot_np = x * ti.cos(array_angle_taichi[angle_idx]) - \
                    y * ti.sin(array_angle_taichi[angle_idx])
                y_rot_np = y * ti.cos(array_angle_taichi[angle_idx]) + \
                    x * ti.sin(array_angle_taichi[angle_idx])
                z_rot_np = z
                
                # x_rot = x_rot_p * bool_apply_pmatrix + x_rot_np *(1 - bool_apply_pmatrix)
                # y_rot = y_rot_p * bool_apply_pmatrix + y_rot_np *(1 - bool_apply_pmatrix)
                # z_rot = z_rot_p * bool_apply_pmatrix + z_rot_np *(1 - bool_apply_pmatrix)
                
                x_rot = x_rot_np
                y_rot = y_rot_np
                z_rot = z_rot_np
                
                x_idx = int(ti.floor((x_rot - x_0) / img_pix_size))
                y_idx = int(ti.floor((y_rot - y_0) / (- img_pix_size)))

                if x_idx >= 0 and x_idx+1 < img_dim and y_idx >= 0 and y_idx+1 < img_dim:
                    x_weight = (
                        x_rot - (x_idx * img_pix_size + x_0)) / img_pix_size
                    y_weight = (
                        y_rot - (y_idx * (- img_pix_size) + y_0)) / (- img_pix_size)

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
                    

            img_sgm_large_taichi[v_idx + det_elem_vertical_recon_range_begin, u_idx] = temp_sgm_val
            #incorporate the vertical recon range


