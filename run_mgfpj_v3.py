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
from run_mgfpj_v2 import *
PI = 3.1415926536
ti.init(default_fp=ti.f32)

def run_mgfpj_v3(file_path):
    ti.reset()
    ti.init(arch=ti.gpu)
    print('Performing FPJ from MandoCT-Taichi (ver 3.0) ...')
    print('This new version of run_mgfpj inherits from run_mgfpj_v2.' )
    print('Add PMatrix forward projection option.' )
    print('View by view forward projection with arbitrary view angle and z position. ')
    print('Is not fully compatible with mgfpj.exe. ')
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
    fpj = Mgfpj_v3(config_dict)
    img_sgm = fpj.MainFunction()
    end_time = time.time()
    execution_time = end_time - start_time  
    if fpj.file_processed_count > 0:
        print(
            f"\nA total of {fpj.file_processed_count:d} files are forward projected!")
        print(f"Time cost is {execution_time:.3} sec\n")
    else:
        print(
            f"\nWarning: Did not find files like {fpj.input_files_pattern:s} in {fpj.input_dir:s}.")
        print("No images are  forward projected!\n")
    gc.collect()  # 
    ti.reset()  # free gpu ram
    return img_sgm


@ti.data_oriented
class Mgfpj_v3(Mgfpj):
    def MainFunction(self):
        self.file_processed_count = 0
        if not self.bool_uneven_scan_angle:
            self.GenerateAngleArray(
                self.view_num, self.img_rot, self.total_scan_angle, self.array_angle_taichi)
        self.GenerateDectPixPosArrayFPJ(self.dect_elem_count_vertical, - self.dect_elem_height, self.dect_offset_vertical, self.array_v_taichi)
        self.GenerateDectPixPosArrayFPJ(self.dect_elem_count_horizontal*self.oversample_size, -self.dect_elem_width/self.oversample_size,
                                     -self.dect_offset_horizontal, self.array_u_taichi)
        
        self.InitializeArrays()#initialize arrays; inherit from mgfpj
        #array_v_taichi is the detector element coordinates along z
        #array_u_taichi is the detector element coordinates along y
        #self.dect_offset is from projection of rotation center to detector center
        #self.dect_offset_horizontal along -y direction is positive as a convention from mgfbp.exe
        #self.dect_offset_vertical along +z direction is positive 
        #+u and +v direction are along -z and -y direction by convention 

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
                                                       self.dect_elem_count_horizontal*self.oversample_size,
                                                       self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                       self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                                       self.helical_scan, self.helical_pitch, view_idx, self.fpj_step_size,
                                                       self.img_center_x, self.img_center_y, self.array_img_center_z_taichi, self.curved_dect,\
                                                       self.matrix_A_each_view_taichi,self.x_s_each_view_taichi,self.bool_apply_pmatrix,\
                                                       self.dect_elem_count_vertical_actual, self.dect_elem_vertical_recon_range_begin)

                        self.BinSinogram(self.img_sgm_large_taichi, self.img_sgm_taichi,
                                         self.dect_elem_count_horizontal, self.dect_elem_count_vertical, self.oversample_size)
                        if self.add_possion_noise:
                            self.AddPossionNoise(self.img_sgm_taichi, self.photon_number, self.dect_elem_count_horizontal, self.dect_elem_count_vertical)

                        self.TransferToRAM(view_idx)

                    print('\nSaving to %s !' % self.output_path)
                    self.SaveSinogram()

        return self.img_sgm

    def __init__(self, config_dict):
        super(Mgfpj_v3,self).__init__(config_dict)
        if 'ScanAngleFile' in config_dict:
            temp_dict = ReadConfigFile(config_dict['ScanAngleFile'])
            if 'Value' in temp_dict:
                self.array_angle = temp_dict['Value']
                if not isinstance(self.array_angle,list):
                    print('ERROR: ScanAngleFile.Value is not an array')
                    sys.exit()
                if len(self.array_angle) != self.view_num:
                    print(f'ERROR: view number is {self.view_num:d} while ScanAngleFile has {len(self.array_angle):d} elements!')
                    sys.exit()
                self.array_angle = np.array(self.array_angle,dtype = np.float32) / 180.0 * PI + self.img_rot
                self.array_angle_taichi.from_numpy(self.array_angle)
                self.bool_uneven_scan_angle = 1
                print("--Scan Angles From File (Original total scan angle value is discarded!)")
            else:
                print("ERROR: ScanAngleFile has no member named 'Value'!")
                sys.exit()
        else:
            self.bool_uneven_scan_angle = False
            
        self.array_img_center_z_taichi =  ti.field(dtype=ti.f32, shape = (1,self.view_num))
        if 'ImageCenterZFile' in config_dict:
            temp_dict = ReadConfigFile(config_dict['ImageCenterZFile'])
            if 'Value' in temp_dict:
                self.array_img_center_z = temp_dict['Value']
                if not isinstance(self.array_img_center_z, list):
                    print('ERROR: ImageCenterZFile.Value is not an array')
                    sys.exit()
                if len(self.array_img_center_z) != self.view_num:
                    print(f'ERROR: view number is {self.view_num:d} while ImageCenterZFile has {len(self.array_img_center_z):d} elements!')
                    sys.exit()
                self.array_img_center_z = np.array(self.array_img_center_z, dtype = np.float32) 
                self.array_img_center_z_taichi.from_numpy(self.array_img_center_z.reshape(1,self.view_num))
                self.bool_image_center_z_from_file = True
                print("--Image Center Z From File (defalt value is discarded)")
            else:
                print("ERROR: ImageCenterZFile has no member named 'Value'!")
                sys.exit()
        else:
            self.array_img_center_z = np.ones(shape = (1,self.view_num), dtype = np.float32) * self.img_center_z
            self.array_img_center_z_taichi.from_numpy(self.array_img_center_z)
            self.bool_image_center_z_from_file = False
            
        del self.img_sgm_large_taichi
        del self.img_sgm_taichi
        del self.array_v_taichi
        self.array_v_taichi = ti.field(dtype = ti.f32,shape = self.dect_elem_count_vertical)
        self.img_sgm_large_taichi = ti.field(dtype=ti.f32, shape=(
            self.dect_elem_count_vertical,self.dect_elem_count_horizontal*self.oversample_size))
        self.img_sgm_taichi = ti.field(dtype=ti.f32, shape=(
            self.dect_elem_count_vertical,self.dect_elem_count_horizontal))
        

    @ti.kernel
    def ForwardProjectionBilinear(self, img_image_taichi: ti.template(), img_sgm_large_taichi: ti.template(),
                                  array_u_taichi: ti.template(), array_v_taichi: ti.template(),
                                  array_angle_taichi: ti.template(), img_dim: ti.i32, img_dim_z: ti.i32,
                                  dect_elem_count_horizontal_oversamplesize: ti.i32,
                                  dect_elem_count_vertical: ti.i32, view_num: ti.i32,
                                  img_pix_size: ti.f32, img_voxel_height: ti.f32, source_isocenter_dis: ti.f32,
                                  source_dect_dis: ti.f32, cone_beam: ti.i32, helical_scan: ti.i32, helical_pitch: ti.f32,
                                  angle_idx: ti.i32, fpj_step_size: ti.f32, img_center_x: ti.f32,
                                  img_center_y: ti.f32, array_img_center_z_taichi: ti.template(), curved_dect: ti.i32, matrix_A_each_view_taichi: ti.template(),\
                                  x_s_each_view_taichi: ti.template(), bool_apply_pmatrix: ti.i32, \
                                  dect_elem_count_vertical_actual: ti.i32, dect_elem_vertical_recon_range_begin:ti.i32):

        # This new version of code assumes that the gantry stays stationary
        # while the image object rotates
        # this can simplify the calculation

        # define aliases
        sid = source_isocenter_dis  # alias
        sdd = source_dect_dis  # alias

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
        z_0 = -(img_dim_z - 1.0) / 2.0 * img_voxel_height + array_img_center_z_taichi[0, angle_idx]
        
        # initialize coordinate for the detector element
        dect_elem_pos_x = dect_elem_pos_y = dect_elem_pos_z = 0.0
        source_dect_elem_dis = 0.0  # initialize detector element to source distance
        # initialize detector element to source unit vector
        unit_vec_lambda_x = unit_vec_lambda_y = unit_vec_lambda_z = 0.0
        # lower range for the line integral
        l_min = sid - (2 * img_dimension ** 2 +\
                       image_dimension_z ** 2)**0.5 / 2.0
        # upper range for the line integral
        l_max = sid + (2 * img_dimension ** 2 +\
                       image_dimension_z ** 2)**0.5 / 2.0
        voxel_diagonal_size = (2*(img_pix_size ** 2) +\
                               (img_voxel_height ** 2))**0.5
        sgm_val_lowerslice = sgm_val_upperslice = 0.0


        #calculate the distance that the gantry moves between adjacent views
        z_dis_per_view = 0.0
        if self.helical_scan:
            total_scan_angle = abs((array_angle_taichi[view_num - 1] - array_angle_taichi[0])) / (view_num - 1) * view_num
            num_laps = total_scan_angle / (PI * 2)
            z_dis_per_view = helical_pitch * (num_laps / view_num) * (abs(
                array_v_taichi[1] - array_v_taichi[0]) * dect_elem_count_vertical) / (sdd / sid)
            #here pitch is calculated from dect_elem_count_vertical, rather than dect_elem_count_vertical_actual

        # number of steps
        count_steps = int(ti.floor(((2 * img_dimension ** 2 +image_dimension_z ** 2)**0.5)/(fpj_step_size * voxel_diagonal_size)))
        

        for u_idx, v_idx in ti.ndrange(dect_elem_count_horizontal_oversamplesize, dect_elem_count_vertical_actual):
            #v range from 0 to dect_elem_count_vertical_actual - 1
            #caluclate the position of the detector element
            if self.curved_dect:
                gamma_prime = ( - array_u_taichi[u_idx]) / sdd #conterclockwise is positive, corresponding to -y direction
                dect_elem_pos_x = -sdd * ti.cos(gamma_prime) + sid
                # positive u direction is - y
                dect_elem_pos_y = -sdd * ti.sin(gamma_prime)#negative gamma_prime corresponds to positive y
            else:
                dect_elem_pos_x = - (sdd - sid)
                # positive u direction is - y
                dect_elem_pos_y = array_u_taichi[u_idx]
                
            #add this distance to z position to simulate helical scan
            dect_elem_pos_z = array_v_taichi[v_idx] + z_dis_per_view * angle_idx
            # assume that the source and the detector moves upward for a helical scan (pitch>0)
            source_pos_z = z_dis_per_view * angle_idx
            #distance between the source and the detector element
            source_dect_elem_dis = ((dect_elem_pos_x - source_pos_x)**2 + (
                dect_elem_pos_y - source_pos_y)**2 + (dect_elem_pos_z - source_pos_z)**2) ** 0.5
            #calculate the unit vector of \vec(x_d - x_s)
            unit_vec_lambda_x = (dect_elem_pos_x - source_pos_x) / source_dect_elem_dis
            unit_vec_lambda_y = (dect_elem_pos_y - source_pos_y) / source_dect_elem_dis
            unit_vec_lambda_z = (dect_elem_pos_z - source_pos_z) / source_dect_elem_dis

            temp_sgm_val = 0.0
            one_over_mag = 0.0 
            # the inverse of the magnification factor for pmatrix forward projection (s)
            for step_idx in ti.ndrange(count_steps):
                # we did not use if bool_apply_pmatrix here is because we found this slows downs the computational speed
                
                #for pmatrix case
                #[x,y,z]^T = A * s * [u,v,1]^T + x_s^T
                one_over_mag = (step_idx * fpj_step_size * voxel_diagonal_size + l_min) / source_dect_elem_dis
                x_p = one_over_mag * (matrix_A_each_view_taichi[angle_idx*9,0] * u_idx \
                                        + matrix_A_each_view_taichi[angle_idx*9+1,0] * v_idx\
                                            + matrix_A_each_view_taichi[angle_idx*9+2,0] * 1) \
                                            + x_s_each_view_taichi[angle_idx*3,0]
                y_p = one_over_mag * (matrix_A_each_view_taichi[angle_idx*9+3,0] * u_idx \
                                        + matrix_A_each_view_taichi[angle_idx*9+4,0] * v_idx\
                                            + matrix_A_each_view_taichi[angle_idx*9+5,0] * 1)\
                                            + x_s_each_view_taichi[angle_idx*3+1,0]
                z_p = one_over_mag * (matrix_A_each_view_taichi[angle_idx*9+6,0] * u_idx \
                                        + matrix_A_each_view_taichi[angle_idx*9+7,0] * v_idx\
                                            + matrix_A_each_view_taichi[angle_idx*9+8,0] * 1)\
                                            + x_s_each_view_taichi[angle_idx*3+2,0] + z_dis_per_view * angle_idx
                                            # for helical scan, if the gantry stay stationary, the object moves downward
                                            # z coordinate of the projected area increases if helical pitch > 0
                x_rot_p = x_p * ti.cos(array_angle_taichi[0]) - \
                    y_p * ti.sin(array_angle_taichi[0])
                y_rot_p = y_p * ti.cos(array_angle_taichi[0]) + \
                    x_p * ti.sin(array_angle_taichi[0])#incorporate the image rotation angle into pmatrix
                z_rot_p = z_p 
                
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
                
                x_rot = x_rot_p * bool_apply_pmatrix + x_rot_np *(1 - bool_apply_pmatrix)
                y_rot = y_rot_p * bool_apply_pmatrix + y_rot_np *(1 - bool_apply_pmatrix)
                z_rot = z_rot_p * bool_apply_pmatrix + z_rot_np *(1 - bool_apply_pmatrix)
                
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

            img_sgm_large_taichi[v_idx + dect_elem_vertical_recon_range_begin, u_idx] = temp_sgm_val
            #incorporate the vertical recon range

    @ti.kernel
    def BinSinogram(self, img_sgm_large_taichi: ti.template(), img_sgm_taichi: ti.template(), dect_elem_count_horizontal: ti.i32,
                    dect_elem_count_vertical: ti.i32, bin_size: ti.i32):
        for v_idx, u_idx in ti.ndrange(dect_elem_count_vertical, dect_elem_count_horizontal):
            img_sgm_taichi[v_idx, u_idx] = 0.0
            for i in ti.ndrange(bin_size):
                img_sgm_taichi[v_idx, u_idx] += img_sgm_large_taichi[v_idx, u_idx * bin_size + i]
            img_sgm_taichi[v_idx, u_idx] /= bin_size


    def TransferToRAM(self, view_idx):
        self.img_sgm[:, view_idx, :] = self.img_sgm_taichi.to_numpy()

    @ti.kernel
    def AddPossionNoise(self, img_sgm_taichi: ti.template(), photon_number: ti.f32, dect_elem_count_horizontal: ti.i32,
                        dect_elem_count_vertical: ti.i32):
        for u_idx, v_idx in ti.ndrange(dect_elem_count_horizontal, dect_elem_count_vertical):
            transmitted_photon_number = photon_number * \
                ti.exp(-img_sgm_taichi[v_idx,u_idx])
            transmitted_photon_number = transmitted_photon_number + \
                ti.randn() * ti.sqrt(transmitted_photon_number)
            if transmitted_photon_number <= 0:
                transmitted_photon_number = 1e-6
            img_sgm_taichi[v_idx,u_idx] = ti.log(photon_number / transmitted_photon_number)

