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
PI = 3.1415926536


def run_mgfpj_v2(file_path):
    ti.reset()
    ti.init(arch=ti.gpu)
    print('Performing FPJ from MandoCT-Taichi (ver 0.2) ...')
    print('This new version of run_mgfpj inherits from run_mgfbp.' )
    print('Add PMatrix forward projection option.' )
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
    fpj = Mgfpj(config_dict)
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
        print("No images are forward projected!\n")
    gc.collect()  # 
    ti.reset()  # free gpu ram
    return img_sgm


@ti.data_oriented
class Mgfpj(Mgfbp):
    def MainFunction(self):
        #self.InitializeSinogramBuffer()
        self.file_processed_count = 0
        self.InitializeArrays()#initialize arrays; inherit from mgfpj
        #array_v_taichi is the detector element coordinates along z
        #array_u_taichi is the detector element coordinates along y
        #self.det_offset is from projection of rotation center to detector center
        #self.det_offset_horizontal along -y direction is positive as a convention from mgfbp.exe
        #self.det_offset_vertical along +z direction is positive 
        #+u and +v direction are along -z and -y direction by convention 

        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):
                if self.ReadImage(file):
                    print('\nForward projecting %s ...' % self.input_path)
                    self.file_processed_count += 1
                    for v_idx in range(self.det_elem_vertical_recon_range_begin,self.det_elem_vertical_recon_range_end + 1):
                        str = 'Forward projecting slice: %4d/%4d' % (v_idx+1, self.det_elem_count_vertical)
                        print('\r' + str, end='')
                        self.ForwardProjectionBilinear(self.img_image_taichi, self.img_sgm_large_taichi, self.array_u_taichi,
                                                       self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                       self.det_elem_count_horizontal*self.oversample_size,
                                                       self.det_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                       self.source_isocenter_dis, self.source_det_dis, self.cone_beam,
                                                       self.helical_scan, self.helical_pitch, v_idx - self.det_elem_vertical_recon_range_begin, self.fpj_step_size,
                                                       self.img_center_x, self.img_center_y, self.img_center_z, self.curved_dect,\
                                                       self.matrix_A_each_view_taichi,self.x_s_each_view_taichi,self.bool_apply_pmatrix)
                                                       #v_idx - self.det_elem_vertical_recon_range_begin takes the recon vertical range into considertaion

                        self.BinSinogram(self.img_sgm_large_taichi, self.img_sgm_taichi,
                                         self.det_elem_count_horizontal, self.view_num, self.oversample_size)
                        if self.add_possion_noise:
                            self.AddPossionNoise(
                                self.img_sgm_taichi, self.photon_number, self.det_elem_count_horizontal, self.view_num)

                        self.TransferToRAM(v_idx)

                    print('\nSaving to %s !' % self.output_path)
                    self.SaveSinogram()

        return self.img_sgm

    def __init__(self, config_dict):
        super(Mgfpj,self).__init__(config_dict)
        self.config_dict = config_dict
        ######## parameters related to input and output filenames ########
        # NEW! Select the form of the output files
        if 'OutputFileForm' in config_dict:
            self.output_file_form = config_dict['OutputFileForm']
            if self.output_file_form == 'sinogram' or self.output_file_form == 'post_log_images':
                pass
            else:
                print("ERROR: OutputFileForm can only be sinogram or post_log_images!")
                sys.exit()
        else:
            self.output_file_form = "sinogram"

        
        ######## parameters related to fpj calculation ########
        if 'OversampleSize' in config_dict:
            # oversample along the horizontal direction
            self.oversample_size = config_dict['OversampleSize']
        else:
            self.oversample_size = 1
        if not isinstance(self.oversample_size, float) and not isinstance(self.oversample_size, int):
            print("ERROR: OversampleSize should be a number!")
            sys.exit()
        if self.oversample_size % 1!=0 or self.oversample_size < 0:
            print("ERROR: OversampleSize must be a positive integer!")
            sys.exit()

        if 'ForwardProjectionStepSize' in config_dict:
            # size of each step for forward projection
            self.fpj_step_size = config_dict['ForwardProjectionStepSize']
        else:
            self.fpj_step_size = 0.2
        
        #we do not need to detector vertical recon range in fpj
        #change the value to full detector range
        # self.det_elem_vertical_recon_range_begin = 0
        # self.det_elem_vertical_recon_range_end = self.det_elem_count_vertical -1
        # self.det_elem_count_vertical_actual =  self.det_elem_count_vertical

        ######### Helical Scan parameters ########
        if 'HelicalPitch' in config_dict:
            self.helical_scan = True
            self.helical_pitch = config_dict['HelicalPitch']
            print('--Helical scan')
        else:
            self.helical_scan = False
            self.helical_pitch = 0.0

        if (self.helical_scan):
            #if ImageCenterZ is not defined or is wrongly auto set from mgfbp
            if ('ImageCenterZ' not in config_dict) or (self.img_center_z_auto_set_from_fbp): 
                self.img_center_z = self.img_voxel_height * \
                    (self.img_dim_z - 1) / 2.0 * np.sign(self.helical_pitch)
                print("Warning: ImageCenterZ is not in the config file or is wrongly set from run_mgfbp!")
                print("For helical scans, the first view begins with the bottom or the top of the image object;")
                print("ImageCenterZ is re-set accordingly to be %.1f mm!" %self.img_center_z)

        # NEW! add poisson noise to generated sinogram
        if 'PhotonNumber' in config_dict:
            self.add_possion_noise = True
            self.photon_number = config_dict['PhotonNumber']
            if not isinstance(self.photon_number,float) and not isinstance(self.photon_number,int):
                print("ERROR: PhotonNumber must be a number!")
                sys.exit()
        else:
            self.add_possion_noise = False
            self.photon_number = 0
        
        #alias for image rotation
        if 'StartAngle' in config_dict:
            self.img_rot = config_dict['StartAngle']
            if not isinstance(self.img_rot,float) and not isinstance(self.img_rot,int):
                print("ERROR: StartAngle must be a number!")
                sys.exit()
            
        
        
        del self.img_recon_taichi
        del self.img_recon
        #re-initialize img_sgm with det_elem_count_vertical, not det_elem_count_vertical_actual
        self.img_sgm = np.zeros((self.det_elem_count_vertical, self.view_num, self.det_elem_count_horizontal),dtype = np.float32)
        self.img_image = np.zeros(
            (self.img_dim_z, self.img_dim, self.img_dim), dtype=np.float32)
        self.array_u_taichi = ti.field(
            dtype=ti.f32, shape=self.det_elem_count_horizontal*self.oversample_size)

        self.img_image_taichi = ti.field(dtype=ti.f32, shape=(
            self.img_dim_z, self.img_dim, self.img_dim))
        # for sgm in gpu ram, we initialize 2D buffer; since gpu ram is limited
        self.img_sgm_large_taichi = ti.field(dtype=ti.f32, shape=(
            1, self.view_num, self.det_elem_count_horizontal*self.oversample_size), order='ijk', needs_dual=True)
        self.img_sgm_taichi = ti.field(dtype=ti.f32, shape=(
            1, self.view_num, self.det_elem_count_horizontal))
        #self.array_angle_taichi = ti.field(dtype=ti.f32, shape=self.view_num)
        self.matrix_A_each_view_taichi = ti.field(dtype=ti.f32, shape=(9*self.view_num, 1))
        self.x_s_each_view_taichi = ti.field(dtype=ti.f32, shape=(3*self.view_num, 1))
        if self.bool_apply_pmatrix:
            self.GenerateMatrixAFromPMatrix()
            self.matrix_A_each_view_taichi.from_numpy(self.matrix_A_each_view)
            self.x_s_each_view_taichi.from_numpy(self.x_s_each_view)
    
    #this function record the A matrix ([e_u, e_v, x_do-x_s]) and x_s for each view
    #These pieces of information will be applied for forward projection using pmatrix
    def GenerateMatrixAFromPMatrix(self):
        self.matrix_A_each_view = np.zeros(shape = (9*self.view_num,1),dtype = np.float32)
        self.x_s_each_view = np.zeros(shape = (3*self.view_num,1),dtype = np.float32)
        for view_idx in range(self.view_num):
            pmatrix_this_view = self.array_pmatrix[(view_idx*12):(view_idx+1)*12]#get the pmatrix for this view
            pmatrix_this_view = np.reshape(pmatrix_this_view,[3,4])#reshape it to be 3x4 matrix
            matrix_A = np.linalg.inv(pmatrix_this_view[:,0:3])#matrix A is the inverse of the 3x3 matrix from the first three columns
            self.matrix_A_each_view[view_idx*9: (view_idx+1) * 9,0] = np.squeeze(matrix_A.reshape((9,1)))
            x_s = - np.matmul(matrix_A,pmatrix_this_view[:,3]).reshape([3,1])#calculate the source position
            self.x_s_each_view[view_idx*3: (view_idx+1) * 3,0] = np.squeeze(x_s.reshape((3,1)))

    @ti.kernel
    def GenerateDectPixPosArrayFPJ(self, det_elem_count_horizontal: ti.i32, det_elem_width: ti.f32, det_offset_horizontal: ti.f32, array_u_taichi: ti.template()):
        for i in ti.ndrange(det_elem_count_horizontal):
            array_u_taichi[i] = (i - (det_elem_count_horizontal - 1) /
                                 2.0) * det_elem_width + det_offset_horizontal

    @ti.kernel
    def ForwardProjectionBilinear(self, img_image_taichi: ti.template(), img_sgm_large_taichi: ti.template(),
                                  array_u_taichi: ti.template(), array_v_taichi: ti.template(),
                                  array_angle_taichi: ti.template(), img_dim: ti.i32, img_dim_z: ti.i32,
                                  det_elem_count_horizontal_oversamplesize: ti.i32,
                                  det_elem_count_vertical: ti.i32, view_num: ti.i32,
                                  img_pix_size: ti.f32, img_voxel_height: ti.f32, source_isocenter_dis: ti.f32,
                                  source_det_dis: ti.f32, cone_beam: ti.i32, helical_scan: ti.i32, helical_pitch: ti.f32,
                                  v_idx: ti.i32, fpj_step_size: ti.f32, img_center_x: ti.f32,
                                  img_center_y: ti.f32, img_center_z: ti.f32, curved_dect: ti.i32, matrix_A_each_view_taichi: ti.template(),\
                                  x_s_each_view_taichi: ti.template(), bool_apply_pmatrix: ti.i32):

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
        z_0 = -(img_dim_z - 1.0) / 2.0 * img_voxel_height + img_center_z

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

        # number of steps
        count_steps = int(
            ti.floor((l_max - l_min)/(fpj_step_size * voxel_diagonal_size)))

        for u_idx, angle_idx in ti.ndrange(det_elem_count_horizontal_oversamplesize, view_num):

            #caluclate the position of the detector element
            if self.curved_dect:
                gamma_prime = ( - array_u_taichi[u_idx]) / sdd #conterclockwise is positive, corresponding to -y direction
                det_elem_pos_x = -sdd * ti.cos(gamma_prime) + sid
                # positive u direction is - y
                det_elem_pos_y = -sdd * ti.sin(gamma_prime)#negative gamma_prime corresponds to positive y
            else:
                det_elem_pos_x = - (sdd - sid)
                # positive u direction is - y
                det_elem_pos_y = array_u_taichi[u_idx]
                
            #add this distance to z position to simulate helical scan
            det_elem_pos_z = array_v_taichi[v_idx] + z_dis_per_view * angle_idx
            # assume that the source and the detector moves upward for a helical scan (pitch>0)
            source_pos_z = z_dis_per_view * angle_idx
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
                one_over_mag = (step_idx * fpj_step_size * voxel_diagonal_size + l_min) / source_det_elem_dis
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

            img_sgm_large_taichi[0, angle_idx, u_idx] = temp_sgm_val

    @ti.kernel
    def BinSinogram(self, img_sgm_large_taichi: ti.template(), img_sgm_taichi: ti.template(), det_elem_count_horizontal: ti.i32,
                    view_num: ti.i32, bin_size: ti.i32):
        for angle_idx, u_idx in ti.ndrange(view_num, det_elem_count_horizontal):
            img_sgm_taichi[0, angle_idx, u_idx] = 0.0
            for i in ti.ndrange(bin_size):
                img_sgm_taichi[0, angle_idx, u_idx] += img_sgm_large_taichi[0,angle_idx, u_idx * bin_size + i]
            img_sgm_taichi[0, angle_idx, u_idx] /= bin_size

    def ReadImage(self, file):
        self.input_path = os.path.join(self.input_dir, file)
        self.output_file = re.sub(
            self.output_file_replace[0], self.output_file_replace[1], file)
        if self.output_file == file:
            # did not file the string in file, so that output_file and file are the same
            print(f"ERROR: did not find string '{self.output_file_replace[0]}' to replace in '{self.output_file}'")
            sys.exit()
        else:
            self.output_path = os.path.join(
                self.output_dir, self.output_file_prefix + self.output_file)
            self.img_image = np.fromfile(self.input_path, dtype=np.float32)
            self.img_image = self.img_image.reshape(
                self.img_dim_z, self.img_dim, self.img_dim)
            if self.convert_to_HU:
                self.img_image = (self.img_image + 1000.0) / \
                    1000.0 * self.water_mu
            self.img_image_taichi.from_numpy(self.img_image)
            return True

    def TransferToRAM(self, v_idx):
        self.img_sgm[v_idx, :, :] = self.img_sgm_taichi.to_numpy()

    @ti.kernel
    def AddPossionNoise(self, img_sgm_taichi: ti.template(), photon_number: ti.f32, det_elem_count_horizontal: ti.i32,
                        view_num: ti.i32):
        for u_idx, angle_idx in ti.ndrange(det_elem_count_horizontal, view_num):
            transmitted_photon_number = photon_number * \
                ti.exp(-img_sgm_taichi[0, angle_idx, u_idx])
            transmitted_photon_number = transmitted_photon_number + \
                ti.randn() * ti.sqrt(transmitted_photon_number)
            if transmitted_photon_number <= 0:
                transmitted_photon_number = 1e-6
            img_sgm_taichi[0, angle_idx, u_idx] = ti.log(photon_number / transmitted_photon_number)

    def SaveSinogram(self):
        if self.cone_beam:
            if self.output_file_form == 'sinogram':
                # by default, first sinogram slice is at bottom row
                # therefore, we need to flip the 0-th axis
                self.img_sgm = np.flip(self.img_sgm, axis=0)
                imwriteRaw(self.img_sgm, self.output_path, dtype=np.float32)
            elif self.output_file_form == 'post_log_images':
                # change view direction as axis 0 and angle direction as axis 1
                imwriteRaw(self.img_sgm.transpose([1, 0, 2]), self.output_path, dtype=np.float32)
        else:
            #if the fpj is not a cone beam type, direct save the generated sgm without flip the dimensions
            imwriteRaw(self.img_sgm, self.output_path, dtype=np.float32)


def remove_comments(jsonc_str):
    pattern = re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL)
    return re.sub(pattern, '', jsonc_str)


def save_jsonc(save_path, data):
    assert save_path.split('.')[-1] == 'jsonc'
    with open(save_path, 'w') as file:
        json.dump(data, file)


def load_jsonc(file_path):
    with open(file_path, 'r') as file:
        jsonc_content = file.read()
        json_content = remove_comments(jsonc_content)
        data = json.loads(json_content)
    return data


def imreadRaw(path: str, height: int, width: int, dtype=np.float32, nSlice: int = 1, offset: int = 0, gap: int = 0):
    with open(path, 'rb') as fp:
        fp.seek(offset)
        if gap == 0:
            arr = np.frombuffer(fp.read(), dtype=dtype, count=nSlice *
                                height * width).reshape((nSlice, height, width)).squeeze()
        else:
            imageBytes = height * width * np.dtype(dtype).itemsize
            arr = np.zeros((nSlice, height, width), dtype=dtype)
            for i in range(nSlice):
                arr[i, ...] = np.frombuffer(fp.read(imageBytes), dtype=dtype).reshape(
                    (height, width)).squeeze()
                fp.seek(gap, os.SEEK_CUR)
    return arr


def imaddRaw(img, path: str, dtype=None, idx=1):
    '''
        Write add file. Convert dtype with `dtype != None`.
    '''

    if dtype is not None:
        img = img.astype(dtype)

    if idx == 0:
        with open(path, 'wb') as fp:
            fp.write(img.flatten().tobytes())
    else:
        with open(path, 'ab') as fp:
            fp.write(img.flatten().tobytes())


def ReadConfigFile(file_path):
    json_data = load_jsonc(file_path)
    return json_data
