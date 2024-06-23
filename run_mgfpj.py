# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:06:09 2024

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
from crip.io import imwriteRaw
from crip.io import imwriteTiff
import time

PI = 3.1415926536


def run_mgfpj(file_path):
    ti.reset()
    ti.init(arch=ti.gpu)
    print('Performing FPJ from MandoCT-Taichi (ver 0.1) ...')
    # record start time point
    start_time = time.time()
    # Delete unnecessary warinings
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='The value of the smallest subnormal for <class \'numpy.float(32|64)\'> type is zero.')
    if not os.path.exists(file_path):
        print(f"ERROR: Config File {file_path} does not exist!")
        # Judge whether the config jsonc file exist
        sys.exit()
    config_dict = ReadConfigFile(file_path)  # 读入jsonc文件并以字典的形式存储在config_dict中
    fpj = Mgfpj(config_dict)  # 将config_dict数据以字典的形式送入对象中
    img_sgm = fpj.MainFunction()
    end_time = time.time()
    execution_time = end_time - start_time  # 计算执行时间
    if fpj.file_processed_count > 0:
        print(
            f"\nA total of {fpj.file_processed_count:d} files are forward projected!")
        print(f"Time cost：{execution_time:.3} sec\n")  # 打印执行时间（以秒为单位）
    else:
        print(
            f"\nWarning: Did not find files like {fpj.input_files_pattern:s} in {fpj.input_dir:s}.")
        print("No images are  forward projected!\n")
    gc.collect()  # 手动触发垃圾回收
    ti.reset()  # free gpu ram
    return img_sgm


@ti.data_oriented
class Mgfpj:
    def MainFunction(self):

        self.file_processed_count = 0
        self.GenerateAngleArray(
            self.view_num, self.start_angle, self.scan_angle, self.array_angle_taichi)
        self.GenerateDectPixPosArray(
            self.dect_elem_count_vertical, - self.dect_elem_height, self.dect_offset_vertical, self.array_v_taichi)
        self.GenerateDectPixPosArray(self.dect_elem_count_horizontal*self.oversample_size, self.dect_elem_width/self.oversample_size,
                                     self.dect_offset_horizontal, self.array_u_taichi)

        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):
                if self.ReadImage(file):
                    print('\nForward projecting %s ...' % self.input_path)
                    self.file_processed_count += 1
                    for v_idx in range(self.dect_elem_count_vertical):

                        str = 'Forward projecting slice: %4d/%4d' % (
                            v_idx+1, self.dect_elem_count_vertical)
                        print('\r' + str, end='')
                        self.ForwardProjectionBilinear(self.img_image_taichi, self.img_sgm_large_taichi, self.array_u_taichi,
                                                       self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                       self.dect_elem_count_horizontal*self.oversample_size,
                                                       self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                       self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                                       self.helican_scan, self.helical_pitch, v_idx, self.fpj_step_size,
                                                       self.img_center_x, self.img_center_y, self.img_center_z, self.curved_dect)

                        self.BinSinogram(self.img_sgm_large_taichi, self.img_sgm_taichi,
                                         self.dect_elem_count_horizontal, self.view_num, self.oversample_size)
                        if self.add_possion_noise:
                            self.AddPossionNoise(
                                self.img_sgm_taichi, self.photon_number, self.dect_elem_count_horizontal, self.view_num)

                        self.TransferToRAM(v_idx)

                    print('\nSaving to %s !' % self.output_path)
                    self.SaveSinogram()

        return self.img_sgm

    def __init__(self, config_dict):
        self.config_dict = config_dict
        ######## parameters related to input and output filenames ########
        self.input_dir = config_dict['InputDir']
        self.output_dir = config_dict['OutputDir']
        self.input_files_pattern = config_dict['InputFiles']
        self.output_file_prefix = config_dict['OutputFilePrefix']
        self.output_file_replace = config_dict['OutputFileReplace']

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

        ########  parameters related to the input image volume (slice) ########
        self.img_dim = config_dict['ImageDimension']
        self.img_pix_size = config_dict['PixelSize']
        # image dimension along z direction
        if 'SliceCount' in config_dict:
            # 根据mgfpj现有的规则用SliceCount表示image的z维度而非用imageSliceCount
            # compatible with C++ version
            self.img_dim_z = config_dict['SliceCount']
        elif 'ImageDimensionZ' in config_dict:
            self.img_dim_z = config_dict['ImageDimensionZ']
        else:
            print(
                "ERROR: Can not find image dimension along Z direction for cone beam recon!")
            sys.exit()
        
        # image center along x and y direction
        if 'ImageCenter' in config_dict:
            self.img_center = config_dict['ImageCenter']
            self.img_center_x = self.img_center[0]
            self.img_center_y = self.img_center[1]
        else:
            self.img_center = [0, 0]
            self.img_center_x = 0
            self.img_center_y = 0

        # img center along z direction
        if 'ImageCenterZ' in config_dict:
            self.img_center_z = config_dict['ImageCenterZ']
        else:
            self.img_center_z = 0

        if 'ConeBeam' in config_dict:
            self.cone_beam = config_dict['ConeBeam']
        else:
            self.cone_beam = False

        ######## parameters related to the detector ########
        # detector type (flat panel or curved)
        if 'CurvedDetector' in config_dict:
            self.curved_dect = config_dict['CurvedDetector']
            if self.curved_dect:
                print("--Curved detector")
        else:
            self.curved_dect = False

        if 'DetectorElementCountHorizontal' in config_dict:
            self.dect_elem_count_horizontal = config_dict['DetectorElementCountHorizontal']
        elif 'SinogramWidth' in config_dict:
            self.dect_elem_count_horizontal = config_dict['SinogramWidth']
        elif 'DetectorElementCount' in config_dict:
            self.dect_elem_count_horizontal = config_dict['DetectorElementCount']
        else:
            print(
                "ERROR: Can not find detector element count along horizontal direction!")
            sys.exit()

        if 'DetectorElementWidth' in config_dict:
            self.dect_elem_width = config_dict['DetectorElementWidth']
        elif 'DetectorElementSize' in config_dict:
            self.dect_elem_width = config_dict['DetectorElementSize']
        else:
            print("ERROR: Can not find detector element width!")
            sys.exit()

        if 'DetectorOffcenter' in config_dict:
            self.dect_offset_horizontal = config_dict['DetectorOffcenter']
        elif 'DetectorOffsetHorizontal' in config_dict:
            self.dect_offset_horizontal = config_dict['DetectorOffsetHorizontal']
        else:
            print(
                "Warning: Can not find horizontal detector offset; Using default value 0")
            # self.return_information += "--Warning: Can not find horizontal detector offset; Using default value 0\n"

        ######## CT scan parameters ########
        self.source_isocenter_dis = config_dict['SourceIsocenterDistance']
        self.source_dect_dis = config_dict['SourceDetectorDistance']
        
        # image rotation (Start Angle) for forward projection
        if 'StartAngle' in config_dict:
            self.start_angle = config_dict['StartAngle'] / 180.0 * PI
        elif 'ImageRotation' in config_dict:
            self.start_angle = config_dict['ImageRotation'] / 180.0 * PI   
        else:
            self.start_angle = 0.0
            
        if 'TotalScanAngle' in config_dict:
            self.scan_angle = config_dict['TotalScanAngle'] / 180.0 * PI
        else:
            self.scan_angle = 2 * PI

        if abs(self.scan_angle % (2*PI)) < (0.01 / 180 * PI):
            print('--Full scan, scan Angle = %.1f degrees' %
                  (self.scan_angle / PI * 180))
            # self.return_information += '--Full scan, scan Angle = %.1f degrees\n' % (self.total_scan_angle / PI * 180)
        else:
            print('--Short scan, scan Angle = %.1f degrees' %
                  (self.scan_angle / PI * 180))
            # self.return_information += '--Short scan, scan Angle = %.1f degrees\n' % (self.total_scan_angle / PI * 180)

        self.view_num = config_dict['Views']

        ######## parameters related to fpj calculation ########
        if 'OversampleSize' in config_dict:
            # oversample along the horizontal direction
            self.oversample_size = config_dict['OversampleSize']
        else:
            self.oversample_size = 1

        if 'ForwardProjectionStepSize' in config_dict:
            # size of each step for forward projection
            self.fpj_step_size = config_dict['ForwardProjectionStepSize']
        else:
            self.fpj_step_size = 0.2

        if 'PmatrixDetectorElementSize' in config_dict:
            self.pmatrix_dect_elem_width = config_dict['PmatrixDetectorElementSize']
        else:
            self.pmatrix_dect_elem_width = self.dect_elem_width

        ######## parameters related to cone beam scan ########
        if self.cone_beam:
            print("--Cone beam forward projection")
            # image voxel height
            if 'VoxelHeight' in config_dict:
                self.img_voxel_height = config_dict['VoxelHeight']
            elif 'ImageSliceThickness' in config_dict:
                self.img_voxel_height = config_dict['ImageSliceThickness']
            else:
                print("ERROR: Can not find image voxel height for cone beam recon!")
                sys.exit()

            if 'DetectorElementCountVertical' in config_dict:
                self.dect_elem_count_vertical = config_dict['DetectorElementCountVertical']
            elif 'DetectorZElementCount' in config_dict:
                self.dect_elem_count_vertical = config_dict['DetectorZElementCount']
            else:
                print(
                    "ERROR: Can not find detector element count along vertical direction!")
                sys.exit()

            # detector element height
            if 'SliceThickness' in config_dict:
                self.dect_elem_height = config_dict['SliceThickness']
            elif 'DetectorElementHeight' in config_dict:
                self.dect_elem_height = config_dict['DetectorElementHeight']
            else:
                print(
                    "ERROR: Can not find detector element height for cone beam recon! ")
                sys.exit()

            # detector offset vertical
            if 'DetectorZOffcenter' in config_dict:
                self.dect_offset_vertical = config_dict['DetectorZOffcenter']
            elif 'DetectorOffsetVertical' in config_dict:
                self.dect_offset_vertical = config_dict['DetectorOffsetVertical']
            else:
                self.dect_offset_vertical = 0
                print(
                    "Warning: Can not find vertical detector offset for cone beam recon; Using default value 0")
        else:
            self.dect_elem_count_vertical = self.img_dim_z
            self.img_voxel_height = 0.0
            self.dect_elem_height = 0.0
            self.dect_offset_vertical = 0.0

        ######## whether images are converted to HU for the input image ########
        if 'WaterMu' in config_dict:
            self.water_mu = config_dict['WaterMu']
            self.convert_to_HU = True
            print("--Converted to HU")
        else:
            self.convert_to_HU = False

        ######### Helical Scan parameters ########
        if 'HelicalPitch' in config_dict:
            self.helican_scan = True
            self.helical_pitch = config_dict['HelicalPitch']
        else:
            self.helican_scan = False
            self.helical_pitch = 0

        if (self.helican_scan) and ('ImageCenterZ' not in config_dict):
            self.img_center_z = self.img_voxel_height * \
                (self.img_dim_z - 1) / 2.0 * np.sign(self.helical_pitch)
            print("Did not find image center along Z direction in the config file!")
            print("For helical scans, the first view begins with the bottom or the top of the image object; the image center along Z direction is set accordingly!")

        # NEW! add poisson noise to generated sinogram
        if 'PhotonNumber' in config_dict:
            self.add_possion_noise = True
            self.photon_number = config_dict['PhotonNumber']
        else:
            self.add_possion_noise = False
            self.photon_number = 0

        self.img_image = np.zeros(
            (self.img_dim_z, self.img_dim, self.img_dim), dtype=np.float32)
        # for sgm in ram, we initialize a 3D buffer
        self.img_sgm = np.zeros((self.dect_elem_count_vertical, self.view_num,
                                self.dect_elem_count_horizontal), dtype=np.float32)
        self.array_u_taichi = ti.field(
            dtype=ti.f32, shape=self.dect_elem_count_horizontal*self.oversample_size)
        self.array_v_taichi = ti.field(
            dtype=ti.f32, shape=self.dect_elem_count_horizontal)
        self.img_image_taichi = ti.field(dtype=ti.f32, shape=(
            self.img_dim_z, self.img_dim, self.img_dim))
        # for sgm in gpu ram, we initialize 2D buffer; since gpu ram is limited
        self.img_sgm_large_taichi = ti.field(dtype=ti.f32, shape=(
            1, self.view_num, self.dect_elem_count_horizontal*self.oversample_size), order='ijk', needs_dual=True)
        self.img_sgm_taichi = ti.field(dtype=ti.f32, shape=(
            1, self.view_num, self.dect_elem_count_horizontal))
        self.array_angle_taichi = ti.field(dtype=ti.f32, shape=self.view_num)

    @ti.kernel
    def GenerateAngleArray(self, view_num: ti.i32, start_angle: ti.f32, scan_angle: ti.f32, array_angle_taichi: ti.template()):
        # 计算beta并用弧度制的形式表示
        for i in ti.ndrange(view_num):
            array_angle_taichi[i] = (scan_angle / view_num * i + start_angle)

    @ti.kernel
    def GenerateDectPixPosArray(self, dect_elem_count_horizontal: ti.i32, dect_elem_width: ti.f32, dect_offset_horizontal: ti.f32, array_u_taichi: ti.template()):
        for i in ti.ndrange(dect_elem_count_horizontal):
            array_u_taichi[i] = (i - (dect_elem_count_horizontal - 1) /
                                 2.0) * dect_elem_width + dect_offset_horizontal

    @ti.kernel
    def ForwardProjectionBilinear(self, img_image_taichi: ti.template(), img_sgm_large_taichi: ti.template(),
                                  array_u_taichi: ti.template(), array_v_taichi: ti.template(),
                                  array_angle_taichi: ti.template(), img_dim: ti.i32, img_dim_z: ti.i32,
                                  dect_elem_count_horizontal_oversamplesize: ti.i32,
                                  dect_elem_count_vertical: ti.i32, view_num: ti.i32,
                                  img_pix_size: ti.f32, img_voxel_height: ti.f32, source_isocenter_dis: ti.f32,
                                  source_dect_dis: ti.f32, cone_beam: ti.i32, helican_scan: ti.i32, helical_pitch: ti.f32,
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
        if self.helican_scan:
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

            unit_vec_lambda_x = (
                dect_elem_pos_x - source_pos_x) / source_dect_elem_dis
            unit_vec_lambda_y = (
                dect_elem_pos_y - source_pos_y) / source_dect_elem_dis
            unit_vec_lambda_z = (
                dect_elem_pos_z - source_pos_z) / source_dect_elem_dis

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
                        z_idx = v_idx
                        sgm_val = (1 - x_weight) * (1 - y_weight) * img_image_taichi[z_idx, y_idx, x_idx]\
                            + x_weight * (1 - y_weight) * img_image_taichi[z_idx, y_idx, x_idx+1]\
                            + (1 - x_weight) * y_weight * img_image_taichi[z_idx, y_idx+1, x_idx]\
                            + x_weight * y_weight * \
                            img_image_taichi[z_idx, y_idx+1, x_idx + 1]
                        temp_sgm_val += sgm_val * fpj_step_size * voxel_diagonal_size

            img_sgm_large_taichi[0, angle_idx, u_idx] = temp_sgm_val

    @ti.kernel
    def BinSinogram(self, img_sgm_large_taichi: ti.template(), img_sgm_taichi: ti.template(), dect_elem_count_horizontal: ti.i32,
                    view_num: ti.i32, bin_size: ti.i32):
        for angle_idx, u_idx in ti.ndrange(view_num, dect_elem_count_horizontal):
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
            print(
                f"ERROR: did not find string '{self.output_file_replace[0]}' to replace in '{self.output_file}'")
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
            # 将正弦图sgm存储到taichi专用的数组中帮助加速程序
            return True

    def TransferToRAM(self, v_idx):
        self.img_sgm[v_idx, :, :] = self.img_sgm_taichi.to_numpy()

    @ti.kernel
    def AddPossionNoise(self, img_sgm_taichi: ti.template(), photon_number: ti.f32, dect_elem_count_horizontal: ti.i32,
                        view_num: ti.i32):
        for u_idx, angle_idx in ti.ndrange(dect_elem_count_horizontal, view_num):
            transmitted_photon_number = photon_number * \
                ti.exp(-img_sgm_taichi[0, angle_idx, u_idx])
            transmitted_photon_number = transmitted_photon_number + \
                ti.randn() * ti.sqrt(transmitted_photon_number)
            if transmitted_photon_number <= 0:
                transmitted_photon_number = 1e-6
            img_sgm_taichi[0, angle_idx, u_idx] = ti.log(
                photon_number / transmitted_photon_number)

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
            #if the fpj is not a bone beam type, direct save the generated sgm without flip the dimensions
            imwriteRaw(self.img_sgm, self.output_path, dtype=np.float32)


def remove_comments(jsonc_str):
    # 使用正则表达式去除注释
    pattern = re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL)
    return re.sub(pattern, '', jsonc_str)


def save_jsonc(save_path, data):
    assert save_path.split('.')[-1] == 'jsonc'
    with open(save_path, 'w') as file:
        json.dump(data, file)


def load_jsonc(file_path):
    # 读取jsonc文件并以字典的形式返回所有数据
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
    # 替换为你的JSONC文件路径
    json_data = load_jsonc(file_path)
    # 现在，json_data包含了从JSONC文件中解析出的数据
    # print(json_data)
    return json_data
