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
    print('Performing FPJ from MandoCT-Taichi (ver 0.11) ...')
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
    gc.collect()  #手动触发垃圾回收
    ti.reset()  #free gpu ram
    return img_sgm


@ti.data_oriented
class Mgfpj:
    def MainFunction(self):

        self.file_processed_count = 0
        self.GenerateAngleArray(
            self.view_num, self.start_angle, self.scan_angle, self.array_angle_taichi)
        self.GenerateDectPixPosArray(
            self.det_elem_count_vertical, - self.det_elem_height, self.det_offset_vertical, self.array_v_taichi)
        self.GenerateDectPixPosArray(self.det_elem_count_horizontal*self.oversample_size, -self.det_elem_width/self.oversample_size,
                                     -self.det_offset_horizontal, self.array_u_taichi)

        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):
                if self.ReadImage(file):
                    print('\nForward projecting %s ...' % self.input_path)
                    self.file_processed_count += 1
                    for v_idx in range(self.det_elem_count_vertical):

                        str = 'Forward projecting slice: %4d/%4d' % (
                            v_idx+1, self.det_elem_count_vertical)
                        print('\r' + str, end='')
                        self.ForwardProjectionBilinear(self.img_image_taichi, self.img_sgm_large_taichi, self.array_u_taichi,
                                                       self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                       self.det_elem_count_horizontal*self.oversample_size,
                                                       self.det_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                       self.source_isocenter_dis, self.source_det_dis, self.cone_beam,
                                                       self.helical_scan, self.helical_pitch, v_idx, self.fpj_step_size,
                                                       self.img_center_x, self.img_center_y, self.img_center_z, self.curved_dect)

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
        self.config_dict = config_dict
        ######## parameters related to input and output filenames ########
        if 'InputDir' in config_dict:
            self.input_dir = config_dict['InputDir']
            if not isinstance(self.input_dir,str):
                print('ERROR: InputDir must be a string!')
                sys.exit()
        else:
            print('ERROR: Can not find InputDir!')
            sys.exit()
            
        if 'OutputDir' in config_dict:
            self.output_dir = config_dict['OutputDir']
            if not isinstance(self.output_dir,str):
                print('ERROR: OutputDir must be a string!')
                sys.exit()
        else:
            print('ERROR: Can not find OutputDir!')
            sys.exit()
        
        if 'InputFiles' in config_dict:
            self.input_files_pattern = config_dict['InputFiles']
            if not isinstance(self.input_files_pattern,str):
                print('ERROR: InputFiles must be a string!')
                sys.exit()
        else:
            print('ERROR: Can not find InputFiles!')
            sys.exit()
        
        if 'OutputFilePrefix' in config_dict:
            self.output_file_prefix = config_dict['OutputFilePrefix']
            if not isinstance(self.output_file_prefix,str):
                print('ERROR: OutputFilePrefix must be a string!')
                sys.exit()
        else:
            print('Warning: Can not find OutputFilePrefix! Set to be an empty string. ')
            self.output_file_prefix = ""
        
        if 'OutputFileReplace' in config_dict:
            self.output_file_replace = config_dict['OutputFileReplace']
        else:
            print("ERROR: Can not find OutputFileReplace in the config file!")
            sys.exit()

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
        if 'ImageDimension' in config_dict:
            self.img_dim = config_dict['ImageDimension']
            if self.img_dim<0 or not isinstance(self.img_dim, int):
                print('ERROR: ImageDimension must be a positive integer!')
                sys.exit()
        else:
            print('ERROR: can not find ImageDimension!')
            sys.exit()
        if 'PixelSize' in config_dict:
            self.img_pix_size = config_dict['PixelSize']
            if not isinstance(self.img_pix_size, float) and not isinstance(self.img_pix_size, int):
                print('ERROR: PixelSize must be a number!')
                sys.exit()
            if self.img_pix_size < 0:
                print('ERROR: PixelSize must be positive!')
                sys.exit()
        else:
            print('ERROR: can not find PixelSize!')
            sys.exit()
               
        if 'SliceCount' in config_dict:
            # 根据mgfpj现有的规则用SliceCount表示image的z维度而非用imageSliceCount
            # compatible with C++ version
            self.img_dim_z = config_dict['SliceCount']
        elif 'ImageDimensionZ' in config_dict:
            self.img_dim_z = config_dict['ImageDimensionZ']
        else:
            print("ERROR: Can not find image dimension along Z direction (ImageDimensionZ or SliceCount)!")
            sys.exit() 
        if not isinstance(self.img_dim_z,int) or self.img_dim_z <= 0:
            print('ERROR: ImageDimensionZ (ImageSliceCount) must be a positive integer!')
            sys.exit()
        
        # image center along x and y direction
        if 'ImageCenter' in config_dict:
            self.img_center = config_dict['ImageCenter']
            if not isinstance(self.img_center,list) or len(self.img_center)!=2:
                print('ERROR: ImageCenter must be an array with two numbers!')
                sys.exit()
            self.img_center_x = self.img_center[0]
            self.img_center_y = self.img_center[1]
        else:
            self.img_center_x = 0.0
            self.img_center_y = 0.0

        # img center along z direction
        if 'ImageCenterZ' in config_dict:
            self.img_center_z = config_dict['ImageCenterZ']
            if not isinstance(self.img_center_z,float) and not isinstance(self.img_center_z,int):
                print('ERROR: ImageCenterZ must be a number!')
                sys.exit()
        else:
            self.img_center_z = 0.0

        if 'ConeBeam' in config_dict:
            self.cone_beam = config_dict['ConeBeam']
            if not isinstance(self.cone_beam,bool):
                print('ERROR: ConeBeam must be true or false!')
                sys.exit()
        else:
            self.cone_beam = False

        ######## parameters related to the detector ########
        # detector type (flat panel or curved)
        if 'CurvedDetector' in config_dict:
            if isinstance(config_dict['CurvedDetector'], bool):
                self.curved_dect = config_dict['CurvedDetector']
            else:
                print("ERROR: CurvedDetector can only be true or false!")
                sys.exit()
            if self.curved_dect:
                print("--Curved detector")
        else:
            self.curved_dect = False 

        if 'DetectorElementCountHorizontal' in config_dict:
            self.det_elem_count_horizontal = config_dict['DetectorElementCountHorizontal']
        elif 'SinogramWidth' in config_dict:
            self.det_elem_count_horizontal = config_dict['SinogramWidth']
        elif 'DetectorElementCount' in config_dict:
            self.det_elem_count_horizontal = config_dict['DetectorElementCount']
        else:
            print("ERROR: Can not find DetectorElementCountHorizontal (SinogramWidth or DetectorElementCount)!")
            sys.exit()
        if self.det_elem_count_horizontal <= 0:
            print("ERROR: DetectorElementCountHorizontal (SinogramWidth) should be larger than 0!")
            sys.exit()
        elif self.det_elem_count_horizontal % 1 != 0:
            print("ERROR: DetectorElementCountHorizontal (SinogramWidth) should be an integer!")
            sys.exit()

        if 'DetectorElementWidth' in config_dict:
            self.det_elem_width = config_dict['DetectorElementWidth']
        elif 'DetectorElementSize' in config_dict:
            self.det_elem_width = config_dict['DetectorElementSize']
        else:
            print("ERROR: Can not find DetectorElementWidth (DetectorElementSize)!")
            sys.exit()
        if self.det_elem_width <= 0:
            print("ERROR: DetectorElementWidth (DetectorElementSize) should be larger than 0!")
            sys.exit()

        if 'DetectorOffcenter' in config_dict:
            self.det_offset_horizontal = config_dict['DetectorOffcenter']
        elif 'DetectorOffsetHorizontal' in config_dict:
            self.det_offset_horizontal = config_dict['DetectorOffsetHorizontal']
        else:
            self.det_offset_horizontal = 0.0
            print("Warning: Can not find DetectorOffsetHorizontal (DetectorOffcenter); Using default value 0")
            # self.return_information += "--Warning: Can not find horizontal detector offset; Using default value 0\n"
        if not isinstance(self.det_offset_horizontal, float) and not isinstance(self.det_offset_horizontal, int):
            print("ERROR: DetectorOffsetHorizontal (DetectorOffcenter) should be a number!")
            sys.exit()

        ######## CT scan parameters ########
        if 'SourceIsocenterDistance' in config_dict:
            self.source_isocenter_dis = config_dict['SourceIsocenterDistance']
            if not isinstance(self.source_isocenter_dis, float) and not isinstance(self.source_isocenter_dis, int):
                print('ERROR: SourceIsocenterDistance must be a number!')
                sys.exit()
            if self.source_isocenter_dis<=0.0:
                print('ERROR: SourceIsocenterDistance must be positive!')
                sys.exit()
        else:
            self.source_isocenter_dis = self.det_elem_count_horizontal * self.det_elem_width * 1000.0
            print('Warning: Did not find SourceIsocenterDistance; Set to infinity!')
            
        if 'SourceDetectorDistance' in config_dict:
            self.source_det_dis = config_dict['SourceDetectorDistance']
            if not isinstance(self.source_det_dis, float) and not isinstance(self.source_det_dis, int):
                print('ERROR: SourceDetectorDistance must be a number!')
                sys.exit()
            if self.source_det_dis<=0.0:
                print('ERROR: SourceDetectorDistance must be positive!')
                sys.exit()
        else:
            self.source_det_dis = self.det_elem_count_horizontal * self.det_elem_width * 1000.0
            print('Warning: Did not find SourceDetectorDistance; Set to infinity!')
        
        # image rotation (Start Angle) for forward projection
        if 'StartAngle' in config_dict:
            self.start_angle = config_dict['StartAngle'] / 180.0 * PI
        elif 'ImageRotation' in config_dict:
            self.start_angle = config_dict['ImageRotation'] / 180.0 * PI   
        else:
            self.start_angle = 0.0
        if not isinstance(self.start_angle, float) and not isinstance(self.start_angle, int):
            print('ERROR: ImageRotation must be a number!')
            sys.exit()
            
        if 'TotalScanAngle' in config_dict:
            self.scan_angle = config_dict['TotalScanAngle'] / 180.0 * PI
        else:
            self.scan_angle = 2 * PI
        if not isinstance(self.scan_angle, float) and not isinstance(self.scan_angle, int):
            print("ERROR: TotalScanAngle should be a number!")
            sys.exit()

        if abs(self.scan_angle % (2*PI)) < (0.01 / 180 * PI):
            print('--Full scan, scan Angle = %.1f degrees' %
                  (self.scan_angle / PI * 180))
            # self.return_information += '--Full scan, scan Angle = %.1f degrees\n' % (self.total_scan_angle / PI * 180)
        else:
            print('--Short scan, scan Angle = %.1f degrees' %
                  (self.scan_angle / PI * 180))
            # self.return_information += '--Short scan, scan Angle = %.1f degrees\n' % (self.total_scan_angle / PI * 180)

        if 'Views' in config_dict:
            self.view_num = config_dict['Views']
        else:
            print("ERROR: Can not find number of views!")
            sys.exit()
        if not isinstance(self.view_num, float) and not isinstance(self.view_num, int):
            print("ERROR: Views should be a number!")
            sys.exit()
        if self.view_num % 1!=0 or self.view_num<0:
            print("ERROR: Views must be must be a positive integer!")
            sys.exit()

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

        if 'PMatrixDetectorElementSize' in config_dict:
            self.pmatrix_det_elem_width = config_dict['PMatrixDetectorElementSize']
        else:
            self.pmatrix_det_elem_width = self.det_elem_width

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
                self.det_elem_count_vertical = config_dict['DetectorElementCountVertical']
            elif 'DetectorZElementCount' in config_dict:
                self.det_elem_count_vertical = config_dict['DetectorZElementCount']
            else:
                print(
                    "ERROR: Can not find detector element count along vertical direction!")
                sys.exit()

            # detector element height
            if 'SliceThickness' in config_dict:
                self.det_elem_height = config_dict['SliceThickness']
            elif 'DetectorElementHeight' in config_dict:
                self.det_elem_height = config_dict['DetectorElementHeight']
            else:
                print("ERROR: Can not find detector element height for cone beam recon! ")
                sys.exit()

            # detector offset vertical
            if 'DetectorZOffcenter' in config_dict:
                self.det_offset_vertical = config_dict['DetectorZOffcenter']
            elif 'DetectorOffsetVertical' in config_dict:
                self.det_offset_vertical = config_dict['DetectorOffsetVertical']
            else:
                self.det_offset_vertical = 0.0
                print("Warning: Can not find vertical detector offset for cone beam recon; Using default value 0")
        else:
            self.det_elem_count_vertical = self.img_dim_z
            self.img_voxel_height = 0.0
            self.det_elem_height = 0.0
            self.det_offset_vertical = 0.0

        ######## whether images are converted to HU for the input image ########
        if 'WaterMu' in config_dict:
            self.water_mu = config_dict['WaterMu']
            self.convert_to_HU = True
            print("--Converted to HU")
        else:
            self.convert_to_HU = False

        ######### Helical Scan parameters ########
        if 'HelicalPitch' in config_dict:
            self.helical_scan = True
            self.helical_pitch = config_dict['HelicalPitch']
        else:
            self.helical_scan = False
            self.helical_pitch = 0

        if (self.helical_scan) and ('ImageCenterZ' not in config_dict):
            self.img_center_z = self.img_voxel_height * \
                (self.img_dim_z - 1) / 2.0 * np.sign(self.helical_pitch)
            print("Warning: Did not find image center along Z direction in the config file!")
            print("For helical scans, the first view begins with the bottom or the top of the image object;") 
            print("ImageCenterZ is re-set accordingly to be %.1f mm!" %self.img_center_z)

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
        self.img_sgm = np.zeros((self.det_elem_count_vertical, self.view_num,
                                self.det_elem_count_horizontal), dtype=np.float32)
        self.array_u_taichi = ti.field(
            dtype=ti.f32, shape=self.det_elem_count_horizontal*self.oversample_size)
        self.array_v_taichi = ti.field(
            dtype=ti.f32, shape=self.det_elem_count_vertical)
        self.img_image_taichi = ti.field(dtype=ti.f32, shape=(
            self.img_dim_z, self.img_dim, self.img_dim))
        # for sgm in gpu ram, we initialize 2D buffer; since gpu ram is limited
        self.img_sgm_large_taichi = ti.field(dtype=ti.f32, shape=(
            1, self.view_num, self.det_elem_count_horizontal*self.oversample_size), order='ijk', needs_dual=True)
        self.img_sgm_taichi = ti.field(dtype=ti.f32, shape=(
            1, self.view_num, self.det_elem_count_horizontal))
        self.array_angle_taichi = ti.field(dtype=ti.f32, shape=self.view_num)

    @ti.kernel
    def GenerateAngleArray(self, view_num: ti.i32, start_angle: ti.f32, scan_angle: ti.f32, array_angle_taichi: ti.template()):
        # 计算beta并用弧度制的形式表示
        for i in ti.ndrange(view_num):
            array_angle_taichi[i] = (scan_angle / view_num * i + start_angle)

    @ti.kernel
    def GenerateDectPixPosArray(self, det_elem_count_horizontal: ti.i32, det_elem_width: ti.f32, det_offset_horizontal: ti.f32, array_u_taichi: ti.template()):
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
                                  img_center_y: ti.f32, img_center_z: ti.f32, curved_dect: ti.i32):

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

        z_dis_per_view = 0.0
        if self.helical_scan:
            total_scan_angle = abs((array_angle_taichi[view_num - 1] - array_angle_taichi[0])) / (view_num - 1) * view_num
            num_laps = total_scan_angle / (PI * 2)
            z_dis_per_view = helical_pitch * (num_laps / view_num) * (abs(
                array_v_taichi[1] - array_v_taichi[0]) * det_elem_count_vertical) / (sdd / sid)

        # count of steps
        count_steps = int(
            ti.floor((l_max - l_min)/(fpj_step_size * voxel_diagonal_size)))

        for u_idx, angle_idx in ti.ndrange(det_elem_count_horizontal_oversamplesize, view_num):

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

            source_det_elem_dis = ((det_elem_pos_x - source_pos_x)**2 + (
                det_elem_pos_y - source_pos_y)**2 + (det_elem_pos_z - source_pos_z)**2) ** 0.5

            unit_vec_lambda_x = (det_elem_pos_x - source_pos_x) / source_det_elem_dis
            unit_vec_lambda_y = (det_elem_pos_y - source_pos_y) / source_det_elem_dis
            unit_vec_lambda_z = (det_elem_pos_z - source_pos_z) / source_det_elem_dis

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
    def AddPossionNoise(self, img_sgm_taichi: ti.template(), photon_number: ti.f32, det_elem_count_horizontal: ti.i32,
                        view_num: ti.i32):
        for u_idx, angle_idx in ti.ndrange(det_elem_count_horizontal, view_num):
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
