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
import matplotlib.pyplot as plt

PI = 3.1415926536

def run_mgfbp(file_path):
    ti.reset()
    ti.init(arch=ti.gpu, device_memory_fraction=0.95)#define device memeory utilization fraction
    print('Performing FBP from MandoCT-Taichi (ver 0.1) ...')
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
    fbp = Mgfbp(config_dict) #将config_dict数据以字典的形式送入对象
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
class Mgfbp:
    def MainFunction(self):
        #Main function for reconstruction
        self.InitializeSinogramBuffer()#initialize sinogram buffer
        self.InitializeArrays()#initialize arrays
        self.InitializeReconKernel()#initialize reconstruction kernel
        self.file_processed_count = 0;#record the number of files processed
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):#match the file pattern
                if self.ReadSinogram(file):
                    self.file_processed_count += 1 
                    print('Reconstructing %s ...' % self.input_path)
                    if self.bool_bh_correction:
                        self.BHCorrection(self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal,self.img_sgm,\
                                          self.array_bh_coefficients_taichi,self.bh_corr_order)#pass img_sgm directly into this function using unified memory
                    self.WeightSgm(self.dect_elem_count_vertical_actual,self.short_scan,self.curved_dect,\
                                   self.total_scan_angle,self.view_num,self.dect_elem_count_horizontal,\
                                       self.source_dect_dis,self.img_sgm,\
                                           self.array_u_taichi,self.array_v_taichi,self.array_angle_taichi)#pass img_sgm directly into this function using unified memory
                    print('Filtering sinogram ...')
                    self.FilterSinogram()
                    self.SaveFilteredSinogram()
                                        
                    print('Back Projection ...')
                    self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                    self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                    self.array_angle_taichi, self.img_rot,self.img_sgm_filtered_taichi,self.img_recon_taichi,\
                                    self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                        self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                            self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
    
                    print('Saving to %s !' % self.output_path)
                    self.SaveReconImg()
        return self.img_recon #函数返回重建
    
    def __init__(self,config_dict):
        self.config_dict = config_dict
        ######## parameters related to input and output filenames ########
        if 'InputDir' in config_dict:
            if type(config_dict['InputDir']) == str:
                self.input_dir = config_dict['InputDir']
            else:
                print("ERROR: InputDir is not a string!")
                sys.exit()
        else:
            print("ERROR: Can not find InputDir in the config file!")
            sys.exit()
        
        if 'OutputDir' in config_dict:
            if type(config_dict['OutputDir']) == str:
                self.output_dir = config_dict['OutputDir']
            else:
                print("ERROR: OutputDir is not a string!")
                sys.exit()
        else:
            print("ERROR: Can not find OutputDir in the config file!")
            sys.exit()
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        if 'InputFiles' in config_dict:
            if type(config_dict['InputFiles']) == str:
                self.input_files_pattern = config_dict['InputFiles']
            else:
                print("ERROR: InputFiles is not a string!")
                sys.exit()
        else:
            print("ERROR: Can not find InputFiles in the config file!")
            sys.exit()
            
        if 'OutputFilePrefix' in config_dict:
            if type(config_dict['OutputFilePrefix']) == str:
                self.output_file_prefix = config_dict['OutputFilePrefix']
            else:
                print("ERROR: OutputFilePrefix is not a string!")
                sys.exit()
        else:
            print("ERROR: Can not find OutputFilePrefix in the config file!")
            sys.exit()

        if 'OutputFileReplace' in config_dict:
            self.output_file_replace = config_dict['OutputFileReplace']
        else:
            print("ERROR: Can not find OutputFileReplace in the config file!")
            sys.exit()
       
        
        #NEW define output file format: tif or raw
        if 'OutputFileFormat' in config_dict:
            self.output_file_format = config_dict['OutputFileFormat']
            if self.output_file_format == 'raw' \
                or self.output_file_format == 'tif'\
                    or self.output_file_format == 'tiff':
                        pass
            else:
                print("ERROR: Output file format can only be 'raw', 'tif' or 'tiff' !")
                sys.exit()
        else:
            self.output_file_format = 'raw'
        
        #NEW! define input file form, sinogram or post_log_images
        if 'InputFileForm' in config_dict:
            self.input_file_form = config_dict['InputFileForm']
            if self.input_file_form == 'sinogram' or self.input_file_form == 'post_log_images':
                pass
            else:
                print("ERROR: InputFileForm can only be sinogram or post_log_images!")
                sys.exit()
        else: 
            self.input_file_form = "sinogram"
            
        #for mgfpj configs; be consistent
        if 'OutputFileForm' in config_dict:
            self.output_file_form = config_dict['OutputFileForm']
            if self.output_file_form == 'sinogram' or self.output_file_form == 'post_log_images':
                pass
            else:
                print("ERROR: OutputFileForm can only be sinogram or post_log_images!")
                sys.exit()
        else:
            self.output_file_form = "sinogram"
        
        #NEW! define whether the first slice of the sinogram corresponds to the 
        #bottom detector row or the top row
        if self.input_file_form == 'post_log_images' or self.output_file_form == 'post_log_images':
            self.first_slice_top_row = True 
            # if the input file form are in post_log_imgs
            # the images are NOT up-side-down
            # first sinogram slice corresponds to top row of the detector
        elif 'FirstSinogramSliceIsDetectorTopRow' in config_dict:
            if isinstance(config_dict['FirstSinogramSliceIsDetectorTopRow'], bool):
                self.first_slice_top_row = config_dict['FirstSinogramSliceIsDetectorTopRow']
            else:
                print("ERROR: FirstSinogramSliceIsDetectorTopRow can only be true or false!")
                sys.exit()
        else:
            self.first_slice_top_row = False # by default, first sgm slice is detector bottom row
            
        if self.first_slice_top_row:
            self.positive_v_is_positive_z = -1
            print('--First sinogram slice corresponds to top detector row')
        else:
            self.positive_v_is_positive_z = 1
        
        if 'SaveFilteredSinogram' in config_dict:
            if isinstance(config_dict['SaveFilteredSinogram'], bool):
                self.save_filtered_sinogram = config_dict['SaveFilteredSinogram']
            else:
                print("ERROR: SaveFilteredSinogram can only be true or false!")
                sys.exit()
                
            if self.save_filtered_sinogram:
                print("--Filtered sinogram is saved")
        else:
            self.save_filtered_sinogram = False
        
        ######## parameters related to detector (fan beam case) ########
        #detector type (flat panel or curved)
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
            self.dect_elem_count_horizontal = config_dict['DetectorElementCountHorizontal']
        elif 'SinogramWidth' in config_dict:
            self.dect_elem_count_horizontal = config_dict['SinogramWidth']
        else:
            print("ERROR: Can not find detector element count along horizontal direction!")
            sys.exit()
        if self.dect_elem_count_horizontal <= 0:
            print("ERROR: DetectorElementCountHorizontal (SinogramWidth) should be larger than 0!")
            sys.exit()
        elif self.dect_elem_count_horizontal % 1 != 0:
            print("ERROR: DetectorElementCountHorizontal (SinogramWidth) should be an integer!")
            sys.exit()
            
        if 'DetectorElementWidth' in config_dict:
            self.dect_elem_width = config_dict['DetectorElementWidth']
        elif 'DetectorElementSize' in config_dict:
            self.dect_elem_width = config_dict['DetectorElementSize']
        else:
            print("ERROR: Can not find detector element width!")
            sys.exit()
        if self.dect_elem_width <= 0:
            print("ERROR: DetectorElementWidth (DetectorElementSize) should be larger than 0!")
            sys.exit()
        
        self.positive_u_is_positive_y = -1 #by default, positive u direction is -y direction
        
        if 'DetectorOffcenter' in config_dict:
            self.dect_offset_horizontal = config_dict['DetectorOffcenter']
        elif 'DetectorOffsetHorizontal' in config_dict:
            self.dect_offset_horizontal = config_dict['DetectorOffsetHorizontal']
        else:
            print("Warning: Can not find horizontal detector offset; Using default value 0")
        if not isinstance(self.dect_offset_horizontal, float) and not isinstance(self.dect_offset_horizontal, int) and not isinstance(self.dect_offset_horizontal, list):
            print("ERROR: DetectorOffsetHorizontal (DetectorOffcenter) should be a number!")
            sys.exit()
            
        if 'DetectorElementCountVertical' in config_dict:
            self.dect_elem_count_vertical = config_dict['DetectorElementCountVertical']
        elif 'SliceCount' in config_dict:
            self.dect_elem_count_vertical = config_dict['SliceCount']
        else:
            print("ERROR: Can not find detector element count along vertical direction!")
            sys.exit()
        if self.dect_elem_count_vertical <= 0:
            print("ERROR: DetectorElementCountVertical (SliceCount) should be larger than 0!")
            sys.exit()
        elif self.dect_elem_count_vertical % 1 != 0:
            print("ERROR: DetectorElementCountVertical (SliceCount) should be an integer!")
            sys.exit()
        
        #NEW! using partial slices of sinogram for reconstruction  
        if 'DetectorElementVerticalReconRange' in config_dict:
            temp_array = config_dict['DetectorElementVerticalReconRange']
            if not isinstance(temp_array, list) or len(temp_array)!=2:
                print("ERROR: DetectorElementVerticalReconRange should be an array with two numbers!")
                sys.exit()
            self.dect_elem_vertical_recon_range_begin = temp_array[0]
            self.dect_elem_vertical_recon_range_end = temp_array[1]
            if self.dect_elem_vertical_recon_range_end > self.dect_elem_count_vertical-1 or \
                self.dect_elem_vertical_recon_range_begin <0:
                print('ERROR: Out of detector row range!')
                sys.exit()
            if self.dect_elem_vertical_recon_range_begin%1!=0 or\
                self.dect_elem_vertical_recon_range_end%1!=0:
                print('ERROR: DetectorElementVerticalReconRange must be integers!')
                sys.exit()
            print(f"--Reconstructing from detector row #{temp_array[0]:d} to #{temp_array[1]:d}")
            self.dect_elem_count_vertical_actual = temp_array[1] - temp_array[0] + 1 
            #actual dect_elem_count_vertical defined by the row range
        else:
            self.dect_elem_vertical_recon_range_begin = 0
            self.dect_elem_vertical_recon_range_end = self.dect_elem_count_vertical-1
            self.dect_elem_count_vertical_actual = self.dect_elem_count_vertical
            
        #NEW! apply gauss smooth along z direction
        if 'DetectorElementVerticalGaussFilterSize' in config_dict:
            self.dect_elem_vertical_gauss_filter_size = config_dict['DetectorElementVerticalGaussFilterSize']
            if self.dect_elem_vertical_gauss_filter_size <= 0:
                print('ERROR: DetectorElementVerticalGaussFilterSize should be larger than 0!')
                sys.exit()
            self.apply_gauss_vertical = True
            print("--Apply Gaussian filter along the vertical direction of the detector")
        else:
            self.apply_gauss_vertical = False
            self.dect_elem_vertical_gauss_filter_size = 0.0001
            
        self.array_kernel_gauss_vertical_taichi = ti.field(dtype=ti.f32, shape=2*self.dect_elem_count_vertical_actual-1)
            
        
        ######## parameters related to CT scan rotation ########
        if 'Views' in config_dict:
            self.view_num = config_dict['Views']
        else:
            print("ERROR: Can not find number of views!")
            sys.exit()
        if self.view_num%1!=0 or self.view_num<0:
            print("ERROR: Views must be larger than 0 and must be an integer!")
            sys.exit()
            
        if 'SinogramHeight' in config_dict:
            self.sgm_height = config_dict['SinogramHeight']
        else:
            self.sgm_height = self.view_num
        if self.sgm_height%1!=0 or self.sgm_height<0:
            print("ERROR: SinogramHeight must be larger than 0 and must be an integer!")
            sys.exit()
        
        if 'TotalScanAngle' in config_dict:
            self.total_scan_angle = config_dict['TotalScanAngle'] / 180.0 * PI
            # TotalScanAngle is originally in degree; change it to rad
            # all angular variables are in rad unit
        else:
            self.total_scan_angle = 2*PI #by default, scan angle is 2*pi
        if not isinstance(self.total_scan_angle, float) and not isinstance(self.total_scan_angle, int):
            print("ERROR: TotalScanAngle should be a number!")
            sys.exit()
       
        if abs(self.total_scan_angle % (2 * PI)) < (0.01 / 180 * PI): 
            self.short_scan = 0
            print('--Full scan, scan angle = %.1f degrees' % (self.total_scan_angle / PI * 180))
        else:
            self.short_scan = 1
            print('--Short scan, scan angle = %.1f degrees' % (self.total_scan_angle / PI * 180))
            
        ######## NEW! Beam Hardening Correction Coefficients ########
        if 'BeamHardeningCorrectionCoefficients' in config_dict:
            print("--BH correction applied")
            self.bool_bh_correction = True
            self.array_bh_coefficients = config_dict['BeamHardeningCorrectionCoefficients']
            if not isinstance(self.array_bh_coefficients, list):
                print('ERROR: BeamHardeningCorrectionCoefficients must be an array!')
                sys.exit()
            self.bh_corr_order = len(self.array_bh_coefficients)
            self.array_bh_coefficients = np.array(self.array_bh_coefficients,dtype = np.float32)
            self.array_bh_coefficients_taichi = ti.field(dtype=ti.f32, shape = self.bh_corr_order)
            self.array_bh_coefficients_taichi.from_numpy(self.array_bh_coefficients)
        else:
            self.bool_bh_correction = False
                
        ######## CT scan geometries ########
        if 'SourceIsocenterDistance' in config_dict:
            self.source_isocenter_dis = config_dict['SourceIsocenterDistance']
            if not isinstance(self.source_isocenter_dis, float) and not isinstance(self.source_isocenter_dis, int):
                print('ERROR: SourceIsocenterDistance must be a number!')
                sys.exit()
            if self.source_isocenter_dis<=0.0:
                print('ERROR: SourceIsocenterDistance must be positive!')
                sys.exit()
        else:
            self.source_isocenter_dis = self.dect_elem_count_horizontal * self.dect_elem_width * 1000.0
            print('Warning: Did not find SourceIsocenterDistance; Set to infinity!')
          
        if 'SourceDetectorDistance' in config_dict:
            self.source_dect_dis = config_dict['SourceDetectorDistance']
            if not isinstance(self.source_dect_dis, float) and not isinstance(self.source_dect_dis, int):
                print('ERROR: SourceDetectorDistance must be a number!')
                sys.exit()
            if self.source_dect_dis<=0.0:
                print('ERROR: SourceDetectorDistance must be positive!')
                sys.exit()
        else:
            self.source_dect_dis = self.dect_elem_count_horizontal * self.dect_elem_width * 1000.0
            print('Warning: Did not find SourceDetectorDistance; Set to infinity!')
        
        
        ######## Reconstruction image size (in-plane) ########
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
            
        if 'ImageRotation' in config_dict:
            self.img_rot = config_dict['ImageRotation'] / 180.0 * PI
            if not isinstance(self.img_rot, float) and not isinstance(self.img_rot, int):
                print('ERROR: ImageRotation must be a number!')
                sys.exit()
        else:
            self.img_rot = 0.0
        # ImageRotation is originally in degree; change it to rad
        # all angular variables are in rad unit
        
        
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
        
        
        
        ######## NEW! reconstruction view mode (axial, coronal or sagittal) ########
        if 'ReconViewMode' in config_dict:
            temp_str = config_dict['ReconViewMode']
            if temp_str == 'axial':
                self.recon_view_mode = 1
            elif temp_str == 'coronal':
                self.recon_view_mode = 2
                print('--Coronal view')
            elif temp_str == 'sagittal':
                self.recon_view_mode = 3
                print('--Sagittal view')
            else:
                print("ERROR: ReconViewMode can only be axial, coronal or sagittal!")
                sys.exit()
        else: 
            self.recon_view_mode = 1
        
        ######## reconstruction kernel parameters ########
        if 'HammingFilter' in config_dict:
            self.kernel_name = 'HammingFilter'
            self.kernel_param = config_dict['HammingFilter']
        elif 'GaussianApodizedRamp' in config_dict:
            self.kernel_name = 'GaussianApodizedRamp'
            self.kernel_param = config_dict['GaussianApodizedRamp'] 
            self.array_kernel_ramp_taichi = ti.field(dtype=ti.f32, shape=2*self.dect_elem_count_horizontal-1)
            self.array_kernel_gauss_taichi = ti.field(dtype=ti.f32, shape=2*self.dect_elem_count_horizontal-1)
            #当进行高斯核运算的时候需要两个额外的数组存储相关数据
        elif 'None' in config_dict:
            self.kernel_name = 'None'
            self.kernel_param = 0.0
        else:
            self.kernel_param = 0.0 #kernel is not defined
        if not isinstance(self.kernel_param,float) and not isinstance(self.kernel_param,int):
            print("ERROR: Kernel parameter must be a number!")
            sys.exit()
            
        if 'DBTMode' in config_dict:#judge whether this recon is a dbt recon
            self.dbt_or_not = config_dict['DBTMode']
            if self.dbt_or_not:
                print("--DBT kernel applied")
        else:
            self.dbt_or_not = 0
        
        ######## whether images are converted to HU ########
        if 'WaterMu' in config_dict: 
            self.water_mu = config_dict['WaterMu']
            if not isinstance(self.water_mu,float) and not isinstance(self.water_mu,int):
                print("ERROR: WaterMu must be a number!")
                sys.exit()
            self.convert_to_HU = True
            print("--Converted to HU")
        else:
            self.convert_to_HU = False
            
        ######## cone beam reconstruction parameters ########
        if 'ConeBeam' in config_dict:
            self.cone_beam = config_dict['ConeBeam']
            if not isinstance(self.cone_beam,bool):
                print('ERROR: ConeBeam must be true or false!')
                sys.exit()
        else:
            self.cone_beam = False
            self.bool_apply_pmatrix = False
        
        if self.cone_beam:
            print("--Cone beam")
            
            #detector element height
            if 'SliceThickness' in config_dict:
                self.dect_elem_height = config_dict['SliceThickness']
            elif 'DetectorElementHeight' in config_dict:
                self.dect_elem_height = config_dict['DetectorElementHeight']
            else:
                print("ERROR: Can not find detector element height for cone beam recon! ")
                sys.exit()
            if not isinstance(self.dect_elem_height,float) and not isinstance(self.dect_elem_height,int):
                print('ERROR: DetectorElementHeight (SliceThickness) must be a number!')
                sys.exit()
            if self.dect_elem_height <= 0:
                print('ERROR: DetectorElementHeight (SliceThickness) must be positive!')
                sys.exit()
                
            #image dimension along z direction
            if 'ImageSliceCount' in config_dict:
                self.img_dim_z = config_dict['ImageSliceCount']
            elif 'ImageDimensionZ' in config_dict:
                self.img_dim_z = config_dict['ImageDimensionZ']
            else:
                print("ERROR: Can not find image dimension along Z direction for cone beam recon!")
                sys.exit() 
            if not isinstance(self.img_dim_z,int) or self.img_dim_z <= 0:
                print('ERROR: ImageDimensionZ (ImageSliceCount) must be a positive integer!')
                sys.exit()
                
            #image voxel height
            if 'VoxelHeight' in config_dict:
                self.img_voxel_height = config_dict['VoxelHeight']
            elif 'ImageSliceThickness' in config_dict:
                self.img_voxel_height = config_dict['ImageSliceThickness']
            else:
                print("ERROR: Can not find image voxel height for cone beam recon!")
                sys.exit()
            if not isinstance(self.img_voxel_height,float) and not isinstance(self.img_voxel_height,int):
                print('ERROR: VoxelHeight (ImageSliceThickness) must be a number!')
                sys.exit()
            if self.img_voxel_height <= 0:
                print('ERROR: VoxelHeight (ImageSliceThickness) must be positive!')
                sys.exit()
            
            ######### projection matrix recon parameters ########
            ##Projection matrix is only effective for cone beam reconstruction##
            self.array_pmatrix_taichi = ti.field(dtype=ti.f32, shape = self.view_num * 12)
            if 'PMatrixFile' in config_dict:
                temp_dict = ReadConfigFile(config_dict['PMatrixFile'])
                if 'Value' in temp_dict:
                    self.array_pmatrix = temp_dict['Value']
                    if not isinstance(self.array_pmatrix,list):
                        print('ERROR: PMatrixFile.Value is not an array')
                        sys.exit()
                    if len(self.array_pmatrix) != self.view_num * 12:
                        print(f'ERROR: view number is {self.view_num:d} while pmatrix has {len(self.array_pmatrix):d} elements!')
                        sys.exit()
                    self.array_pmatrix = np.array(self.array_pmatrix,dtype = np.float32)
                    self.array_pmatrix_taichi.from_numpy(self.array_pmatrix)
                    self.bool_apply_pmatrix = 1
                    print("--PMatrix applied")
                else:
                    print("ERROR: PMatrixFile has no member named 'Value'!")
                    sys.exit()
            else:
                self.bool_apply_pmatrix = 0
            
            if self.bool_apply_pmatrix:
                #whether to change the pmatrix source trajectory to standard form 
                #(source positions will be on the z=0 plane with center at the origin)
                #(source position for the first view will be on the +x axis)
                if 'ModifyPMatrixToStandardForm' in config_dict:
                    self.modify_pmatrix_to_standard_form = config_dict['ModifyPMatrixToStandardForm']
                    if not isinstance(self.modify_pmatrix_to_standard_form,bool):
                        print("ERROR: ModifyPMatrixToStandardForm must be True or False!")
                        sys.exit();
                else:
                    self.modify_pmatrix_to_standard_form = False # false by default, since only data from Wandong requires pmatrix file modification
                
                ## Change the projection matrix Values if a different detector binning is applied for obj CT scan
                ## compared with the binning mode for pmatrix
                if 'PMatrixDetectorElementWidth' in config_dict:
                    print('--PMatrix detector pixel width is different from the CT scan')
                    self.pmatrix_elem_width = config_dict['PMatrixDetectorElementWidth'];
                    if not isinstance(self.pmatrix_elem_width,float) and not isinstance(self.pmatrix_elem_width,int):
                        print('ERROR: PMatrixDetectorElementWidth must be a number!')
                        sys.exit()
                    if self.pmatrix_elem_width <= 0:
                        print('ERROR: PMatrixDetectorElementWidth must be positive!')
                        sys.exit()
                else:
                    self.pmatrix_elem_width = self.dect_elem_width;
                
                if 'PMatrixDetectorElementHeight' in config_dict:
                    print('--PMatrix detector pixel height is different from the CT scan')
                    self.pmatrix_elem_height = config_dict['PMatrixDetectorElementHeight'];
                    if not isinstance(self.pmatrix_elem_height,float) and not isinstance(self.pmatrix_elem_height,int):
                        print('ERROR: PMatrixDetectorElementHeight must be a number!')
                        sys.exit()
                    if self.pmatrix_elem_height <= 0:
                        print('ERROR: PMatrixDetectorElementHeight must be positive!')
                        sys.exit()
                else:
                    self.pmatrix_elem_height = self.dect_elem_height;
                
                
                self.ChangePMatrix_SourceTrajectory() 
                #if self.modify_pmatrix_to_standard_form
                #this function will change the pmatrix source trajectory to standard form 
                #(source positions will be on the z=0 plane with center at the origin)
                #(source position for the first view will be on the +x axis)
                #if not, this function will output the parameters from pmatrix
                #the parmaters include: source_isocenter_dis,source_dect_dis, 
                #dect_offset_horizontal, dect_offset_vertical and total_scan_angle. 
                #save the updated parameters values to the dictionary
                config_dict['SourceIsocenterDistance'] = self.source_isocenter_dis
                config_dict['SourceDetectorDistance'] = self.source_dect_dis
                config_dict['DetectorOffsetHorizontal'] = self.dect_offset_horizontal
                config_dict['DetectorOffsetVertical'] = self.dect_offset_vertical
                config_dict['TotalScanAngle'] = self.total_scan_angle / PI * 180.0
                self.ChangePMatrix_PMatrixPixelSize()
                self.ChangePMatrix_VerticalReconRange()
                self.array_pmatrix_taichi.from_numpy(self.array_pmatrix) #update the taichi array
                
            if not self.bool_apply_pmatrix:
                #detector offset vertical (is only effective when pmatrix is not applied)
                if 'SliceOffCenter' in config_dict:
                    self.dect_offset_vertical = config_dict['SliceOffCenter'] 
                elif 'DetectorOffsetVertical' in config_dict:
                    self.dect_offset_vertical = config_dict['DetectorOffsetVertical']
                else: 
                    self.dect_offset_vertical = 0
                    print("Warning: Can not find vertical detector offset for cone beam recon; Using default value 0")
                if not isinstance(self.dect_offset_vertical,float) and not isinstance(self.dect_offset_vertical,int):
                    print('ERROR: DetectorOffsetVertical (SliceOffCenter) must be a number!')
                    sys.exit()
            
                
            #img center along z direction
            if 'ImageCenterZ' in config_dict:
                self.img_center_z = config_dict['ImageCenterZ']
                self.img_center_z_auto_set_from_fbp = False
                if not isinstance(self.img_center_z,float) and not isinstance(self.img_center_z,int):
                    print('ERROR: ImageCenterZ must be a number!')
                    sys.exit()
            else:
                #ImageCenterZ is calculated from detector offset along vertical direction
                #may use the updated value
                current_center_row_idx = (self.dect_elem_vertical_recon_range_end +  self.dect_elem_vertical_recon_range_begin)/2
                distance_to_original_detector_center_row = (current_center_row_idx - (self.dect_elem_count_vertical-1)/2) * self.dect_elem_height
                if self.first_slice_top_row:
                    distance_to_original_detector_center_row = distance_to_original_detector_center_row * (-1)
                self.img_center_z = (self.dect_offset_vertical + distance_to_original_detector_center_row)\
                    * self.source_isocenter_dis / self.source_dect_dis
                print("Warning: Did not find image center along z direction! ")
                print("Use default setting (central slice of the given detector recon row range)")
                print("Image center at Z direction is %.4f mm (from run_mgfbp). " %self.img_center_z)
                self.img_center_z_auto_set_from_fbp = True
                config_dict['ImageCenterZ'] = self.img_center_z
        else:
            print("--Fan beam")
            self.dect_elem_height = 0.0
            self.dect_offset_vertical = 0.0
            self.img_dim_z = self.dect_elem_count_vertical
            self.img_voxel_height = 0.0
            self.img_center_z = 0.0
            self.bool_apply_pmatrix = False
            self.array_pmatrix_taichi = ti.field(dtype=ti.f32, shape=self.view_num * 12)
        
            
        self.img_recon = np.zeros((self.img_dim_z,self.img_dim,self.img_dim),dtype = np.float32)
        
        
        self.img_recon_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z,self.img_dim, self.img_dim),order='ikj')
        #img_recon_taichi is the reconstructed img
        self.array_angle_taichi = ti.field(dtype=ti.f32, shape=self.view_num)
        #angel_taichi存储旋转角度，且经过计算之后以弧度制表示
        
        self.array_recon_kernel_taichi = ti.field(dtype=ti.f32, shape=2*self.dect_elem_count_horizontal-1)
        #存储用于对正弦图进行卷积的核
        self.array_u_taichi = ti.field(dtype=ti.f32,shape=self.dect_elem_count_horizontal)
        #存储数组u
        self.array_v_taichi = ti.field(dtype = ti.f32,shape = self.dect_elem_count_vertical_actual)
        
        #save the parameters from pmatrix
        if 'SaveModifiedConfigFolder' in config_dict:
            if isinstance(config_dict['SaveModifiedConfigFolder'], str):
                save_config_folder_name = config_dict['SaveModifiedConfigFolder']
                del config_dict['SaveModifiedConfigFolder']
                if not os.path.exists(save_config_folder_name):
                    os.makedirs(save_config_folder_name)
                save_jsonc(save_config_folder_name + "/config_mgfbp.jsonc", config_dict)
                print('Modified config files are saved to %s folder.' %(save_config_folder_name))
                if self.bool_apply_pmatrix:
                    config_pmatrix = {}
                    config_pmatrix['Value'] = self.array_pmatrix.tolist()
                    save_jsonc(save_config_folder_name + "/pmatrix_file.jsonc", config_pmatrix)
                    config_sid = {}
                    config_sid['Value'] = self.source_isocenter_dis_each_view.tolist()
                    save_jsonc(save_config_folder_name + "/sid_file.jsonc", config_sid)
                    config_sdd = {}
                    config_sdd['Value'] = self.source_dect_dis_each_view.tolist()
                    save_jsonc(save_config_folder_name + "/sdd_file.jsonc", config_sdd)
                    config_dect_offset_horizontal = {}
                    config_dect_offset_horizontal['Value'] = np.squeeze(self.dect_offset_horizontal_each_view).tolist()
                    save_jsonc(save_config_folder_name + "/dect_offset_horizontal_file.jsonc", config_dect_offset_horizontal)
                    config_dect_offset_vertical = {}
                    config_dect_offset_vertical['Value'] = np.squeeze(self.dect_offset_vertical_each_view).tolist()
                    save_jsonc(save_config_folder_name + "/dect_offset_vertical_file.jsonc", config_dect_offset_vertical)
                    config_scan_angle = {}
                    config_scan_angle['Value'] = np.squeeze(self.scan_angle_each_view).tolist()
                    save_jsonc(save_config_folder_name + "/scan_angle_file.jsonc", config_scan_angle)
            else:
                print('ERROR: SaveModifiedConfigFolder must be a string!')
                sys.exit()
            
    def ChangePMatrix_SourceTrajectory(self):
        ## Change the projection matrix values
        ## If the plane of the source position is not perpendicular to the z-axis
        ## and the source position of the first view is not located on the +x-axis 
        ## The original pmatrix need to be modified
        x_s_rec = np.zeros(shape = (3,self.view_num))
        for view_idx in range(self.view_num): 
            pmatrix_this_view = self.array_pmatrix[(view_idx*12):(view_idx+1)*12]#get the pmatrix for this view
            pmatrix_this_view = np.reshape(pmatrix_this_view,[3,4])#reshape it to be 3x4 matrix
            matrix_A = np.linalg.inv(pmatrix_this_view[:,0:3])#matrix A is the inverse of the 3x3 matrix from the first three columns
            x_s_rec[:,view_idx:view_idx+1]=- np.matmul(matrix_A,pmatrix_this_view[:,3]).reshape([3,1])#calculate the source position
        #after getting the position of the source, we need to calculate the plane of the source trajectory
        #assume the plane is defined as a*x + b*y + c*z = 1
        matrix_B = x_s_rec.transpose()
        vec_y = np.ones(shape = (self.view_num,1))
        sol = XuLeastSquareSolution(matrix_B, vec_y) #solve a, b and c using least squared method
        
        angle_y = np.arctan(sol[0] / sol[2]) #first rotate along y axis
        angle_x = np.arctan(sol[1] / np.sqrt( sol[0]**2 + sol[2] ** 2 )) * np.sign(sol[2]) #then rotate along x axis
        rotation_matrix_y = np.array([[math.cos(angle_y),0,-math.sin(angle_y)],[0,1,0],[math.sin(angle_y),0,math.cos(angle_y)]])
        rotation_matrix_x = np.array([[1,0,0],[0,math.cos(angle_x),-math.sin(angle_x)],[0,math.sin(angle_x),math.cos(angle_x)]])
        x_s_rec_rot = np.matmul(rotation_matrix_x,np.matmul(rotation_matrix_y,x_s_rec))
        z_c = np.mean(x_s_rec_rot[2,:]) #center of the circle along the z-direction
        
        x_s_xy_plane = x_s_rec_rot[0:2,:] #get the position when the points are projected onto the xy plane
        #after the projection operation, we need to determine the center for a 2D circle, which is easy
        #assume the circle is x^2 + y^2 - a*x - b*y + c = 0
        matrix_B = np.concatenate((-x_s_xy_plane.transpose(),np.ones(shape=(self.view_num,1))),axis = 1)
        vec_y = -np.sum(x_s_xy_plane**2,axis = 0)
        sol = XuLeastSquareSolution(matrix_B, vec_y) # solution for a b and c; center is a/2 and b/2
        x_s_rec_rot_shift_xyz = x_s_rec_rot - np.array([[sol[0]/2],[sol[1]/2],[z_c]]) 
        # get the position of the source when it is shifted so that the center of the circle is at the origin.
        # now the source trajectory is in the x-y plane and the circle of the center is located as the origin.
        # we need to rotate it along the z-axis so that for the first view, the source is located at +x axis. 
        angle_z = math.atan2(x_s_rec_rot_shift_xyz[1,0],x_s_rec_rot_shift_xyz[0,0])
        rotation_matrix_z = np.array([[math.cos(angle_z),math.sin(angle_z),0],[-math.sin(angle_z),math.cos(angle_z),0],[0,0,1]])
        x_s_rec_final = np.matmul(rotation_matrix_z,x_s_rec_rot_shift_xyz)#final source positions
        rotation_matrix_total = np.matmul(rotation_matrix_z,np.matmul(rotation_matrix_x,rotation_matrix_y)) 
        #multiply three rotation operations together
        
        if self.modify_pmatrix_to_standard_form == False:
            #if we do not want to modify the pmatrix, rotation_matrix_total is set to be an identity matrix
            rotation_matrix_total = np.eye(3)
            x_s_rec_final = x_s_rec
        
        v_center_rec = np.zeros(shape = (self.view_num,1))#array to record the center along v direction
        u_center_rec = np.zeros(shape = (self.view_num,1))#array to record the center along u direction
        scan_angle_summation = 0.0
        self.scan_angle_each_view = np.zeros(shape = (self.view_num,1))
        x_d_center_x_s_rec_final = np.zeros(shape = (3, self.view_num))#array to record the center along u direction
        for view_idx in range(self.view_num):
            pmatrix_this_view = self.array_pmatrix[(view_idx*12):(view_idx+1)*12]#get the pmatrix for this view
            pmatrix_this_view = np.reshape(pmatrix_this_view,[3,4])#reshape it to be 3x4 matrix
            matrix_A = np.linalg.inv(pmatrix_this_view[:,0:3])#matrix A is the inverse of the 3x3 matrix from the first three columns
            e_v_0 = matrix_A[:,1] #calculate the unit vector along detector vertical direction
            e_u_0 = matrix_A[:,0] #calculate the unit vector along detector horizontal direction
            pixel_size_ratio = (self.pmatrix_elem_width / np.linalg.norm(e_u_0) + self.pmatrix_elem_height / np.linalg.norm(e_v_0)) * 0.5
            #confirm the pixel size from pmatrix is the same with the preset pmatrix pixel size;
            pmatrix_this_view = pmatrix_this_view / pixel_size_ratio #if not, need to normalize the pmatrix
            
            matrix_A = np.linalg.inv(pmatrix_this_view[:,0:3]) #matrix A is the inverse of the 3x3 matrix from the first three columns
            x_s = x_s_rec_final[:,view_idx:view_idx+1] #calculate the source position
            if view_idx <= self.view_num - 2: #calculate total scan angle
                x_s_next_view =  x_s_rec_final[:,view_idx+1:view_idx+2]
                delta_angle = math.atan2(x_s_next_view[1], x_s_next_view[0]) - math.atan2(x_s[1], x_s[0])
                if abs(delta_angle) > PI:
                    delta_angle = delta_angle - 2 * PI *np.sign(delta_angle)
                scan_angle_summation = scan_angle_summation + delta_angle
                self.scan_angle_each_view[view_idx+1,0] = scan_angle_summation / PI * 180.0
            e_v_0 = matrix_A[:,1] #calculate the unit vector along detector vertical direction
            e_u_0 = matrix_A[:,0] #calculate the unit vector along detector horizontal direction
            e_v = np.matmul(rotation_matrix_total, e_v_0) #change the unit vector along detector vertical direction
            e_u = np.matmul(rotation_matrix_total, e_u_0) #change the unit vector along detector horizontal direction
            if view_idx == 0:#judge the positive direction of u and v from e_u and e_v in the first view
                self.positive_u_is_positive_y = np.sign(e_u[1]) # if positive_u_is_positive_y, this outputs 1; otherwise -1; 
                if self.positive_u_is_positive_y == 1:
                    print("Attention: +u direction is along +y based on pmatrix! ")
                self.positive_v_is_positive_z = np.sign(e_v[2]) # if positive_v_is_positive_z, this outputs 1; otherwise -1; 
                if self.positive_v_is_positive_z == 1:
                    print("Attention: +v direction is along +z based on pmatrix! ")
                
            x_do_x_s = np.matmul(rotation_matrix_total, matrix_A[:,2]) #change x_do - x_s
            matrix_A[:,0] = e_u; matrix_A[:,1] = e_v; matrix_A[:,2] = x_do_x_s; #update the matrix A
            matrix_A_inverse = np.linalg.inv(matrix_A) #recalculate the inverse of matrix A
            pmatrix_this_view = np.append(matrix_A_inverse, -np.matmul(matrix_A_inverse,x_s), axis = 1)#get the new pmatrix for this view
            u_center_rec[view_idx, 0]= pmatrix_this_view[0,3] / pmatrix_this_view[2,3]#get center of the pixel along u direction
            v_center_rec[view_idx, 0]= pmatrix_this_view[1,3] / pmatrix_this_view[2,3]#get center of the pixel along v direction
            v_center_rec[view_idx, 0]= -np.dot(x_do_x_s,e_v)/(self.pmatrix_elem_height**2)  #get center of the pixel along v direction
            x_d_center_x_s_rec_final[:,view_idx:view_idx+1] = x_do_x_s.reshape((3,1)) + \
                u_center_rec[view_idx, 0] * e_u.reshape((3,1)) + v_center_rec[view_idx, 0] * e_v.reshape((3,1))
            
            pmatrix_this_view = np.reshape(pmatrix_this_view,[12,1])#reshape the matrix to be a vector
            self.array_pmatrix[(view_idx*12):(view_idx+1)*12] = pmatrix_this_view[:,0] #update the pmatrix array
        #get offset value for each view
        self.dect_offset_vertical_each_view = self.positive_v_is_positive_z * ((self.dect_elem_count_vertical *self.dect_elem_height / self.pmatrix_elem_height - 1) / 2.0\
                                       - v_center_rec) * self.pmatrix_elem_height #+z is positive
        self.dect_offset_horizontal_each_view = -self.positive_u_is_positive_y * ((self.dect_elem_count_horizontal *self.dect_elem_width/ self.pmatrix_elem_width - 1) / 2.0\
                                       - u_center_rec) * self.pmatrix_elem_width #-y is positive based on mgfbp.exe convention
        #update the parameters from the pmatrix
        self.dect_offset_vertical = np.squeeze(np.mean(self.dect_offset_vertical_each_view,axis = 0)).tolist()#calculate the mean offset
        self.dect_offset_horizontal = np.squeeze(np.mean(self.dect_offset_horizontal_each_view,axis = 0)).tolist()#calculate the mean offset 
        self.source_isocenter_dis_each_view = np.sqrt(np.sum(np.multiply(x_s_rec_final,x_s_rec_final), axis = 0))
        self.source_dect_dis_each_view = np.sqrt(np.sum(np.multiply(x_d_center_x_s_rec_final,x_d_center_x_s_rec_final), axis = 0))
        self.source_isocenter_dis =  np.squeeze( np.mean(self.source_isocenter_dis_each_view, axis = 0)).tolist()
        self.source_dect_dis =  np.squeeze( np.mean( self.source_dect_dis_each_view, axis = 0)).tolist()
        print('Parameters are updated from PMatrix:')
        print('Mean Offset values are %.2f mm and %.2f mm for horizontal and vertical direction respectively;' \
              %( self.dect_offset_horizontal, self.dect_offset_vertical))
        print('Mean Source to Isocenter Distance is %.2f mm;' %(self.source_isocenter_dis))
        print('Mean Source to Detector Distance is %.2f mm;' %(self.source_dect_dis))
        if self.short_scan:
            #update the total scan angle only when the scan is not 360 degree full scan
            #for 360 degree scan, if total scan angle is updated (e.g. 359.5 degree)
            #the div_factor calculation in back projection may have problem
            self.total_scan_angle = scan_angle_summation / (self.view_num - 1) * self.view_num
            print('Total Scan Angle is %.2f degrees.'  %( self.total_scan_angle / PI * 180.0))
        else:
            print('Total Scan Angle is not updated for full scan.')
            
    def ChangePMatrix_PMatrixPixelSize(self):
        ## Change the projection matrix values if the detector pixel size for pmatrix is different from the CT scan
        ## The original pmatrix need to be modified
        for view_idx in range(self.view_num):
            pmatrix_this_view = self.array_pmatrix[(view_idx*12):(view_idx+1)*12]#get the pmatrix for this view
            pmatrix_this_view = np.reshape(pmatrix_this_view,[3,4])#reshape it to be 3x4 matrix
            matrix_A = np.linalg.inv(pmatrix_this_view[:,0:3])#matrix A is the inverse of the 3x3 matrix from the first three columns
            x_s = - np.matmul(matrix_A,pmatrix_this_view[:,3]).reshape([3,1])#calculate the source position
            e_v_0 = matrix_A[:,1] #calculate the unit vector along detector vertical direction
            e_u_0 = matrix_A[:,0] #calculate the unit vector along detector horizontal direction
            e_v = e_v_0 * self.dect_elem_height / self.pmatrix_elem_height #change the unit vector along detector vertical direction
            e_u = e_u_0 * self.dect_elem_width / self.pmatrix_elem_width #change the unit vector along detector horizontal direction
            x_do_x_s = matrix_A[:,2] + np.multiply(0.5, e_v - e_v_0) + np.multiply(0.5, e_u - e_v_0)#change x_do - x_s
            matrix_A[:,0] = e_u; matrix_A[:,1] = e_v; matrix_A[:,2] = x_do_x_s; #update the matrix A
            matrix_A_inverse = np.linalg.inv(matrix_A) #recalculate the inverse of matrix A
            pmatrix_this_view = np.append(matrix_A_inverse, -np.matmul(matrix_A_inverse,x_s), axis = 1)#get the new pmatrix for this view
            pmatrix_this_view = np.reshape(pmatrix_this_view,[12,1])#reshape the matrix to be a vector
            self.array_pmatrix[(view_idx*12):(view_idx+1)*12] = pmatrix_this_view[:,0] #update the pmatrix array
            
    def ChangePMatrix_VerticalReconRange(self):
        ## Change the projection matrix values if vertical recon range is applied
        ## The original pmatrix need to be modified
        for view_idx in range(self.view_num):
            pmatrix_this_view = self.array_pmatrix[(view_idx*12):(view_idx+1)*12]#get the pmatrix for this view
            pmatrix_this_view = np.reshape(pmatrix_this_view,[3,4])#reshape it to be 3x4 matrix
            matrix_A = np.linalg.inv(pmatrix_this_view[:,0:3])#matrix A is the inverse of the 3x3 matrix from the first three columns
            x_s = - np.matmul(matrix_A,pmatrix_this_view[:,3]).reshape([3,1])#calculate the source position
            e_v = matrix_A[:,1]#calculate the unit vector along detector vertical direction
            matrix_A[:,2] = matrix_A[:,2] + np.multiply(self.dect_elem_vertical_recon_range_begin, e_v)
            #calculate the new x_do - x_s
            
            matrix_A_inverse = np.linalg.inv(matrix_A) #recalculate the inverse of matrix A
            pmatrix_this_view = np.append(matrix_A_inverse, -np.matmul(matrix_A_inverse,x_s), axis = 1)#get the new pmatrix for this view
            pmatrix_this_view = np.reshape(pmatrix_this_view,[12,1])#reshape the matrix to be a vector
            self.array_pmatrix[(view_idx*12):(view_idx+1)*12] = pmatrix_this_view[:,0] #update the pmatrix array
        
        
    @ti.kernel
    def GenerateHammingKernel(self,dect_elem_count_horizontal:ti.i32,dect_elem_width:ti.f32,kernel_param:ti.f32,\
                              source_dect_dis:ti.f32, source_isocenter_dis:ti.f32, array_recon_kernel_taichi:ti.template(),curved_dect:ti.i32 ,dbt_or_not:ti.i32):
        #计算hamming核分两步处理
        n = 0
        bias = dect_elem_count_horizontal - 1
        t = kernel_param
        for i in ti.ndrange(2 * dect_elem_count_horizontal - 1):  
            n = i - bias
            #part 1 ramp
            if n == 0:
                array_recon_kernel_taichi[i] = t / (4 * dect_elem_width * dect_elem_width)
            elif n % 2 == 0:
                array_recon_kernel_taichi[i] = 0
            else:
                if curved_dect:
                    temp_val = float(n) * dect_elem_width / source_dect_dis
                    array_recon_kernel_taichi[i] = -t / (PI * PI * (source_dect_dis **2) * (temp_val - temp_val**3/3/2/1 + temp_val**5/5/4/3/2/1)**2)
                    #use taylor expansion to replace the built-in taichi.sin function
                    #the built-in taichi.sin function leads to 1% bias in calculation
                else:
                    array_recon_kernel_taichi[i] = -t / (PI * PI * (float(n) **2) * (dect_elem_width **2))
            #part 2 cosine
            sgn = 1 if n % 2 == 0 else -1
            array_recon_kernel_taichi[i] += (1-t)*(sgn/(2 * PI * dect_elem_width * dect_elem_width)*(1/(1 + 2 * n)+ 1 / (1 - 2 * n))- 1 / (PI * PI * dect_elem_width * dect_elem_width) * (1 / (1 + 2 * n) / (1 + 2 * n) + 1 / (1 - 2 * n) / (1 - 2 * n)))
            
            # #modified ramp for DBT
            if dbt_or_not == 1:
                k_t = 0.01 / (2 * (dect_elem_width ))
                if n == 0:
                    array_recon_kernel_taichi[i] += k_t ** 2 * source_isocenter_dis / source_dect_dis 
                    # add this * source_isocenter_dis / source_dect_dis factor so that images with different magnification look the same
                    # there is an 1/source_dect_dis factor factor in the weightsgm function
                else:
                    temp_val = n * dect_elem_width * k_t * PI
                    array_recon_kernel_taichi[i] += (ti.sin(temp_val) **2) / ((temp_val) **2) * (k_t **2) * source_isocenter_dis / source_dect_dis
            else:
                pass
            

    @ti.kernel
    def GenerateGassianKernel(self,dect_elem_count_horizontal:ti.i32,dect_elem_width:ti.f32,kernel_param:ti.f32,array_kernel_gauss_taichi:ti.template()):
        #计算高斯
        temp_sum = 0.0
        delta = kernel_param
        for i in ti.ndrange(2 * dect_elem_count_horizontal - 1):
            n = i - (dect_elem_count_horizontal - 1)
            array_kernel_gauss_taichi[i] = ti.exp(-n*n/2/delta/delta)
            temp_sum+=array_kernel_gauss_taichi[i]
        for i in ti.ndrange(2 * dect_elem_count_horizontal - 1):
            array_kernel_gauss_taichi[i] = array_kernel_gauss_taichi[i]/temp_sum / dect_elem_width


    @ti.kernel
    def GenerateAngleArray(self,view_num:ti.i32,img_rot:ti.f32,scan_angle:ti.f32,array_angle_taichi:ti.template()):
        #计算beta并用弧度制的形式表示
        for i in ti.ndrange(view_num):
            array_angle_taichi[i] = (scan_angle / view_num * i ) + img_rot
  

    @ti.kernel
    def GenerateDectPixPosArray(self,dect_elem_count_horizontal:ti.i32,dect_elem_count_horizontal_actual:ti.i32,dect_elem_width:ti.f32,\
                                dect_offset_horizontal:ti.f32,array_u_taichi:ti.template(),dect_elem_begin_idx:ti.i32):
        # dect_elem_begin_idx is for recon of partial slices of the sinogram
        # since the slice idx may not begin with 0
        for i in ti.ndrange(dect_elem_count_horizontal_actual):
            array_u_taichi[i] = (i + dect_elem_begin_idx - (dect_elem_count_horizontal - 1) / 2.0) \
                * dect_elem_width + dect_offset_horizontal
                
    @ti.kernel
    def BHCorrection(self, dect_elem_count_vertical_actual:ti.i32, view_num:ti.i32, dect_elem_count_horizontal:ti.i32,img_sgm_taichi:ti.types.ndarray(dtype=ti.f32, ndim=3),\
                     array_bh_coefficients_taichi:ti.template(),bh_corr_order:ti.i32):
        #对正弦图做加权，包括fan beam的cos加权和短扫面加权
        for  i, j, s in ti.ndrange(view_num, dect_elem_count_horizontal, dect_elem_count_vertical_actual):
            temp_val = 0.0
            for t in ti.ndrange(bh_corr_order):
                temp_val = temp_val + array_bh_coefficients_taichi[t] * (img_sgm_taichi[s,i,j]**(t+1))#apply ploynomial calculation
            img_sgm_taichi[s,i,j] = temp_val

    @ti.kernel
    def WeightSgm(self, dect_elem_count_vertical_actual:ti.i32, short_scan:ti.i32, curved_dect:ti.i32, scan_angle:ti.f32,\
                  view_num:ti.i32, dect_elem_count_horizontal:ti.i32, source_dect_dis:ti.f32,img_sgm_taichi:ti.types.ndarray(dtype=ti.f32, ndim=3),\
                      array_u_taichi:ti.template(),array_v_taichi:ti.template(),array_angle_taichi:ti.template()):
        #对正弦图做加权，包括fan beam的cos加权和短扫面加权
        for  i, j in ti.ndrange(view_num, dect_elem_count_horizontal):
            u_actual = array_u_taichi[j]
            for s in ti.ndrange(dect_elem_count_vertical_actual):
                v_actual = array_v_taichi[s]
                if curved_dect:
                    img_sgm_taichi[s,i,j] = img_sgm_taichi[s,i,j] * source_dect_dis * ti.math.cos( (-1)*u_actual/source_dect_dis) \
                        * source_dect_dis / ((source_dect_dis**2 + v_actual**2)**0.5)
                else:
                    img_sgm_taichi[s,i,j]=(img_sgm_taichi[s,i,j] * source_dect_dis * source_dect_dis ) \
                        / (( source_dect_dis **2 + u_actual**2 + v_actual **2) ** 0.5)
                if short_scan:
                    #for scans longer than 360 degrees but not multiples of 360, we also need to apply parker weighting
                    #for example, for a 600 degrees scan, we also need to apply parker weighting
                    num_rounds = ti.floor(abs(scan_angle) / (PI * 2))
                    remain_angle = abs(scan_angle) - num_rounds * PI * 2
                    #angle remains: e.g., if totalScanAngle = 600 degree, remain_angle = 240 degree
                    beta = abs(array_angle_taichi[i] - array_angle_taichi[0])
                    rotation_direction =  abs(scan_angle) / (scan_angle)
                    gamma = 0.0
                    if curved_dect:
                        gamma = ((-1) * u_actual / source_dect_dis) * rotation_direction
                        #positive y corresponds to clockwise -> negative gamma
                    else:
                        gamma = ti.atan2((-1) *u_actual, source_dect_dis) * rotation_direction
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
                    img_sgm_taichi[s,i,j] *= weighting
                
    @ti.kernel
    def ConvolveSgmAndKernel(self, dect_elem_count_vertical_actual:ti.i32, view_num:ti.i32, \
                             dect_elem_count_horizontal:ti.i32, dect_elem_width:ti.f32, img_sgm_taichi:ti.types.ndarray(dtype=ti.f32, ndim=3), \
                                 array_recon_kernel_taichi:ti.template(),array_kernel_gauss_vertical_taichi:ti.template(),\
                                     dect_elem_height:ti.f32, apply_gauss_vertical:ti.i32,img_sgm_filtered_intermediate_taichi:ti.template(),\
                                         img_sgm_filtered_taichi:ti.template()):
        #apply filter along vertical direction
        for i, j, k in ti.ndrange(dect_elem_count_vertical_actual, view_num, dect_elem_count_horizontal):
            temp_val = 0.0
            if apply_gauss_vertical:
                # if vertical filter is applied, apply vertical filtering and 
                # save the intermediate result to img_sgm_filtered_intermediate_taichi
                for n in ti.ndrange(dect_elem_count_vertical_actual):
                    if i - n <= 10 and  i - n >=-10: #set a 10 pixel threshold to accelerate the program  
                        temp_val += img_sgm_taichi[n, j, k] \
                            * array_kernel_gauss_vertical_taichi[i + (dect_elem_count_vertical_actual - 1) - n]
                img_sgm_filtered_intermediate_taichi[i, j, k] = temp_val * dect_elem_height
            else:
                pass
                
        for i, j, k in ti.ndrange(dect_elem_count_vertical_actual, view_num, dect_elem_count_horizontal):
            temp_val = 0.0 
            if apply_gauss_vertical:
                # if vertical filter is applied, use img_sgm_filtered_intermediate_taichi
                # for horizontal filtering
                for m in ti.ndrange(dect_elem_count_horizontal):
                    temp_val += img_sgm_filtered_intermediate_taichi[i, j, m] \
                        * array_recon_kernel_taichi[ k + (dect_elem_count_horizontal - 1) - m]
            else:
                # if not, use img_sgm_taichi
                for m in ti.ndrange(dect_elem_count_horizontal):
                    temp_val += img_sgm_taichi[i, j, m] \
                            * array_recon_kernel_taichi[ k + (dect_elem_count_horizontal - 1) - m]
            img_sgm_filtered_taichi[i, j, k] = temp_val * dect_elem_width

    @ti.kernel
    def ConvolveKernelAndKernel(self, dect_elem_count_horizontal:ti.i32, \
                                dect_elem_width:ti.f32, array_kernel_ramp_taichi:ti.template(), \
                                    array_kernel_gauss_taichi:ti.template(), array_recon_kernel_taichi:ti.template()):
        #当核为高斯核时卷积计算的过程之一
        for i in ti.ndrange(2 * dect_elem_count_horizontal - 1):
            reconKernel_conv_local = 0.0
            for j in ti.ndrange(2 * dect_elem_count_horizontal - 1):
                if i - (j - (dect_elem_count_horizontal - 1)) < 0 or i - (j - (dect_elem_count_horizontal - 1)) > 2 * dect_elem_count_horizontal - 2:
                    pass
                else:
                    reconKernel_conv_local = reconKernel_conv_local + array_kernel_gauss_taichi[j] * array_kernel_ramp_taichi[i - (j - (dect_elem_count_horizontal - 1))]
            array_recon_kernel_taichi[i]= reconKernel_conv_local * dect_elem_width  
        
    @ti.kernel
    def BackProjectionPixelDriven(self, dect_elem_count_vertical_actual:ti.i32, img_dim:ti.i32, dect_elem_count_horizontal:ti.i32, \
                                  view_num:ti.i32, dect_elem_width:ti.f32,\
                                  img_pix_size:ti.f32, source_isocenter_dis:ti.f32, source_dect_dis:ti.f32,total_scan_angle:ti.f32,\
                                      array_angle_taichi:ti.template(),img_rot:ti.f32,img_sgm_filtered_taichi:ti.template(),img_recon_taichi:ti.template(),\
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
                elif recon_view_mode == 3: #sagittal view (from right to left)
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
                    if bool_apply_pmatrix == 0:
                        mag_factor = source_dect_dis / pix_to_source_parallel_dis
                        y_after_rotation_angle_this_view = - x*ti.sin(angle_this_view_exclude_img_rot) + y*ti.cos(angle_this_view_exclude_img_rot)
                        if curved_dect:
                            pix_proj_to_dect_u = source_dect_dis * ti.atan2(y_after_rotation_angle_this_view, pix_to_source_parallel_dis)
                        else:
                            pix_proj_to_dect_u = mag_factor * y_after_rotation_angle_this_view
                        pix_proj_to_dect_u_idx = (pix_proj_to_dect_u - array_u_taichi[0]) / (array_u_taichi[1] - array_u_taichi[0])
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
                            pix_proj_to_dect_v_idx = (array_pmatrix_taichi[12*j + 4] * x +\
                                array_pmatrix_taichi[12*j + 5] * y +\
                                    array_pmatrix_taichi[12*j + 6] * z +\
                                        array_pmatrix_taichi[12*j + 7] * 1) * mag_factor
                                
                        temp_v_idx_floor = int(ti.floor(pix_proj_to_dect_v_idx)) 
                        if temp_v_idx_floor < 0 or temp_v_idx_floor + 1 > dect_elem_count_vertical_actual - 1:
                            img_recon_taichi[i_z, i_y, i_x] = 0
                            break
                        else:
                            ratio_v = pix_proj_to_dect_v_idx - temp_v_idx_floor
                            part_0 = img_sgm_filtered_taichi[temp_v_idx_floor,j,temp_u_idx_floor] * (1 - ratio_u) + \
                                img_sgm_filtered_taichi[temp_v_idx_floor,j,temp_u_idx_floor + 1] * ratio_u
                            part_1 = img_sgm_filtered_taichi[temp_v_idx_floor + 1,j,temp_u_idx_floor] * (1 - ratio_u) +\
                                  img_sgm_filtered_taichi[temp_v_idx_floor + 1,j,temp_u_idx_floor + 1] * ratio_u
                                  
                            img_recon_taichi[i_z, i_y, i_x] += (source_isocenter_dis * distance_weight) * \
                                ((1 - ratio_v) * part_0 + ratio_v * part_1) * delta_angle * div_factor
                    else: 
                        val_0 = img_sgm_filtered_taichi[i_z , j , temp_u_idx_floor]
                        val_1 = img_sgm_filtered_taichi[i_z , j , temp_u_idx_floor + 1]
                        img_recon_taichi[i_z, i_y, i_x] += (source_isocenter_dis * distance_weight) * \
                            ((1 - ratio_u) * val_0 + ratio_u * val_1) * delta_angle * div_factor
    
    def ReadSinogram(self,file):
        self.input_path = os.path.join(self.input_dir, file)
        self.output_file = re.sub(self.output_file_replace[0], self.output_file_replace[1], file)
        if self.output_file == file:
            #did not find the string in file, so that output_file and file are the same
            print(f"ERROR: did not file string '{self.output_file_replace[0]}' to replace in '{self.output_file}'")
            sys.exit()
        else:
            print('\nLoading %s to RAM...' % self.input_path)
            if self.output_file_format == 'tif' or self.output_file_format == 'tiff':
                #to save to tif, '*.raw' need to be changed to '*.tif'
                self.output_file = re.sub('.raw', '.tif', self.output_file)
            self.output_path = os.path.join(self.output_dir, self.output_file_prefix + self.output_file)
            #对一些文件命名的处理都遵循的过去程序的命名规
            if self.input_file_form == 'sinogram':
                file_offset = self.dect_elem_vertical_recon_range_begin * 4 * self.sgm_height * self.dect_elem_count_horizontal
                # '4' is size of a float numer in bytes
                item_count = self.dect_elem_count_vertical_actual * self.sgm_height * self.dect_elem_count_horizontal
                temp_buffer = np.fromfile(self.input_path, dtype = np.float32, offset = file_offset, count = item_count)
                temp_buffer = temp_buffer.reshape(self.dect_elem_count_vertical_actual,self.sgm_height,self.dect_elem_count_horizontal)
            elif self.input_file_form == 'post_log_images':
                file_offset = self.dect_elem_vertical_recon_range_begin * self.dect_elem_count_horizontal * 4
                file_gap = (self.dect_elem_vertical_recon_range_begin + self.dect_elem_count_vertical - 1 - self.dect_elem_vertical_recon_range_end)\
                    * self.dect_elem_count_horizontal * 4
                temp_buffer = imreadRaw(self.input_path, height = self.dect_elem_count_vertical_actual, width = self.dect_elem_count_horizontal,\
                                         offset = file_offset, gap = file_gap, nSlice = self.sgm_height)
                #print(self.dect_elem_count_vertical_actual,self.dect_elem_count_horizontal, self.sgm_height)
                temp_buffer = np.transpose(temp_buffer,[1,0,2])
    
                
            self.img_sgm = temp_buffer[:,0:self.view_num,:]
            self.img_sgm = np.ascontiguousarray(self.img_sgm) #only contiguous arrays can be passed to ti kernel functions
            
            del temp_buffer
            #no longer need this: self.img_sgm_taichi.from_numpy(self.img_sgm)
            #since img_sgm can be directly passed to ti kernel functions using unified memory
            return True
    
    def InitializeSinogramBuffer(self):
        self.img_sgm = np.zeros((self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal),dtype = np.float32)
              
        ######### initialize taichi components ########
        #img_sgm_filtered_taichi存储卷积后的正弦
        self.img_sgm_filtered_taichi = ti.field(dtype=ti.f32, shape=(self.dect_elem_count_vertical_actual,self.view_num, self.dect_elem_count_horizontal))
        #img_sgm_filtered_taichi 纵向卷积后正弦图的中间结果，then apply horizontal convolution
        
        if self.apply_gauss_vertical:
            self.img_sgm_filtered_intermediate_taichi = ti.field(dtype=ti.f32, shape=(self.dect_elem_count_vertical_actual,self.view_num, self.dect_elem_count_horizontal))
        else:
            self.img_sgm_filtered_intermediate_taichi = ti.field(dtype=ti.f32, shape=(1,1,1))
            #if vertical gauss filter is not applied, initialize this intermediate sgm with a small size to save GPU memory
    
    def InitializeArrays(self):
        #calculate u array; by default, +u is along -y direction
        #by convention, detector_offset_horizontal is along -y direction, we need to multiply -1. 
        self.GenerateDectPixPosArray(self.dect_elem_count_horizontal,  self.dect_elem_count_horizontal,\
                                     (self.positive_u_is_positive_y) *self.dect_elem_width, (-1) * self.dect_offset_horizontal,self.array_u_taichi, 0)
        #计算数组v
        if self.cone_beam == True:
            self.GenerateDectPixPosArray(self.dect_elem_count_vertical,self.dect_elem_count_vertical_actual,\
                                         (self.positive_v_is_positive_z) *self.dect_elem_height, self.dect_offset_vertical, self.array_v_taichi,
                                         self.dect_elem_vertical_recon_range_begin)
        else:
            self.GenerateDectPixPosArray(self.dect_elem_count_vertical,self.dect_elem_count_vertical_actual,\
                                         0,0,self.array_v_taichi,0)
        #计算angle数组    
        self.GenerateAngleArray(self.view_num,self.img_rot,self.total_scan_angle,self.array_angle_taichi) 
        #img_rot is added to array_angle_taichi 
    
    def InitializeReconKernel(self):
        if 'HammingFilter' in self.config_dict:
            self.GenerateHammingKernel(self.dect_elem_count_horizontal,self.dect_elem_width,\
                                       self.kernel_param,self.source_dect_dis,self.source_isocenter_dis,self.array_recon_kernel_taichi,self.curved_dect, self.dbt_or_not)
            #计算hamming核存储在array_recon_kernel_taichi
            
        elif 'GaussianApodizedRamp' in self.config_dict:
            self.GenerateGassianKernel(self.dect_elem_count_horizontal,self.dect_elem_width,\
                                       self.kernel_param,self.array_kernel_gauss_taichi)
            #计算高斯核存储在array_kernel_gauss_taichi
            self.GenerateHammingKernel(self.dect_elem_count_horizontal,self.dect_elem_width,1,\
                                       self.source_dect_dis,self.source_isocenter_dis,self.array_kernel_ramp_taichi,self.curved_dect, self.dbt_or_not)
            #1.以hamming参数1调用一次hamming核处理运算结果存储在array_kernel_ramp_taichi
            self.ConvolveKernelAndKernel(self.dect_elem_count_horizontal,self.dect_elem_width,\
                                         self.array_kernel_ramp_taichi,self.array_kernel_gauss_taichi,self.array_recon_kernel_taichi)
            #2.将计算出来的高斯核array_kernel_gauss_taichi与以hamming参数1计算出来的hamming核array_kernel_ramp_taichi进行一次运算得到新的高斯核存储在array_recon_kernel_taichi
        
        self.GenerateGassianKernel(self.dect_elem_count_vertical_actual,self.dect_elem_height,\
                                       self.dect_elem_vertical_gauss_filter_size,self.array_kernel_gauss_vertical_taichi)

            
    def FilterSinogram(self):
        if self.kernel_name == 'None':
            self.img_sgm_filtered_taichi.from_numpy(self.img_sgm)
            #non filtration is performed
        else:
            self.ConvolveSgmAndKernel(self.dect_elem_count_vertical_actual,self.view_num,self.dect_elem_count_horizontal,\
                                      self.dect_elem_width,self.img_sgm,self.array_recon_kernel_taichi,\
                                          self.array_kernel_gauss_vertical_taichi,self.dect_elem_height, self.apply_gauss_vertical,
                                          self.img_sgm_filtered_intermediate_taichi, self.img_sgm_filtered_taichi)
            #pass img_sgm directly into this function using unified memory
            #用hamming核计算出的array_recon_kernel_taichi计算卷积后的正弦图img_sgm_filtered_taichi
        
    def SaveFilteredSinogram(self):
        if self.save_filtered_sinogram:
            sgm_filtered = self.img_sgm_filtered_taichi.to_numpy()
            sgm_filtered = sgm_filtered.astype(np.float32)
            imwriteRaw(sgm_filtered,self.output_dir+'/'+ self.output_file +'_sgm_filtered.raw',dtype=np.float32)
            del sgm_filtered

            
    def SaveReconImg(self):
        self.img_recon = self.img_recon_taichi.to_numpy()
        if self.convert_to_HU:
            self.img_recon = (self.img_recon / self.water_mu - 1)*1000
        if self.output_file_format == 'raw':
            imwriteRaw(self.img_recon,self.output_path,dtype=np.float32)
        elif self.output_file_format == 'tif' or self.output_file_format == 'tiff':
            imwriteTiff(self.img_recon, self.output_path,dtype=np.float32)
        self.img_recon_taichi.from_numpy(np.zeros_like(self.img_recon))
    
    
def remove_comments(jsonc_str):
    # 使用正则表达式去除注
    pattern = re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL)
    return re.sub(pattern, '', jsonc_str)

def save_jsonc(save_path,data):
    assert save_path.split('.')[-1] == 'jsonc'
    with open(save_path,'w') as file:
        json.dump(data,file)

def load_jsonc(file_path):
    #读取jsonc文件并以字典的形式返回所有数
    with open(file_path, 'r') as file:
        jsonc_content = file.read()
        json_content = remove_comments(jsonc_content)
        data = json.loads(json_content)
    return data

def imreadRaw(path: str, height: int, width: int, dtype = np.float32, nSlice: int = 1, offset: int = 0, gap: int = 0):
    with open(path, 'rb') as fp:
        fp.seek(offset)
        if gap == 0:
            arr = np.frombuffer(fp.read(), dtype = dtype,count = nSlice * height * width).reshape((nSlice, height, width))
        else:
            imageBytes = height * width * np.dtype(dtype).itemsize
            arr = np.zeros((nSlice, height, width), dtype=dtype)
            for i in range(nSlice):
                arr[i, ...] = np.frombuffer(fp.read(imageBytes), dtype=dtype).reshape((height, width)).squeeze()
                fp.seek(gap, os.SEEK_CUR)
    return arr

def ReadConfigFile(file_path):
    # 替换为你的JSONC文件路径
    json_data = load_jsonc(file_path)
    # 现在，json_data包含了从JSONC文件中解析出的数
    # print(json_data)
    return json_data

#get the least squared solution of Bx=y
#basic principle is pseudo-inverse: x = (B^T*B)^{-1}*B^T*y
def XuLeastSquareSolution(matrix_B,vec_y):
    temp = np.matmul(np.linalg.inv(np.matmul(matrix_B.transpose(), matrix_B)), matrix_B.transpose())
    return np.matmul(temp,vec_y)
    

                

