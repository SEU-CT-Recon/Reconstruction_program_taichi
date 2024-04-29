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
#所有函数以单词首字母大写命名，不加连字符；例如GenerateHammingKernel
#所有变量以小写单词加连字符分隔命名，例如dect_elem_width
#变量命名意义要清晰，不要是单个字母，会看不懂
#所有一维数组加上array_前缀，如果是taichi类型，就以taichi结尾，例如array_u_taichi
#所有二维及以上维数组加上img_前缀，如果是taichi类型，就以taichi结尾，例如img_sgm_taichi
#ToDo: 目前GenerateHammingKernel等函数的形式参数还没有跟self里的变量名统一，你给改下
#这个不影响结果，但是要改下，保证可读性
#然后加上cone beam 的功能，新的参数命名按照上述规则

PI = 3.1415926536

def run_mgfbp(file_path):
    ti.reset()
    ti.init(arch=ti.gpu)
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
    config_dict = ReadConfigFile(file_path)#读入jsonc文件并以字典的形式存储在config_dict中
    fbp = Mgfbp(config_dict) #将config_dict数据以字典的形式送入对象中
    # Ensure output directory exists; if not, create the directory
    if not os.path.exists(fbp.output_dir):
        os.makedirs(fbp.output_dir)
    img_recon = fbp.MainFunction()
    end_time = time.time()# record end time point
    execution_time = end_time - start_time# 计算执行时间
    if fbp.file_processed_count > 0:
        print(f"\nA total of {fbp.file_processed_count:d} file(s) are reconstructed!")
        print(f"Time cost：{execution_time:.3} sec\n")# 打印执行时间（以秒为单位）
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
        self.InitializeArrays()  
        self.InitializeReconKernel()    
        self.file_processed_count = 0;
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):
                if self.ReadSinogram(file):
                    self.file_processed_count +=1 
                    print('\nReconstructing %s ...' % self.input_path)
                    self.WeightSgm(self.dect_elem_count_vertical_actual,self.short_scan,self.curved_dect,\
                                   self.total_scan_angle,self.view_num,self.dect_elem_count_horizontal,\
                                       self.source_dect_dis,self.img_sgm_taichi,\
                                           self.array_u_taichi,self.array_v_taichi,self.array_angel_taichi)
                    print('Filtering sinogram ...')
                    self.FilterSinogram()
                    self.SaveFilteredSinogram()
                    
                    
                    print('Back Projection ...')
                    self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                    self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                    self.array_angel_taichi, self.img_rot,self.img_sgm_filtered_taichi,self.img_recon_taichi,\
                                    self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                        self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                            self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                self.bool_apply_pmatrix,self.array_pmatrix_taichi)
    
                    print('Saving to %s !' % self.output_path)
                    self.SaveReconImg()
        return self.img_recon #函数返回重建图
    
    def __init__(self,config_dict):
        self.config_dict = config_dict
        ######## parameters related to input and output filenames ########
        self.input_dir = config_dict['InputDir']
        self.output_dir = config_dict['OutputDir']
        self.input_files_pattern = config_dict['InputFiles']
        self.output_file_prefix = config_dict['OutputFilePrefix']
        self.output_file_replace = config_dict['OutputFileReplace']
        
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
        
        #NEW! define whether the first slice of the sinogram corresponds to the 
        #bottom detector row or the top row
        if self.input_file_form == 'post_log_images':
            self.first_slice_top_row = True 
            # if the input file form are in post_log_imgs
            # the images are NOT up-side-down
            # first sinogram slice corresponds to top row of the detector
        elif 'FirstSinogramSliceIsDetectorTopRow' in config_dict:
            self.first_slice_top_row = config_dict['FirstSinogramSliceIsDetectorTopRow']
        else:
            self.first_slice_top_row = False # by default, first sgm slice is detector bottom row
        if self.first_slice_top_row:
            print('--First sinogram slice corresponds to top detector row')
        
        if 'SaveFilteredSinogram' in config_dict:
            self.save_filtered_sinogram = config_dict['SaveFilteredSinogram']
            if self.save_filtered_sinogram:
                print("--Filtered sinogram is saved")
        else:
            self.save_filtered_sinogram = False
        
        ######## parameters related to detector (fan beam case) ########
        #detector type (flat panel or curved)
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
        else:
            print("ERROR: Can not find detector element count along horizontal direction!")
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
            print("Warning: Can not find horizontal detector offset; Using default value 0")
            
        if 'DetectorElementCountVertical' in config_dict:
            self.dect_elem_count_vertical = config_dict['DetectorElementCountVertical']
        elif 'SliceCount' in config_dict:
            self.dect_elem_count_vertical = config_dict['SliceCount']
        else:
            print("ERROR: Can not find detector element count along vertical direction!")
            sys.exit()
        
        #NEW! using partial slices of sinogram for reconstruction  
        if 'DetectorElementVerticalReconRange' in config_dict:
            temp_array = config_dict['DetectorElementVerticalReconRange']
            self.dect_elem_vertical_recon_range_begin = temp_array[0]
            self.dect_elem_vertical_recon_range_end = temp_array[1]
            if self.dect_elem_vertical_recon_range_end > self.dect_elem_count_vertical-1 or \
                self.dect_elem_vertical_recon_range_begin <0:
                print('ERROR: Out of detector row range!')
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
            
        if 'SinogramHeight' in config_dict:
            self.sgm_height = config_dict['SinogramHeight']
        else:
            self.sgm_height = self.view_num
        
        if 'TotalScanAngle' in config_dict:
            self.total_scan_angle = config_dict['TotalScanAngle'] / 180.0 * PI
            # TotalScanAngle is originally in degree; change it to rad
            # all angular variables are in rad unit
        else:
            self.total_scan_angle = 2*PI #by default, scan angle is 2*pi
       
        if abs(self.total_scan_angle % PI) < (0.01 / 180 * PI):
            self.short_scan = 0
            print('--Full scan, scan Angle = %.1f degrees' % (self.total_scan_angle / PI * 180))
        else:
            self.short_scan = 1
            print('--Short scan, scan Angle = %.1f degrees' % (self.total_scan_angle / PI * 180))
            
        ######### projection matrix recon parameters ########
        self.array_pmatrix_taichi = ti.field(dtype=ti.f32, shape=self.view_num * 12)
        if 'PMatrixFile' in config_dict:
            temp_dict = ReadConfigFile(config_dict['PMatrixFile'])
            if 'Value' in temp_dict:
                self.array_pmatrix = np.array(temp_dict['Value'],dtype = np.float32)
                if len(self.array_pmatrix) != self.view_num * 12:
                    print(f'ERROR: view number is {self.view_num:d} while pmatrix has {len(self.array_pmatrix):d} elements!')
                    sys.exit()
                self.array_pmatrix_taichi.from_numpy(self.array_pmatrix)
                self.bool_apply_pmatrix = 1
                print("--Pmatrix applied")
            else:
                print(f"ERROR: PMatrixFile has no member named 'Value'!")
                sys.exit()
        else:
            self.bool_apply_pmatrix = 0
        
        
        ######## CT scan geometries ########        
        self.source_isocenter_dis = config_dict['SourceIsocenterDistance']
        self.source_dect_dis = config_dict['SourceDetectorDistance']
        
        ######## Reconstruction image size (in-plane) ########
        self.img_dim = config_dict['ImageDimension']
        self.img_pix_size = config_dict['PixelSize']
        self.img_rot = config_dict['ImageRotation'] / 180.0 * PI
        # ImageRotation is originally in degree; change it to rad
        # all angular variables are in rad unit
        self.img_center = config_dict['ImageCenter']
        self.img_center_x = self.img_center[0]
        self.img_center_y = self.img_center[1]
        
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
        
        ######## whether images are converted to HU ########
        if 'WaterMu' in config_dict: 
            self.water_mu = config_dict['WaterMu']
            self.convert_to_HU = True
            print("--Converted to HU")
        else:
            self.convert_to_HU = False
            
        ######## cone beam reconstruction parameters ########
        if 'ConeBeam' in config_dict:
            self.cone_beam = config_dict['ConeBeam']
        else:
            self.cone_beam = False
        
        if self.cone_beam:
            print("--Cone beam recon")
                        
            #detector element height
            if 'SliceThickness' in config_dict:
                self.dect_elem_height = config_dict['SliceThickness']
            elif 'DetectorElementHeight' in config_dict:
                self.dect_elem_height = config_dict['DetectorElementHeight']
            else:
                print("ERROR: Can not find detector element height for cone beam recon! ")
                sys.exit()
                
            #detector offset vertical
            if 'SliceOffCenter' in config_dict:
                self.dect_offset_vertical = config_dict['SliceOffCenter'] 
            elif 'DetectorOffsetVertical' in config_dict:
                self.dect_offset_vertical = config_dict['DetectorOffsetVertical']
            else: 
                self.dect_offset_vertical = 0
                print("Warning: Can not find vertical detector offset for cone beam recon; Using default value 0")
            
            #image dimension along z direction
            if 'ImageSliceCount' in config_dict:
                self.img_dim_z = config_dict['ImageSliceCount']
            elif 'ImageDimensionZ' in config_dict:
                self.img_dim_z = config_dict['ImageDimensionZ']
            else:
                print("ERROR: Can not find image dimension along Z direction for cone beam recon!")
                sys.exit() 
                
            #image voxel height
            if 'VoxelHeight' in config_dict:
                self.img_voxel_height = config_dict['VoxelHeight']
            elif 'ImageSliceThickness' in config_dict:
                self.img_voxel_height = config_dict['ImageSliceThickness']
            else:
                print("ERROR: Can not find image voxel height for cone beam recon!")
                sys.exit()
                
            #img center along z direction
            if 'ImageCenterZ' in config_dict:
                self.img_center_z = config_dict['ImageCenterZ']
            else:
                current_center_row_idx = (self.dect_elem_vertical_recon_range_end +  self.dect_elem_vertical_recon_range_begin)/2
                distance_to_original_detector_center_row = (current_center_row_idx - (self.dect_elem_count_vertical-1)/2) * self.dect_elem_height
                if self.first_slice_top_row:
                    distance_to_original_detector_center_row = distance_to_original_detector_center_row * (-1)
                self.img_center_z = (self.dect_offset_vertical + distance_to_original_detector_center_row)\
                    * self.source_isocenter_dis / self.source_dect_dis
                print("Warning: Did not find image center along z direction! Use default setting (central slice of the given detector recon row range)")
                print("Image center at Z direction is %.4f mm. " %self.img_center_z)
        else:
            print("--Fan beam recon")
            self.dect_elem_height = 0.0
            self.dect_offset_vertical = 0.0
            self.img_dim_z = self.dect_elem_count_vertical
            self.img_voxel_height = 0.0
            self.img_center_z = 0
            
        self.img_recon = np.zeros((self.img_dim_z,self.img_dim,self.img_dim),dtype = np.float32)
        self.img_sgm = np.zeros((self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal),dtype = np.float32)
              
        ######### initialize taichi components ########
        #img_sgm_filtered_taichi存储卷积后的正弦图
        self.img_sgm_filtered_taichi = ti.field(dtype=ti.f32, shape=(self.dect_elem_count_vertical_actual,self.view_num, self.dect_elem_count_horizontal))
        #img_sgm_filtered_taichi 纵向卷积后正弦图的中间结果，then apply horizontal convolution
        self.img_sgm_filtered_intermediate_taichi = ti.field(dtype=ti.f32, shape=(self.dect_elem_count_vertical_actual,self.view_num, self.dect_elem_count_horizontal))
        
        self.img_recon_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z,self.img_dim, self.img_dim),order='ikj')
        #img_recon_taichi is the reconstructed img
        self.array_angel_taichi = ti.field(dtype=ti.f32, shape=self.view_num)
        #angel_taichi存储旋转角度，且经过计算之后以弧度制表示
        self.img_sgm_taichi = ti.field(dtype=ti.f32, shape=(self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal))
        #存储读取的正弦图
        self.array_recon_kernel_taichi = ti.field(dtype=ti.f32, shape=2*self.dect_elem_count_horizontal-1)
        #存储用于对正弦图进行卷积的核
        self.array_u_taichi = ti.field(dtype=ti.f32,shape=self.dect_elem_count_horizontal)
        #存储数组u
        self.array_v_taichi = ti.field(dtype = ti.f32,shape = self.dect_elem_count_vertical_actual)
        
    
        
        
    @ti.kernel
    def GenerateHammingKernel(self,dect_elem_count_horizontal:ti.i32,dect_elem_width:ti.f32,kernel_param:ti.f32,\
                              source_dect_dis:ti.f32,array_recon_kernel_taichi:ti.template(),curved_dect:ti.i32):
        #计算hamming核分两步处理
        n = 0
        bias = dect_elem_count_horizontal - 1
        t = kernel_param
        for i in ti.ndrange(2 * dect_elem_count_horizontal - 1):  
            n = i - bias
            #part 1 ramp核
            if n == 0:
                array_recon_kernel_taichi[i] = t / (4 * dect_elem_width * dect_elem_width)
            elif n % 2 == 0:
                array_recon_kernel_taichi[i] = 0
            else:
                if curved_dect:
                    temp_val = float(n) * dect_elem_width / source_dect_dis
                    array_recon_kernel_taichi[i] = -t / (PI * PI * (source_dect_dis **2) * (temp_val - temp_val**3/3/2/1 + temp_val**5/5/4/3/2/1)**2 )
                    #use taylor expansion to replace the built-in taichi.sin function
                    #this function leads to 1% bias in calculation
                else:
                    array_recon_kernel_taichi[i] = -t / (PI * PI * (float(n) **2) * (dect_elem_width **2))
            #part 2 cosine核
            sgn = 1 if n % 2 == 0 else -1
            array_recon_kernel_taichi[i] += (1-t)*(sgn/(2 * PI * dect_elem_width * dect_elem_width)*(1/(1 + 2 * n)+ 1 / (1 - 2 * n))- 1 / (PI * PI * dect_elem_width * dect_elem_width) * (1 / (1 + 2 * n) / (1 + 2 * n) + 1 / (1 - 2 * n) / (1 - 2 * n)))

    @ti.kernel
    def GenerateGassianKernel(self,dect_elem_count_horizontal:ti.i32,dect_elem_width:ti.f32,kernel_param:ti.f32,array_kernel_gauss_taichi:ti.template()):
        #计算高斯核
        temp_sum = 0.0
        delta = kernel_param
        for i in ti.ndrange(2 * dect_elem_count_horizontal - 1):
            n = i - (dect_elem_count_horizontal - 1)
            array_kernel_gauss_taichi[i] = ti.exp(-n*n/2/delta/delta)
            temp_sum+=array_kernel_gauss_taichi[i]
        for i in ti.ndrange(2 * dect_elem_count_horizontal - 1):
            array_kernel_gauss_taichi[i] = array_kernel_gauss_taichi[i]/temp_sum / dect_elem_width


    @ti.kernel
    def GenerateAngleArray(self,view_num:ti.i32,img_rot:ti.f32,scan_angle:ti.f32,array_angel_taichi:ti.template()):
        #计算beta并用弧度制的形式表示
        for i in ti.ndrange(view_num):
            array_angel_taichi[i] = (scan_angle / view_num * i ) + img_rot
  

    @ti.kernel
    def GenerateDectPixPosArray(self,dect_elem_count_horizontal:ti.i32,dect_elem_count_horizontal_actual:ti.i32,dect_elem_width:ti.f32,\
                                dect_offset_horizontal:ti.f32,array_u_taichi:ti.template(),dect_elem_begin_idx:ti.i32, first_slice_top_row:ti.i32):
        #计算u数组，并加入偏移方便后续做处理
        flag = 0
        if first_slice_top_row:
            flag = -1 # if first sinogram is the top detector row, it corresponds to a positive v value
        else:
            flag = 1
        # dect_elem_begin_idx is for recon of partial slices of the sinogram
        # since the slice idx may not begin with 0
        for i in ti.ndrange(dect_elem_count_horizontal_actual):
            array_u_taichi[i] = flag * (i + dect_elem_begin_idx - (dect_elem_count_horizontal - 1) / 2.0) \
                * dect_elem_width + dect_offset_horizontal

    @ti.kernel
    def WeightSgm(self, dect_elem_count_vertical_actual:ti.i32, short_scan:ti.i32, curved_dect:ti.i32, scan_angle:ti.f32,\
                  view_num:ti.i32, dect_elem_count_horizontal:ti.i32, source_dect_dis:ti.f32,img_sgm_taichi:ti.template(),\
                      array_u_taichi:ti.template(),array_v_taichi:ti.template(),array_angel_taichi:ti.template()):
        #对正弦图做加权，包括fan beam的cos加权和短扫面加权
        for  i, j in ti.ndrange(view_num, dect_elem_count_horizontal):
            u_actual = array_u_taichi[j]
            for s in ti.ndrange(dect_elem_count_vertical_actual):
                v_actual = array_v_taichi[s]
                if curved_dect:
                    img_sgm_taichi[s,i,j] = img_sgm_taichi[s,i,j] * source_dect_dis * ti.math.cos(u_actual/source_dect_dis) \
                        * source_dect_dis / ((source_dect_dis**2 + v_actual**2)**0.5)
                else:
                    img_sgm_taichi[s,i,j]=(img_sgm_taichi[s,i,j] * source_dect_dis * source_dect_dis ) \
                        / (( source_dect_dis **2 + u_actual**2 + v_actual **2) ** 0.5)
                if short_scan:
                    #for scans longer than 360 degrees but not muliples of 360, we also need to apply parker weighting
                    #for example, for a 600 degrees scan, we also need to apply parker weighting
                    num_rounds = ti.floor(abs(scan_angle) / (PI * 2))
                    remain_angle = abs(scan_angle) - num_rounds * PI * 2
                    #angle remains: e.g., if totalScanAngle = 600 degree, remain_angle = 240 degree
                    beta = abs(array_angel_taichi[i] - array_angel_taichi[0])
                    rotation_direction =  abs(scan_angle) / (scan_angle)
                    gamma = 0.0
                    if curved_dect:
                        gamma = (u_actual / source_dect_dis) * rotation_direction
                    else:
                        gamma = ti.atan2(u_actual, source_dect_dis) * rotation_direction
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
                            weighting = 1
                        elif (PI * (2 * num_rounds + 1) - 2 * gamma) <= beta <= (PI * (2 * num_rounds + 1) + gamma_max):
                            weighting = ti.sin(PI / 2 * (PI + gamma_max - (beta - PI * 2 * num_rounds)) / (gamma_max + 2 * gamma))
                            weighting = weighting * weighting
                    else:
                        weighting = 1    
                    img_sgm_taichi[s,i,j] *= weighting
                
    @ti.kernel
    def ConvolveSgmAndKernel(self, dect_elem_count_vertical_actual:ti.i32, view_num:ti.i32, \
                             dect_elem_count_horizontal:ti.i32, dect_elem_width:ti.f32, img_sgm_taichi:ti.template(), \
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
                                      array_angel_taichi:ti.template(),img_rot:ti.f32,img_sgm_filtered_taichi:ti.template(),img_recon_taichi:ti.template(),\
                                          array_u_taichi:ti.template(), short_scan:ti.i32,cone_beam:ti.i32,dect_elem_height:ti.f32,\
                                              array_v_taichi:ti.template(),img_dim_z:ti.i32,img_voxel_height:ti.f32, \
                                                  img_center_x:ti.f32,img_center_y:ti.f32,img_center_z:ti.f32,curved_dect:ti.i32,\
                                                      bool_apply_pmatrix:ti.i32, array_pmatrix_taichi:ti.template()):
        
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
        #计算重建图并保存到img_recon_taichi中
        for i_x, i_y in ti.ndrange(img_dim, img_dim):
            for i_z in ti.ndrange(img_dim_z):
                img_recon_taichi[i_z, i_y, i_x] = 0.0
                x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                y_after_rot = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_y
                z = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_z
                x = + x_after_rot * ti.cos(img_rot) + y_after_rot * ti.sin(img_rot)
                y = - x_after_rot * ti.sin(img_rot) + y_after_rot * ti.cos(img_rot)
                for j in ti.ndrange(view_num):
                    #calculate angular interval for this view
                    delta_angle = 0.0
                    if j == view_num - 1:
                        delta_angle =  abs(array_angel_taichi[view_num-1] - array_angel_taichi[0]) / (view_num-1)
                    else:
                        delta_angle = abs(array_angel_taichi[j+1] - array_angel_taichi[j])
                    
                    pix_to_source_parallel_dis = 0.0
                    mag_factor = 0.0
                    temp_u_idx_floor = 0
                    pix_proj_to_dect_u = 0.0
                    pix_proj_to_dect_v = 0.0
                    pix_proj_to_dect_u_idx = 0.0
                    pix_proj_to_dect_v_idx = 0.0
                    ratio_u = 0.0
                    ratio_v = 0.0
                    pix_to_source_parallel_dis = source_isocenter_dis - x * ti.cos(array_angel_taichi[j]) - y * ti.sin(array_angel_taichi[j])
                    if self.bool_apply_pmatrix == 0:
                        mag_factor = source_dect_dis / pix_to_source_parallel_dis
                        if curved_dect:
                            pix_proj_to_dect_u = source_dect_dis * ti.atan2(x*ti.sin(array_angel_taichi[j])-y*ti.cos(array_angel_taichi[j]),pix_to_source_parallel_dis)
                        else:
                            pix_proj_to_dect_u = mag_factor * (x*ti.sin(array_angel_taichi[j])-y*ti.cos(array_angel_taichi[j]))
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
                                        
                    
                    distance_weight = 0.0
                    if curved_dect:
                        distance_weight = 1.0 / ((pix_to_source_parallel_dis * pix_to_source_parallel_dis) + (x * ti.sin(array_angel_taichi[j]) - y * ti.cos(array_angel_taichi[j])) \
                                                * (x * ti.sin(array_angel_taichi[j]) - y * ti.cos(array_angel_taichi[j])))
                    else:
                        distance_weight = 1.0 / (pix_to_source_parallel_dis * pix_to_source_parallel_dis)

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
            #did not file the string in file, so that output_file and file are the same
            print(f"ERROR: did not file string '{self.output_file_replace[0]}' to replace in '{self.output_file}'")
            sys.exit()
        else:
            if self.output_file_format == 'tif' or self.output_file_format == 'tiff':
                #to save to tif, '*.raw' need to be changed to '*.tif'
                self.output_file = re.sub('.raw', '.tif', self.output_file)
            self.output_path = os.path.join(self.output_dir, self.output_file_prefix + self.output_file)
            #对一些文件命名的处理都遵循的过去程序的命名规则
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
            del temp_buffer
            self.img_sgm_taichi.from_numpy(self.img_sgm)
            #将正弦图sgm存储到taichi专用的数组中帮助加速程序
            return True
    
    def InitializeArrays(self):
        #计算数组u
        self.GenerateDectPixPosArray(self.dect_elem_count_horizontal,self.dect_elem_count_horizontal,\
                                     self.dect_elem_width,self.dect_offset_horizontal,self.array_u_taichi,0,False)
        #计算数组v
        if self.cone_beam == True:
            self.GenerateDectPixPosArray(self.dect_elem_count_vertical,self.dect_elem_count_vertical_actual,\
                                         self.dect_elem_height,self.dect_offset_vertical,self.array_v_taichi,
                                         self.dect_elem_vertical_recon_range_begin,self.first_slice_top_row)
        else:
            self.GenerateDectPixPosArray(self.dect_elem_count_vertical,self.dect_elem_count_vertical_actual,\
                                         0,0,self.array_v_taichi,0,False)
        #计算angle数组    
        self.GenerateAngleArray(self.view_num,0,self.total_scan_angle,self.array_angel_taichi)
        #img_rot is not incorporated here;
        #image rotation is only performed in the backprojection process
    
    def InitializeReconKernel(self):
        if 'HammingFilter' in self.config_dict:
            self.GenerateHammingKernel(self.dect_elem_count_horizontal,self.dect_elem_width,\
                                       self.kernel_param,self.source_dect_dis,self.array_recon_kernel_taichi,self.curved_dect)
            #计算hamming核存储在array_recon_kernel_taichi中
            
        elif 'GaussianApodizedRamp' in self.config_dict:
            self.GenerateGassianKernel(self.dect_elem_count_horizontal,self.dect_elem_width,\
                                       self.kernel_param,self.array_kernel_gauss_taichi)
            #计算高斯核存储在array_kernel_gauss_taichi中
            self.GenerateHammingKernel(self.dect_elem_count_horizontal,self.dect_elem_width,1,\
                                       self.source_dect_dis,self.array_kernel_ramp_taichi,self.curved_dect)
            #1.以hamming参数为1调用一次hamming核处理运算结果存储在array_kernel_ramp_taichi
            self.ConvolveKernelAndKernel(self.dect_elem_count_horizontal,self.dect_elem_width,\
                                         self.array_kernel_ramp_taichi,self.array_kernel_gauss_taichi,self.array_recon_kernel_taichi)
            #2.将计算出来的高斯核array_kernel_gauss_taichi与以hamming参数为1计算出来的hamming核array_kernel_ramp_taichi进行一次运算得到新的高斯核存储在array_recon_kernel_taichi中
        
        self.GenerateGassianKernel(self.dect_elem_count_vertical_actual,self.dect_elem_height,\
                                       self.dect_elem_vertical_gauss_filter_size,self.array_kernel_gauss_vertical_taichi)

            
    def FilterSinogram(self):
        self.ConvolveSgmAndKernel(self.dect_elem_count_vertical_actual,self.view_num,self.dect_elem_count_horizontal,\
                                  self.dect_elem_width,self.img_sgm_taichi,self.array_recon_kernel_taichi,\
                                      self.array_kernel_gauss_vertical_taichi,self.dect_elem_height, self.apply_gauss_vertical,
                                      self.img_sgm_filtered_intermediate_taichi, self.img_sgm_filtered_taichi)
        #用hamming核计算出的array_recon_kernel_taichi计算卷积后的正弦图img_sgm_filtered_taichi
        
    def SaveFilteredSinogram(self):
        if self.save_filtered_sinogram:
            sgm_filtered = self.img_sgm_filtered_taichi.to_numpy()
            sgm_filtered = sgm_filtered.astype(np.float32)
            imwriteRaw(sgm_filtered,self.output_dir+'/'+ self.output_file +'sgm_filtered.raw',dtype=np.float32)
            del sgm_filtered

            
    def SaveReconImg(self):
        self.img_recon = self.img_recon_taichi.to_numpy()
        if self.convert_to_HU:
            self.img_recon = (self.img_recon / self.water_mu - 1)*1000
        if self.output_file_format == 'raw':
            imwriteRaw(self.img_recon,self.output_path,dtype=np.float32)
        elif self.output_file_format == 'tif' or self.output_file_format == 'tiff':
            imwriteTiff(self.img_recon, self.output_path,dtype=np.float32)
    
    
def remove_comments(jsonc_str):
    # 使用正则表达式去除注释
    pattern = re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL)
    return re.sub(pattern, '', jsonc_str)

def save_jsonc(save_path,data):
    assert save_path.split('.')[-1] == 'jsonc'
    with open(save_path,'w') as file:
        json.dump(data,file)

def load_jsonc(file_path):
    #读取jsonc文件并以字典的形式返回所有数据
    with open(file_path, 'r') as file:
        jsonc_content = file.read()
        json_content = remove_comments(jsonc_content)
        data = json.loads(json_content)
    return data

def imreadRaw(path: str, height: int, width: int, dtype = np.float32, nSlice: int = 1, offset: int = 0, gap: int = 0):
    with open(path, 'rb') as fp:
        fp.seek(offset)
        if gap == 0:
            arr = np.frombuffer(fp.read(), dtype = dtype,count = nSlice * height * width).reshape((nSlice, height, width)).squeeze()
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
    # 现在，json_data包含了从JSONC文件中解析出的数据
    # print(json_data)
    return json_data
    

                

