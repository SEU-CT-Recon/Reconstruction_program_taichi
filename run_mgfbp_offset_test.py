# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:34:06 2024

@author: xji
"""

import taichi as ti
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from crip.io import imwriteRaw
from run_mgfbp import *
import gc


def run_mgfbp_offset_test(file_path):
    ti.reset()
    ti.init(arch=ti.gpu)
    print('Performing FBP from MandoCT-Taichi to select optimal offset value ...')
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
    fbp = Mgfbp_offset_test(config_dict) #将config_dict数据以字典的形式送入对象中
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

#inherit a class from Mgfbp
@ti.data_oriented
class Mgfbp_offset_test(Mgfbp):
    def __init__(self,config_dict):
        super(Mgfbp_offset_test,self).__init__(config_dict)
        self.array_dect_offset_horizontal = self.dect_offset_horizontal
        self.offset_num = len(self.array_dect_offset_horizontal)
        self.dect_offset_horizontal = 0
        self.img_recon_combine = np.zeros((self.img_dim_z*self.offset_num,self.img_dim,self.img_dim),dtype = np.float32)
    
    def MainFunction(self):
        #Main function for reconstruction
        self.InitializeSinogramBuffer()
        self.InitializeReconKernel()    
        self.file_processed_count = 0;
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):
                if self.ReadSinogram(file):
                    self.file_processed_count +=1 
                    print('\nReconstructing %s ...' % self.input_path)
                    for offset_idx in range(self.offset_num):
                        if offset_idx == 0:
                            self.dect_offset_horizontal = self.array_dect_offset_horizontal[offset_idx]
                            self.InitializeArrays()  
                            self.WeightSgm(self.dect_elem_count_vertical_actual,self.short_scan,self.curved_dect,\
                                           self.total_scan_angle,self.view_num,self.dect_elem_count_horizontal,\
                                               self.source_dect_dis,self.img_sgm,\
                                                   self.array_u_taichi,self.array_v_taichi,self.array_angle_taichi)
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
                        else:
                            self.dect_offset_horizontal = self.array_dect_offset_horizontal[offset_idx]
                            self.InitializeArrays()  
                            self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                            self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                            self.array_angle_taichi, self.img_rot,self.img_sgm_filtered_taichi,self.img_recon_taichi,\
                                            self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                                self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                    self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                        self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
                            
                        self.img_recon_combine[offset_idx*self.img_dim_z: (offset_idx+1)*self.img_dim_z ,:,:] = self.img_recon_taichi.to_numpy()
    
                    print('Saving to %s !' % self.output_path)
                    self.SaveReconImg()
        return self.img_recon_combine #函数返回重建图
    
    def SaveReconImg(self):
        if self.convert_to_HU:
            self.img_recon_combine = (self.img_recon_combine / self.water_mu - 1)*1000
        if self.output_file_format == 'raw':
            imwriteRaw(self.img_recon_combine,self.output_path,dtype=np.float32)
        elif self.output_file_format == 'tif' or self.output_file_format == 'tiff':
            imwriteTiff(self.img_recon_combine, self.output_path,dtype=np.float32)