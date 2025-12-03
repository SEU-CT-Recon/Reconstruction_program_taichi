# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:19:43 2024

@author: xji
"""

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

def run_mgfbp_helical(file_path):
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
    fbp = Mgfbp_helical(config_dict) #将config_dict数据以字典的形式送入对象中
    img_recon = fbp.MainFunction()
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
class Mgfbp_helical(Mgfbp):
    def __init__(self,config_dict):
        super(Mgfbp_helical,self).__init__(config_dict)
        if 'HelicalPitch' in config_dict:
            self.helical_pitch = config_dict['HelicalPitch'] 
        else: 
            print("ERROR: Can not find helical pitch in the config file!")
            sys.exit()
        self.mag_ratio_isocenter = self.source_det_dis / self.source_isocenter_dis
        self.isocenter_coverage = self.det_elem_count_vertical_actual * self.det_elem_height / self.mag_ratio_isocenter
        self.dis_per_round = self.helical_pitch * self.isocenter_coverage
        self.num_rounds = abs(self.total_scan_angle) / (2 * PI)
        self.angle_per_view = self.total_scan_angle / (self.view_num)
        self.dis_per_view = self.dis_per_round / (self.view_num / self.num_rounds)
        self.gamma_max = 2 * np.arctan(0.5 * self.det_elem_count_horizontal * self.det_elem_width / self.source_det_dis) / PI
        self.short_scan_range = 2 * PI / abs(self.helical_pitch)
        
        if self.short_scan_range < PI + self.gamma_max:
            self.short_scan_range = PI + self.gamma_max
            
        self.det_offset_vertical_at_isocenter = self.det_offset_vertical/self.mag_ratio_isocenter
        z_begin = self.isocenter_coverage / 2.0 * np.sign(self.helical_pitch) + self.det_offset_vertical_at_isocenter
        z_end = self.dis_per_view * self.view_num - self.isocenter_coverage / 2.0 * np.sign(self.helical_pitch) + self.det_offset_vertical_at_isocenter
        
        self.array_z_pos = np.arange(z_begin,z_end,self.img_voxel_height * np.sign(self.helical_pitch), dtype = np.float32)
        print("Reconstruction Z coverage: %.1f mm to %.1f mm" %(self.array_z_pos[0],self.array_z_pos[-1]))
        self.view_count = int(np.ceil( self.short_scan_range / abs(self.angle_per_view))) #view_count for helical scan
        self.short_scan_range_actual = self.view_count * abs(self.angle_per_view)
        print("Size of rebinned sinogram is %d x %d (%.1f degrees scan)" %(self.det_elem_count_horizontal,\
                                                                           self.view_count, np.rad2deg(self.short_scan_range_actual)))
        print("There are a total of %d slices to be reconstructed" %len(self.array_z_pos))
        
        self.array_z_pos_taichi = ti.field(dtype=ti.f32, shape=len(self.array_z_pos))
        self.array_z_pos_taichi.from_numpy(self.array_z_pos)

        
        #reinitialize buffer for reconstructed image
        self.img_recon = np.zeros((len(self.array_z_pos),self.img_dim,self.img_dim),dtype = np.float32)
        self.img_rot_add = 0.0 #initialize added image rotation value
        
        
    
    def MainFunction(self):
        self.InitializeSinogramBuffer()
        self.file_processed_count = 0
        if not os.path.exists(self.input_dir + '/temp'):
            os.mkdir(self.input_dir + '/temp')
            
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):
                if self.ReadSinogram(file): # 读取正弦图
                    self.ChangeReconParameterValues()
                    self.file_processed_count +=1 
                    print('\nReconstructing %s ...' % self.input_path)
                    print('Initializing recon kernel ... ')
                    self.InitializeReconKernel()
                    print('Initializing arrays ... ')
                    self.InitializeArrays()

                    for z_idx in range(len(self.array_z_pos)): 
                        str = 'Reconstructing slice: %4d/%4d' %(z_idx+1, len(self.array_z_pos))
                        print('\r' + str, end = '')
                        self.CalculateAddedImgRotation(z_idx)#calculate the added rotation angle for each slice
                        self.GenerateAngleArray(self.view_num,self.img_rot + self.img_rot_add,self.total_scan_angle,self.array_angle_taichi)
                        #array_angle_taichi need to include the image rotation; reinitialize it
                        self.SinogramRebinning(self.view_num_original, self.det_offset_vertical, self.det_elem_count_horizontal,\
                                               self.det_elem_height, self.det_elem_count_vertical_original, self.source_det_dis,\
                                               self.mag_ratio_isocenter, self.dis_per_view,self.view_count, self.angle_per_view,\
                                               self.img_sgm_taichi_original,self.array_z_pos_taichi, self.img_sgm, \
                                                   z_idx)
                        
                        
                        imaddRaw(self.img_sgm,self.input_dir + '/temp/sgm_rebin.raw', idx = z_idx)

                        self.img_sgm = np.ascontiguousarray(self.img_sgm)
                        if self.bool_bh_correction:
                            self.BHCorrection(self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal,self.img_sgm,\
                                              self.array_bh_coefficients_taichi,self.bh_corr_order)
                        self.img_sgm = np.ascontiguousarray(self.img_sgm)  
                        self.WeightSgm(self.det_elem_count_vertical_actual,self.short_scan,self.curved_dect,\
                                    self.total_scan_angle,self.view_num,self.det_elem_count_horizontal,\
                                        self.source_det_dis,self.img_sgm,\
                                            self.array_u_taichi,self.array_v_taichi,self.array_angle_taichi)

                        self.FilterSinogram()
                        self.SaveFilteredSinogram()
                        self.img_sgm = np.ascontiguousarray(self.img_sgm)  
                        self.BackProjectionPixelDriven(self.det_elem_count_vertical_actual, self.img_dim, self.det_elem_count_horizontal, \
                                        self.view_num, self.det_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_det_dis,self.total_scan_angle,\
                                        self.array_angle_taichi, self.img_rot + self.img_rot_add,self.img_sgm_filtered_taichi,self.img_recon_taichi,\
                                        self.array_u_taichi,self.short_scan,self.cone_beam,self.det_elem_height,\
                                            self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                    self.bool_apply_pmatrix,self.array_pmatrix_taichi,self.recon_view_mode)
                            
                        self.img_recon[z_idx,:,:] = self.img_recon_taichi.to_numpy()
    
                    print('\nSaving to %s !' % self.output_path)
                    self.SaveReconImg()
                    self.ChangeReconParameterValuesBack()
        return self.img_recon
    
    def CalculateAddedImgRotation(self,z_idx):
        z_pos = self.array_z_pos[z_idx]
        view_idx_v_equal_0 = (z_pos - self.det_offset_vertical_at_isocenter) / self.dis_per_view
        self.img_rot_add = np.floor(view_idx_v_equal_0 - 0.5 * (self.view_count - 1.0 )) * self.angle_per_view
        
    @ti.kernel
    def SinogramRebinning(self, view_num:ti.i32, det_offset_vertical:ti.f32,\
              det_elem_count_horizontal:ti.i32, det_elem_height:ti.f32,\
              det_elem_count_vertical:ti.i32, source_det_dis:ti.f32,\
              mag_ratio_isocenter:ti.f32, dis_per_view:ti.f32,view_count:ti.i32, angle_per_view:ti.f32,\
              img_sgm_taichi_original:ti.template(), array_z_pos_taichi:ti.template(), img_sgm_rebin_taichi:ti.types.ndarray(dtype=ti.f32, ndim=3), z_idx: ti.i32):

        for i, j in ti.ndrange(view_count, det_elem_count_horizontal):

            z_pos = array_z_pos_taichi[z_idx]
            view_idx_v_equal_0 = (z_pos - det_offset_vertical / mag_ratio_isocenter) / dis_per_view

            view_begin = ti.floor(view_idx_v_equal_0 - 0.5 * view_count + 0.5)
            
            # 注意这里的view_index均为浮点数
            view_idx = view_begin + i
            
            if 0 <= view_idx <= view_num - 2:
                v_pos = (z_pos - (view_idx) * dis_per_view) * mag_ratio_isocenter - det_offset_vertical 
                v_idx = - v_pos / det_elem_height + (det_elem_count_vertical - 1.0) / 2.0
                if v_idx >= det_elem_count_vertical - 1:                    
                    img_sgm_rebin_taichi[0, i, j] = img_sgm_taichi_original[det_elem_count_vertical - 1, int(view_idx), j] 
                elif v_idx <= 0:                       
                    img_sgm_rebin_taichi[0, i, j] = img_sgm_taichi_original[0, int(view_idx), j] 
                else:                
                    w = v_idx - ti.floor(v_idx)
                    data_1 = img_sgm_taichi_original[int(ti.floor(v_idx)), int(view_idx), j] 
                        
                    data_2 = img_sgm_taichi_original[int(ti.floor(v_idx)) + 1, int(view_idx), j] 
                    img_sgm_rebin_taichi[0, i, j] = data_1 * (1 - w) + data_2 * w

                img_sgm_rebin_taichi[0, i, j] *= source_det_dis / ti.sqrt(source_det_dis ** 2 + v_pos ** 2)
            else:
                img_sgm_rebin_taichi[0, i, j] = 0.0
                
    def ChangeReconParameterValues(self):
        self.view_num_original = self.view_num #record the original value
        self.det_elem_count_vertical_original = self.det_elem_count_vertical #record the original value
        self.det_elem_count_vertical_actual_original = self.det_elem_count_vertical_actual
        self.total_scan_angle_original = self.total_scan_angle #record the original value
        
        self.img_sgm_taichi_original = ti.field(dtype=ti.f32, shape=(self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal))
        #here we did not use np.zeros and unified memory; because this may slow down the compuational speed
        self.img_sgm_taichi_original.from_numpy(self.img_sgm) #record the original value input sinogram
        #ti.template has the drawback of using more gpu memory
        
        self.det_elem_count_vertical = 1 #we are processing each sinogram slice, so this number is 1
        self.det_elem_count_vertical_actual = 1
        self.cone_beam = False
        #recalculate total scan angle
        self.total_scan_angle = self.short_scan_range_actual * np.sign(self.total_scan_angle)
        # re-judge whether the scan is a total scan
        if abs(self.total_scan_angle % (2*PI)) < (0.01 / 180 * PI):
            self.short_scan = 0
            print('--Rebinned sinogram is full scan, scan Angle = %.1f degrees' % (self.total_scan_angle / PI * 180))
        else:
            self.short_scan = 1
            print('--Rebinned sinogram is short scan, scan Angle = %.1f degrees' % (self.total_scan_angle / PI * 180))
        self.sgm_height = self.view_count
        self.view_num = self.view_count
        self.img_dim_z = 1
        self.img_sgm = np.zeros(dtype = np.float32, shape= (1,self.view_num, self.det_elem_count_horizontal) )
        self.img_sgm = np.ascontiguousarray(self.img_sgm) #only contiguous arrays can be passed to ti kernel functions
        self.img_sgm_filtered_taichi = ti.field(dtype=ti.f32, shape=(1,self.view_num, self.det_elem_count_horizontal))
        self.img_sgm_filtered_intermediate_taichi = ti.field(dtype=ti.f32, shape=(1,self.view_num, self.det_elem_count_horizontal))
        
        
        self.img_recon_taichi = ti.field(dtype=ti.f32, shape=(1,self.img_dim, self.img_dim),order='ikj')
        
        
        self.array_angle_taichi = ti.field(dtype=ti.f32, shape=self.view_num)
        self.array_recon_kernel_taichi = ti.field(dtype=ti.f32, shape=2*self.det_elem_count_horizontal-1)
        self.array_u_taichi = ti.field(dtype = ti.f32,shape = self.det_elem_count_horizontal)
        self.array_v_taichi = ti.field(dtype = ti.f32,shape = self.det_elem_count_vertical_actual)
        
    def SaveReconImg(self):
        if self.convert_to_HU:
            self.img_recon = (self.img_recon / self.water_mu - 1)*1000
        if self.output_file_format == 'raw':
            imwriteRaw(self.img_recon,self.output_path,dtype=np.float32)
        elif self.output_file_format == 'tif' or self.output_file_format == 'tiff':
            imwriteTiff(self.img_recon, self.output_path,dtype=np.float32)
    
    #change values of the parameters back to process next input file
    def ChangeReconParameterValuesBack(self):
        self.view_num = self.view_num_original #record the original value
        self.det_elem_count_vertical = self.det_elem_count_vertical_original #record the original value
        self.det_elem_count_vertical_actual = self.det_elem_count_vertical_actual_original #record the original value
        self.total_scan_angle = self.total_scan_angle_original #record the original value
        self.img_sgm = np.zeros(dtype = np.float32, shape=(self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal))
        self.img_sgm = np.ascontiguousarray(self.img_sgm) #only contiguous arrays can be passed to ti kernel functions
        self.sgm_height = self.view_num
        
     
