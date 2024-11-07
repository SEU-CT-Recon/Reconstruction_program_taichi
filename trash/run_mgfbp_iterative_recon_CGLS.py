# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:10:08 2024

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
from run_mgfpj_ver2 import *
import gc

def run_mgfbp_ir(file_path):
    ti.reset()
    ti.init(arch=ti.gpu)
    print('Performing Iterative Recon from MandoCT-Taichi (ver 0.1) ...')
    # record start time point
    
    #Delete unnecessary warinings
    warnings.filterwarnings('ignore', category=UserWarning, \
                            message='The value of the smallest subnormal for <class \'numpy.float(32|64)\'> type is zero.')
    
    if not os.path.exists(file_path):
        print(f"ERROR: Config File {file_path} does not exist!")
        #Judge whether the config jsonc file exist
        sys.exit()
    config_dict = ReadConfigFile(file_path)#读入jsonc文件并以字典的形式存储在config_dict中
    
    print("Generating seed image from FBP ...")
    fbp =  Mgfbp(config_dict) #将config_dict数据以字典的形式送入对象中
    img_recon_seed = fbp.MainFunction() #generate a seed from fbp reconstruction
    #img_recon_seed = np.zeros((fbp.img_dim_z,fbp.img_dim, fbp.img_dim),dtype=np.float32)
    start_time = time.time()
    print("\nPerform Iterative Recon ...")
    fbp = Mgfbp_ir(config_dict) #将config_dict数据以字典的形式送入对象中
    img_recon = fbp.MainFunction(img_recon_seed)#use the seed to initialize the iterative process
    gc.collect()# 手动触发垃圾回收
    ti.reset()# free gpu ram
    end_time = time.time()# record end time point
    execution_time = end_time - start_time# 计算执行时间
    if fbp.file_processed_count > 0:
        print(f"\nA total of {fbp.file_processed_count:d} file(s) are reconstructed!")
        print(f"Time cost：{execution_time:.3} sec ({execution_time/fbp.num_iter/fbp.file_processed_count:.3} sec per iteration). \n")
    else:
        print(f"\nWarning: Did not find files like {fbp.input_files_pattern:s} in {fbp.input_dir:s}.")
        print("No images are reconstructed!")
    del fbp #delete the fbp object
    return img_recon

# inherit a class from Mgfbp
@ti.data_oriented
class Mgfbp_ir(Mgfpj):
    def __init__(self,config_dict):
        super(Mgfbp_ir,self).__init__(config_dict)
        
        if 'NumberOfIterations' in config_dict:
            self.num_iter = config_dict['NumberOfIterations']
            if not isinstance(self.num_iter,int) or self.num_iter<0:
                print("ERROR: NumberOfIterations must be a positive integer!")
                sys.exit()
        else:
            self.num_iter = 100
            print("Warning: Did not find NumberOfIterations! Use default value 100. ")
        
        if 'HelicalPitch' in config_dict:
            self.helical_scan = True
            self.helical_pitch = config_dict['HelicalPitch']
        else:
            self.helical_scan = False
            self.helical_pitch = 0.0
        
        self.img_sgm_taichi = ti.field(dtype=ti.f32, shape=(
            self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal), order='ijk')
        
        self.img_recon = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_recon_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        self.img_x = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_x_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        
        self.img_d = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_d_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        
        self.img_bp_fp_x = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_fp_x_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        self.img_r = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_r_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        self.img_bp_b = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_b_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        
        self.img_fp_x = np.zeros((self.dect_elem_count_vertical, self.view_num,
                                self.dect_elem_count_horizontal), dtype=np.float32)
        self.img_fp_x_taichi = ti.field(dtype=ti.f32, shape=(
            self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal), order='ijk')
        
        self.img_fp_x_taichi_single_slice = ti.field(dtype=ti.f32, shape=(
            1, self.view_num, self.dect_elem_count_horizontal), order='ijk')
        
        self.img_fp_d = np.zeros((self.dect_elem_count_vertical, self.view_num,
                                self.dect_elem_count_horizontal), dtype=np.float32)
        self.img_fp_d_taichi = ti.field(dtype=ti.f32, shape=(
            self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal), order='ijk')
        self.img_fp_d_taichi_single_slice = ti.field(dtype=ti.f32, shape=(
            1, self.view_num, self.dect_elem_count_horizontal), order='ijk')
        
        self.img_bp_fp_d = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_fp_d_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
   
    def MainFunction(self,img_recon_seed):
        #Main function for reconstruction
        self.InitializeArrays()#initialize arrays
        self.file_processed_count = 0;#record the number of files processed
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):#match the file pattern
                if self.ReadSinogram(file):
                    self.file_processed_count += 1 
                    print('Reconstructing %s ...' % self.input_path)
                    if self.convert_to_HU:
                        img_recon_seed = (img_recon_seed/1000 + 1 ) * self.water_mu
                    self.img_x = img_recon_seed
                    self.img_x_taichi.from_numpy(self.img_x)
                    self.img_sgm_taichi.from_numpy(self.img_sgm)
                    
                    #P^T b
                    self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                    self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                    self.array_angle_taichi, self.img_rot,self.img_sgm_taichi,self.img_bp_b_taichi,\
                                    self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                        self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                            self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
                        
                    #P^T P x
                    for v_idx in range(self.dect_elem_count_vertical_actual): 
                        str = 'fpj slice: %4d/%4d' % (v_idx+1, self.dect_elem_count_vertical_actual)
                        print('\r' + str, end='')
                        self.ForwardProjectionBilinear(self.img_x_taichi, self.img_fp_x_taichi_single_slice, self.array_u_taichi,
                                                       self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                       self.dect_elem_count_horizontal,
                                                       self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                       self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                                       self.helical_scan, self.helical_pitch, v_idx, self.fpj_step_size,
                                                       self.img_center_x, self.img_center_y, self.img_center_z, self.curved_dect,
                                                       self.matrix_A_each_view_taichi, self.x_s_each_view_taichi, self.bool_apply_pmatrix)
                        self.TaichiReadFromSingleSlice(self.img_fp_x_taichi,self.img_fp_x_taichi_single_slice,v_idx,self.view_num,self.dect_elem_count_horizontal)
                    
                    self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                    self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                    self.array_angle_taichi, self.img_rot,self.img_fp_x_taichi,self.img_bp_fp_x_taichi,\
                                    self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                        self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                            self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
                    
                    
                    self.TaichiFieldSubtraction(self.img_bp_b_taichi,self.img_bp_fp_x_taichi,self.img_r_taichi,self.img_dim,self.img_dim_z)
                    self.img_d_taichi = self.img_r_taichi #d
                    
                    loss = np.zeros(shape = [1,self.num_iter])
                    
                    for idx in range(self.num_iter):
                        #P^T P d
                        for v_idx in range(self.dect_elem_count_vertical_actual): 
                            str_1 = 'Running iterations: %4d/%4d; ' % (idx+1, self.num_iter)
                            str_2 = 'fpj slice: %4d/%4d' % (v_idx+1, self.dect_elem_count_vertical_actual)
                            print('\r' + str_1 + str_2, end='')
                            self.ForwardProjectionBilinear(self.img_d_taichi, self.img_fp_d_taichi_single_slice, self.array_u_taichi,
                                                           self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                           self.dect_elem_count_horizontal,
                                                           self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                           self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                                           self.helical_scan, self.helical_pitch, v_idx, self.fpj_step_size,
                                                           self.img_center_x, self.img_center_y, self.img_center_z, self.curved_dect,
                                                           self.matrix_A_each_view_taichi, self.x_s_each_view_taichi, self.bool_apply_pmatrix)
                            self.TaichiReadFromSingleSlice(self.img_fp_d_taichi,self.img_fp_d_taichi_single_slice,v_idx,self.view_num,self.dect_elem_count_horizontal)
                        
                        self.BackProjectionPixelDriven(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                        self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                        self.array_angle_taichi, self.img_rot,self.img_fp_d_taichi,self.img_bp_fp_d_taichi,\
                                        self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                            self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                    self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
                        
                        self.img_x = self.img_x_taichi.to_numpy()
                        self.img_r = self.img_r_taichi.to_numpy()
                        self.img_d = self.img_d_taichi.to_numpy()
                        self.img_bp_fp_d = self.img_bp_fp_d_taichi.to_numpy()
                        
                        r_l2_norm = np.sum(np.multiply(self.img_r,self.img_r))
                        alpha = r_l2_norm / np.sum(np.multiply(self.img_d,self.img_bp_fp_d)) 
                        self.img_x = self.img_x + np.multiply(alpha, self.img_d) 
                        self.img_r = self.img_r - np.multiply(alpha, self.img_bp_fp_d) 
                        
                        beta = np.sum(np.multiply(self.img_r,self.img_r)) / r_l2_norm
                        self.img_d = self.img_r + beta * self.img_d
                        
                        self.img_x_taichi.from_numpy(self.img_x)
                        self.img_d_taichi.from_numpy(self.img_d)
                        self.img_r_taichi.from_numpy(self.img_r)
     
                        # r_i_l2_norm = self.TaichiInnerProduct(self.img_r_taichi,self.img_r_taichi,self.img_dim_z,self.img_dim,self.img_dim)
                        # alpha = r_i_l2_norm /self.TaichiInnerProduct(self.img_d_taichi,self.img_bp_fp_d_taichi,self.img_dim_z,self.img_dim,self.img_dim)
                        
                        # self.TaichiFieldAdd(self.img_x_taichi, self.img_d_taichi, self.img_x_taichi, alpha, self.img_dim,self.img_dim, self.img_dim_z)
                        # self.TaichiFieldAdd(self.img_r_taichi, self.img_bp_fp_d_taichi, self.img_r_taichi, - alpha, self.img_dim,self.img_dim, self.img_dim_z)

                        # beta = self.TaichiInnerProduct(self.img_r_taichi,self.img_r_taichi,self.img_dim_z,self.img_dim,self.img_dim) / r_i_l2_norm
                        # self.TaichiFieldAdd(self.img_d_taichi, self.img_r_taichi, self.img_d_taichi, beta,self.img_dim,self.img_dim, self.img_dim_z)
                        
                        loss[0,idx] = r_l2_norm
                        if idx%10 == 0:
                            plt.plot(range(idx),loss[0,0:idx])                 
                            plt.show()
                            
                        if idx%1==0:
                            self.img_x = self.img_x_taichi.to_numpy()
                            imaddRaw(self.img_x[:,:,:],self.output_path, dtype = np.float32, idx = idx)
                        
                    
                    print('\nSaving to %s !' % self.output_path)
                    #self.SaveReconImg()
        return self.img_recon #函数返回重建图
    @ti.kernel
    def TaichiInnerProduct(self,img_1_taichi:ti.template(),img_2_taichi:ti.template(),img_dim_1:ti.i32,img_dim_2:ti.i32,img_dim_3:ti.i32) -> float:
        output = 0.0
        for x_idx, y_idx,z_idx in ti.ndrange(img_dim_1,img_dim_2, img_dim_3):
            output = output + img_1_taichi[x_idx,y_idx,z_idx] * img_2_taichi[x_idx,y_idx,z_idx]
        return output
    
    @ti.kernel
    def TaichiReadFromSingleSlice(self,img_taichi:ti.template(),img_taichi_single_slice:ti.template(),img_idx_1:ti.i32,img_dim_2:ti.i32,img_dim_3:ti.i32):
        for x_idx, y_idx in ti.ndrange(img_dim_2, img_dim_3):
            img_taichi[img_idx_1,x_idx,y_idx] = img_taichi_single_slice[0,x_idx,y_idx]
    
    @ti.kernel
    def TaichiFieldSubtraction(self,img_1:ti.template(),img_2:ti.template(),img_3:ti.template(),img_dim:ti.i32,img_dim_z:ti.i32):
        for x_idx, y_idx, z_idx in ti.ndrange(img_dim, img_dim, img_dim_z):
            img_3[z_idx,x_idx,y_idx] = img_1[z_idx,x_idx,y_idx] - img_2[z_idx,x_idx,y_idx] 
            
    @ti.kernel        
    def TaichiFieldAdd(self, img_1:ti.template(), img_2:ti.template(), img_3:ti.template(), alpha:ti.f32, img_dim_1:ti.i32, img_dim_2:ti.i32, img_dim_3:ti.i32):
        for x_idx, y_idx, z_idx in ti.ndrange(img_dim_1, img_dim_2, img_dim_3):
            img_3[z_idx,x_idx,y_idx] = img_1[z_idx,x_idx,y_idx] + alpha * img_2[z_idx,x_idx,y_idx] 
    
                            
    def SaveReconImg(self):
        self.img_recon = self.img_x
        if self.convert_to_HU:
            self.img_recon = (self.img_recon / self.water_mu - 1)*1000
        if self.output_file_format == 'raw':
            imwriteRaw(self.img_recon,self.output_path,dtype=np.float32)
        elif self.output_file_format == 'tif' or self.output_file_format == 'tiff':
            imwriteTiff(self.img_recon, self.output_path,dtype=np.float32)
        
     
