# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:44:41 2024

@author: xji
"""

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
from run_mgfpj_v2 import *
import gc


def run_mgfbp_ir(file_path):
    ti.reset()
    ti.init(arch = ti.gpu)
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
        
        if 'Lambda' in config_dict:
            self.coef_lambda = config_dict['Lambda']
            if (not isinstance(self.coef_lambda,int) and not isinstance(self.coef_lambda,float)) or self.coef_lambda < 0.0:
                print("ERROR: Lambda must be a positive number!")
                sys.exit()
        else:
            self.coef_lambda = 1e-5
            print("Warning: Did not find Lambda! Use default value 1e-5. ")
        
        if 'BetaTV' in config_dict:
            self.beta_tv = config_dict['BetaTV']
            if (not isinstance(self.beta_tv,float) and not isinstance(self.beta_tv,int)) or self.beta_tv < 0.0:
                print("ERROR: BetaTV must be a positive number!")
                sys.exit()
        else:
            self.beta_tv = 1e-6
            print("Warning: Did not find BetaTV! Use default value 1e-6. ")
        
        if 'NumberOfIterations' in config_dict:
            self.num_iter = config_dict['NumberOfIterations']
            if not isinstance(self.num_iter,int) or self.num_iter<0.0:
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
            self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal), order='ijk')
        
        self.img_recon = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)

        self.img_x = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_x_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        
        self.img_bp_fp_x = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_fp_x_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))

        self.img_bp_b = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_b_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        
        self.img_fp_x = np.zeros((self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal), dtype=np.float32)
        self.img_fp_x_taichi = ti.field(dtype=ti.f32, shape=(
            self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal), order='ijk')
        
        self.img_fp_x_taichi_single_slice = ti.field(dtype=ti.f32, shape=(
            1, self.view_num, self.det_elem_count_horizontal), order='ijk')

        
        self.img_d_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        
        self.img_d = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_r = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_gradient_tv = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)

        
        self.img_bp_fp_d = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_fp_d_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))

        self.img_fp_d = np.zeros((self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal), dtype=np.float32)
        self.img_fp_d_taichi = ti.field(dtype=ti.f32, shape=(
            self.det_elem_count_vertical_actual, self.view_num, self.det_elem_count_horizontal), order='ijk')
        
        self.img_fp_d_taichi_single_slice = ti.field(dtype=ti.f32, shape=(
            1, self.view_num, self.det_elem_count_horizontal), order='ijk')
        
        self.sgm_total_pixel_count = self.det_elem_count_vertical_actual*self.view_num*self.det_elem_count_horizontal
        self.img_total_pixel_count = self.img_dim_z * self.img_dim * self.img_dim
        self.pixel_count_ratio = float( self.sgm_total_pixel_count/self.img_total_pixel_count)
        
   
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
                    self.BackProjectionPixelDriven(self.det_elem_count_vertical_actual, self.img_dim, self.det_elem_count_horizontal, \
                                    self.view_num, self.det_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_det_dis,self.total_scan_angle,\
                                    self.array_angle_taichi, self.img_rot,self.img_sgm_taichi,self.img_bp_b_taichi,\
                                    self.array_u_taichi,self.short_scan,self.cone_beam,self.det_elem_height,\
                                        self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                            self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
                    
                    for irn_iter_idx in range(4):
                        for v_idx in range(self.det_elem_count_vertical_actual):                              
                            self.ForwardProjectionBilinear(self.img_x_taichi, self.img_fp_x_taichi_single_slice, self.array_u_taichi,
                                                           self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                           self.det_elem_count_horizontal,
                                                           self.det_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                           self.source_isocenter_dis, self.source_det_dis, self.cone_beam,
                                                           self.helical_scan, self.helical_pitch, v_idx, self.fpj_step_size,
                                                           self.img_center_x, self.img_center_y, self.img_center_z, self.curved_dect,
                                                           self.matrix_A_each_view_taichi, self.x_s_each_view_taichi, self.bool_apply_pmatrix)
                            self.TaichiReadFromSingleSlice(self.img_fp_x_taichi,self.img_fp_x_taichi_single_slice,v_idx,self.view_num,self.det_elem_count_horizontal)
                        #P^T P x
                        self.BackProjectionPixelDriven(self.det_elem_count_vertical_actual, self.img_dim, self.det_elem_count_horizontal, \
                                        self.view_num, self.det_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_det_dis,self.total_scan_angle,\
                                        self.array_angle_taichi, self.img_rot,self.img_fp_x_taichi,self.img_bp_fp_x_taichi,\
                                        self.array_u_taichi,self.short_scan,self.cone_beam,self.det_elem_height,\
                                            self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                    self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
                        
                        self.img_bp_fp_x = self.img_bp_fp_x_taichi.to_numpy() 
                        self.img_bp_b = self.img_bp_b_taichi.to_numpy()
                        
                        WR = self.GenerateWR(self.img_x,self.beta_tv)
                        self.img_gradient_tv = self.GradientTVCalc(self.img_x, WR)
                        self.img_d = self.img_bp_b - self.img_bp_fp_x-  self.coef_lambda * self.img_gradient_tv *self.pixel_count_ratio
                        self.img_r = self.img_d
                        loss = np.zeros(shape = [1,self.num_iter])
                        imwriteRaw(self.img_d,'img_d.raw')
                        for idx in range(self.num_iter):
                            #P^T P d
                            self.img_d_taichi.from_numpy(self.img_d)
                            for v_idx in range(self.det_elem_count_vertical_actual):
                                str_0 = 'Running IRN iterations: %4d/%4d; ' % (irn_iter_idx+1, 15)
                                str_1 = 'Running iterations: %4d/%4d; ' % (idx+1, self.num_iter)
                                str_2 = 'fpj slice: %4d/%4d' % (v_idx+1, self.det_elem_count_vertical_actual)
                                print('\r' + str_0 + str_1 + str_2, end='')    
                                self.ForwardProjectionBilinear(self.img_d_taichi, self.img_fp_d_taichi_single_slice, self.array_u_taichi,
                                                               self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                               self.det_elem_count_horizontal,
                                                               self.det_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                               self.source_isocenter_dis, self.source_det_dis, self.cone_beam,
                                                               self.helical_scan, self.helical_pitch, v_idx, self.fpj_step_size,
                                                               self.img_center_x, self.img_center_y, self.img_center_z, self.curved_dect,
                                                               self.matrix_A_each_view_taichi, self.x_s_each_view_taichi, self.bool_apply_pmatrix)
                                self.TaichiReadFromSingleSlice(self.img_fp_d_taichi,self.img_fp_d_taichi_single_slice,v_idx,self.view_num,self.det_elem_count_horizontal)
                            #P^T P d
                            self.BackProjectionPixelDriven(self.det_elem_count_vertical_actual, self.img_dim, self.det_elem_count_horizontal, \
                                            self.view_num, self.det_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_det_dis,self.total_scan_angle,\
                                            self.array_angle_taichi, self.img_rot,self.img_fp_d_taichi,self.img_bp_fp_d_taichi,\
                                            self.array_u_taichi,self.short_scan,self.cone_beam,self.det_elem_height,\
                                                self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                    self.img_center_x,self.img_center_y,self.img_center_z,self.curved_dect,\
                                                        self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode)
                            
                            self.img_bp_fp_d = self.img_bp_fp_d_taichi.to_numpy()
                            imwriteRaw(self.img_bp_fp_d,'img_bp_fp_d.raw')
                            self.img_bp_fp_d = self.img_bp_fp_d  + self.coef_lambda * self.GradientTVCalc(self.img_d, WR) * self.pixel_count_ratio

                            
                            r_l2_norm = np.sum(np.multiply(self.img_r,self.img_r))
                            alpha = r_l2_norm / np.sum(np.multiply(self.img_d, self.img_bp_fp_d)) 
                            
                            if np.sqrt(r_l2_norm/ (self.img_dim**2 * self.img_dim_z)) / self.water_mu * 1000 < 1e-1:
                                #print('\nIteration Terminated!')
                                break
                            
                            self.img_x = self.img_x + np.multiply(alpha, self.img_d) 
                            self.img_r = self.img_r - np.multiply(alpha, self.img_bp_fp_d) 
                            
                            beta = np.sum(np.multiply(self.img_r, self.img_r)) / r_l2_norm
                            self.img_d = self.img_r + beta * self.img_d
                            
                            self.img_x_taichi.from_numpy(self.img_x)
                            self.img_d_taichi.from_numpy(self.img_d)
                            
                            if idx%1==0:
                                if self.convert_to_HU:
                                    plt.figure(dpi=300)
                                    plt.imshow((self.img_x[:,:,128]/ self.water_mu - 1)*1000,cmap = 'gray',vmin = -0, vmax = 50)
                                    plt.show()
                                    #imaddRaw((self.img_x / self.water_mu - 1)*1000, self.output_path, dtype = np.float32, idx = idx* (irn_iter_idx) + 1)
                        
                        
                    print('\nSaving to %s !' % self.output_path)
                    self.SaveReconImg()
        return self.img_recon #函数返回重建图
    
    def GenerateWR(self, img_x, beta_tv):
        ux = np.zeros_like(img_x)
        uy = np.zeros_like(img_x)
        uz = np.zeros_like(img_x)
        ux[:, 0:-1, :]= img_x[:,1:,:] - img_x[:,:-1,:]
        uy[:, :, 0:-1]= img_x[:,:,1:] - img_x[:,:,:-1]
        if img_x.shape[0]>2:
            uz[0:-1, :, :]= img_x[1:,:,:] - img_x[:-1,:,:]
        WR = 2 / np.sqrt(ux**2 +uy ** 2+uz ** 2 + self.beta_tv) 
        return WR
    
    def GradientTVCalc(self, img_x, WR):
        ux = np.zeros_like(img_x)
        uy = np.zeros_like(img_x)
        uz = np.zeros_like(img_x)
        ux[:, 0:-1, :]= img_x[:, 1: ,:] - img_x[:, :-1 ,:]
        uy[:, :, 0:-1]= img_x[:, :, 1:] - img_x[:, : , :-1]
        if img_x.shape[0]>2:
            uz[0:-1, :, :]= img_x[1:,:,:] - img_x[:-1,:,:]
        ux = np.multiply(WR,ux)
        uy = np.multiply(WR,uy)
        uz = np.multiply(WR,uz)
        uxx = np.zeros_like(img_x)
        uyy = np.zeros_like(img_x)
        uzz = np.zeros_like(img_x)
        uxx[:, 1:-1, :]= ux[:,1:-1,:] - ux[:,0:-2,:]
        uyy[:, :, 1:-1 ]= uy[:,:,1:-1] - uy[:,:,0:-2]
        if img_x.shape[0]>2:
            uzz[1:-1, :, : ]= uz[1:-1,:,:] - uz[0:-2,:,:]
        output = -uxx - uyy - uzz
        return output
    
    @ti.kernel
    def BackProjectionPixelDriven(self, det_elem_count_vertical_actual:ti.i32, img_dim:ti.i32, det_elem_count_horizontal:ti.i32, \
                                  view_num:ti.i32, det_elem_width:ti.f32,\
                                  img_pix_size:ti.f32, source_isocenter_dis:ti.f32, source_det_dis:ti.f32,total_scan_angle:ti.f32,\
                                      array_angle_taichi:ti.template(),img_rot:ti.f32,img_sgm_filtered_taichi:ti.template(),img_recon_taichi:ti.template(),\
                                          array_u_taichi:ti.template(), short_scan:ti.i32,cone_beam:ti.i32,det_elem_height:ti.f32,\
                                              array_v_taichi:ti.template(),img_dim_z:ti.i32,img_voxel_height:ti.f32, \
                                                  img_center_x:ti.f32,img_center_y:ti.f32,img_center_z:ti.f32,curved_dect:ti.i32,\
                                                      bool_apply_pmatrix:ti.i32, array_pmatrix_taichi:ti.template(), recon_view_mode: ti.i32):
        
        
        for i_x, i_y in ti.ndrange(img_dim, img_dim):
            for i_z in ti.ndrange(img_dim_z):
                img_recon_taichi[i_z, i_y, i_x] = 0.0
                x_after_rot = 0.0; y_after_rot = 0.0; x=0.0; y=0.0;z=0.0;
                if recon_view_mode == 1: #axial view (from bottom to top)
                    x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                    y_after_rot = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_y
                    z = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_z
                elif recon_view_mode == 2: #coronal view (from fron to back)
                    x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                    z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_z
                    y_after_rot = - (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_y
                elif recon_view_mode == 3: #sagittal view (from left to right)
                    z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_z
                    y_after_rot = - img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_y
                    x_after_rot = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_x
                    
                x = + x_after_rot * ti.cos(img_rot) + y_after_rot * ti.sin(img_rot)
                y = - x_after_rot * ti.sin(img_rot) + y_after_rot * ti.cos(img_rot)
                for j in ti.ndrange(view_num):               
                    pix_to_source_parallel_dis = 0.0
                    mag_factor = 0.0
                    temp_u_idx_floor = 0
                    pix_proj_to_det_u = 0.0
                    pix_proj_to_det_v = 0.0
                    pix_proj_to_det_u_idx = 0.0
                    pix_proj_to_det_v_idx = 0.0
                    ratio_u = 0.0
                    ratio_v = 0.0
                    angle_this_view_exclude_img_rot = array_angle_taichi[j] - img_rot
                    
                    pix_to_source_parallel_dis = source_isocenter_dis - x * ti.cos(angle_this_view_exclude_img_rot) - y * ti.sin(angle_this_view_exclude_img_rot)
                    if bool_apply_pmatrix == 0:
                        mag_factor = source_det_dis / pix_to_source_parallel_dis
                        if curved_dect:
                            pix_proj_to_det_u = source_det_dis * ti.atan2(x*ti.sin(angle_this_view_exclude_img_rot)-y*ti.cos(angle_this_view_exclude_img_rot),pix_to_source_parallel_dis)
                        else:
                            pix_proj_to_det_u = mag_factor * (x*ti.sin(angle_this_view_exclude_img_rot)-y*ti.cos(angle_this_view_exclude_img_rot))
                        pix_proj_to_det_u_idx = (pix_proj_to_det_u - array_u_taichi[0]) / det_elem_width
                    else:
                        mag_factor = 1.0 / (array_pmatrix_taichi[12*j + 8] * x +\
                            array_pmatrix_taichi[12*j + 9] * y +\
                                array_pmatrix_taichi[12*j + 10] * z +\
                                    array_pmatrix_taichi[12*j + 11] * 1)
                        pix_proj_to_det_u_idx = (array_pmatrix_taichi[12*j + 0] * x +\
                            array_pmatrix_taichi[12*j + 1] * y +\
                                array_pmatrix_taichi[12*j + 2] * z +\
                                    array_pmatrix_taichi[12*j + 3] * 1) * mag_factor
                    if pix_proj_to_det_u_idx < 0 or  pix_proj_to_det_u_idx + 1 > det_elem_count_horizontal - 1:
                        img_recon_taichi[i_z, i_y, i_x] = 0
                        break
                    temp_u_idx_floor = int(ti.floor(pix_proj_to_det_u_idx))
                    ratio_u = pix_proj_to_det_u_idx - temp_u_idx_floor
                                        
                    
                    distance_weight = 0.0
                    if curved_dect:
                        distance_weight = 1.0 / ((pix_to_source_parallel_dis * pix_to_source_parallel_dis) + \
                                                 (x * ti.sin(angle_this_view_exclude_img_rot) - y * ti.cos(angle_this_view_exclude_img_rot)) \
                                                * (x * ti.sin(angle_this_view_exclude_img_rot) - y * ti.cos(angle_this_view_exclude_img_rot)))
                    else:
                        distance_weight = 1.0 / (pix_to_source_parallel_dis * pix_to_source_parallel_dis)

                    if cone_beam == True:
                        if bool_apply_pmatrix == 0:
                            pix_proj_to_det_v = mag_factor * z
                            pix_proj_to_det_v_idx = (pix_proj_to_det_v - array_v_taichi[0]) / det_elem_height \
                                * abs(array_v_taichi[1] - array_v_taichi[0]) / (array_v_taichi[1] - array_v_taichi[0])
                                #abs(array_v_taichi[1] - array_v_taichi[0]) / (array_v_taichi[1] - array_v_taichi[0]) defines whether the first 
                                #sinogram slice corresponds to the top row
                        else:
                            pix_proj_to_det_v_idx = (array_pmatrix_taichi[12*j + 4] * x +\
                                array_pmatrix_taichi[12*j + 5] * y +\
                                    array_pmatrix_taichi[12*j + 6] * z +\
                                        array_pmatrix_taichi[12*j + 7] * 1) * mag_factor
                                
                        temp_v_idx_floor = int(ti.floor(pix_proj_to_det_v_idx))   #mark
                        if temp_v_idx_floor < 0 or temp_v_idx_floor + 1 > det_elem_count_vertical_actual - 1: 
                            img_recon_taichi[i_z, i_y, i_x] = 0
                            break
                        else:
                            ratio_v = pix_proj_to_det_v_idx - temp_v_idx_floor
                            part_0 = img_sgm_filtered_taichi[temp_v_idx_floor,j,temp_u_idx_floor] * (1 - ratio_u) + \
                                img_sgm_filtered_taichi[temp_v_idx_floor,j,temp_u_idx_floor + 1] * ratio_u
                            part_1 = img_sgm_filtered_taichi[temp_v_idx_floor + 1,j,temp_u_idx_floor] * (1 - ratio_u) +\
                                  img_sgm_filtered_taichi[temp_v_idx_floor + 1,j,temp_u_idx_floor + 1] * ratio_u
                            img_recon_taichi[i_z, i_y, i_x] += ((1 - ratio_v) * part_0 + ratio_v * part_1)
                    else: 
                        val_0 = img_sgm_filtered_taichi[i_z , j , temp_u_idx_floor]
                        val_1 = img_sgm_filtered_taichi[i_z , j , temp_u_idx_floor + 1]
                        img_recon_taichi[i_z, i_y, i_x] += ((1 - ratio_u) * val_0 + ratio_u * val_1)
    
    def TVMap(self, beta ):
        self.img_tv_map = np.sqrt (np.multiply(np.gradient(self.img_x, axis = 1), np.gradient(self.img_x, axis = 1)) +\
                np.multiply(np.gradient(self.img_x, axis = 2), np.gradient(self.img_x, axis = 2))  + beta)
        
    
    @ti.kernel 
    def TaichiGradient(self,img:ti.template(), img_output:ti.template(),img_dim_0:ti.i32,img_dim_1:ti.i32,img_dim_2:ti.i32,dim_idx:ti.i32):
        for idx_0, idx_1, idx_2 in ti.ndrange(img_dim_0, img_dim_1, img_dim_2):
            if dim_idx == 0:
                if idx_0 != 0 and idx_0 != img_dim_0-1 :
                    img_output[idx_0,idx_1,idx_2] = (img[idx_0+1,idx_1,idx_2] - img[idx_0-1,idx_1,idx_2]) /2.0
                else:
                    img_output[idx_0,idx_1,idx_2] = 0
                    
            elif dim_idx == 1:
                if idx_1 != 0 and idx_1 != img_dim_1-1 :
                    img_output[idx_0,idx_1,idx_2] = (img[idx_0,idx_1+1,idx_2] - img[idx_0,idx_1-1,idx_2]) /2.0
                else:
                    img_output[idx_0,idx_1,idx_2] = 0
        
            elif dim_idx == 2:
                if idx_2 != 0 and idx_2 != img_dim_2-1 :
                    img_output[idx_0,idx_1,idx_2] = (img[idx_0,idx_1,idx_2+1] - img[idx_0,idx_1,idx_2-1]) /2.0
                else:
                    img_output[idx_0,idx_1,idx_2] = 0
    
    
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
    
    
    def build_weights(x):
        Dxx = np.zeros_like(x, dtype=np.float32)
        Dyx = np.zeros_like(x, dtype=np.float32)
        # Dzx = np.zeros_like(x, dtype=np.float32)
    
        Dxx[:-1, :] = x[:-1, :] - x[1:, :]
        Dyx[:, :-1] = x[:, :-1] - x[:, 1:]
        # Dzx[:, :, :-1] = x[:, :, :-1] - x[:, :, 1:]
    
        W = (Dxx ** 2 + Dyx ** 2 + 1e-6) ** (-1 / 4)
        return W

    def Lx(W, img):
        Dxx = np.copy(img)
        Dyx = np.copy(img)
        # Dzx = np.copy(img)
    
        Dxx[0:-2, :] = img[0:-2, :] - img[1:-1, :]
        Dyx[:, 0:-2] = img[:, 0:-2] - img[:, 1:-1]
        # Dzx[:, :, 0:-2] = img[:, :, 0:-2] - img[:, :, 1:-1]
    
        return np.stack((W * Dxx, W * Dyx), axis=0)
    
    def Ltx(W, img3):
        Wx_1 = W * img3[0, :, :]
        Wx_2 = W * img3[1, :, :]
        # Wx_3 = W * img3[2, :, :, :]
    
        DxtWx_1 = Wx_1
        DytWx_2 = Wx_2
        # DztWx_3 = Wx_3
    
        DxtWx_1[1:-2, :] = Wx_1[1:-2, :] - Wx_1[0:-3, :]
        DxtWx_1[-1, :] = -Wx_1[-2, :]
    
        DytWx_2[:, 1:-2] = Wx_2[:, 1:-2] - Wx_2[:, 0:-3]
        DytWx_2[:, -1] = -Wx_2[:, -2]
    
        # DztWx_3[:, :, 1:-2] = Wx_3[:, :, 1:-2] - Wx_3[:, :, 0:-3]
        # DztWx_3[:, :, -1] = -Wx_3[:, :, -2]
    
        return DxtWx_1 + DytWx_2# + DztWx_3        
    
    def SaveReconImg(self):
        self.img_recon = self.img_x
        if self.convert_to_HU:
            self.img_recon = (self.img_recon / self.water_mu - 1)*1000
        if self.output_file_format == 'raw':
            imwriteRaw(self.img_recon,self.output_path,dtype=np.float32)
        elif self.output_file_format == 'tif' or self.output_file_format == 'tiff':
            imwriteTiff(self.img_recon, self.output_path,dtype=np.float32)
        
     
