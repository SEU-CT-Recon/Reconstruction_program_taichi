# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:12:03 2024

@author: xji
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:26:08 2024

@author: xji
"""

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
from run_mgfpj_nmwj import *
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
    #img_recon_seed = fbp.MainFunction() #generate a seed from fbp reconstruction
    img_recon_seed = np.zeros((fbp.img_dim_z,fbp.img_dim, fbp.img_dim),dtype=np.float32)
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
class Mgfbp_ir(Mgfpj_nmwj):
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
        
        if 'NumberOfIRNIterations' in config_dict:
            self.num_irn_iter = config_dict['NumberOfIRNIterations']
            if not isinstance(self.num_irn_iter,int) or self.num_irn_iter<0.0:
                print("ERROR: NumberOfIRNIterations must be a positive integer!")
                sys.exit()
        else:
            self.num_irn_iter = 15
            print("Warning: Did not find NumberOfIRNIterations! Use default value 15. ")
        
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
            self.dect_elem_count_vertical_actual, self.dect_elem_count_horizontal), order='ij')
        
        self.img_recon = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)

        self.img_x = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_x_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        
        self.img_bp_fp_x = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_fp_x_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))

        self.img_bp_b = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_b_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        
        self.img_fp_x = np.zeros((self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal), dtype=np.float32)

        self.img_fp_x_taichi_single_view = ti.field(dtype=ti.f32, shape=(
            self.dect_elem_count_vertical, self.dect_elem_count_horizontal))

        
        self.img_d_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        
        self.img_d = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_r = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_gradient_tv = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)

        
        self.img_bp_fp_d = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_bp_fp_d_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))

        self.img_fp_d = np.zeros((self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal), dtype=np.float32)

        self.img_fp_d_taichi_single_view = ti.field(dtype=ti.f32, shape=(
            self.dect_elem_count_vertical, self.dect_elem_count_horizontal), order='ij')
        
        self.sgm_total_pixel_count = self.dect_elem_count_vertical_actual*self.view_num*self.dect_elem_count_horizontal
        self.img_total_pixel_count = self.img_dim_z * self.img_dim * self.img_dim
        self.pixel_count_ratio = float( self.sgm_total_pixel_count/self.img_total_pixel_count)
        
        self.img_x_truncation_flag_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        
   
    def MainFunction(self,img_recon_seed):
        #Main function for reconstruction
        if not self.bool_uneven_scan_angle:
            self.GenerateAngleArray(
                self.view_num, self.img_rot, self.total_scan_angle, self.array_angle_taichi)
        self.GenerateDectPixPosArrayFPJ(self.dect_elem_count_vertical, - self.dect_elem_height, self.dect_offset_vertical, self.array_v_taichi)
        self.GenerateDectPixPosArrayFPJ(self.dect_elem_count_horizontal*self.oversample_size,-self.dect_elem_width/self.oversample_size,
                                     -self.dect_offset_horizontal, self.array_u_taichi)
        self.file_processed_count = 0;#record the number of files processed
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):#match the file pattern
                if self.ReadSinogram(file):
                    self.file_processed_count += 1 
                    print('Reconstructing %s ...' % self.input_path)
                    
                    #read prior image
                    img_prior = imreadRaw('./rec/rec_24.raw', width = self.img_dim, height = self.img_dim, nSlice = self.img_dim_z)
                    img_prior = img_prior.reshape((self.img_dim_z,self.img_dim,self.img_dim))
                    alpha_prior = 0.5
                    
                    if self.convert_to_HU:
                          img_recon_seed = (img_recon_seed/1000 + 1 ) * self.water_mu
                          img_prior = (img_prior/1000 + 1 ) * self.water_mu
                    img_recon_seed = img_prior
                    self.img_x = img_recon_seed
                    self.img_x_taichi.from_numpy(self.img_x)
                    
                    #P^T b
                    self.img_x_truncation_flag_taichi.from_numpy(np.ones_like(self.img_x) )
                    self.img_bp_b_taichi.from_numpy(np.zeros_like(self.img_x))
                    for view_idx in range(self.view_num):   
                        str_2 = 'BP of input sinogram, view: %4d/%4d' % (view_idx+1, self.view_num)
                        print('\r' + str_2, end='')   
                        self.img_sgm_taichi.from_numpy(self.img_sgm[:,view_idx,:])
                        self.BackProjectionPixelDrivenPerView(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                        self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                        self.array_angle_taichi, self.img_rot,self.img_sgm_taichi,self.img_bp_b_taichi,\
                                        self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                            self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                self.img_center_x,self.img_center_y,self.array_img_center_z_taichi,self.curved_dect,\
                                                    self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode, view_idx, self.img_x_truncation_flag_taichi,\
                                                        self.array_source_pos_z_taichi)
                    self.SetTruncatedRegionToZero(self.img_bp_b_taichi,self.img_x_truncation_flag_taichi, self.img_dim, self.img_dim_z)
                    self.img_bp_b = self.img_bp_b_taichi.to_numpy()
                    imwriteRaw(self.img_bp_b,'img_bp_b.raw')
                    
                    
                    for irn_iter_idx in range(self.num_irn_iter):
                        self.img_bp_fp_x_taichi.from_numpy(np.zeros_like(self.img_x))
                        for view_idx in range(self.view_num):   
                            str_2 = 'FPJ and BP of recon seed, view: %4d/%4d' % (view_idx+1, self.view_num)
                            print('\r' + str_2, end='') 
                            self.ForwardProjectionBilinear(self.img_x_taichi, self.img_fp_x_taichi_single_view, self.array_u_taichi,
                                                            self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                            self.dect_elem_count_horizontal,
                                                            self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                            self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                                            self.helical_scan, self.helical_pitch, view_idx, self.fpj_step_size,
                                                            self.img_center_x, self.img_center_y, self.array_img_center_z_taichi, self.curved_dect,
                                                            self.matrix_A_each_view_taichi, self.x_s_each_view_taichi, self.bool_apply_pmatrix,\
                                                            self.dect_elem_count_vertical_actual, self.dect_elem_vertical_recon_range_begin,\
                                                            self.array_source_pos_z_taichi)
                            #imaddRaw(self.img_fp_x_taichi_single_view.to_numpy(),'img_fp_x_single_view.raw',idx = view_idx)    
                            self.BackProjectionPixelDrivenPerView(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                            self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                            self.array_angle_taichi, self.img_rot,self.img_fp_x_taichi_single_view,self.img_bp_fp_x_taichi,\
                                            self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                                self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                    self.img_center_x,self.img_center_y,self.array_img_center_z_taichi,self.curved_dect,\
                                                        self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode,view_idx, self.img_x_truncation_flag_taichi,\
                                                            self.array_source_pos_z_taichi)
                        self.SetTruncatedRegionToZero(self.img_bp_fp_x_taichi,self.img_x_truncation_flag_taichi, self.img_dim, self.img_dim_z)
                        self.img_bp_fp_x = self.img_bp_fp_x_taichi.to_numpy() 
                        imwriteRaw(self.img_bp_fp_x,'img_bp_fp_x.raw')
                        
                        
                        WR = self.GenerateWR(self.img_x,self.beta_tv)
                        img_DT_WR_D_x = self.GradientTVCalc(self.img_x, WR)
                        WR_prior = self.GenerateWR(self.img_x - img_prior,self.beta_tv)
                        img_DT_WRpiror_D_prior = self.GradientTVCalc(img_prior, WR_prior)
                        img_DT_WRpiror_D_x = self.GradientTVCalc(self.img_x, WR_prior)
                        
                        
                        self.img_d = self.img_bp_b + self.coef_lambda * alpha_prior*self.pixel_count_ratio*img_DT_WRpiror_D_prior\
                            - self.img_bp_fp_x - self.coef_lambda* (alpha_prior)*img_DT_WRpiror_D_x * self.pixel_count_ratio \
                                - self.coef_lambda* (1-alpha_prior)*img_DT_WR_D_x * self.pixel_count_ratio 
                        self.img_r = self.img_d
                        
                        # imwriteRaw(img_DT_WR_D_x,'img_DT_WR_D_x.raw')
                        # imwriteRaw(self.img_x,'img_x.raw')
                        
                        
                        # imwriteRaw(self.img_d,'img_d.raw')
                        
                        loss = np.zeros(shape = [1,self.num_iter])
                        #imaddRaw(self.img_DT_WR_D_x, 'img_DT_WR_D_x.raw', idx = irn_iter_idx)
                        for iter_idx in range(self.num_iter):
                            #P^T P d
                            self.img_bp_fp_d_taichi.from_numpy(np.zeros_like(self.img_x))
                            self.img_d_taichi.from_numpy(self.img_d)
                            for view_idx in range(self.view_num): 
                                str_0 = 'Running IRN iterations: %4d/%4d; ' % (irn_iter_idx+1, self.num_irn_iter)
                                str_1 = 'Running iterations: %4d/%4d; ' % (iter_idx+1, self.num_iter)
                                str_2 = 'FPJ and BP view: %4d/%4d' % (view_idx+1, self.view_num)
                                print('\r' + str_0 + str_1 + str_2, end='')
                                self.ForwardProjectionBilinear(self.img_d_taichi, self.img_fp_d_taichi_single_view, self.array_u_taichi,
                                                                self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                                                self.dect_elem_count_horizontal,
                                                                self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                                                self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                                                self.helical_scan, self.helical_pitch, view_idx, self.fpj_step_size,
                                                                self.img_center_x, self.img_center_y, self.array_img_center_z_taichi, self.curved_dect,
                                                                self.matrix_A_each_view_taichi, self.x_s_each_view_taichi, self.bool_apply_pmatrix, \
                                                                self.dect_elem_count_vertical_actual, self.dect_elem_vertical_recon_range_begin,\
                                                                self.array_source_pos_z_taichi)
                                self.BackProjectionPixelDrivenPerView(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                                                self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                                                self.array_angle_taichi, self.img_rot,self.img_fp_d_taichi_single_view,self.img_bp_fp_d_taichi,\
                                                self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                                    self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                                        self.img_center_x,self.img_center_y,self.array_img_center_z_taichi,self.curved_dect,\
                                                            self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode,view_idx, self.img_x_truncation_flag_taichi,\
                                                                self.array_source_pos_z_taichi)
                            self.SetTruncatedRegionToZero(self.img_bp_fp_d_taichi,self.img_x_truncation_flag_taichi, self.img_dim, self.img_dim_z)
                            self.img_bp_fp_d = self.img_bp_fp_d_taichi.to_numpy()
                            #imwriteRaw(self.img_bp_fp_d,'img_bp_fp_d.raw')
                            self.img_bp_fp_d = self.img_bp_fp_d  + self.coef_lambda *(alpha_prior) * self.GradientTVCalc(self.img_d, WR_prior) * self.pixel_count_ratio\
                                + self.coef_lambda *(1-alpha_prior)* self.GradientTVCalc(self.img_d, WR) * self.pixel_count_ratio

                            
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
                            
                            if iter_idx%1==0:
                                if self.convert_to_HU:
                                    plt.figure(dpi=300)
                                    #plt.imshow((self.img_x[:,:,int(round(self.img_dim/2))]/ self.water_mu - 1)*1000,cmap = 'gray',vmin = -50, vmax = 100)
                                    plt.imshow((self.img_x[int(round(self.img_dim_z/2)),:,:]/ self.water_mu - 1)*1000,cmap = 'gray',vmin = -100, vmax = 100)
                                    #plt.imshow((self.img_x[0,:,:]/ self.water_mu - 1)*1000,cmap = 'gray',vmin = -30, vmax = 100)
                                    plt.show()
                                    
                            if self.output_file_format == 'tif' or self.output_file_format == 'tiff':
                                imwriteTiff((self.img_x/ self.water_mu - 1)*1000, self.output_path,dtype=np.float32)

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
    def BackProjectionPixelDrivenPerView(self, dect_elem_count_vertical_actual:ti.i32, img_dim:ti.i32, dect_elem_count_horizontal:ti.i32, \
                                  view_num:ti.i32, dect_elem_width:ti.f32,\
                                  img_pix_size:ti.f32, source_isocenter_dis:ti.f32, source_dect_dis:ti.f32,total_scan_angle:ti.f32,\
                                      array_angle_taichi:ti.template(),img_rot:ti.f32,img_sgm_filtered_taichi:ti.template(),img_recon_taichi:ti.template(),\
                                          array_u_taichi:ti.template(), short_scan:ti.i32,cone_beam:ti.i32,dect_elem_height:ti.f32,\
                                              array_v_taichi:ti.template(),img_dim_z:ti.i32,img_voxel_height:ti.f32, \
                                                  img_center_x:ti.f32,img_center_y:ti.f32,array_img_center_z_taichi:ti.template(),curved_dect:ti.i32,\
                                                      bool_apply_pmatrix:ti.i32, array_pmatrix_taichi:ti.template(), recon_view_mode: ti.i32, view_idx:ti.i32,\
                                                          img_x_truncation_flag_taichi:ti.template(), array_source_pos_z_taichi:ti.template()):
        
        
        for i_x, i_y in ti.ndrange(img_dim, img_dim):
            for i_z in ti.ndrange(img_dim_z):
                x_after_rot = 0.0; y_after_rot = 0.0; x=0.0; y=0.0;z=0.0;
                if recon_view_mode == 1: #axial view (from bottom to top)
                    x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                    y_after_rot = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_y
                    z = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + array_img_center_z_taichi[0,view_idx] - array_source_pos_z_taichi[view_idx]
                elif recon_view_mode == 2: #coronal view (from fron to back)
                    x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                    z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + array_img_center_z_taichi[0,view_idx] - array_source_pos_z_taichi[view_idx]
                    y_after_rot = - (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_y
                elif recon_view_mode == 3: #sagittal view (from left to right)
                    z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + array_img_center_z_taichi[0,view_idx] - array_source_pos_z_taichi[view_idx]
                    y_after_rot = - img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_y
                    x_after_rot = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_x
                    
                x = + x_after_rot * ti.cos(img_rot) + y_after_rot * ti.sin(img_rot)
                y = - x_after_rot * ti.sin(img_rot) + y_after_rot * ti.cos(img_rot)
                
                
                pix_to_source_parallel_dis = 0.0
                mag_factor = 0.0
                temp_u_idx_floor = 0
                pix_proj_to_dect_u = 0.0
                pix_proj_to_dect_v = 0.0
                pix_proj_to_dect_u_idx = 0.0
                pix_proj_to_dect_v_idx = 0.0
                ratio_u = 0.0
                ratio_v = 0.0
                angle_this_view_exclude_img_rot = array_angle_taichi[view_idx] - img_rot
                
                
                pix_to_source_parallel_dis = source_isocenter_dis - x * ti.cos(angle_this_view_exclude_img_rot) - y * ti.sin(angle_this_view_exclude_img_rot)
                
                direction_flag = ti.abs(x - source_isocenter_dis* ti.cos(angle_this_view_exclude_img_rot)) > ti.abs(y - source_isocenter_dis* ti.sin(angle_this_view_exclude_img_rot))
                distance_factor = ti.sqrt( (x - source_isocenter_dis* ti.cos(angle_this_view_exclude_img_rot)) ** 2 + \
                                          (y - source_isocenter_dis* ti.sin(angle_this_view_exclude_img_rot)) ** 2 \
                                              + z**2) /(ti.abs(x - source_isocenter_dis* ti.cos(angle_this_view_exclude_img_rot)) * direction_flag + \
                                                        ti.abs(y - source_isocenter_dis* ti.sin(angle_this_view_exclude_img_rot)) * (1- direction_flag))
                
                
                gamma_fan = ti.atan2(-x*ti.sin(angle_this_view_exclude_img_rot)+y*ti.cos(angle_this_view_exclude_img_rot),pix_to_source_parallel_dis)
                alpha_fan = ti.asin(source_isocenter_dis * ti.sin(gamma_fan) /(source_dect_dis - source_isocenter_dis))
                pix_proj_to_dect_u = (source_dect_dis - source_isocenter_dis) * (gamma_fan + alpha_fan)
                pix_proj_to_dect_u_idx = (pix_proj_to_dect_u - array_u_taichi[0]) / (array_u_taichi[1] - array_u_taichi[0])
                
                
                
                pix_source_dis_xy = pix_to_source_parallel_dis / ti.cos(gamma_fan)
                pix_element_dis_xy = (source_isocenter_dis) * ti.cos(gamma_fan) +  (source_dect_dis - source_isocenter_dis) * ti.cos(alpha_fan)
                mag_factor = pix_element_dis_xy / pix_source_dis_xy
                
                
                
                    
                if pix_proj_to_dect_u_idx < 0 or  pix_proj_to_dect_u_idx + 1 > dect_elem_count_horizontal - 1:
                    img_x_truncation_flag_taichi[i_z, i_y, i_x] = 0.0 #mark the truncated region
                else:
                    temp_u_idx_floor = int(ti.floor(pix_proj_to_dect_u_idx))
                    ratio_u = pix_proj_to_dect_u_idx - temp_u_idx_floor
                    
                    pix_proj_to_dect_v = mag_factor * z
                    pix_proj_to_dect_v_idx = (pix_proj_to_dect_v - array_v_taichi[0])/(array_v_taichi[1] - array_v_taichi[0])

                                
                    temp_v_idx_floor = int(ti.floor(pix_proj_to_dect_v_idx))   #mark
                    
                    
                    if temp_v_idx_floor < 0 or temp_v_idx_floor + 1 > dect_elem_count_vertical_actual - 1:
                        #img_x_truncation_flag_taichi[i_z, i_y, i_x] = 0.0 #mark the truncated region with -10000
                        img_recon_taichi[i_z, i_y, i_x] +=0.0
                    else:
                        ratio_v = pix_proj_to_dect_v_idx - temp_v_idx_floor
                        part_0 = img_sgm_filtered_taichi[temp_v_idx_floor,temp_u_idx_floor] * (1 - ratio_u) + \
                            img_sgm_filtered_taichi[temp_v_idx_floor,temp_u_idx_floor + 1] * ratio_u
                        part_1 = img_sgm_filtered_taichi[temp_v_idx_floor + 1,temp_u_idx_floor] * (1 - ratio_u) +\
                              img_sgm_filtered_taichi[temp_v_idx_floor + 1,temp_u_idx_floor + 1] * ratio_u
                        img_recon_taichi[i_z, i_y, i_x] += ((1 - ratio_v) * part_0 + ratio_v * part_1) * distance_factor * img_pix_size

    
    
    @ti.kernel 
    def SetTruncatedRegionToZero(self,img_recon_taichi:ti.template(),img_x_truncation_flag_taichi:ti.template(),img_dim:ti.i32,img_dim_z:ti.i32):
        for i_x, i_y in ti.ndrange(img_dim, img_dim):
            for i_z in ti.ndrange(img_dim_z):
                img_recon_taichi[i_z,i_x,i_y] *=img_x_truncation_flag_taichi[i_z,i_x,i_y]
    
    def TVMap(self, beta ):
        self.img_tv_map = np.sqrt (np.multiply(np.gradient(self.img_x, axis = 1), np.gradient(self.img_x, axis = 1)) +\
                np.multiply(np.gradient(self.img_x, axis = 2), np.gradient(self.img_x, axis = 2))  + beta)
        
    
    
    
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
        
     
