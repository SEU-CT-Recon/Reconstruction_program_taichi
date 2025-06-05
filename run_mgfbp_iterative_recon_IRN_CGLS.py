

# %reset -f
# %clear
import taichi as ti
import numpy as np
import os
from crip.io import imwriteRaw
from run_mgfbp import *
from run_mgfpj_v3 import *
import gc


def run_mgfbp_ir(file_path):
    ti.reset()
    ti.init(arch = ti.gpu)
    print('Performing Iterative Recon from MandoCT-Taichi (ver 0.3) ...')
    # this version simplfied some expressions
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
        print(f"Time cost：{execution_time:.3} sec ({execution_time/fbp.num_iter_runned/fbp.file_processed_count:.3} sec per iteration). \n")
    else:
        print(f"\nWarning: Did not find files like {fbp.input_files_pattern:s} in {fbp.input_dir:s}.")
        print("No images are reconstructed!")
    del fbp #delete the fbp object
    return img_recon

# inherit a class from Mgfbp
@ti.data_oriented
class Mgfbp_ir(Mgfpj_v3):
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
        
        self.img_fp_effective_map_taichi_single_view = ti.field(dtype=ti.f32, shape=(
            self.dect_elem_count_vertical, self.dect_elem_count_horizontal))
 
        self.img_d = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
        self.img_r = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)
 
        self.img_bp_fp_d = np.zeros((self.img_dim_z,self.img_dim, self.img_dim),dtype=np.float32)

        self.sgm_total_pixel_count = self.dect_elem_count_vertical_actual*self.view_num*self.dect_elem_count_horizontal
        self.img_total_pixel_count = self.img_dim_z * self.img_dim * self.img_dim
        self.pixel_count_ratio = float( self.sgm_total_pixel_count/self.img_total_pixel_count)
        
        self.img_x_truncation_flag_taichi = ti.field(dtype=ti.f32, shape=(self.img_dim_z, self.img_dim, self.img_dim))
        self.loss = np.empty(0)
        
        self.img_fp_effective_map = np.zeros_like(self.img_sgm)
        
        self.num_iter_runned = 0
   
    def MainFunction(self,img_recon_seed):
        #Main function for reconstruction
        if not self.bool_uneven_scan_angle:
            self.GenerateAngleArray(
                self.view_num, self.img_rot, self.total_scan_angle, self.array_angle_taichi)
        self.GenerateDectPixPosArrayFPJ(self.dect_elem_count_vertical, - self.dect_elem_height, self.dect_offset_vertical, self.array_v_taichi)
        self.GenerateDectPixPosArrayFPJ(self.dect_elem_count_horizontal*self.oversample_size,-self.dect_elem_width/self.oversample_size,
                                     -self.dect_offset_horizontal, self.array_u_taichi)
        self.file_processed_count = 0; # record the number of files processed
        for file in os.listdir(self.input_dir):
            if re.match(self.input_files_pattern, file):#match the file pattern
                if self.ReadSinogram(file):
                    self.file_processed_count += 1 
                    print('Reconstructing %s ...' % self.input_path)
                    
                    if self.convert_to_HU:
                          img_recon_seed = (img_recon_seed/1000 + 1) * self.water_mu
                    self.img_x = img_recon_seed
                    
                    self.img_fp_effective_map = self.GenEffectiveMapForwardProjection(self.img_x)
                    #imwriteRaw(self.ForwardProjection(self.img_x).transpose(1,0,2)\
                    #- (self.img_sgm * self.img_fp_effective_map).transpose(1,0,2), "img_fp_x.raw")
                    
                    #P^T b
                    self.img_bp_b = self.BackProjection(self.img_sgm)
                    
                    WR = np.ones_like(self.img_x)
                    self.img_x = self.TikhonovSol(self.img_x,WR,-1)
                    self.SaveLossValAndPlot()
                    for irn_iter_idx in range(self.num_irn_iter):  
                        WR = self.GenerateWR(self.img_x,self.beta_tv)
                        self.img_x = self.TikhonovSol(self.img_x,WR,irn_iter_idx)
                        self.SaveLossValAndPlot()
                        self.SaveReconImg()
                        

                    print('\nSaving to %s !' % self.output_path)
                    self.SaveReconImg()
        return self.img_recon #函数返回重建图
    
    def SaveLossValAndPlot(self):
        loss_val = self.LossValCalc()
        self.loss = np.append(self.loss,loss_val)
        plt.semilogy(range(len(self.loss)),(self.loss))
        plt.show()
    
    def LossValCalc(self):
        return 0.5 * np.sum((self.ForwardProjection(self.img_x) - self.img_sgm*self.img_fp_effective_map)**2)/self.sgm_total_pixel_count\
            + self.coef_lambda *  self.TVPenaltyVal(self.img_x) / self.img_total_pixel_count
    
    def TikhonovSol(self,img_seed,WR,irn_idx):
        self.img_bp_fp_x = self.ForwardProjectionAndBackProjection(img_seed)
        imwriteTiff(self.img_bp_fp_x.transpose(1,0,2),'img_bp_fp_x.tif')
        #imwriteRaw(self.img_bp_fp_x, "img_bp_fp_x.raw")
        img_output = img_seed
        self.img_d = self.img_bp_b - self.img_bp_fp_x - self.coef_lambda * self.Dt_W_D(self.img_x, WR) *self.pixel_count_ratio
        imwriteTiff(self.img_d.transpose(1,0,2),'img_d.tif')
        #imwriteRaw(self.img_d, "img_d.raw")
        self.img_r = self.img_d

        for iter_idx in range(self.num_iter):
            #P^T P d
            self.img_bp_fp_d = self.ForwardProjectionAndBackProjection(self.img_d)
            self.img_bp_fp_d = self.img_bp_fp_d  + self.coef_lambda * self.Dt_W_D(self.img_d, WR) * self.pixel_count_ratio
            r_l2_norm = np.sum(np.multiply(self.img_r,self.img_r))
            alpha = r_l2_norm / np.sum(np.multiply(self.img_d, self.img_bp_fp_d))  
            delta_img_x =  np.multiply(alpha, self.img_d) 
            delta_img_x_max_hu = np.max(abs(delta_img_x))/ self.water_mu * 1000
            
            str_0 = 'Reweight index: %4d/%4d, ' % (irn_idx+1, self.num_irn_iter)
            str_1 = 'Iterative index: %4d/%4d, max update value: %6.2f HU' % (iter_idx+1, self.num_iter,delta_img_x_max_hu)
            print('\r' +str_0 + str_1, end='') 

            img_output = img_output + delta_img_x
            self.img_x = img_output ###
            self.SaveReconImg() ###
            self.img_r = self.img_r - np.multiply(alpha, self.img_bp_fp_d)
            
            beta = np.sum(np.multiply(self.img_r, self.img_r)) / r_l2_norm
            self.img_d = self.img_r + beta * self.img_d
            

            if self.num_iter_runned%3==0:
                if self.convert_to_HU:
                    plt.figure(dpi=300)
                    plt.imshow((img_output[:,:,int(round(self.img_dim/2))]/ self.water_mu - 1)*1000,cmap = 'gray',vmin = -50, vmax = 50)
                    plt.show()
                    plt.figure(dpi=300)
                    plt.imshow((img_output[int(round(self.img_dim_z/2)),:,:]/ self.water_mu - 1)*1000,cmap = 'gray',vmin = -50, vmax = 50)
                    #plt.imshow((self.img_x[0,:,:]/ self.water_mu - 1)*1000,cmap = 'gray',vmin = -30, vmax = 100)
                    plt.show()
                    
                    self.SaveLossValAndPlot()
                    
        return img_output
    
    def GenerateWR(self, img_x, beta_tv):
        ux = np.zeros_like(img_x)
        uy = np.zeros_like(img_x)
        uz = np.zeros_like(img_x)
        ux[:, 0:-1, :]= img_x[:,1:,:] - img_x[:,:-1,:]
        uy[:, :, 0:-1]= img_x[:,:,1:] - img_x[:,:,:-1]
        if img_x.shape[0]>2:
            uz[0:-1, :, :]= img_x[1:,:,:] - img_x[:-1,:,:]
        WR = 1 / np.sqrt(ux**2 +uy ** 2+uz ** 2 + self.beta_tv) 
        return WR
    
    def TVPenaltyVal(self,img_x):
        ux = np.zeros_like(img_x)
        uy = np.zeros_like(img_x)
        uz = np.zeros_like(img_x)
        ux[:, 0:-1, :]= img_x[:, 1: ,:] - img_x[:, :-1 ,:]
        uy[:, :, 0:-1]= img_x[:, :, 1:] - img_x[:, : , :-1]
        if img_x.shape[0]>2:
            uz[0:-1, :, :]= img_x[1:,:,:] - img_x[:-1,:,:]
        output = np.sum(np.sqrt(ux * ux + uy * uy + uz * uz + self.beta_tv))
        return output
    
    def TikhonovPenaltyVal(self,img_x,WR):
        ux = np.zeros_like(img_x)
        uy = np.zeros_like(img_x)
        uz = np.zeros_like(img_x)
        ux[:, 0:-1, :]= img_x[:, 1: ,:] - img_x[:, :-1 ,:]
        uy[:, :, 0:-1]= img_x[:, :, 1:] - img_x[:, : , :-1]
        if img_x.shape[0]>2:
            uz[0:-1, :, :]= img_x[1:,:,:] - img_x[:-1,:,:]
        output = np.sum(ux * WR * ux + uy * WR * uy + uz * WR * uz)
        return output
    
    def Dt_W_D(self, img_x, WR):
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
    
    def ForwardProjectionAndBackProjection(self,img_x):
        img_bp_fp_x = np.zeros_like(img_x)
        self.img_bp_fp_x_taichi.from_numpy(np.zeros_like(img_x))
        self.img_x_taichi.from_numpy(img_x)
        self.img_x_truncation_flag_taichi.from_numpy(np.ones_like(self.img_x))
        for view_idx in range(self.view_num):
            self.img_fp_effective_map_taichi_single_view.from_numpy(self.img_fp_effective_map[:,view_idx,:])
            self.img_fp_x_taichi_single_view.from_numpy(np.zeros((self.dect_elem_count_vertical_actual, self.dect_elem_count_horizontal)))
            self.ForwardProjectionBilinear(self.img_x_taichi, self.img_fp_x_taichi_single_view, self.array_u_taichi,
                                            self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                            self.dect_elem_count_horizontal,
                                            self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                            self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                            self.helical_scan, self.helical_pitch, view_idx, self.fpj_step_size,
                                            self.img_center_x, self.img_center_y, self.array_img_center_z_taichi, self.curved_dect,
                                            self.matrix_A_each_view_taichi, self.x_s_each_view_taichi, self.bool_apply_pmatrix,\
                                            self.dect_elem_count_vertical_actual, self.dect_elem_vertical_recon_range_begin,self.img_fp_effective_map_taichi_single_view)
            self.BackProjectionPixelDrivenPerView(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                            self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                            self.array_angle_taichi, self.img_rot,self.img_fp_x_taichi_single_view,self.img_bp_fp_x_taichi,\
                            self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                    self.img_center_x,self.img_center_y,self.array_img_center_z_taichi,self.curved_dect,\
                                        self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode,view_idx, self.img_x_truncation_flag_taichi)
        self.SetTruncatedRegionToZero(self.img_bp_fp_x_taichi,self.img_x_truncation_flag_taichi, self.img_dim, self.img_dim_z)
        img_bp_fp_x = self.img_bp_fp_x_taichi.to_numpy()
        self.num_iter_runned += 1
        
        return img_bp_fp_x
    
    
    def BackProjection(self, img_sgm):
        self.img_x_truncation_flag_taichi.from_numpy(np.ones_like(self.img_x) )
        img_bp_b = np.zeros_like(self.img_x)
        self.img_bp_b_taichi.from_numpy(np.zeros_like(self.img_x))
        for view_idx in range(self.view_num):   
            str_2 = 'BP of input sinogram ...' 
            print('\r' + str_2, end='')   
            self.img_sgm_taichi.from_numpy(img_sgm[:,view_idx,:]*self.img_fp_effective_map[:,view_idx,:])
            self.BackProjectionPixelDrivenPerView(self.dect_elem_count_vertical_actual, self.img_dim, self.dect_elem_count_horizontal, \
                            self.view_num, self.dect_elem_width,self.img_pix_size, self.source_isocenter_dis, self.source_dect_dis,self.total_scan_angle,\
                            self.array_angle_taichi, self.img_rot,self.img_sgm_taichi,self.img_bp_b_taichi,\
                            self.array_u_taichi,self.short_scan,self.cone_beam,self.dect_elem_height,\
                                self.array_v_taichi,self.img_dim_z,self.img_voxel_height,\
                                    self.img_center_x,self.img_center_y,self.array_img_center_z_taichi,self.curved_dect,\
                                        self.bool_apply_pmatrix,self.array_pmatrix_taichi, self.recon_view_mode, view_idx, self.img_x_truncation_flag_taichi)
        self.SetTruncatedRegionToZero(self.img_bp_b_taichi,self.img_x_truncation_flag_taichi, self.img_dim, self.img_dim_z)
        img_bp_b = self.img_bp_b_taichi.to_numpy()
        return img_bp_b
    
    def GenEffectiveMapForwardProjection(self,img_x):
        self.img_x_taichi.from_numpy(img_x)
        img_fp_x = np.zeros((self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal), dtype=np.float32)
        for view_idx in range(self.view_num):   
            self.GenEffectiveMapForwardProjectionAgent(self.img_x_taichi, self.img_fp_x_taichi_single_view, self.array_u_taichi,
                                            self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                            self.dect_elem_count_horizontal,
                                            self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                            self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                            self.helical_scan, self.helical_pitch, view_idx, self.fpj_step_size,
                                            self.img_center_x, self.img_center_y, self.array_img_center_z_taichi, self.curved_dect,
                                            self.matrix_A_each_view_taichi, self.x_s_each_view_taichi, self.bool_apply_pmatrix,\
                                            self.dect_elem_count_vertical_actual, self.dect_elem_vertical_recon_range_begin)
            img_fp_x[:,view_idx,:] = self.img_fp_x_taichi_single_view.to_numpy()   
        return img_fp_x
    
    def ForwardProjection(self,img_x):
        self.img_x_taichi.from_numpy(img_x)
        img_fp_x = np.zeros((self.dect_elem_count_vertical_actual, self.view_num, self.dect_elem_count_horizontal), dtype=np.float32)
        for view_idx in range(self.view_num): 
            self.img_fp_effective_map_taichi_single_view.from_numpy(self.img_fp_effective_map[:,view_idx,:])
            self.img_fp_x_taichi_single_view.from_numpy(np.zeros((self.dect_elem_count_vertical_actual, self.dect_elem_count_horizontal)))
            self.ForwardProjectionBilinear(self.img_x_taichi, self.img_fp_x_taichi_single_view, self.array_u_taichi,
                                            self.array_v_taichi, self.array_angle_taichi, self.img_dim, self.img_dim_z,
                                            self.dect_elem_count_horizontal,
                                            self.dect_elem_count_vertical, self.view_num, self.img_pix_size, self.img_voxel_height,
                                            self.source_isocenter_dis, self.source_dect_dis, self.cone_beam,
                                            self.helical_scan, self.helical_pitch, view_idx, self.fpj_step_size,
                                            self.img_center_x, self.img_center_y, self.array_img_center_z_taichi, self.curved_dect,
                                            self.matrix_A_each_view_taichi, self.x_s_each_view_taichi, self.bool_apply_pmatrix,\
                                            self.dect_elem_count_vertical_actual, self.dect_elem_vertical_recon_range_begin,self.img_fp_effective_map_taichi_single_view)
            img_fp_x[:,view_idx,:] = self.img_fp_x_taichi_single_view.to_numpy()   
        return img_fp_x
    
    @ti.kernel
    def BackProjectionPixelDrivenPerView(self, dect_elem_count_vertical_actual:ti.i32, img_dim:ti.i32, dect_elem_count_horizontal:ti.i32, \
                                  view_num:ti.i32, dect_elem_width:ti.f32,\
                                  img_pix_size:ti.f32, source_isocenter_dis:ti.f32, source_dect_dis:ti.f32,total_scan_angle:ti.f32,\
                                      array_angle_taichi:ti.template(),img_rot:ti.f32,img_sgm_filtered_taichi:ti.template(),img_recon_taichi:ti.template(),\
                                          array_u_taichi:ti.template(), short_scan:ti.i32,cone_beam:ti.i32,dect_elem_height:ti.f32,\
                                              array_v_taichi:ti.template(),img_dim_z:ti.i32,img_voxel_height:ti.f32, \
                                                  img_center_x:ti.f32,img_center_y:ti.f32,array_img_center_z_taichi:ti.template(),curved_dect:ti.i32,\
                                                      bool_apply_pmatrix:ti.i32, array_pmatrix_taichi:ti.template(), recon_view_mode: ti.i32, view_idx:ti.i32,\
                                                          img_x_truncation_flag_taichi:ti.template()):
        
        
        for i_x, i_y in ti.ndrange(img_dim, img_dim):
            for i_z in ti.ndrange(img_dim_z):
                x_after_rot = 0.0; y_after_rot = 0.0; x=0.0; y=0.0;z=0.0;
                if recon_view_mode == 1: #axial view (from bottom to top)
                    x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                    y_after_rot = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + img_center_y
                    z = (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + array_img_center_z_taichi[0,view_idx]
                elif recon_view_mode == 2: #coronal view (from fron to back)
                    x_after_rot = img_pix_size * (i_x - (img_dim - 1) / 2.0) + img_center_x
                    z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + array_img_center_z_taichi[0,view_idx]
                    y_after_rot = - (i_z - (img_dim_z - 1) / 2.0) * img_voxel_height + img_center_y
                elif recon_view_mode == 3: #sagittal view (from left to right)
                    z = - img_pix_size * (i_y - (img_dim - 1) / 2.0) + array_img_center_z_taichi[0,view_idx]
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
                           
                if bool_apply_pmatrix == 0:
                    mag_factor = source_dect_dis / pix_to_source_parallel_dis
                    if curved_dect:
                        pix_proj_to_dect_u =source_dect_dis * ti.atan2(-x*ti.sin(angle_this_view_exclude_img_rot)+y*ti.cos(angle_this_view_exclude_img_rot),pix_to_source_parallel_dis)
                    else:
                        pix_proj_to_dect_u = mag_factor * (-x*ti.sin(angle_this_view_exclude_img_rot)+y*ti.cos(angle_this_view_exclude_img_rot))
                    pix_proj_to_dect_u_idx = (pix_proj_to_dect_u - array_u_taichi[0]) / (array_u_taichi[1] - array_u_taichi[0])
                else:
                    mag_factor = 1.0 / (array_pmatrix_taichi[12*view_idx + 8] * x +\
                        array_pmatrix_taichi[12*view_idx + 9] * y +\
                            array_pmatrix_taichi[12*view_idx + 10] * z +\
                                array_pmatrix_taichi[12*view_idx + 11] * 1)
                    pix_proj_to_dect_u_idx = (array_pmatrix_taichi[12*view_idx + 0] * x +\
                        array_pmatrix_taichi[12*view_idx + 1] * y +\
                            array_pmatrix_taichi[12*view_idx + 2] * z +\
                                array_pmatrix_taichi[12*view_idx + 3] * 1) * mag_factor
                if pix_proj_to_dect_u_idx < 0 or  pix_proj_to_dect_u_idx + 1 > dect_elem_count_horizontal - 1:
                    img_x_truncation_flag_taichi[i_z, i_y, i_x] = 0.0 #mark the truncated region
                    # img_recon_taichi[i_z, i_y, i_x] += 0.0
                else:
                    temp_u_idx_floor = int(ti.floor(pix_proj_to_dect_u_idx))
                    ratio_u = pix_proj_to_dect_u_idx - temp_u_idx_floor
                                        
        
                    if cone_beam == True:
                        if bool_apply_pmatrix == 0:
                            pix_proj_to_dect_v = mag_factor * z
                            pix_proj_to_dect_v_idx = (pix_proj_to_dect_v - array_v_taichi[0]) / dect_elem_height \
                                * abs(array_v_taichi[1] - array_v_taichi[0]) / (array_v_taichi[1] - array_v_taichi[0])
                                #abs(array_v_taichi[1] - array_v_taichi[0]) / (array_v_taichi[1] - array_v_taichi[0]) defines whether the first 
                                #sinogram slice corresponds to the top row
                        else:
                            pix_proj_to_dect_v_idx = (array_pmatrix_taichi[12*view_idx + 4] * x +\
                                array_pmatrix_taichi[12*view_idx + 5] * y +\
                                    array_pmatrix_taichi[12*view_idx + 6] * z +\
                                        array_pmatrix_taichi[12*view_idx + 7] * 1) * mag_factor
                                
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
                    else: 
                        val_0 = img_sgm_filtered_taichi[i_z , temp_u_idx_floor]
                        val_1 = img_sgm_filtered_taichi[i_z , temp_u_idx_floor + 1]
                        img_recon_taichi[i_z, i_y, i_x] += ((1 - ratio_u) * val_0 + ratio_u * val_1) * distance_factor * img_pix_size
    
    
    @ti.kernel 
    def SetTruncatedRegionToZero(self,img_recon_taichi:ti.template(),img_x_truncation_flag_taichi:ti.template(),img_dim:ti.i32,img_dim_z:ti.i32):
        for i_x, i_y in ti.ndrange(img_dim, img_dim):
            for i_z in ti.ndrange(img_dim_z):
                img_recon_taichi[i_z,i_x,i_y] *=img_x_truncation_flag_taichi[i_z,i_x,i_y]

       
    
    def SaveReconImg(self):
        self.img_recon = self.img_x
        if self.convert_to_HU:
            self.img_recon = (self.img_recon / self.water_mu - 1)*1000
        if self.output_file_format == 'raw':
            imwriteRaw(self.img_recon,self.output_path,dtype=np.float32)
        elif self.output_file_format == 'tif' or self.output_file_format == 'tiff':
            imwriteTiff(self.img_recon, self.output_path,dtype=np.float32)
    
    @ti.kernel
    def ForwardProjectionBilinear(self, img_image_taichi: ti.template(), img_sgm_large_taichi: ti.template(),
                                  array_u_taichi: ti.template(), array_v_taichi: ti.template(),
                                  array_angle_taichi: ti.template(), img_dim: ti.i32, img_dim_z: ti.i32,
                                  dect_elem_count_horizontal_oversamplesize: ti.i32,
                                  dect_elem_count_vertical: ti.i32, view_num: ti.i32,
                                  img_pix_size: ti.f32, img_voxel_height: ti.f32, source_isocenter_dis: ti.f32,
                                  source_dect_dis: ti.f32, cone_beam: ti.i32, helical_scan: ti.i32, helical_pitch: ti.f32,
                                  angle_idx: ti.i32, fpj_step_size: ti.f32, img_center_x: ti.f32,
                                  img_center_y: ti.f32, array_img_center_z_taichi: ti.template(), curved_dect: ti.i32, matrix_A_each_view_taichi: ti.template(),\
                                  x_s_each_view_taichi: ti.template(), bool_apply_pmatrix: ti.i32, \
                                  dect_elem_count_vertical_actual: ti.i32, dect_elem_vertical_recon_range_begin:ti.i32,\
                                  img_fp_effective_map:ti.template()):

        # This new version of code assumes that the gantry stays stationary
        # while the image object rotates
        # this can simplify the calculation

        # define aliases
        sid = source_isocenter_dis  # alias
        sdd = source_dect_dis  # alias

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
        z_0 = -(img_dim_z - 1.0) / 2.0 * img_voxel_height + array_img_center_z_taichi[0, angle_idx]

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

        #calculate the distance that the gantry moves between adjacent views
        z_dis_per_view = 0.0
        if self.helical_scan:
            total_scan_angle = abs((array_angle_taichi[view_num - 1] - array_angle_taichi[0])) / (view_num - 1) * view_num
            num_laps = total_scan_angle / (PI * 2)
            z_dis_per_view = helical_pitch * (num_laps / view_num) * (abs(
                array_v_taichi[1] - array_v_taichi[0]) * dect_elem_count_vertical) / (sdd / sid)
            #here pitch is calculated from dect_elem_count_vertical, rather than dect_elem_count_vertical_actual

        # number of steps
        count_steps = int(
            ti.floor((l_max - l_min)/(fpj_step_size * voxel_diagonal_size)))

        for u_idx, v_idx in ti.ndrange(dect_elem_count_horizontal_oversamplesize, dect_elem_count_vertical_actual):
            
            if img_fp_effective_map[v_idx + dect_elem_vertical_recon_range_begin, u_idx] == 1:
                #v range from 0 to dect_elem_count_vertical_actual - 1
                #caluclate the position of the detector element
                if self.curved_dect:
                    gamma_prime = ( - array_u_taichi[u_idx]) / sdd #conterclockwise is positive, corresponding to -y direction
                    dect_elem_pos_x = -sdd * ti.cos(gamma_prime) + sid
                    # positive u direction is - y
                    dect_elem_pos_y = -sdd * ti.sin(gamma_prime)#negative gamma_prime corresponds to positive y
                else:
                    dect_elem_pos_x = - (sdd - sid)
                    # positive u direction is - y
                    dect_elem_pos_y = array_u_taichi[u_idx]
                    
                #add this distance to z position to simulate helical scan
                dect_elem_pos_z = array_v_taichi[v_idx] + z_dis_per_view * angle_idx
                # assume that the source and the detector moves upward for a helical scan (pitch>0)
                source_pos_z = z_dis_per_view * angle_idx
                #distance between the source and the detector element
                source_dect_elem_dis = ((dect_elem_pos_x - source_pos_x)**2 + (
                    dect_elem_pos_y - source_pos_y)**2 + (dect_elem_pos_z - source_pos_z)**2) ** 0.5
                #calculate the unit vector of \vec(x_d - x_s)
                unit_vec_lambda_x = (dect_elem_pos_x - source_pos_x) / source_dect_elem_dis
                unit_vec_lambda_y = (dect_elem_pos_y - source_pos_y) / source_dect_elem_dis
                unit_vec_lambda_z = (dect_elem_pos_z - source_pos_z) / source_dect_elem_dis
    
                temp_sgm_val = 0.0
                one_over_mag = 0.0 
                # the inverse of the magnification factor for pmatrix forward projection (s)
                for step_idx in ti.ndrange(count_steps):
                    # we did not use if bool_apply_pmatrix here is because we found this slows downs the computational speed
                    
                    #for pmatrix case
                    #[x,y,z]^T = A * s * [u,v,1]^T + x_s^T
                    one_over_mag = (step_idx * fpj_step_size * voxel_diagonal_size + l_min) / source_dect_elem_dis
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
    
                img_sgm_large_taichi[v_idx + dect_elem_vertical_recon_range_begin, u_idx] = temp_sgm_val
                #incorporate the vertical recon range
        
    @ti.kernel
    def GenEffectiveMapForwardProjectionAgent(self, img_image_taichi: ti.template(), img_sgm_large_taichi: ti.template(),
                                  array_u_taichi: ti.template(), array_v_taichi: ti.template(),
                                  array_angle_taichi: ti.template(), img_dim: ti.i32, img_dim_z: ti.i32,
                                  dect_elem_count_horizontal_oversamplesize: ti.i32,
                                  dect_elem_count_vertical: ti.i32, view_num: ti.i32,
                                  img_pix_size: ti.f32, img_voxel_height: ti.f32, source_isocenter_dis: ti.f32,
                                  source_dect_dis: ti.f32, cone_beam: ti.i32, helical_scan: ti.i32, helical_pitch: ti.f32,
                                  angle_idx: ti.i32, fpj_step_size: ti.f32, img_center_x: ti.f32,
                                  img_center_y: ti.f32, array_img_center_z_taichi: ti.template(), curved_dect: ti.i32, matrix_A_each_view_taichi: ti.template(),\
                                  x_s_each_view_taichi: ti.template(), bool_apply_pmatrix: ti.i32, \
                                  dect_elem_count_vertical_actual: ti.i32, dect_elem_vertical_recon_range_begin:ti.i32):

        # This new version of code assumes that the gantry stays stationary
        # while the image object rotates
        # this can simplify the calculation

        # define aliases
        sid = source_isocenter_dis  # alias
        sdd = source_dect_dis  # alias

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
        z_0 = -(img_dim_z - 1.0) / 2.0 * img_voxel_height + array_img_center_z_taichi[0, angle_idx]

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

        #calculate the distance that the gantry moves between adjacent views
        z_dis_per_view = 0.0
        if self.helical_scan:
            total_scan_angle = abs((array_angle_taichi[view_num - 1] - array_angle_taichi[0])) / (view_num - 1) * view_num
            num_laps = total_scan_angle / (PI * 2)
            z_dis_per_view = helical_pitch * (num_laps / view_num) * (abs(
                array_v_taichi[1] - array_v_taichi[0]) * dect_elem_count_vertical) / (sdd / sid)
            #here pitch is calculated from dect_elem_count_vertical, rather than dect_elem_count_vertical_actual

        # number of steps
        count_steps = int(
            ti.floor((l_max - l_min)/(fpj_step_size * voxel_diagonal_size)))

        for u_idx, v_idx in ti.ndrange(dect_elem_count_horizontal_oversamplesize, dect_elem_count_vertical_actual):
            img_sgm_large_taichi[v_idx + dect_elem_vertical_recon_range_begin, u_idx] = 1.0
            #v range from 0 to dect_elem_count_vertical_actual - 1
            #caluclate the position of the detector element
            if self.curved_dect:
                gamma_prime = ( - array_u_taichi[u_idx]) / sdd #conterclockwise is positive, corresponding to -y direction
                dect_elem_pos_x = -sdd * ti.cos(gamma_prime) + sid
                # positive u direction is - y
                dect_elem_pos_y = -sdd * ti.sin(gamma_prime)#negative gamma_prime corresponds to positive y
            else:
                dect_elem_pos_x = - (sdd - sid)
                # positive u direction is - y
                dect_elem_pos_y = array_u_taichi[u_idx]
                
            #add this distance to z position to simulate helical scan
            dect_elem_pos_z = array_v_taichi[v_idx] + z_dis_per_view * angle_idx
            # assume that the source and the detector moves upward for a helical scan (pitch>0)
            source_pos_z = z_dis_per_view * angle_idx
            #distance between the source and the detector element
            source_dect_elem_dis = ((dect_elem_pos_x - source_pos_x)**2 + (
                dect_elem_pos_y - source_pos_y)**2 + (dect_elem_pos_z - source_pos_z)**2) ** 0.5
            #calculate the unit vector of \vec(x_d - x_s)
            unit_vec_lambda_x = (dect_elem_pos_x - source_pos_x) / source_dect_elem_dis
            unit_vec_lambda_y = (dect_elem_pos_y - source_pos_y) / source_dect_elem_dis
            unit_vec_lambda_z = (dect_elem_pos_z - source_pos_z) / source_dect_elem_dis

            temp_sgm_val = 0.0
            one_over_mag = 0.0 
            # the inverse of the magnification factor for pmatrix forward projection (s)
            for step_idx in ti.ndrange(count_steps):
                # we did not use if bool_apply_pmatrix here is because we found this slows downs the computational speed
                
                #for pmatrix case
                #[x,y,z]^T = A * s * [u,v,1]^T + x_s^T
                one_over_mag = (step_idx * fpj_step_size * voxel_diagonal_size + l_min) / source_dect_elem_dis
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
                    if self.cone_beam:
                        z_idx = int(ti.floor((z_rot - z_0) / img_voxel_height))

                        if ((x_idx - img_dim/2.0)**2 + (y_idx - img_dim/2.0)**2) < (img_dim/2.0)**2:

                            z_idx = int(ti.floor((z_rot - z_0) / img_voxel_height))

                            if z_idx < 0 or z_idx + 1 >= img_dim_z:
                                img_sgm_large_taichi[v_idx + dect_elem_vertical_recon_range_begin, u_idx] = 0.0
        
     
