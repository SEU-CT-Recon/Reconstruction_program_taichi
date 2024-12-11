
# %reset -f
# %clear
import taichi as ti
import numpy as np
import os
from crip.io import imwriteRaw
from run_mgfbp import *
from run_mgfbp_iterative_recon_IRN_CGLS import *
import gc


def run_mgfbp_ir(file_path):
    ti.reset()
    ti.init(arch = ti.gpu)
    print('Performing Iterative Recon from MandoCT-Taichi (PICCS ver 0.2) ...')
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
    start_time = time.time()
    print("\nPerform Iterative Recon ...")
    fbp = Mgfbp_ir_piccs(config_dict) #将config_dict数据以字典的形式送入对象中
    img_recon = fbp.MainFunction()#use the seed to initialize the iterative process
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
class Mgfbp_ir_piccs(Mgfbp_ir):
    def __init__(self,config_dict):
        super(Mgfbp_ir_piccs,self).__init__(config_dict)
        #read prior image
        
        
        if 'Alpha' in config_dict:
            self.coef_alpha = config_dict['Alpha']
            if (not isinstance(self.coef_alpha,int) and not isinstance(self.coef_alpha,float)) or self.coef_alpha < 0.0:
                print("ERROR: Alpha must be a positive number!")
                sys.exit()
        else:
            self.coef_alpha = 1e-5
            print("Warning: Did not find Alpha! Use default value 1e-5. ")
            
        self.img_prior = imreadRaw(config_dict['PriorImageFile'], width = self.img_dim, height = self.img_dim, nSlice = self.img_dim_z)
        self.img_prior = self.img_prior.reshape((self.img_dim_z,self.img_dim,self.img_dim))
        if self.convert_to_HU:
              self.img_prior = (self.img_prior/1000 + 1 ) * self.water_mu
        
   
    def MainFunction(self):
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

                    if self.convert_to_HU:
                          self.img_prior = (self.img_prior/1000.0 + 1 ) * self.water_mu
                          
                    self.img_fp_effective_map = self.GenEffectiveMapForwardProjection(self.img_x)
                    self.img_x = self.img_prior

                    #P^T b
                    self.img_bp_b = self.BackProjection(self.img_sgm)

                    WR = np.ones_like(self.img_x)
                    WR_prior = np.ones_like(self.img_x)
                    self.img_x = self.TikhonovSol(self.img_x,WR,WR_prior,-1)
                    self.SaveLossValAndPlot()
                    for irn_iter_idx in range(self.num_irn_iter):  
                        WR = self.GenerateWR(self.img_x,self.beta_tv)
                        WR_prior = self.GenerateWR(self.img_x - self.img_prior, self.beta_tv)
                        self.img_x = self.TikhonovSol(self.img_x,WR,WR_prior,irn_iter_idx)
                        if self.num_iter_runned%5==0:
                            self.SaveLossValAndPlot()
                        self.SaveReconImg()
                        

                    print('\nSaving to %s !' % self.output_path)
                    self.SaveReconImg()
        return self.img_recon #函数返回重建图
    
    def TikhonovSol(self,img_seed,WR,WR_prior, irn_idx):
        img_output = img_seed
        self.img_d = self.img_bp_b + self.coef_lambda * self.coef_alpha*self.pixel_count_ratio * self.Dt_W_D(self.img_prior, WR_prior)\
            - self.FunctionFx(img_seed,WR, WR_prior)
        self.img_r = self.img_d

        for iter_idx in range(self.num_iter):
            #P^T P d
            self.img_bp_fp_d =  self.FunctionFx(self.img_d, WR, WR_prior)
            r_l2_norm = np.sum(np.multiply(self.img_r,self.img_r))
            alpha = r_l2_norm / np.sum(np.multiply(self.img_d, self.img_bp_fp_d))  
            delta_img_x =  np.multiply(alpha, self.img_d) 
            delta_img_x_max_hu = np.max(abs(delta_img_x))/ self.water_mu * 1000
            
            str_0 = 'Reweight index: %4d/%4d, ' % (irn_idx+1, self.num_irn_iter)
            str_1 = 'Iterative index: %4d/%4d, max update value: %6.2f HU' % (iter_idx+1, self.num_iter,delta_img_x_max_hu)
            print('\r' +str_0 + str_1, end='') 

            img_output = img_output + delta_img_x

            self.img_r = self.img_r - np.multiply(alpha, self.img_bp_fp_d)
            
            beta = np.sum(np.multiply(self.img_r, self.img_r)) / r_l2_norm
            self.img_d = self.img_r + beta * self.img_d
            

            if self.num_iter_runned%1==0:
                if self.convert_to_HU:
                    plt.figure(dpi=300)
                    #plt.imshow((self.img_x[:,:,int(round(self.img_dim/2))]/ self.water_mu - 1)*1000,cmap = 'gray',vmin = -50, vmax = 100)
                    plt.imshow((img_output[int(round(self.img_dim_z/2)),:,:]/ self.water_mu - 1)*1000,cmap = 'gray',vmin = -50, vmax = 100)
                    #plt.imshow((self.img_x[0,:,:]/ self.water_mu - 1)*1000,cmap = 'gray',vmin = -30, vmax = 100)
                    plt.show()
        return img_output
    
    def FunctionFx(self,img_x, WR, WR_prior):
        img_output = self.ForwardProjectionAndBackProjection(img_x) + self.coef_lambda* (1-self.coef_alpha) * self.Dt_W_D(img_x, WR) *self.pixel_count_ratio\
            + self.coef_lambda* self.coef_alpha * self.Dt_W_D(img_x, WR_prior) *self.pixel_count_ratio
        return img_output
    
    def SaveLossValAndPlot(self):
        loss_val = self.LossValCalc()
        self.loss = np.append(self.loss,loss_val)
        plt.semilogy(range(len(self.loss)),(self.loss))
        plt.show()
    
    def LossValCalc(self):
        return 0.5 * np.sum((self.ForwardProjection(self.img_x) - self.img_sgm * self.img_fp_effective_map)**2)/self.sgm_total_pixel_count\
            + self.coef_lambda *(1-self.coef_alpha)*  self.TVPenaltyVal(self.img_x) / self.img_total_pixel_count\
                + self.coef_lambda *self.coef_alpha*  self.TVPenaltyVal(self.img_x -self.img_prior) / self.img_total_pixel_count
    
        
     
