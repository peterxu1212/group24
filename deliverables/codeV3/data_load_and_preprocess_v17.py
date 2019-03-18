# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:32:13 2019

@author: PeterXu
"""


#import scipy

#from scipy.misc import imsave

import numpy as np
 # linear algebra

import pandas as pd 
# data processing, CSV file I/O (e.g. pd.read_csv)


#from skimage import io, measure, morphology

from skimage import img_as_ubyte

from skimage.filters import threshold_otsu

from skimage.morphology import closing, square, disk, remove_small_objects, erosion, dilation

#import skimage.morphology as sm

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

from skimage import exposure

#from skimage.measure import label, regionprops
from skimage.color import label2rgb

from skimage.transform import resize, rescale

#from skimage.io import imsave
from scipy.misc import imsave as misc_imsave


import os


import matplotlib.pyplot as plt

import matplotlib.patches as mpatches


import logging
import logging.config


import json



import time


i_too_many_regions_threshold = 3


i_try_erosion_max_wh_threshold = 28


def apply_rmv_sm_obj(in_arr_image, i_threshold):
    
    
    #print("\n apply_rmv_sm_obj: in_arr_image.dtype = ", in_arr_image.dtype)
    
    labeled_image, i_label_num = label(in_arr_image, return_num=True)
    #print("\n apply_rmv_sm_obj i_label_num = ", i_label_num)
    
    #regions = regionprops(labeled_image)
    #print("len of regions : ", len(regions))

    out_arr_image_mask = remove_small_objects(labeled_image, min_size=i_area_threshold, connectivity=1)
    #out_arr_image[out_arr_image > 0] = 255
    
    
    #print("\n apply_rmv_sm_obj: out_arr_image_mask.dtype = ", out_arr_image_mask.dtype, out_arr_image_mask.shape)
    
    in_arr_image[out_arr_image_mask == False] = i_threshold
    
    out_arr_image = np.copy(in_arr_image)
    
    #out_arr_image = in_arr_image[out_arr_image_mask]
    
    #print("\n apply_rmv_sm_obj: out_arr_image.dtype = ", out_arr_image.dtype)
        
    out_arr_image = img_as_ubyte(out_arr_image)
    
    return out_arr_image


def img_info_analysis(in_arr_image):
        
    labeled_image, i_ideal_label_num = label(in_arr_image, return_num=True)
    
    
    img_region_set = []
    for region in regionprops(labeled_image):
        # take regions with large enough areas
        if region.area >= i_area_threshold:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            
            img_region_item = {}
                        
            region_width = maxc - minc
            region_height = maxr - minr
            
            max_of_wh = max(region_width, region_height)
            
            img_region_item['max_of_wh'] = max_of_wh
            img_region_item['region_area'] = int(region.area)            
            
            img_region_set.append(img_region_item)

    i_ideal_region_count = 0
    
    i_ideal_least_area = 0
    
    i_ideal_least_max_of_wh = 0
    
    len_irs = len(img_region_set)
    i_ideal_region_count = len_irs
    
    if len_irs > 0:
        
        img_region_set_sorted = sorted(img_region_set, key=lambda x: (x['max_of_wh']), reverse=False)
        i_ideal_least_max_of_wh = img_region_set_sorted[0]['max_of_wh']
                
        img_region_set_sorted = sorted(img_region_set, key=lambda x: (x['region_area']), reverse=False)
        i_ideal_least_area = img_region_set_sorted[0]['region_area']
        
         
        
    return i_ideal_region_count, i_ideal_least_area, i_ideal_least_max_of_wh


def img_process_to_remove_bk(in_arr_image, in_i_index, in_str_folder_prefix, in_b_train_or_test):    
    
    #logger.info("img_process_to_remove_bk: in_i_index = %d, in_b_train_or_test %d ", in_i_index, in_b_train_or_test)
     
    i_label1_num = 0
    i_label2_num = 0
    i_label3_num = 0
    i_label4_num = 0
    i_label5_num = 0
    i_label6_num = 0
    
    i_otsu_threshold_s1 = 0
    i_otsu_threshold_s2 = 0
    i_otsu_threshold_s3 = 0
    
    i_out_bkr_stage_count = 0
    
    
    out_arr_image = np.copy(in_arr_image)    
       
   
    b_do_practical = True
    
    
    if b_do_practical:
    
        #out_arr_image = np.copy(tmp_arr_image)
        
        
        # practical way, using filters.threshold_otsu, probably with multi-stage   
        
        out_fn = in_str_folder_prefix + "_bkr_debug" + "/" + str(in_i_index) + "_start" + ".png"
    
        #misc_imsave(out_fn, tmp_ideal_arr_image)
        plt.imsave(out_fn, in_arr_image)
        
        flt_threshold_s1_image = np.copy(in_arr_image)
        i_otsu_threshold_s1 = threshold_otsu(flt_threshold_s1_image)   #返回一个阈值
        #print("\n i_otsu_threshold_s1 = ", i_otsu_threshold_s1)
    
    
        flt_threshold_s1_image[flt_threshold_s1_image <= i_otsu_threshold_s1] = i_otsu_threshold_s1
        
        
        out_fn = in_str_folder_prefix + "_bkr_debug" + "/" + str(in_i_index) + "_s1" + ".png"
    
        #misc_imsave(out_fn, tmp_ideal_arr_image)
        plt.imsave(out_fn, flt_threshold_s1_image)
    
        
        #flt_threshold_s1_image = img_as_ubyte(flt_threshold_s1_image)
        labeled_image, i_label1_num = label(flt_threshold_s1_image, return_num=True)
    
        #print("\n i_label1_num = ", i_label1_num)
        
        if i_label1_num >= i_too_many_regions_threshold:
            #apply second stage
            
            
            flt_threshold_s2_image = np.copy(flt_threshold_s1_image)
    
            #b_ilc = exposure.is_low_contrast(flt_threshold_s2_image)
            #print("\n b_ilc = ", b_ilc)
    
            i_otsu_threshold_s2 = threshold_otsu(flt_threshold_s2_image)
            #print("\n i_otsu_threshold_s2 = ", i_otsu_threshold_s2)
    
            if i_otsu_threshold_s2 <= i_otsu_threshold_s1:
                #logger.info("img_process_to_remove_bk warning: in_i_index = %d, i_otsu_threshold_s2 %d <= i_otsu_threshold_s1 %d ", in_i_index, i_otsu_threshold_s2, i_otsu_threshold_s1)
            
                flt_threshold_s1_image = apply_rmv_sm_obj(flt_threshold_s1_image, i_otsu_threshold_s1)
                
                flt_threshold_s1_image[flt_threshold_s1_image == i_otsu_threshold_s1] = 0
                #flt_threshold_s1_image[flt_threshold_s1_image > i_otsu_threshold_s1] = 255
                            
                i_out_bkr_stage_count = 1
                out_arr_image = np.copy(flt_threshold_s1_image)
            
            else:
                i_out_bkr_stage_count = 2
                
                flt_threshold_s2_image[flt_threshold_s2_image <= i_otsu_threshold_s2] = i_otsu_threshold_s2
                #flt_threshold_s2_image[flt_threshold_s2_image > to2_edges9] = 255
    
                
                out_fn = in_str_folder_prefix + "_bkr_debug" + "/" + str(in_i_index) + "_s2" + ".png"
    
                #misc_imsave(out_fn, tmp_ideal_arr_image)
                plt.imsave(out_fn, flt_threshold_s2_image)
                
                labeled_image, i_label2_num = label(flt_threshold_s2_image, return_num=True)
    
                
                if i_label2_num >= i_too_many_regions_threshold:
                    #apply third stage
                    
                    
                    flt_threshold_s3_image = np.copy(flt_threshold_s2_image)
            
                    #b_ilc = exposure.is_low_contrast(flt_threshold_s2_image)
                    #print("\n b_ilc = ", b_ilc)
            
                    i_otsu_threshold_s3 = threshold_otsu(flt_threshold_s3_image)
                    
                    if i_otsu_threshold_s3 <= i_otsu_threshold_s2:
                        #logger.info("img_process_to_remove_bk warning: in_i_index = %d, i_otsu_threshold_s3 %d <= i_otsu_threshold_s2 %d ", in_i_index, i_otsu_threshold_s3, i_otsu_threshold_s2)
                        flt_threshold_s2_image = apply_rmv_sm_obj(flt_threshold_s2_image, i_otsu_threshold_s2)
                
                    
                        flt_threshold_s2_image[flt_threshold_s2_image == i_otsu_threshold_s2] = 0
                        #flt_threshold_s1_image[flt_threshold_s1_image > i_otsu_threshold_s1] = 255
                        
                        
                                    
                        i_out_bkr_stage_count = 2
                        out_arr_image = np.copy(flt_threshold_s2_image)
                    
                    else:
                        
                        
                        i_out_bkr_stage_count = 3
                
                        flt_threshold_s3_image[flt_threshold_s3_image <= i_otsu_threshold_s3] = i_otsu_threshold_s3
                        #flt_threshold_s2_image[flt_threshold_s2_image > to2_edges9] = 255
                        
                        
                        
                        out_fn = in_str_folder_prefix + "_bkr_debug" + "/" + str(in_i_index) + "_s3" + ".png"
    
                        #misc_imsave(out_fn, tmp_ideal_arr_image)
                        plt.imsave(out_fn, flt_threshold_s3_image)
                
                        
                        
                        labeled_image, i_label3_num = label(flt_threshold_s3_image, return_num=True)
                        
                        #print("\n i_label2_num = ", i_label2_num)
                        
                        
                        
                        #print("\n before remove sm obj flt_threshold_s2_image.dtype = ", flt_threshold_s2_image.dtype, flt_threshold_s2_image.shape) 
                        
                        flt_threshold_s3_image = apply_rmv_sm_obj(flt_threshold_s3_image, i_otsu_threshold_s3)
                        
                        labeled_image, i_label4_num = label(flt_threshold_s3_image, return_num=True)
    
                        #print("\n i_label3_num = ", i_label3_num)
                        
                        flt_threshold_s3_image[flt_threshold_s3_image == i_otsu_threshold_s3] = 0 

                        out_arr_image = np.copy(flt_threshold_s3_image)
                  
                else:
                
                
                    flt_threshold_s2_image[flt_threshold_s2_image == i_otsu_threshold_s2] = 0    
                
                    out_arr_image = np.copy(flt_threshold_s2_image)
            
            #pass
        else:
            flt_threshold_s1_image[flt_threshold_s1_image == i_otsu_threshold_s1] = 0
            #flt_threshold_s1_image[flt_threshold_s1_image > i_otsu_threshold_s1] = 255
            
                    
            i_out_bkr_stage_count = 1
            out_arr_image = np.copy(flt_threshold_s1_image)
        
       
        out_arr_image = img_as_ubyte(out_arr_image)
    
        out_fnp = in_str_folder_prefix + "_bkr_practical" + "/" + str(in_i_index) + ".png"
        misc_imsave(out_fnp, out_arr_image)
    
    
    else:
        pass

    
    
    labeled_image, i_label6_num = label(out_arr_image, return_num=True)
    
    
    
    out_fn = in_str_folder_prefix + "_bkr_debug" + "/" + str(in_i_index) + "_final" + ".png"
    
    plt.imsave(out_fn, out_arr_image)
    
    return out_arr_image, i_out_bkr_stage_count, i_label1_num, i_label2_num, i_label3_num, i_label4_num, i_label5_num, i_label6_num, i_otsu_threshold_s1, i_otsu_threshold_s2, i_otsu_threshold_s3, i_ideal_region_count, i_ideal_least_area, i_ideal_least_max_of_wh


def img_process_to_reduce_objs(in_arr_image, in_i_index, in_str_folder_prefix, in_b_train_or_test):    
    
    tmp_arr_image = np.copy(in_arr_image)       
    
    # and further remove small objects, only maintain the two largest objects for now    
    #labeled_image, i_label_num = label(tmp_arr_image, return_num=True, connectivity=1)   
    labeled_image, i_label_num = label(tmp_arr_image, return_num=True)   
    
    b_keep_largest_two = True
    
    
    out_arr_image = None
        
    img_region_set = []
    for region in regionprops(labeled_image):
        # take regions with large enough areas
        if region.area >= i_area_threshold:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            
            img_region_item = {}
                        
            region_width = maxc - minc
            region_height = maxr - minr
            
            max_of_wh = max(region_width, region_height)
            
            img_region_item['max_of_wh'] = max_of_wh
            img_region_item['region_obj'] = region
            
            
            img_region_set.append(img_region_item)

    #i_origin_region_count = 0
    b_process_onlyone_or_likely = False
    
    len_irs = len(img_region_set)
    #i_origin_region_count = len_irs
    
    img_region_set_sorted = []
    
    if len_irs > 0:
        
        img_region_set_sorted = sorted(img_region_set, key=lambda x: (x['max_of_wh']), reverse=True)
            
        tmp_image = np.copy(labeled_image)
        if len_irs == 1:
            logger.info("img_process_to_reduce_objs alert: only 1 region in the image which is big enough for in_i_index = %d ", in_i_index)
            
            #to double check for the threshold, whether they are indeed only one, or one with very small but still valid other numbers?            
            out_fn = in_str_folder_prefix + "_onlyone" + "/" + str(in_i_index) + ".png"
            misc_imsave(out_fn, tmp_arr_image)
            
            
            img_region_item_0 = img_region_set_sorted[0]        
            region_obj_0 = img_region_item_0['region_obj']
            
            tmp_image[tmp_image != region_obj_0.label] = 0
            tmp_image[tmp_image == region_obj_0.label] = 1
            
            
            tmp_arr_image[tmp_image == 0] = 0
            
            if b_process_onlyone_or_likely:
                #tmp_image = img_as_ubyte(tmp_image)
                tmp_onlyone_img = np.copy(tmp_arr_image)
                #i_region_remained_count = 1
                
                out_fn_psd = in_str_folder_prefix + "_onlyone_processed" + "/" + str(in_i_index) + ".png"
                misc_imsave(out_fn_psd, tmp_onlyone_img)
            
            
        else:
            #len_irs >= 2
            
            if b_keep_largest_two:
                
                img_region_item_0 = img_region_set_sorted[0]        
                region_obj_0 = img_region_item_0['region_obj']
                
                img_region_item_1 = img_region_set_sorted[1]        
                region_obj_1 = img_region_item_1['region_obj']
                
                   
                #only keep these two region, and nullify all the others
                #tmp_image[tmp_image == -1] = 0
                tmp_image[tmp_image == region_obj_0.label] = -1
                tmp_image[tmp_image == region_obj_1.label] = -1
                
                tmp_image[tmp_image != -1] = 0
                tmp_image[tmp_image == -1] = 1
                
                tmp_arr_image[tmp_image == 0] = 0
                
            else:                
            
                if len_irs == 2:
                    img_region_item_0 = img_region_set_sorted[0]        
                    region_obj_0 = img_region_item_0['region_obj']
                    
                    img_region_item_1 = img_region_set_sorted[1]        
                    region_obj_1 = img_region_item_1['region_obj']
                    
                       
                    #only keep these two region, and nullify all the others
                    #tmp_image[tmp_image == -1] = 0
                    tmp_image[tmp_image == region_obj_0.label] = -1
                    tmp_image[tmp_image == region_obj_1.label] = -1
                    
                    tmp_image[tmp_image != -1] = 0
                    tmp_image[tmp_image == -1] = 1
                    
                    tmp_arr_image[tmp_image == 0] = 0
                    
                else:
                    #len_irs >= 3:
                    
                    img_region_item_0 = img_region_set_sorted[0]        
                    region_obj_0 = img_region_item_0['region_obj']
                    
                    img_region_item_1 = img_region_set_sorted[1]        
                    region_obj_1 = img_region_item_1['region_obj']
                    
                    img_region_item_2 = img_region_set_sorted[2]        
                    region_obj_2 = img_region_item_2['region_obj']
                    
                    tmp_image[tmp_image == region_obj_0.label] = -1
                    tmp_image[tmp_image == region_obj_1.label] = -1
                    tmp_image[tmp_image == region_obj_2.label] = -1
                    
                    tmp_image[tmp_image != -1] = 0
                    tmp_image[tmp_image == -1] = 1
                    
                    tmp_arr_image[tmp_image == 0] = 0                
                    
        
        out_arr_image = np.copy(tmp_arr_image)
        
        
    else:
        logger.info("img_process_to_reduce_objs warning: no big enough region exist in the image for in_i_index = %d ", in_i_index)
    
        
    #return out_arr_image, i_origin_region_count
    return out_arr_image, img_region_set_sorted, labeled_image



def do_img_separation(in_arr_image, in_i_index, in_str_folder_prefix):
    
    out_i_separate_success = 0
    
    tmp_arr_image = np.copy(in_arr_image)    
    tmp_arr_image_to_separate = np.copy(in_arr_image)        
    
    tmp_arr_image_eroded = None
        
    labeled_image = None
    
    i_current_region_cnt = 1
    
    
    
    
    out_fn = in_str_folder_prefix + "_spd_debug" + "/" + str(in_i_index) + "_start" + ".png"
    
    plt.imsave(out_fn, in_arr_image)
    
    tmp_arr_image_eroded = erosion(tmp_arr_image_to_separate, square(1))       
    
    labeled_image, i_label_num = label(tmp_arr_image_eroded, return_num=True)        
    if i_label_num == (i_current_region_cnt + 1):
        #success, and separate according to region
        out_i_separate_success = 1
    else:
        
        tmp_arr_image_eroded = erosion(tmp_arr_image_to_separate, disk(1))
        labeled_image, i_label_num = label(tmp_arr_image_eroded, return_num=True)
        if i_label_num == (i_current_region_cnt + 1):
            #success, and separate according to region
            out_i_separate_success = 2
        else:
            
            tmp_arr_image_eroded = erosion(tmp_arr_image_to_separate, square(2))
            
            labeled_image, i_label_num = label(tmp_arr_image_eroded, return_num=True)
            if i_label_num == (i_current_region_cnt + 1):
                #success, and separate according to region
                out_i_separate_success = 3                   
            else:
                tmp_arr_image_eroded = erosion(tmp_arr_image_to_separate, disk(2))
                
                labeled_image, i_label_num = label(tmp_arr_image_eroded, return_num=True)                    
                if i_label_num == (i_current_region_cnt + 1):
                    #success, and separate according to region
                    out_i_separate_success = 4                    
                else:
                    pass                   
        
    if out_i_separate_success == 0:
        logger.info("do_img_separation warning: fail to separate image for in_i_index = %d ", in_i_index)
        
    else:
        #pass    

        
    
        out_fn = in_str_folder_prefix + "_spd_debug" + "/" + str(in_i_index) + "_eroded" + ".png"
    
        plt.imsave(out_fn, tmp_arr_image_eroded)
        
        img_region_set = []
        for region in regionprops(labeled_image):
            # take regions with large enough areas
            if True:
            #if region.area >= i_area_threshold:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                
                img_region_item = {}
                            
                region_width = maxc - minc
                region_height = maxr - minr
                
                max_of_wh = max(region_width, region_height)
                
                img_region_item['max_of_wh'] = max_of_wh
                img_region_item['region_obj'] = region
                                    
                img_region_set.append(img_region_item)

        
        len_irs = len(img_region_set)
        img_region_set_sorted = []

        if len_irs > 0:
            
            img_region_set_sorted = sorted(img_region_set, key=lambda x: (x['max_of_wh']), reverse=True)
                
            #tmp_image = np.copy(labeled_image)                
             
            img_region_item_0 = img_region_set_sorted[0]        
            region_obj_0 = img_region_item_0['region_obj']
               
            labeled_image[labeled_image != region_obj_0.label] = 0
            
            tmp_arr_image_eroded[labeled_image == 0] = 0
            
            
            out_fn = in_str_folder_prefix + "_spd_debug" + "/" + str(in_i_index) + "_cleaned" + ".png"
            
            plt.imsave(out_fn, tmp_arr_image_eroded)
            

            
            tmp_arr_image_recovered = None
            
            if out_i_separate_success == 1:
                
                tmp_arr_image_recovered = dilation(tmp_arr_image_eroded, square(1))
                
            elif out_i_separate_success == 2:
                
                tmp_arr_image_recovered = dilation(tmp_arr_image_eroded, disk(1))
                
            elif out_i_separate_success == 3:
            
                tmp_arr_image_recovered = dilation(tmp_arr_image_eroded, square(2))
                
            elif out_i_separate_success == 4:
        
                tmp_arr_image_recovered = dilation(tmp_arr_image_eroded, disk(2))
                
            else:
                pass
            
            
            
            out_fn = in_str_folder_prefix + "_spd_debug" + "/" + str(in_i_index) + "_recovered" + ".png"
    
            plt.imsave(out_fn, tmp_arr_image_recovered)
            
            tmp_arr_image = np.copy(tmp_arr_image_recovered)

    
        
    out_arr_image = np.copy(tmp_arr_image)
    
    
            
    out_fn = in_str_folder_prefix + "_spd_debug" + "/" + str(in_i_index) + "_final" + ".png"
    
    plt.imsave(out_fn, out_arr_image)
            
    return out_arr_image, out_i_separate_success


def get_grey_scale_value_for_specific_region(tmp_arr_image, region_obj):
    
    i_gc_val = -1
    
    tmp_0_rco = region_obj.coords[0]
    
    i_gc_val = tmp_arr_image[tmp_0_rco[0]][tmp_0_rco[1]]
    
    return i_gc_val
        


def img_process_to_remove_attachment(in_arr_image, in_i_index, in_str_folder_prefix, in_b_train_or_test, in_img_region_set_sorted, in_tmp_labeled_image):    

    
    tmp_arr_image = np.copy(in_arr_image)
    
    #assume that there are only 2 regions at most till this stage, for we have done remove objects to ensure this.    
    i_to_try_separation = 0
    i_separate_success = 0    
    
    i_len_total_regions = len(in_img_region_set_sorted)
    
    img_region_item_largest = in_img_region_set_sorted[0]
    
    if i_len_total_regions == 1:
        
        if img_region_item_largest['max_of_wh'] > i_try_erosion_max_wh_threshold:
        
            i_to_try_separation = 1        
            
            tmp_arr_image_to_separate = np.copy(tmp_arr_image)
            
            region_obj_0 = img_region_item_largest['region_obj']        
            
            i_greyscale_val = get_grey_scale_value_for_specific_region(tmp_arr_image_to_separate, region_obj_0)       
            
            
            tmp_arr_image_to_separate[in_tmp_labeled_image == region_obj_0.label] = 0        
            
            tmp_arr_image_to_for_operation = np.zeros((64, 64), dtype=np.uint8)
            
            tmp_arr_image_to_for_operation[in_tmp_labeled_image == region_obj_0.label] = i_greyscale_val
            
          
            
            tmp_arr_image_separated, i_separate_success = do_img_separation(tmp_arr_image_to_for_operation, in_i_index, in_str_folder_prefix)
            
                            
            labeled_image, i_label_num = label(tmp_arr_image_separated, return_num=True)
            
            img_region_set = []
            for region in regionprops(labeled_image):
                # take regions with large enough areas
                if True:
                #if region.area >= i_area_threshold:
                    # draw rectangle around segmented coins
                    
                    img_region_item = {}
                                    
                    img_region_item['region_obj'] = region
                                        
                    img_region_set.append(img_region_item)
            
            new_img_region_item_0 = img_region_set[0]
            new_region_obj_0 = new_img_region_item_0['region_obj']
                    
            i_greyscale_val = get_grey_scale_value_for_specific_region(tmp_arr_image_separated, new_region_obj_0)
            
            tmp_arr_image_to_separate[labeled_image == new_region_obj_0.label] = i_greyscale_val        
            
            tmp_arr_image = np.copy(tmp_arr_image_to_separate)
        
    else:        
        #i_len_total_regions >= 2
        
        img_region_item_second_largest = in_img_region_set_sorted[1]
          
        if img_region_item_largest['max_of_wh'] > i_try_erosion_max_wh_threshold or img_region_item_second_largest['max_of_wh'] > i_try_erosion_max_wh_threshold:
            
            i_to_try_separation = 1
            tmp_arr_image_to_separate = np.copy(tmp_arr_image)
            
            for orig_img_region_item in (img_region_item_largest, img_region_item_second_largest):
                
                if orig_img_region_item['max_of_wh'] > i_try_erosion_max_wh_threshold:           
        
                    tmp_region_obj = orig_img_region_item['region_obj']                    
        
                    i_greyscale_val = get_grey_scale_value_for_specific_region(tmp_arr_image_to_separate, tmp_region_obj)      
                
                    tmp_arr_image_to_separate[in_tmp_labeled_image == tmp_region_obj.label] = 0        
        
                    tmp_arr_image_to_for_operation = np.zeros((64, 64), dtype=np.uint8)
        
                    tmp_arr_image_to_for_operation[in_tmp_labeled_image == tmp_region_obj.label] = i_greyscale_val
        
                    
                    tmp_arr_image_separated, i_any_separate_success = do_img_separation(tmp_arr_image_to_for_operation, in_i_index, in_str_folder_prefix)
                    
                    if i_any_separate_success > 0:
                        i_separate_success = i_any_separate_success
                    
                                
                    labeled_image, i_label_num = label(tmp_arr_image_separated, return_num=True)
                    
                    img_region_set = []
                    for region in regionprops(labeled_image):
                        # take regions with large enough areas
                        if True:
                            
                            img_region_item = {}
                                            
                            img_region_item['region_obj'] = region
                                                
                            img_region_set.append(img_region_item)
                    
                    new_img_region_item_0 = img_region_set[0]
                    new_region_obj_0 = new_img_region_item_0['region_obj']
                            
                    i_greyscale_val = get_grey_scale_value_for_specific_region(tmp_arr_image_separated, new_region_obj_0)
                    
                    tmp_arr_image_to_separate[labeled_image == new_region_obj_0.label] = i_greyscale_val       
               
            tmp_arr_image = np.copy(tmp_arr_image_to_separate)    
            
        else:
            pass
        
    
    if i_separate_success > 0:
                
        out_fns = in_str_folder_prefix + "_separated" + "/" + str(in_i_index) + ".png"
        misc_imsave(out_fns, tmp_arr_image)
    else:
    
        tmp_arr_image = np.copy(in_arr_image)
        
                
    
    out_arr_image = np.copy(tmp_arr_image)
    
    return out_arr_image, i_to_try_separation, i_separate_success



def op_img(in_i_index, in_str_folder_prefix, in_b_train_or_test=True, b_display=False):
    
    
    #s1
    
    st = time.time()
    

    
    ti_tmp = ""
    
    if in_b_train_or_test:
    
        ti_tmp = train_images_set[in_i_index]
    else:
        ti_tmp = test_images_set[in_i_index]
    
    #np.set_printoptions(threshold=np.inf)
    
    
    
    et = time.time()
    ut = et - st
    
    global total_oi_s1_time
    
    total_oi_s1_time += ut
    
    
    
    
    #s2
    
    st = time.time()
    
    
    flt_ti = np.copy(ti_tmp)
    
    
    flt_ti = flt_ti.astype(np.uint8)
    flt_ti = img_as_ubyte(flt_ti)
    
    
    
    flt_ti, i_bkr_stage_count, i_bkr_label1_num, i_bkr_label2_num, i_bkr_label3_num, i_bkr_label4_num, i_bkr_label5_num, i_bkr_label6_num, i_bkr_otsu_threshold_s1, i_bkr_otsu_threshold_s2, i_bkr_otsu_threshold_s3, i_ideal_region_count, i_ideal_least_area, i_ideal_least_max_of_wh = img_process_to_remove_bk(flt_ti, in_i_index, in_str_folder_prefix, in_b_train_or_test)

    i_practical_region_count, i_practical_least_area, i_practical_least_max_of_wh = img_info_analysis(flt_ti)

    flt_ti, out_img_region_set_sorted, tmp_labeled_image = img_process_to_reduce_objs(flt_ti, in_i_index, in_str_folder_prefix, in_b_train_or_test)
    
    out_fn_process = in_str_folder_prefix + "_process" + "/" + str(in_i_index) + ".png"

    misc_imsave(out_fn_process, flt_ti)
    
    out_i_to_try_separation = 0
    out_i_separate_success = 0    
    
    flt_ti, out_i_to_try_separation, out_i_separate_success = img_process_to_remove_attachment(flt_ti, in_i_index, in_str_folder_prefix, in_b_train_or_test, out_img_region_set_sorted, tmp_labeled_image)
    
     
    et = time.time()
    ut = et - st
    
    
    global total_oi_s2_time
    total_oi_s2_time += ut
    
    
    #s3
    
    img_region_set = []
    
    st = time.time()
    
    i_out_label_count = 0
    
    tmp_image_for_crop = np.copy(flt_ti)
    
    
    
    
    
            
    out_fn = in_str_folder_prefix + "_slb_debug" + "/" + str(in_i_index) + "_start" + ".png"
    
    plt.imsave(out_fn, tmp_image_for_crop)

    
    
    labeled_image, i_label_num = label(tmp_image_for_crop, return_num=True)
   
    i_out_label_count = i_label_num
    
       
    for region in regionprops(labeled_image):
        # take regions with large enough areas
        if region.area >= i_area_threshold:
            minr, minc, maxr, maxc = region.bbox
            
            img_region_item = {}
            
            
            region_width = maxc - minc
            region_height = maxr - minr
            
            max_of_wh = max(region_width, region_height)
            
            img_region_item['max_of_wh'] = max_of_wh
            
            img_region_item['i_area'] = int(region.area)
            

            img_region_item['region_obj'] = region
            
            
            img_region_set.append(img_region_item)
            

            
            
                
    
    et = time.time()
    ut = et - st
    
    
    global total_oi_s3_time
    total_oi_s3_time += ut


    #s5
    
    
    st = time.time()
    
    out_fn_raw = in_str_folder_prefix + "_raw" + "/" + str(in_i_index) + ".png"
    
    misc_imsave(out_fn_raw, ti_tmp)
    
     
    et = time.time()
    ut = et - st
    
    
    global total_oi_s5_time
    total_oi_s5_time += ut
    
    
    #s4
    
    
    
    
    st = time.time()    
     
    img_tw = np.zeros((28, 28), dtype=np.uint8)
    
       
    i_out_max_wh = 0
    i_out_region_area = 0
    
    out_arr = None
    
    i_out_region_area_2nd = 0
    i_out_max_wh_2nd = 0 
       
    i_out_region_area_3rd = 0
    i_out_max_wh_3rd = 0 
    
    len_irs = len(img_region_set)
    
    if len_irs > 0:
        
        
        img_region_set_sorted = sorted(img_region_set, key=lambda x: (x['max_of_wh']), reverse=True)
    
        
        img_region_item_0 = img_region_set_sorted[0]
        
        region_obj_0 = img_region_item_0['region_obj']
        
        tmp_label_image = np.copy(labeled_image)
        
        
        tmp_label_image[tmp_label_image != region_obj_0.label] = 0
        tmp_label_image[tmp_label_image == region_obj_0.label] = 1
        
        minr, minc, maxr, maxc = region_obj_0.bbox
        
        #i_out_region_area = region_obj_0.area
        
        i_out_region_area = img_region_item_0['i_area']
        i_out_max_wh = img_region_item_0['max_of_wh']
        
        if i_out_region_area != int(region_obj_0.area):
            logger.info("op_img warning: for in_i_index = %d, i_out_region_area: %d != region_obj_0.area: %d ", in_i_index, i_out_region_area, int(region_obj_0.area))
    
        
        if len_irs > 1:
            
            img_region_item_1 = img_region_set_sorted[1]
        
            region_obj_1 = img_region_item_1['region_obj']
            
            #i_out_region_area_2nd = region_obj_1.area
            
            i_out_region_area_2nd = img_region_item_1['i_area']
        
            i_out_max_wh_2nd = img_region_item_1['max_of_wh']
        
        
            if i_out_region_area_2nd != int(region_obj_1.area):
                logger.info("op_img warning: for in_i_index = %d, i_out_region_area_2nd: %d != region_obj_1.area: %d ", in_i_index, i_out_region_area_2nd, int(region_obj_1.area))
    
            
            if len_irs > 2:
                
                img_region_item_2 = img_region_set_sorted[2]
        
                region_obj_2 = img_region_item_2['region_obj']
                
                #i_out_region_area_2nd = region_obj_1.area

                
                i_out_region_area_3rd = img_region_item_2['i_area']            
                i_out_max_wh_3rd = img_region_item_2['max_of_wh']
            
            
                if i_out_region_area_3rd != int(region_obj_2.area):
                    logger.info("op_img warning: for in_i_index = %d, i_out_region_area_3rd: %d != region_obj_2.area: %d ", in_i_index, i_out_region_area_3rd, int(region_obj_2.area))
    
            
            
        tmp_image_for_crop[tmp_label_image == 0] = 0
        
        
            
        out_fn = in_str_folder_prefix + "_slb_debug" + "/" + str(in_i_index) + "_s1" + ".png"
    
        plt.imsave(out_fn, tmp_image_for_crop)
        
        
    


    
        img_cropped = tmp_image_for_crop[minr - i_pad : maxr + i_pad, minc - i_pad : maxc + i_pad]
        
        out_fn = in_str_folder_prefix + "_slb_debug" + "/" + str(in_i_index) + "_s2" + ".png"
    
        plt.imsave(out_fn, img_cropped)

        
        f_rs_ratio = float(26.0 / float(i_out_max_wh))
        
        img_rescaled = rescale(img_cropped, f_rs_ratio, preserve_range=True, anti_aliasing=True)
        
        
        
        
        out_fn = in_str_folder_prefix + "_slb_debug" + "/" + str(in_i_index) + "_s3" + ".png"
    
        plt.imsave(out_fn, img_rescaled)
        
         
        i_width_rsd = img_rescaled.shape[0]
        i_height_rsd = img_rescaled.shape[1]
        
        
        i_x = int((28 - i_width_rsd) / 2)
        
        i_y = int((28 - i_height_rsd) / 2)
        
        img_tw[i_x : (i_x + i_width_rsd), i_y : (i_y + i_height_rsd)] = img_rescaled
        
          
        
        
        out_fn = in_str_folder_prefix + "_slb_debug" + "/" + str(in_i_index) + "_s3" + ".png"
    
        plt.imsave(out_fn, img_tw)
        
        #out_fn_seg = in_str_folder_prefix + "/" +  str(in_i_index) + ".png"
        out_fn = in_str_folder_prefix + "_small" + "/" + str(in_i_index) + ".png"
        misc_imsave(out_fn, img_tw)
        
    else:
        logger.info("len_irs is 0, error may occur, for in_i_index = %d and in_str_folder_prefix = %s !!!!!!", in_i_index, in_str_folder_prefix)
    
     
    out_arr_small = np.reshape(img_tw, 28 * 28)
    
    out_arr = np.reshape(flt_ti, 64 * 64)

    out_fn_ro = in_str_folder_prefix + "" + "/" + str(in_i_index) + ".png"

    misc_imsave(out_fn_ro, flt_ti)
    
    
    et = time.time()
    ut = et - st
    
    
    global total_oi_s4_time
    total_oi_s4_time += ut
   
    return out_arr_small, out_arr, i_out_label_count, i_out_region_area, i_out_max_wh, i_out_region_area_2nd, i_out_max_wh_2nd, i_out_region_area_3rd, i_out_max_wh_3rd, i_ideal_region_count, i_ideal_least_area, i_ideal_least_max_of_wh, i_practical_region_count, i_practical_least_area, i_practical_least_max_of_wh, i_bkr_stage_count, i_bkr_label1_num, i_bkr_label2_num, i_bkr_label3_num, i_bkr_label4_num, i_bkr_label5_num, i_bkr_label6_num, i_bkr_otsu_threshold_s1, i_bkr_otsu_threshold_s2, i_bkr_otsu_threshold_s3, out_i_to_try_separation, out_i_separate_success
     
    
i_area_threshold = 38


i_pad = 0



logging.config.fileConfig('logging.conf')

# create logger
logger = logging.getLogger('Project3Group24')



logger.info("\n\n\n\n\n\n\n\n\n\nprogram begins. ")

input_data_folder = "../../../comp551_Modified_MNIST/input/"

output_data_folder = "../../../comp551_Modified_MNIST/output/"

logger.info(os.listdir(input_data_folder))


train_images_set = pd.read_pickle(input_data_folder + 'train_images.pkl')
train_labels_set = pd.read_csv(input_data_folder + 'train_labels.csv')
test_images_set = pd.read_pickle(input_data_folder + 'test_images.pkl')

print("\n train_images_set.shape = ", train_images_set.shape)
print("\n train_labels_set.shape = ", train_labels_set.shape)
print("\n test_images_set.shape = ", test_images_set.shape)

label_set = [] 
label_set = train_labels_set.iloc[:, 1]


label_set = label_set.tolist()

print("\n label_set.shape = ", type(label_set), np.asarray(label_set).shape)



#quit()




b_try = False
#b_try = True



b_generate_csv_file = True
#b_generate_csv_file = False



total_oi_time = 0.0

total_df_time = 0.0

total_ii_time = 0.0


total_oi_s1_time = 0.0

total_oi_s2_time = 0.0

total_oi_s3_time = 0.0

total_oi_s4_time = 0.0

total_oi_s5_time = 0.0






train_folder_prefix = ""

if b_try:
    train_folder_prefix = output_data_folder + "try_imgs"
else:
    
    train_folder_prefix = output_data_folder + "train_imgs"




#df = pd.DataFrame(columns = ["idx"] * (28 * 28 + 1))


col_labels = ["idx"] * (64 * 64 + 1)

col_small_labels = ["idx"] * (28 * 28 + 1)

list_of_op_data_set = []

list_of_op_data_small_set = []

list_of_op_data_set_w_tidy = []


img_info_set = []


#in_i_index = 22189

#for i_index in (22189, 74, 784, 1080, 1579):

#for i_index in range(0, 1100, 1):
#for i_index in range(0, 160, 1):
#for i_index in (3, 613, 1080, 1420, 2388, 2405, 6397, 37124):
for i_index in range(0, 40000, 1):
    
    
    st = time.time()
    
    #i_out_max_wh, out_arr, out_region_area, i_out_label_count, i_orig_region_count, i_region_area_2nd, i_max_wh_2nd = op_img(i_index, train_folder_prefix, True, False)
   
    out_arr_small, out_arr, i_out_label_count, i_out_region_area, i_out_max_wh, i_out_region_area_2nd, i_out_max_wh_2nd, i_out_region_area_3rd, i_out_max_wh_3rd, i_ideal_region_count, i_ideal_least_area, i_ideal_least_max_of_wh, i_practical_region_count, i_practical_least_area, i_practical_least_max_of_wh, i_bkr_stage_count, i_bkr_label1_num, i_bkr_label2_num, i_bkr_label3_num, i_bkr_label4_num, i_bkr_label5_num, i_bkr_label6_num, i_bkr_otsu_threshold_s1, i_bkr_otsu_threshold_s2, i_bkr_otsu_threshold_s3, i_to_try_separation, i_separate_success = op_img(i_index, train_folder_prefix, True, False)
    

    et = time.time()
    ut = et - st
    
    total_oi_time += ut
    
    
    
    st = time.time()
    
    
    tmp_tl = label_set[i_index]
    
    arr_small_int = out_arr_small.astype(int)
    
    list_small_row = []
    
    
    list_small_row.append(tmp_tl)
    
    list_small_arr = arr_small_int.tolist()
    
    list_small_row.extend(list_small_arr)
    
    list_of_op_data_small_set.append(list_small_row)
    
    
    arr_int = out_arr.astype(int)
    
    list_row = []
    
    #tmp_tl = train_labels_set.iloc[i_index]['Category']
    
    
    #tmp_tl = train_labels_set[i_index]
    
    list_row.append(tmp_tl)
    
    list_arr = arr_int.tolist()
    
    list_row.extend(list_arr)
    
    #df.loc[i_index] = list_row
    
    list_of_op_data_set.append(list_row)
    
    """
    if i_out_max_wh <= 28:
        list_of_op_data_set_w_tidy.append(list_row)
    """   
    
    
    et = time.time()
    ut = et - st
    
    total_df_time += ut
    
    
    
    st = time.time()
    
    img_info_item = {}


    img_info_item['i_index'] = i_index
    
    img_info_item['i_max_wh'] = int(i_out_max_wh)    
    img_info_item['i_area'] = int(i_out_region_area)
    
    img_info_item['i_max_wh_2nd'] = int(i_out_max_wh_2nd)
    img_info_item['i_area_2nd'] = int(i_out_region_area_2nd)
    
    img_info_item['i_max_wh_3rd'] = int(i_out_max_wh_3rd)
    img_info_item['i_area_3rd'] = int(i_out_region_area_3rd)
    
    
    img_info_item['i_final_label_count'] = int(i_out_label_count)
    
    img_info_item['i_ideal_region_count'] = int(i_ideal_region_count)
    img_info_item['i_ideal_least_area'] = int(i_ideal_least_area)
    img_info_item['i_ideal_least_max_of_wh'] = int(i_ideal_least_max_of_wh)
    
    img_info_item['i_practical_region_count'] = int(i_practical_region_count)
    img_info_item['i_practical_least_area'] = int(i_practical_least_area)
    img_info_item['i_practical_least_max_of_wh'] = int(i_practical_least_max_of_wh)
    
    img_info_item['i_bkr_stage_count'] = int(i_bkr_stage_count)
    
    img_info_item['i_bkr_label1_num'] = int(i_bkr_label1_num)
    img_info_item['i_bkr_label2_num'] = int(i_bkr_label2_num)
    img_info_item['i_bkr_label3_num'] = int(i_bkr_label3_num)
    img_info_item['i_bkr_label4_num'] = int(i_bkr_label4_num)
    
    
    img_info_item['i_bkr_label5_num'] = int(i_bkr_label5_num)
    img_info_item['i_bkr_label6_num'] = int(i_bkr_label6_num)
    
    
    img_info_item['i_bkr_otsu_threshold_s1'] = int(i_bkr_otsu_threshold_s1)    
    img_info_item['i_bkr_otsu_threshold_s2'] = int(i_bkr_otsu_threshold_s2)
    
    img_info_item['i_bkr_otsu_threshold_s3'] = int(i_bkr_otsu_threshold_s3)
    
    img_info_item['i_to_try_separation'] = int(i_to_try_separation)
    img_info_item['i_separate_success'] = int(i_separate_success)
    
        
    img_info_set.append(img_info_item)
    
    
    et = time.time()
    ut = et - st
    
    total_ii_time += ut
    
    
    if i_index % 500 == 0:
        logger.info(" i_index = %d", i_index)


print("\n total_oi_time = ", total_oi_time)
print("\n total_df_time = ", total_df_time)
print("\n total_ii_time = ", total_ii_time)
    

print("\n total_oi_s1_time = ", total_oi_s1_time)
print("\n total_oi_s2_time = ", total_oi_s2_time)
print("\n total_oi_s3_time = ", total_oi_s3_time)
print("\n total_oi_s4_time = ", total_oi_s4_time)
print("\n total_oi_s5_time = ", total_oi_s5_time)


df = None
if b_generate_csv_file:
    df = pd.DataFrame.from_records(list_of_op_data_set, columns=col_labels)


if b_try:
    df.to_csv(output_data_folder + 'try_train.csv', index = False, header = True)
else:
    if b_generate_csv_file:
        df.to_csv(output_data_folder + 'train.csv', index = False, header = True)
    
if b_generate_csv_file:
    #df = pd.DataFrame.from_records(list_of_op_data_set_w_tidy, columns=col_labels)
    df = pd.DataFrame.from_records(list_of_op_data_small_set, columns=col_small_labels)
    

if b_try:
    pass
else:
    if b_generate_csv_file:    
        #df.to_csv(output_data_folder + 'train_w_tidy.csv', index = False, header = True)
        df.to_csv(output_data_folder + 'train_small.csv', index = False, header = True)
        
df = None	

list_of_op_data_set = []

list_of_op_data_small_set = []

list_of_op_data_set_w_tidy = []

with open(output_data_folder + "train_preprocess_stat.json", "w") as wf_json_set:
    json.dump(img_info_set, wf_json_set)

img_info_set_sorted = sorted(img_info_set, key=lambda x: (x['i_max_wh']), reverse=True)

i_buffer = len(img_info_set_sorted)

"""
print("\n\n\n\n  largest: \n", i_buffer)

for x in range(0, 0 + i_buffer, 1):
    print("", img_info_set_sorted[x])
"""



quit()



test_folder_prefix = ""

if b_try:
    test_folder_prefix = output_data_folder + "try2_imgs"
else:
    
    test_folder_prefix = output_data_folder + "test_imgs"


#df = pd.DataFrame(columns = ["idx"] * (28 * 28))

col_labels = ["idx"] * (64 * 64)

col_small_labels = ["idx"] * (28 * 28)

list_of_op_data_set = []

list_of_op_data_small_set = []



img_info_set = []

#len_of_train = len()

st = time.time()

et = time.time()

ut = et - st
print(st)





total_oi_time = 0.0

total_df_time = 0.0

total_ii_time = 0.0


total_oi_s1_time = 0.0

total_oi_s2_time = 0.0

total_oi_s3_time = 0.0

total_oi_s4_time = 0.0

total_oi_s5_time = 0.0



#for i_index in range(0, 1100, 1):
#for i_index in range(0, 160, 1):
#for i_index in (128, 403, 6503, 6507, 6916):
for i_index in range(0, 10000, 1):
    
    
    st = time.time()
    
    
    #i_out_max_wh, out_arr, out_region_area, i_out_label_count, i_orig_region_count, i_region_area_2nd, i_max_wh_2nd  = op_img(i_index, test_folder_prefix, False, False)
    
    out_arr_small, out_arr, i_out_label_count, i_out_region_area, i_out_max_wh, i_out_region_area_2nd, i_out_max_wh_2nd, i_out_region_area_3rd, i_out_max_wh_3rd, i_ideal_region_count, i_ideal_least_area, i_ideal_least_max_of_wh, i_practical_region_count, i_practical_least_area, i_practical_least_max_of_wh, i_bkr_stage_count, i_bkr_label1_num, i_bkr_label2_num, i_bkr_label3_num, i_bkr_label4_num, i_bkr_label5_num, i_bkr_label6_num, i_bkr_otsu_threshold_s1, i_bkr_otsu_threshold_s2, i_bkr_otsu_threshold_s3, i_to_try_separation, i_separate_success = op_img(i_index, test_folder_prefix, False, False)
    
    
    
    et = time.time()
    ut = et - st
    
    total_oi_time += ut
    
    
    
    
    st = time.time()
    
    
    arr_small_int = out_arr_small.astype(int)
    
    list_small_row = []
    
    
    #list_small_row.append(tmp_tl)
    
    list_small_arr = arr_small_int.tolist()
    
    list_small_row.extend(list_small_arr)
    
    list_of_op_data_small_set.append(list_small_row)
    
    
    
    
    
    arr_int = out_arr.astype(int)
    
    list_row = []
    
    #tmp_tl = train_labels_set.iloc[i_index]['Category']
    #tmp_tl = train_labels_set[i_index]
    
    #list_row.append(tmp_tl)
    
    list_arr = arr_int.tolist()
    
    list_row.extend(list_arr)
    
    
    list_of_op_data_set.append(list_row)
      
    et = time.time()
    ut = et - st
    
    total_df_time += ut
    
       
    
    st = time.time()
    
    img_info_item = {}

    img_info_item['i_index'] = i_index
    
    img_info_item['i_max_wh'] = int(i_out_max_wh)    
    img_info_item['i_area'] = int(i_out_region_area)
    
    img_info_item['i_max_wh_2nd'] = int(i_out_max_wh_2nd)
    img_info_item['i_area_2nd'] = int(i_out_region_area_2nd)
    
    img_info_item['i_max_wh_3rd'] = int(i_out_max_wh_3rd)
    img_info_item['i_area_3rd'] = int(i_out_region_area_3rd)
    
    
    img_info_item['i_final_label_count'] = int(i_out_label_count)
    
    img_info_item['i_ideal_region_count'] = int(i_ideal_region_count)
    img_info_item['i_ideal_least_area'] = int(i_ideal_least_area)
    img_info_item['i_ideal_least_max_of_wh'] = int(i_ideal_least_max_of_wh)
    
    img_info_item['i_practical_region_count'] = int(i_practical_region_count)
    img_info_item['i_practical_least_area'] = int(i_practical_least_area)
    img_info_item['i_practical_least_max_of_wh'] = int(i_practical_least_max_of_wh)
    
    img_info_item['i_bkr_stage_count'] = int(i_bkr_stage_count)
    
    img_info_item['i_bkr_label1_num'] = int(i_bkr_label1_num)
    img_info_item['i_bkr_label2_num'] = int(i_bkr_label2_num)
    img_info_item['i_bkr_label3_num'] = int(i_bkr_label3_num)
    img_info_item['i_bkr_label4_num'] = int(i_bkr_label4_num)
    
    
    img_info_item['i_bkr_label5_num'] = int(i_bkr_label5_num)
    img_info_item['i_bkr_label6_num'] = int(i_bkr_label6_num)
    
    
    
    img_info_item['i_bkr_otsu_threshold_s1'] = int(i_bkr_otsu_threshold_s1)    
    img_info_item['i_bkr_otsu_threshold_s2'] = int(i_bkr_otsu_threshold_s2)
    
    img_info_item['i_bkr_otsu_threshold_s3'] = int(i_bkr_otsu_threshold_s3)
    
    img_info_item['i_to_try_separation'] = int(i_to_try_separation)
    img_info_item['i_separate_success'] = int(i_separate_success)
   
    
    img_info_set.append(img_info_item)
    
    
    et = time.time()
    ut = et - st
    
    total_ii_time += ut
    
    
    
    if i_index % 500 == 0:
        logger.info(" i_index = %d", i_index)
        
        

logger.info("\n finish of big loop")        
    
print("\n total_oi_time = ", total_oi_time)
print("\n total_df_time = ", total_df_time)
print("\n total_ii_time = ", total_ii_time)
    

print("\n total_oi_s1_time = ", total_oi_s1_time)
print("\n total_oi_s2_time = ", total_oi_s2_time)
print("\n total_oi_s3_time = ", total_oi_s3_time)
print("\n total_oi_s4_time = ", total_oi_s4_time)
print("\n total_oi_s5_time = ", total_oi_s5_time)

df = None
if b_generate_csv_file:
    df = pd.DataFrame.from_records(list_of_op_data_set, columns=col_labels)


if b_try:
    df.to_csv(output_data_folder + 'try_test.csv', index = False, header = True)
else:
    if b_generate_csv_file:
        df.to_csv(output_data_folder + 'test.csv', index = False, header = True)


if b_generate_csv_file:
    #df = pd.DataFrame.from_records(list_of_op_data_set_w_tidy, columns=col_labels)
    df = pd.DataFrame.from_records(list_of_op_data_small_set, columns=col_small_labels)
    

if b_try:
    pass
else:
    if b_generate_csv_file:    
        #df.to_csv(output_data_folder + 'train_w_tidy.csv', index = False, header = True)
        df.to_csv(output_data_folder + 'test_small.csv', index = False, header = True)




logger.info("\n finish of generate data csv")


list_of_op_data_set = []

with open(output_data_folder + "test_preprocess_stat.json", "w") as wf_json_set:
    json.dump(img_info_set, wf_json_set)


img_info_set_sorted = sorted(img_info_set, key=lambda x: (x['i_max_wh']), reverse=True)



i_buffer = len(img_info_set_sorted)




#exp_folder_prefix = output_data_folder + "exp_imgs"


logger.info("\n\n\n\n\n\n\n\n\n\nprogram ends. ")