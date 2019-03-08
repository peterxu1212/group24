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

from skimage.morphology import closing, square

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

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






def op_img(in_i_index, in_str_folder_prefix, in_b_train_or_test=True, b_display=False):
    
    #lc_tmp = -1
    
    """
    if in_b_train_or_test:
        lc_tmp = train_labels_set.iloc[in_i_index]['Category']
    else:
        pass
    
    """
    
    
    #s1
    
    st = time.time()
    
    #print("\n lc_tmp = ", lc_tmp)
    
    
    
    #fig, axes = plt.subplots(1, 1)
    
    
    #axes[0, 0].set_title('Label: {}'.format(lc_tmp))
    
    #plt.title('Label: {}'.format(lc_tmp))
    
    ti_tmp = ""
    
    if in_b_train_or_test:
    
        ti_tmp = train_images_set[in_i_index]
    else:
        ti_tmp = test_images_set[in_i_index]
    
    #np.set_printoptions(threshold=np.inf)
    
    #print("\n ti_tmp.shape = ", ti_tmp.shape, type(ti_tmp))
    
    #print("\n\n ti_tmp = \n", ti_tmp)
    
    
    et = time.time()
    ut = et - st
    
    global total_oi_s1_time
    
    total_oi_s1_time += ut
    
    
    
    
    #s2
    
    st = time.time()
    
    #thresh = threshold_otsu(ti_tmp)

    #print("\n thresh = ", thresh)
    
    flt_ti = np.copy(ti_tmp)
    #flt_ti = ti_tmp
    
    #flt_ti[flt_ti <= int(thresh)] = 0
    
    
    
    #this line is not necessary, but with this line, 
    #the speed increase for the label function, for tiny region will disappear. 
    #flt_ti[flt_ti <= 254] = 0
    
    
    
    
    #flt_ti = ti_tmp[ti_tmp>=254]
    
    
    #print("\n flt_ti.shape = ", flt_ti.shape, type(flt_ti))
    
    #print("\n\n flt_ti = \n", flt_ti)
    
    """    
    
    bw = closing(flt_ti > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)
    """    
    
    et = time.time()
    ut = et - st
    
    
    global total_oi_s2_time
    total_oi_s2_time += ut
    
    
    #s3
    
    
    st = time.time()
    
    # label image regions
    #label_image = label(cleared)
    
    label_image = label(flt_ti)
    #image_label_overlay = label2rgb(label_image, image=flt_ti)
    """
    np.set_printoptions(threshold=np.inf)
    print("\n label_image = \n", label_image)
    """
    
    #mpo_label = morphology.label(flt_ti, connectivity=2)
    
    #print("\n\n label_image = \n", label_image)
    
    
    #axes[0, 0].imshow(ti_tmp)
    
    #plt.imshow(ti_tmp)
    
    
    #plt.imshow(ti_tmp, cmap="gray")
    fig = None
    ax = None
    
    if b_display:
        #fig, ax = plt.subplots(figsize=(10, 10))
        fig, ax = plt.subplots()
        #ax.imshow(image_label_overlay)
    
        #ax.imshow(flt_ti)
        
        ax.imshow(ti_tmp)
        #plt.imshow
        
        #ax.imshow(bw)
    
    
    img_region_set = []
    
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 50:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            
            img_region_item = {}
            
            
            region_width = maxc - minc
            region_height = maxr - minr
            
            max_of_wh = max(region_width, region_height)
            
            img_region_item['max_of_wh'] = max_of_wh
            
            """
            img_region_item['minr'] = minr
            
            img_region_item['minc'] = minc
            
            img_region_item['maxr'] = maxr
            
            img_region_item['maxc'] = maxc
            """
            
            img_region_item['region_obj'] = region
            
            
            img_region_set.append(img_region_item)
            
            """
            print("\n region.label = \n", type(region.label), region.label)
            print("\n region.area = \n", type(region.area), region.area)
            print("\n region.bbox = \n", type(region.bbox), region.bbox)
            print("\n region.coords = \n", type(region.coords), region.coords.shape, region.coords, )
            """
            
            #print("\n region.bbox = ", region.bbox, region.area, maxc - minc, maxr - minr)
            
            #rect = mpatches.Rectangle((minc - 1, minr - 1), maxc - minc + 1, maxr - minr + 1, fill=False, edgecolor='red', linewidth=1)
            #rect = mpatches.Rectangle((minc - 1, minr - 1), region_width, region_height, fill=False, edgecolor='red', linewidth=1)
            #rect = mpatches.Rectangle((minc, minr), region_width, region_height, fill=False, edgecolor='red', linewidth=1)
            
            if b_display:
                rect = mpatches.Rectangle((minc - 1, minr - 1), region_width + 1, region_height + 1, fill=False, edgecolor='red', linewidth=1)
            
            
                ax.add_patch(rect)
    

    et = time.time()
    ut = et - st
    
    
    global total_oi_s3_time
    total_oi_s3_time += ut


    #s5
    
    
    st = time.time()
    
    out_fn = in_str_folder_prefix + "_raw" + "/" + str(in_i_index) + ".png"
    
    #imsave(out_fn, ti_tmp) 
    plt.imsave(out_fn, ti_tmp)
    
    #plt.show()
    #plt.savefig(out_fn)
    
    
    et = time.time()
    ut = et - st
    
    
    global total_oi_s5_time
    total_oi_s5_time += ut
    
    
    #s4
    
    st = time.time()    
            
    i_out_max_wh = 0
    i_out_region_area = 0
    
    out_arr = None
    
    len_irs = len(img_region_set)
    
    if len_irs > 0:
        
        
        img_region_set_sorted = sorted(img_region_set, key=lambda x: (x['max_of_wh']), reverse=True)
        """
        for img_region_item in img_region_set_sorted:
            print("\n img_region_item = ", img_region_item)
        """    
        
        img_region_item_0 = img_region_set_sorted[0]
        
        region_obj = img_region_item_0['region_obj']
        
        tmp_label_image = np.copy(label_image)
        
        
        
        #print("\n region_obj.label = ", region_obj.label)
        
        tmp_label_image[tmp_label_image != region_obj.label] = 0
        tmp_label_image[tmp_label_image == region_obj.label] = 255
        
        minr, minc, maxr, maxc = region_obj.bbox
        
        i_out_region_area = region_obj.area
        i_out_max_wh = img_region_item_0['max_of_wh']
        
        tmp_label_image = img_as_ubyte(tmp_label_image)
        
        img_cropped = tmp_label_image[minr - i_pad : maxr + i_pad, minc - i_pad : maxc + i_pad]
        
        #resize()
        """
        print("\n img_cropped = ", img_cropped)
        
        print("img_cropped.dtype.name = ", img_cropped.dtype.name)
        """
        
        #img_tw = None
        
        
        #img_tw = resize(img_cropped, (28, 28), anti_aliasing=True)
        
        #img_resized = resize(img_cropped, (20, 20), anti_aliasing=True)
        img_resized = resize(img_cropped, (28, 28), anti_aliasing=True)
        
        
        """
        print("\n img_resized = ", img_resized)
        
        print("img_resized.dtype.name = ", img_resized.dtype.name)
        """
        
        img_dst = img_as_ubyte(img_resized)
        
        """
        print("\n img_dst = ", img_dst)
        
        print("img_dst.dtype.name = ", img_dst.dtype.name)
        """
        
        img_tw = np.zeros((28, 28), dtype=np.uint8)
        
        #img_tw[4:24, 4:24] = img_dst
        img_tw[0:28, 0:28] = img_dst
        
        
        #print("img_tw.dtype.name = ", img_tw.dtype.name)
        
        #img_tw[4:24, 4:24] = 1
        
        """
        region_width = maxc - minc
        region_height = maxr - minr
        
        
        i_out_min_wh = min(region_width, region_height)
        
        if i_out_max_wh > 20:
            
            
            
        elif i_out_max_wh < 20:
            
        else:
            #i_out_max_wh == 20
        
        f_ratio = 
        
        img_normalized = rescale(img_cropped, f_ratio, anti_aliasing=True)
        """
        
        #rescale(img_cropped)
        
        
        #np.zeros((28, 28))
        
        """
        if i_out_max_wh > 28:
                    
            img_tw = resize(img_cropped, (28, 28), anti_aliasing=True)
            
        else：
            #i_out_max_wh <= 28       
            
            img_tw = np.zeros((28, 28))
         """   
            
        #print("\n img_tw = ", img_tw)
            
        out_arr = np.reshape(img_tw, 28 * 28)
        
        
        #print("\n out_arr = ", type(out_arr), out_arr)
        
        #print("\n img_cropped.shape = ", img_cropped.shape, type(img_cropped))
    
        #print("\n\n img_cropped = \n", img_cropped)
        
        
        out_fn_seg = in_str_folder_prefix + "/" +  str(in_i_index) + ".png"
        
        #imsave(out_fn_seg, img_cropped)
        misc_imsave(out_fn_seg, img_tw)
        #plt.imsave(out_fn_seg, img_tw)
        
    else:
        logger.info("len_irs is 0, error may occur, for in_i_index = %d and in_str_folder_prefix = %s !!!!!!", in_i_index, in_str_folder_prefix)
    
    
    
    
    
    
    et = time.time()
    ut = et - st
    
    
    global total_oi_s4_time
    total_oi_s4_time += ut
    
    
    
    
    
    
    
    
    
    if b_display:
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
    
    
    
    
    
    
    
    
    
    """
    if b_display:
        
        plt.imshow(flt_ti)
    
        plt.show()
    """
    
    return i_out_max_wh, out_arr, i_out_region_area



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







total_oi_time = 0.0

total_df_time = 0.0

total_ii_time = 0.0


total_oi_s1_time = 0.0

total_oi_s2_time = 0.0

total_oi_s3_time = 0.0

total_oi_s4_time = 0.0

total_oi_s5_time = 0.0



"""


train_folder_prefix = ""

if b_try:
    train_folder_prefix = output_data_folder + "try_imgs"
else:
    
    train_folder_prefix = output_data_folder + "train_imgs"




#df = pd.DataFrame(columns = ["idx"] * (28 * 28 + 1))


col_labels = ["idx"] * (28 * 28 + 1)

list_of_op_data_set = []

list_of_op_data_set_w_tidy = []


img_info_set = []

#for i_index in range(0, 10, 1):
#for i_index in range(0, 2000, 1):
for i_index in range(0, 40000, 1):
    
    
    st = time.time()
    
    i_out_max_wh, out_arr, out_region_area = op_img(i_index, train_folder_prefix, True, False)
    
    
    et = time.time()
    ut = et - st
    
    total_oi_time += ut
    
    
    
    st = time.time()
    
    arr_int = out_arr.astype(int)
    
    list_row = []
    
    #tmp_tl = train_labels_set.iloc[i_index]['Category']
    tmp_tl = label_set[i_index]
    
    #tmp_tl = train_labels_set[i_index]
    
    list_row.append(tmp_tl)
    
    list_arr = arr_int.tolist()
    
    list_row.extend(list_arr)
    
    #df.loc[i_index] = list_row
    
    list_of_op_data_set.append(list_row)
    
    if i_out_max_wh <= 28:
        list_of_op_data_set_w_tidy.append(list_row)
    
    
    et = time.time()
    ut = et - st
    
    total_df_time += ut
    
    
    
    st = time.time()
    
    img_info_item = {}

    img_info_item['i_max_wh'] = i_out_max_wh
    img_info_item['i_index'] = i_index
    img_info_item['i_area'] = int(out_region_area)
    
    img_info_set.append(img_info_item)
    
    
    et = time.time()
    ut = et - st
    
    total_ii_time += ut
    
    
    if i_index % 500 == 0:
        logger.info("\n i_index = %d", i_index)



print("\n total_oi_time = ", total_oi_time)
print("\n total_df_time = ", total_df_time)
print("\n total_ii_time = ", total_ii_time)
    

print("\n total_oi_s1_time = ", total_oi_s1_time)
print("\n total_oi_s2_time = ", total_oi_s2_time)
print("\n total_oi_s3_time = ", total_oi_s3_time)
print("\n total_oi_s4_time = ", total_oi_s4_time)
print("\n total_oi_s5_time = ", total_oi_s5_time)




df = pd.DataFrame.from_records(list_of_op_data_set, columns=col_labels)

#df.to_csv(output_data_folder + 'train.csv', index = False, header = True)


if b_try:
    df.to_csv(output_data_folder + 'try_train.csv', index = False, header = True)
else:
    df.to_csv(output_data_folder + 'train.csv', index = False, header = True)
    

df = pd.DataFrame.from_records(list_of_op_data_set_w_tidy, columns=col_labels)

if b_try:
    pass
else:    
    df.to_csv(output_data_folder + 'train_w_tidy.csv', index = False, header = True)


with open(output_data_folder + "train_preprocess_stat.json", "w") as wf_json_set:
    json.dump(img_info_set, wf_json_set)

img_info_set_sorted = sorted(img_info_set, key=lambda x: (x['i_max_wh']), reverse=True)



i_buffer = len(img_info_set_sorted)


print("\n\n\n\n  largest: \n", i_buffer)

for x in range(0, 0 + i_buffer, 1):
    print("", img_info_set_sorted[x])







test_folder_prefix = ""

if b_try:
    test_folder_prefix = output_data_folder + "try2_imgs"
else:
    
    test_folder_prefix = output_data_folder + "test_imgs"


#df = pd.DataFrame(columns = ["idx"] * (28 * 28))

col_labels = ["idx"] * (28 * 28)



list_of_op_data_set = []


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



#for i_index in range(0, 10, 1):
#for i_index in range(0, 2000, 1):
for i_index in range(0, 10000, 1):
    
    
    st = time.time()
    
    
    
    
    i_out_max_wh, out_arr, out_region_area = op_img(i_index, test_folder_prefix, False, False)
        
    
    
    
    et = time.time()
    ut = et - st
    
    total_oi_time += ut
    
    
    
    
    st = time.time()
    
    
    
    
    
    
    arr_int = out_arr.astype(int)
    
    list_row = []
    
    #tmp_tl = train_labels_set.iloc[i_index]['Category']
    #tmp_tl = train_labels_set[i_index]
    
    #list_row.append(tmp_tl)
    
    list_arr = arr_int.tolist()
    
    list_row.extend(list_arr)
    
    
    list_of_op_data_set.append(list_row)
    
    #df.loc[i_index] = list_row
    
    
    
    
    
    
    et = time.time()
    ut = et - st
    
    total_df_time += ut
    
    
    
    
    
    st = time.time()
    
    img_info_item = {}

    img_info_item['i_max_wh'] = i_out_max_wh
    img_info_item['i_index'] = i_index
    img_info_item['i_area'] = int(out_region_area)
    
    img_info_set.append(img_info_item)
    
    
    et = time.time()
    ut = et - st
    
    total_ii_time += ut
    
    
    
    if i_index % 500 == 0:
        logger.info("\n i_index = %d", i_index)
        
        

logger.info("\n finish of big loop")        
    
print("\n total_oi_time = ", total_oi_time)
print("\n total_df_time = ", total_df_time)
print("\n total_ii_time = ", total_ii_time)
    

print("\n total_oi_s1_time = ", total_oi_s1_time)
print("\n total_oi_s2_time = ", total_oi_s2_time)
print("\n total_oi_s3_time = ", total_oi_s3_time)
print("\n total_oi_s4_time = ", total_oi_s4_time)
print("\n total_oi_s5_time = ", total_oi_s5_time)



df = pd.DataFrame.from_records(list_of_op_data_set, columns=col_labels)


if b_try:
    df.to_csv(output_data_folder + 'try_test.csv', index = False, header = True)
else:
    df.to_csv(output_data_folder + 'test.csv', index = False, header = True)


logger.info("\n finish of generate data csv")


with open(output_data_folder + "test_preprocess_stat.json", "w") as wf_json_set:
    json.dump(img_info_set, wf_json_set)


img_info_set_sorted = sorted(img_info_set, key=lambda x: (x['i_max_wh']), reverse=True)



i_buffer = len(img_info_set_sorted)


print("\n\n\n\n  largest: \n", i_buffer)

for x in range(0, 0 + i_buffer, 1):
    print("", img_info_set_sorted[x])

"""







exp_folder_prefix = output_data_folder + "exp_imgs"


#Let's show image with id 16
img_idx = 784

img_idx = 13
img_idx = 16
img_idx = 33071
i_out_max_wh, out_arr, out_region_area = op_img(img_idx, exp_folder_prefix, True, True)


#op_img(27, "./ceshi/")


logger.info("\n\n\n\n\n\n\n\n\n\nprogram ends. ")