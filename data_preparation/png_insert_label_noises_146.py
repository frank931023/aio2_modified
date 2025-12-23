# -*- coding: utf-8 -*-
"""
Noisy label injection for building extraction task 
    - For samples stored as single files and opened with cv2 (Massachussets dataset)

@author: liu_ch
"""

import os
import argparse
import cv2
import numpy as np

from random import seed, shuffle, randint, choice, random
    

def get_args():
    parser = argparse.ArgumentParser(description='Insert label noises into building extractiond data.')
    # data settings
    parser.add_argument('--data_dir', type=str, help='path-to-massachusetts-dataset',)
    parser.add_argument('--partition', type=str, choices=['train','test','val'], default='train')
    # saving directory
    parser.add_argument('--save_dir_name', type=str, default='ns_seg_rm_3')  
    # noise insertion settings
    parser.add_argument('--ns_types', nargs="+", default=['remove'], #['shift','erosion','dilation','rotation'], 
                        help='Candidate noise types.') 
    parser.add_argument('--ns_rates',  nargs="+", type=float, default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], #, 0.8],
                        help='Candidate noise rates.')
    # generate index layer on the fly or directly load generated ones
    parser.add_argument('--generate_index', dest='gind', action='store_false')  
    # seed
    parser.add_argument('--seed',  type=int, default=86,
                        help='Random seed.')
    return parser.parse_args()

# first verison
# def shift_noise(mask):
#     # mask_u8 = (mask > 0).astype(np.uint8) * 255
#     # contours, _ = cv2.findContours(mask_u8, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#     contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#     cnt = contours[0]
#     print("cnt: ", cnt)
#     e = max([cnt[:,:,i].max()-cnt[:,:,i].min()+1 for i in [0,1]]+[12])
#     shift_bound = int(e/20)
#     print("shift_bound: ", shift_bound)
    
#     # shift
#     rows,cols = mask.shape
#     M = np.float32([[1,0,randint(-shift_bound,shift_bound)],[0,1,randint(-shift_bound,shift_bound)]])
#     print("rols: ", rows, " cols: ", cols, " M: ", M)
#     dst = cv2.warpAffine(mask, M, (cols,rows))
#     print("dst: ", dst)
#     return dst

# second version
# def shift_noise(mask):
#     # 若整張都是 0，就直接回傳
#     if np.count_nonzero(mask) == 0:
#         return mask

#     mask_u8 = mask.astype(np.uint8)
#     contours, _ = cv2.findContours(mask_u8, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) == 0:
#         return mask

#     cnt = contours[0]
#     e = max([cnt[:,:,i].max()-cnt[:,:,i].min()+1 for i in [0,1]]+[12])
#     shift_bound = int(e/2)
    
#     rows, cols = mask.shape
#     M = np.float32([[1,0,randint(-shift_bound,shift_bound)],
#                     [0,1,randint(-shift_bound,shift_bound)]])
#     dst = cv2.warpAffine(mask, M, (cols,rows))
#     return dst

# third version
def shift_noise(mask, debug=False):
    print("np.unique(mask): ", np.unique(mask))

    if np.count_nonzero(mask) == 0:
        return mask

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("contours: ", contours)
    if len(contours) == 0:
        return mask

    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    rows, cols = mask_u8.shape

    # 最大可移動距離
    max_left = x
    max_right = cols - (x + w)
    max_up = y
    max_down = rows - (y + h)

    max_dx = min(max_left, max_right)
    max_dy = min(max_up, max_down)

    if max_dx > 0:
        dx = randint(-max_dx, max_dx)
    else:
        dx = 0
    if max_dy > 0:
        dy = randint(-max_dy, max_dy)
    else:
        dy = 0

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    dst = cv2.warpAffine(mask_u8, M, (cols, rows), flags=cv2.INTER_NEAREST, borderValue=0)
    print("dst: ", dst)

    # 若結果全黑 = 代表掉出去了 → 回傳原 mask
    # if np.count_nonzero(dst) == 0:
    #     if debug:
    #         print("[SHIFT WARNING] 建物消失！改回原 mask")
    #     return mask

    return dst



def erosion_noise(mask):
    kss = np.arange(5,15) # 3-7
    # kernel size
    while 1:
        ks1 = choice(kss)
        ks2 = choice(kss)
        if ks1+ks2>2:
            break
    # erosion
    kernel = np.ones((ks1,ks2), np.uint8)
    dst = cv2.erode(mask, kernel)
    return dst


def dilate_noise(mask):
    kss = np.arange(5,16) # 3-8
    # kernel size
    while 1:
        ks1 = choice(kss)
        ks2 = choice(kss)
        if ks1+ks2>2:
            break
    # dilation
    kernel = np.ones((ks1,ks2), np.uint8)
    dst = cv2.dilate(mask, kernel)
    return dst

# first version
# def rotate_noise(mask):
#     # img shape
#     rows,cols = mask.shape
#     # find origin of object
#     contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#     cnt = contours[0]
#     origin = [(cnt[:,:,i].max()+cnt[:,:,i].min())/2 for i in [0,1]]
#     # rotate
#     M = cv2.getRotationMatrix2D(origin,randint(10,350),1)
#     dst = cv2.warpAffine(mask, M, (cols,rows))  # rotating the image by random degree with respect to building center without any scaling
#     return dst

# second version
# def rotate_noise(mask):
#     if np.count_nonzero(mask) == 0:
#         return mask

#     mask_u8 = mask.astype(np.uint8)
#     rows, cols = mask_u8.shape
#     contours, _ = cv2.findContours(mask_u8, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours) == 0:
#         return mask

#     cnt = contours[0]
#     origin = [(cnt[:,:,i].max()+cnt[:,:,i].min())/2 for i in [0,1]]
#     M = cv2.getRotationMatrix2D(origin, randint(10,350), 1)
#     dst = cv2.warpAffine(mask_u8, M, (cols,rows))
#     return dst


# third verison
def rotate_noise(mask, debug=False):
    if np.count_nonzero(mask) == 0:
        return mask

    m = mask.astype(np.uint8)
    rows, cols = m.shape
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    if len(contours) == 0:
        return mask

    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    # 旋轉中心設在建物 bbox 中心，不會跑太遠
    cx = x + w / 2
    cy = y + h / 2

    angle = randint(15, 345)

    if debug:
        print(f"[ROTATE] angle={angle}, center=({cx:.1f},{cy:.1f})")

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    dst = cv2.warpAffine(m, M, (cols, rows))

    # 若結果全黑 → 表示建物掉出去了 → 回原 mask
    if np.count_nonzero(dst) == 0:
        if debug:
            print("[ROTATE WARNING] 建物消失！改回原 mask")
        return mask

    return dst

def remove_noise(mask):
    return np.zeros_like(mask, dtype=np.uint8)


def add_noise(mask):
    if random()>0.5:
        mask = rotate_noise(mask)
    if random()>0.5:
        mask = shift_noise(mask)
    if random()>0.5:
        mask = dilate_noise(mask)
    return mask


def insert_single_item_noise(mask, nst):
    # ===== debug =====
    # if np.sum(mask>0) == 0:
    #     print("警告: mask 是空的, 無法加 noise")
    #     return mask
    # =================

    if nst=='shift':
        dst = shift_noise(mask)
    elif nst=='erosion':
        dst = erosion_noise(mask)
    elif nst=='dilation':
        dst = dilate_noise(mask)
    elif nst=='rotation':
        dst = rotate_noise(mask)
    elif nst=='remove':
        dst = remove_noise(mask)
    else:
        raise ValueError('Given noise type is invalid!')
    return dst


def count_and_index_building(gt):
    # find contours
    contours, hierarchy = cv2.findContours(gt,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    # index buildings
    # mask = np.zeros(list(gt.shape)+[3], dtype=np.int_)
    mask = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    for ci, cnt in enumerate(contours):
        cv2.drawContours(mask, [cnt], 0, (ci+1,ci+1,ci+1), thickness=cv2.FILLED)
    # convert from RGB to gray
    mask = mask[:,:,0] # cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask*gt  # to refine the shapes of objects in index file
    return mask

# def count_and_index_building(gt):
#     """
#     將二值 gt 轉成 building index mask：
#       - 背景 = 0
#       - 第 i 個建物 = i  (i 從 1 開始)
#     """
#     # 先確保是 0/1 二值
#     gt_bin = (gt > 0).astype(np.uint8)

#     # 找外部輪廓
#     contours, hierarchy = cv2.findContours(
#         gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
#     )
    
#     # 建立單通道 index mask
#     idx_mask = np.zeros_like(gt_bin, dtype=np.int32)

#     # 注意 enumerate 從 1 開始，這樣 label = 1,2,3,...
#     for ci, cnt in enumerate(contours, start=1):
#         cv2.drawContours(idx_mask, [cnt], 0, ci, thickness=cv2.FILLED)

#     return idx_mask





if __name__ == '__main__':
    # 0 - preparations
    args = get_args()
    seed(args.seed)
    nsrs = np.array(args.ns_rates)
    
    # directories
    # original data with building indexes
    if args.gind:
        ind_dir = os.path.join(args.data_dir,args.partition,'seg')
    else:
        ind_dir = os.path.join(args.data_dir,args.partition,'index')
    # saving directory for generated noisy labels
    save_dir = os.path.join(args.data_dir,args.partition, args.save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # get all file names
    fnames = os.listdir(ind_dir)
    
    # 1 - check whether require adding noises for no-object patches
    if 'add' in args.ns_types:
        add_bd = True
        add_obj = None
        args.ns_types.remove('add')
    else:
        add_bd = False
    
    # candidates of numbers/types of label noise insertion into each patch
    nsts = np.arange(len(args.ns_types))+1
    
    # 2 - starting inserting label noises one by one
    for fname in fnames:
        fpath = os.path.join(ind_dir,fname)
        
        gt = cv2.imread(fpath,0)
        _, gt = cv2.threshold(gt, 10, 255, cv2.THRESH_BINARY)
        # convert gt matrix to building indexes
        if args.gind: 
            gt = count_and_index_building(gt)
        mask = np.zeros_like(gt,dtype=np.uint8)
        n_bd = gt.max()


        # gt = cv2.imread(fpath, 0)

        # # 先把灰階變成 0/1
        # _, gt = cv2.threshold(gt, 10, 255, cv2.THRESH_BINARY)
        # gt = (gt > 0).astype(np.uint8)

        # # 產生 index mask
        # if args.gind:
        #     gt = count_and_index_building(gt)

        # mask = np.zeros_like(gt, dtype=np.uint8)
        # n_bd = int(gt.max())   # 現在就是「建物個數」，不會是 59022 這種鬼東西

        
        if n_bd>0: # buidlings exist in current patch
            # 1> building list
            bd_ilist = list(np.arange(n_bd)+1)
            
            # 2> select number and types of label noises
            # number
            shuffle(nsts)
            n_nst = nsts[0]
            # types
            shuffle(nsts)
            inds_nst = nsts[:n_nst]-1
            
            # 3> determine noise rate
            shuffle(nsrs)
            nsr = nsrs[0]
            
            # 4> insert label noises type by type
            portion_rest = 1.
            for ti in range(n_nst):
                shuffle(bd_ilist)
                # noise type (str) to insert
                nst = args.ns_types[inds_nst[ti]]
                # corresponding portion
                p = portion_rest if ti == n_nst-1 else portion_rest*random()
                portion_rest -= p
                # number of buildings to modify
                n_mbd = int(n_bd*nsr*p)
                # insert label noises one building by one building
                for bi in range(n_mbd):
                    bind = bd_ilist[0]
                    org_mask = (gt==bind).astype(np.uint8)

                    # ======== debug ========
                    print("處理檔案:", fname)
                    print("noise type:", nst)
                    print("building index:", bind)
                    print("org_mask shape:", org_mask.shape)
                    print("org_mask 非零像素數:", np.sum(org_mask>0))
                    # =======================

                    try:
                       mask += insert_single_item_noise(org_mask,nst)
                    except Exception as e:
                        print("Insert noise error: ", e)

                    # remove current processed building id
                    bd_ilist.pop(0)
            
            # 5> for add_noise, update add_obj
            if add_bd:
                if random()>0.7:
                    add_obj_ind = randint(1,n_bd)
                    add_obj = np.zeros_like(gt,dtype=np.uint8)
                    add_obj[gt==add_obj_ind] = 1
            
            # 6> add untouched objects
            for bi in bd_ilist:
                mask += (gt==bi).astype(np.uint8)
                
        else: # no buildings in current patch
            if add_bd:    
                if (add_obj is not None) and random()>0.5:
                    mask += add_noise(add_obj)
        
        # convert mask to binary map
        mask[mask>1] = 1
        
        # save new modified mask
        psave = os.path.join(save_dir,fname)
        cv2.imwrite(psave, mask)
        
    
    
    
    
    
    
    
    
    
    