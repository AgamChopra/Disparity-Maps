import numpy as np
from PIL import Image
from tqdm import trange

MAX = 999999

def pad_zeros(a,w):
    w_ = int(w/2)
    a = np.pad(a,((w_,w_),(w_,w_)),'constant')
    return a
    

def compute_err_rate(ground, estimated):  #computes error % of int(pix_values/64) - that of ground truth > 1
    ground = np.round((np.asarray(Image.open(ground).convert('L'))/4))
    estimated = np.round((np.asarray(Image.open(estimated))/4))    
    bad_pix = np.sum(np.where(np.abs(estimated - ground) > 1, 1, 0))
    err_rate = round(100 * bad_pix / (ground.shape[0] * ground.shape[1]), 2)   
    return err_rate


def rank_transform(image, tran_windowsize):
    w, h = image.size
    img_arr = np.asarray(image)
    trans_img = np.zeros((w,h), np.uint8)
    trans_img.shape = h,w
    img_arr = pad_zeros(img_arr,tran_windowsize)
    trans_img = pad_zeros(trans_img,tran_windowsize)
    window_ = int(tran_windowsize / 2)
    for y in range(window_,h-window_):
        for x in range(window_,w-window_):
            rank = 0
            for v in range(-window_,window_+1):
                for u in range(-window_,window_+1):
                    if (img_arr[y + v, x + u] > img_arr[y, x]):
                        rank = rank+1
            trans_img[y, x] = rank	
    return trans_img[window_:h+window_, window_:w+window_]


def stereo_match(left_img, right_img, kernel, max_offset):
    left_img = Image.open(left_img).convert('L')
    right_img = Image.open(right_img).convert('L')
    w, h = left_img.size  # assume that both images are same size   
    left = rank_transform(left_img,5)
    right = rank_transform(right_img,5)
    depth = np.zeros((h, w), np.uint8)
    kernel_ = int(kernel / 2)      
    for y in trange(kernel_, h - kernel_):              
        for x in range(kernel_, w - kernel_):
            best_offset = 0
            prev_sad = MAX           
            for offset in range(max_offset):               
                sad = 0                                          
                for dy in range(-kernel_, kernel_+1):
                    for dx in range(-kernel_, kernel_+1): 
                        sad += abs(int(left[y+dy, x+dx]) - int(right[y+dy, (x+dx) - offset]))                           
                if sad < prev_sad:
                    prev_sad = sad
                    best_offset = offset                           
            depth[y, x] = best_offset * 255 / max_offset   
    im = Image.fromarray(depth)
    im.show()                            
    Image.fromarray(depth).save('R:/classes 2020-22/Spring 2022/CS 532 3D CV/HW2/teddy 2/teddy/disp_map_%d.pgm'%(kernel))


def confidence_cost(ground, estimated, d = 12):
    epsilon = 1E-9
    [w, h] = ground.shape
    C = np.zeros((w,h,d)) 
    c1, c2 =  np.ones((w,h)) * MAX, np.ones((w,h)) * MAX   
    for i in trange(w):
        for j in range(d,h):
            for k in range(d):
                C[i,j,k] = np.abs(ground[i,j] - estimated[i,j-d])
                if c1[i,j] == MAX:
                    c1[i,j] = C[i,j,k]
                elif c1[i,j] < C[i,j,k] and c2[i,j] == MAX:
                    c2[i,j] = C[i,j,k]
                elif c2[i,j] != MAX:
                    break
    return c1+epsilon,c2+epsilon


def confidence_analysis(ground_, estimated_, k):
    ground = np.round((np.asarray(Image.open(ground_).convert('L'))/4))
    estimated = np.round((np.asarray(Image.open(estimated_))/4))    
    c1,c2 = confidence_cost(ground, estimated) #returns c1 and c2 values for all pixel locations c1.shape=c2.shape=ground.shape
    C_PKRN = c2 / c1    
    C_flat = C_PKRN.flatten()
    mid_C_PKRN = np.sort(C_flat)[::-1][:int(len(C_flat)/2)][-1]    
    mask = np.where(C_PKRN >= mid_C_PKRN, 1, 0)    
    pix_kept = np.sum(mask)    
    print('# of pix kept =', pix_kept, '(', round(100*pix_kept/len(C_flat),2),'% )')    
    bad_pix = np.sum(np.where(np.abs(estimated - ground) > 1, 1, 0) * mask)
    err_rate = round(100 * bad_pix / pix_kept, 2)    
    print('Error rate of sparse disp map =', err_rate, '%')    
    disp_map_sparse = Image.open(estimated_) * mask
    im = Image.fromarray(disp_map_sparse.astype(np.uint8))
    im.show()
    im.save('R:/classes 2020-22/Spring 2022/CS 532 3D CV/HW2/teddy 2/teddy/disp_map_%d_sparse.pgm'%(k))    
    return pix_kept, err_rate, disp_map_sparse


if __name__ == '__main__':
    print('---------3x3---------')
    k = 3
    stereo_match("R:/classes 2020-22/Spring 2022/CS 532 3D CV/HW2/teddy 2/teddy/teddyL.pgm", "R:/classes 2020-22/Spring 2022/CS 532 3D CV/HW2/teddy 2/teddy/teddyR.pgm", k, 63)
    ground = "R:/classes 2020-22/Spring 2022/CS 532 3D CV/HW2/teddy 2/teddy/disp2.pgm"
    estimated = "R:/classes 2020-22/Spring 2022/CS 532 3D CV/HW2/teddy 2/teddy/disp_map_%d.pgm"%(k)
    print("Error rate for kernel size of",k,"=",compute_err_rate(ground, estimated),"%")
    confidence_analysis(ground, estimated, k)
    print('---------\n')
    print('---------15x15---------')
    k = 15
    stereo_match("R:/classes 2020-22/Spring 2022/CS 532 3D CV/HW2/teddy 2/teddy/teddyL.pgm", "R:/classes 2020-22/Spring 2022/CS 532 3D CV/HW2/teddy 2/teddy/teddyR.pgm", k, 63)
    ground = "R:/classes 2020-22/Spring 2022/CS 532 3D CV/HW2/teddy 2/teddy/disp2.pgm"
    estimated = "R:/classes 2020-22/Spring 2022/CS 532 3D CV/HW2/teddy 2/teddy/disp_map_%d.pgm"%(k)
    print("Error rate for kernel size of",k,"=",compute_err_rate(ground, estimated),"%")
    confidence_analysis(ground, estimated, k)
    print('---------')