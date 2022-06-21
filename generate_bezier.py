import numpy as np
import cv2

from fitCurves import fitCurve


def patch_edge(p_img):
    kernel = np.zeros((3,3), dtype=np.uint8)
    kernel[0,1] = 1
    kernel[1,1] = 1
    kernel[2,1] = 1
    kernel[1,0] = 1
    kernel[1,2] = 1
    eroded_img = cv2.erode(p_img, kernel,
                            borderType=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    border = p_img - eroded_img 
    # border_comp = cv2.bitwise_not(border)
    # kernel = np.ones((3,3), dtype=np.uint8)
    # kernel[0,1] = 1

    # return cv2.bitwise_or(cv2.bitwise_or(border1, border2), cv2.bitwise_or(border3, border4))
    return border

def edge_to_points(e_img, x_offset, y_offset):
    mask = (e_img > 200 )*255
    # img = e_img[mask]
    return np.int64(np.squeeze(cv2.findNonZero(mask))+ np.array([x_offset, y_offset]))
    
def find_path_recurseive(points, current_pint, ordered, initial_cost):
    # print("@-"*15)
    # print('points: ')
    # print(points)
    # print("current_pint")
    # print(current_pint)
    # print("ordered")
    # print(ordered)
    # print("initial cost")
    # print(initial_cost)

    if len(points) == 1:
        # ordered.append(current_pint)
        return  initial_cost + 1

    err = np.sum((points - current_pint)**2, axis=1, keepdims=True)
    mask = err < 0.5 
    idx = np.where(mask == True)[0][0]
    points = np.vstack((points[:idx], points[idx+1:]))
    # print ("err ", err)
    # print (' points:\n ', points)
    err = np.sum((points - current_pint)**2, axis=1, keepdims=True)
    idxs = np.argsort(err, axis=0)
    nearests = [points[idxs[0]][0]]            
    # print ("ids \n", idxs)
    # print('NEARESTs \n ', nearests)
    # print("$-"*12)
    for i in idxs[1:]:
        # print()
        if len(idxs)==1:
            break
        if err[i[0]][0] == err[idxs[0]][0]:
            nearests.append(points[i[0]])
        else:
            break
    
    # print("Nearests: \n", nearests)
    cost_opt = 99999999 
    for n in nearests:
        # print(n, current_pint)
        if np.sum((n - current_pint)**2) > 4:
            # print("too far: ", n, current_pint)
            return initial_cost
        # print("RRRR", points, n)
        ol = [n]
        cost = find_path_recurseive(points, n, ol, initial_cost+1)
        if cost < cost_opt:
            # next_p = n
            ol_opt = ol
            cost_opt = cost

    # print("ol_opt: \n" , ol_opt, cost_opt)

    ordered += ol_opt   
    # ordered.append(next_p)
    # print("out: ", ol_opt)
    return cost_opt

def find_path_greedy(points, initial):
    out = [initial]
    #removint initial point
    err = np.sum((points - initial)**2, axis=1, keepdims=True)
    mask = err < 0.5 
    idx = np.where(mask == True)[0][0]
    points = np.vstack((points[:idx], points[idx+1:]))
    
    while len(points)!=0: 
        print('out', out[-1])
        print('points', points)
        err = np.sum((points - out[-1])**2, axis=1, keepdims=True)
        idxs = np.argsort(err, axis=0)
        nearest = points[idxs[0]][0]
        print('nearest', nearest)
        print("err", err[idxs[0]])
        if err[idxs[0]] > 4:
            break
        out.append(nearest)

        #removint p 
        err = np.sum((points - nearest)**2, axis=1, keepdims=True)
        mask = err < 0.5 
        idx = np.where(mask == True)[0][0]
        print(idx)
        points = np.vstack((points[:idx], points[idx+1:]))

    return out


def generate_ordered_points_cnt(img_p, x_offset, y_offset):
    _, thresh = cv2.threshold(img_p, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    points_orderd = contours[0]
    points_orderd += np.array([x_offset, y_offset])
    points_orderd = np.squeeze(points_orderd)
    _, e, r = np.unique(points_orderd, return_inverse=True, return_counts=True, axis=0)
    points_orderd = points_orderd[r[e]==1]
    points_orderd = points_orderd.tolist()
    # points_orderd.append(points_orderd[0])

    points_orderd = list(map(lambda x : np.array(x), points_orderd))
    return points_orderd


def divide_to_four_segment(beziers):
    """return a dictionary of four (top, right,left, bottom) curves."""
    num_total_cp = len(beziers) * len(beziers[0])
    min_num_chunks = int(num_total_cp/4.1)
    
    length_chunk = 0 
    segments = []
    segment = []

    for bezier in beziers:
        for point in bezier:
            if len(segments) >= 3:
                segment.append(point)    
                continue

            if length_chunk < min_num_chunks:
                segment.append(point)
            else:
                if np.sum((point - segment[-1])**2) < 0.000001: # those 2 must be the same.
                    segments.append(segment)
                    segment = [point]
                    length_chunk = 0

                else:
                    segment.append(point)
            
            length_chunk += 1

    segments.append(segment)
    return segments


def generate_ordered_points(p_img, x_offset, y_offset, method=0):
    img_edge = patch_edge(p_img)

    points = edge_to_points(img_edge, x_offset, y_offset)
    init_p = points[0]
    points_orderd = [init_p]

    if method == 0:
        points_orderd = generate_ordered_points_cnt(p_img, x_offset, y_offset)
    
    elif method == 1:
        points_orderd = find_path_greedy(points, init_p)
        
    elif method == 2:
        cost = find_path_recurseive(points, init_p, points_orderd, 0)
    

    points_orderd.append(points_orderd[0])
    return points_orderd


def generate_bzcs_patch(p_img, x_offset, y_offset, method=0, max_err=0.1):
    
    points_orderd = generate_ordered_points(p_img, x_offset, y_offset, method)
    beziers = fitCurve(points_orderd, float(max_err))

    return beziers

def generate_4curves_patch(p_img, x_offset, y_offset, method=0, max_err=0.1):
    
    points_orderd = generate_ordered_points(p_img, x_offset, y_offset, method)
    beziers = fitCurve(points_orderd, float(max_err))

    segments = divide_to_four_segment(beziers)

    return segments
