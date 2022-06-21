import argparse
import json
import cv2
import numpy as np
import os
import math
from svgpathtools import Path, CubicBezier, wsvg

from generate_bezier import *


parser = argparse.ArgumentParser(description='generate cut path from result.json')
parser.add_argument('-i', help='input result.json')
# parser.add_argument('-o', help='out dir name')

args = parser.parse_args()
# print(args.i)


result = open(os.path.join(args.i, 'result.json'))
result = json.load(result)

patches = result['patches']
sources = result['textures_source']
sources_img = []
# target = result['texture_target']
sources_output = []
sources_output1 = []
sources_output2 = []
for i in range(len(sources)):
    source_texture = cv2.imread(os.path.join(args.i, sources[i]['filename']))
    sources_img.append(source_texture)
    sources_output.append(np.ones(source_texture.shape, dtype=np.uint8)*255)
    sources_output1.append(np.ones(source_texture.shape, dtype=np.uint8)*255)
    sources_output2.append(np.ones(source_texture.shape, dtype=np.uint8)*255)
    # cv2.imshow('qwer'+str(i), source_texture)


final_curves = []
c_points = []
patch_path = []
cu_curves = [[] for _ in range(len(sources))] 
for i in range(len(patches)):
# for i in range(1):

    p_mask = cv2.imread(os.path.join(args.i, patches[i]['region_target']['mask']))
    bounding_box = patches[i]['region_target']['bounding_box']

    source_index = int(patches[i]['source_index'])
    source_texture = cv2.imread(os.path.join(args.i, sources[source_index]['filename']))

    anchor_src = patches[i]['anchor_source']

    ts = patches[i]['transformation_source']
    ts = np.array([float(x) for x in ts['mat']]).reshape(2,3)

    ts_inv = patches[i]['transformation_source_inv']
    ts_inv = np.array([float(x) for x in ts_inv['mat']]).reshape(2,3)
#######################


    # anc_point = np.array([anchor_src['x'], anchor_src['y']])
    # theta = math.acos(ts[0,0])*180/math.pi
    
    # curve_sides = ['curves_top', 'curves_right', 'curves_bot', 'curves_left']
    # for side in curve_sides:
    #     side_list = patches[i]['region_target'][side]
    #     for curve in side_list:
    #         c_points_list = curve['control_points']
    #         # print("33: ", len(c_points), "sad ", c_points[0]['x'])
    #         for point in c_points_list:
    #             c_points.append([point['x'], point['y']])
    
    #         c_points = np.array(c_points, dtype=np.float64).reshape(len(c_points), 2)
    #         transformed_points = np.dot(c_points, ts_inv[:, :2]) + ts_inv[:, 2]
    #         ps = transformed_points
            
            
    #         c = complex
    #         seg = CubicBezier(c(ps[0,0]+ps[0,1]), c(ps[1,0]+ps[1,1]), c(ps[2,0]+ps[2,1]), c(ps[3,0]+ps[3,1]))
    #         patch_path.append(seg)
    #         cu_curves[source_index].append(seg)
    #         c_points = []
    # if i != len(patches)-1: final_curves = []
######################
    w = int(bounding_box['width'])
    h = int(bounding_box['height'])
    src_t_size = (int(np.ceil(abs(source_texture.shape[1]*ts[0,0]) + abs(source_texture.shape[0]*ts[1,0]))), int(np.ceil(abs(source_texture.shape[1]*ts[0,1]) + abs(source_texture.shape[0]*ts[0,0]))))

    # cv2.imshow('0', source_texture)
    src_t = cv2.warpAffine(source_texture, ts, src_t_size)
    
    kernel = np.ones((3,3),np.uint8)
    msk_e = cv2.erode(p_mask, kernel)
    msk_e[0,:] =0
    msk_e[-1,:] =0
    msk_e[:,0] =0
    msk_e[:,-1] =0
    edge_p = p_mask - msk_e
    edge_p[:,:,2] = 0

    cut_src0 = np.ones((src_t_size[1], src_t_size[0], 3), dtype=np.uint8) * 255
    cut_src1 = np.ones((src_t_size[1], src_t_size[0], 3), dtype=np.uint8) * 255
    cut_src2 = np.ones((src_t_size[1], src_t_size[0], 3), dtype=np.uint8) * 255
    print(i, cut_src0[int(anchor_src['y']):int(anchor_src['y'])+h, int(anchor_src['x']):int(anchor_src['x'])+w].shape, edge_p.shape)
    cut_src0[int(anchor_src['y']):int(anchor_src['y'])+h, int(anchor_src['x']):int(anchor_src['x'])+w] = cut_src0[int(anchor_src['y']):int(anchor_src['y'])+h, int(anchor_src['x']):int(anchor_src['x'])+w]* (1- p_mask//255)
    cut_src1[int(anchor_src['y']):int(anchor_src['y'])+h, int(anchor_src['x']):int(anchor_src['x'])+w] = cut_src1[int(anchor_src['y']):int(anchor_src['y'])+h, int(anchor_src['x']):int(anchor_src['x'])+w]* (1- edge_p//255)
    cut_src2[int(anchor_src['y']):int(anchor_src['y'])+h, int(anchor_src['x']):int(anchor_src['x'])+w] = cut_src2[int(anchor_src['y']):int(anchor_src['y'])+h, int(anchor_src['x']):int(anchor_src['x'])+w]* (1- edge_p//255)
    
    # cv2.imshow('1', cut_src)
    
    org =  (int(anchor_src['x'])+int(w/2.7), int(anchor_src['y'])+int(h/1.7))
    cut_src2 = cv2.putText(cut_src2, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 210, 50), 1)

    cut_src0 = cv2.warpAffine(cut_src0, ts_inv, (source_texture.shape[1], source_texture.shape[0]))
    cut_src1 = cv2.warpAffine(cut_src1, ts_inv, (source_texture.shape[1], source_texture.shape[0]))
    cut_src2 = cv2.warpAffine(cut_src2, ts_inv, (source_texture.shape[1], source_texture.shape[0]))

    # cv2.imshow('1.5', p_mask)
    # print(anchor_src)
    _, thresh = cv2.threshold(cut_src0, 180, 255, 0)
    # cv2.imshow('2', cut_src)
    thresh = 255-thresh
    # cv2.imshow('2.1', thresh)
    # op = generate_ordered_points_cnt(thresh[:,:,0], 0,0)
    bzs = generate_bzcs_patch(thresh[:,:,0], 0,0, max_err=0.5)
    # print("1"*13)
    # print(bzs)
    cu_curves[source_index].append(bzs)
    
    src_t[int(anchor_src['y']):int(anchor_src['y'])+h, int(anchor_src['x']):int(anchor_src['x'])+w] = src_t[int(anchor_src['y']):int(anchor_src['y'])+h, int(anchor_src['x']):int(anchor_src['x'])+w]* (1- edge_p//255)
    # src_t_t = cv2.warpAffine(src_t, ts_inv, (source_texture.shape[1], source_texture.shape[0]))

    # if len(sources_output) == 0:
    # print(cut_src.shape, sources_output[source_index].shape)
    sources_output[source_index] = np.fmin(sources_output[source_index], cut_src0)
    sources_output1[source_index] = np.fmin(sources_output1[source_index], cut_src1)
    sources_output2[source_index] = np.fmin(sources_output2[source_index], cut_src2)

    # cv2.imshow('source_texture', source_texture)
    # cv2.imshow('s1tt', src_t_t)
    # cv2.imshow('src_t', src_t)
    # cv2.imshow('cut', cut_src)
    # cv2.imwrite(f'./p/{i}_{source_index}.jpg', cut_src)

    cv2.waitKey(0)
cv2.destroyAllWindows()

os.makedirs(os.path.join(args.i, "cut_path"), exist_ok = True)
for i in range(len(sources)):
    # cv2.imwrite(os.path.join(args.i, "cut_path", f'src_indx_{i}.jpg'), sources_output[i])
    cv2.imwrite(os.path.join(args.i, "cut_path", f'a_src_indx_{i}.jpg'), sources_output1[i])
    cv2.imwrite(os.path.join(args.i, "cut_path", f'b_src_indx_{i}.jpg'), sources_output2[i])
    cv2.imwrite(os.path.join(args.i, "cut_path", f'c_src_indx_{i}.jpg'), cv2.add(255-sources_output2[i], sources_img[i]))
# print("3"*13)

for i in range(len(sources)):
    attributes = []
    curves_in_src = []
    path = Path()
    for patch in cu_curves[i]:
        for ps in patch:
            # print("aaa: ", ps) 
            c = complex
            seg = CubicBezier(c(ps[0][0],ps[0][1]), c(ps[1][0],ps[1][1]), c(ps[2][0],ps[2][1]), c(ps[3][0],ps[3][1]))
            path.append(seg)
            # attributes.append({'styles': 'fill:"none";stroke:#000000;stroke-width:0.5;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1'})
            attributes.append({'fill': 'none', 'stroke': "#000000", 'stroke-width': '1'})
    h, w, _ = sources_output[i].shape
    svg_attr = {'height': f'{h}px', 'width': f'{w}px'}
    wsvg(path, filename=os.path.join(args.i, "cut_path", f'output_{i}.svg'), attributes=attributes, svg_attributes=svg_attr)


