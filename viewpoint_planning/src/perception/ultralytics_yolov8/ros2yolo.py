#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import torch
import cv2
import numpy as np
import sys
import time
import pandas as pd
import message_filters
from offboard_py.msg import Orientation 

sys.path.append('/home/lee/catkin_ws/src/yolov5')
yolov5_path="/home/lee/catkin_ws/src/yolov5"
weight_path="/home/lee/catkin_ws/src/yolov5/weight/yolo5s_best.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = torch.load('/home/lee/catkin_ws/src/offboard_py/scripts/model/yolomodel.pth')
model = torch.hub.load(yolov5_path, 'custom',path=weight_path, source='local',force_reload=True)
model = model.cuda()


model.conf = 0.5


moving= Orientation()

bridge = CvBridge()
pub = rospy.Publisher('/yolo/detections', Image, queue_size=10)
pub_moving = rospy.Publisher('orientation_topic', Orientation, queue_size=10)


last_time = 0
camera_info = None
foward_state=0
key=0

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 如果不允许放大，则r不能大于1
        r = min(r, 1.0)

    ratio = r, r  # 宽度和高度的比率
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 调整后的新尺寸

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 需要添加的总边框宽度和高度

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 确保边框宽度是stride的倍数
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # 分割到两侧
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)



    return im




def callback(color_msg, depth_msg):
    global last_time
    # current_time = time.time()
    
    # # 控制处理速率，约每秒30帧
    # if current_time - last_time < 1.0 / 30:
    #     return
    # last_time = current_time
    
    try:
        # 将ROS图像消息转换为OpenCV图像
        cv_image = bridge.imgmsg_to_cv2(color_msg, "bgr8")

        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    except CvBridgeError as e:
        rospy.logerr(e)
        return
    # print(depth_msg)
    # 处理彩色图像以适应YOLO模型
    img = letterbox(cv_image, new_shape=(640, 640), auto=False)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = np.transpose(img, (2, 0, 1))

    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float()

    img /= 255.0  # 归一化

    img = img.unsqueeze(0)  # 增加批次维度


    # 使用YOLO模型进行物体检测
    # with torch.no_grad(): 
    detections = model(img)
    
    #TODO understand layers of detection

    detections=detections[detections[ : , : ,4] > model.conf]
    # box_values = detections[:, :4]
    dectshow(img,detections)
    orientation(depth_image,detections)
    # print(box_values)
    # print(detections[:,2,:])
    # print("detections:",detections,detections.shape)
    # print("detections_type:",type(detections))
    
def orientation(d_img,boxs):   # input: d_img: depth Image,  boxs: detected result BB box from the RGB image
    #image size 640*640
    c_d_img=d_img.copy()
    c_d_img[c_d_img == 0] = 9999
    origin=[640//2,640//2]
    norm_vector=np.array([0,0])
    foward_state=0
    radius = 200
    threshold =600
  
    check_center_x,check_center_y=np.ogrid[:640,:640]
    square_mask = (check_center_x >= (origin[0] - radius)) & (check_center_x < (origin[0] + radius)) & \
            (check_center_y >= (origin[1] - radius)) & (check_center_y < (origin[1] + radius))
    depth_in_square = c_d_img[square_mask]
    depth_in_square[depth_in_square == 0] = 9999
    if ((boxs.nelement() != 0) & (np.any((c_d_img > (threshold)))) ):
        sum_box_h_w= boxs[:, 2] * boxs[:, 3]
        max_value, max_index = torch.max(sum_box_h_w, dim=0)
        box=boxs[int(max_index)]
        global key
        x=int(box[0])
        y=int(box[1])
        h=int(box[2])//2
        w=int(box[3])//2
        # box_origin=[x,y]
        vctor=np.array([-(origin[0] - x) , (origin[1]- y-40)])

        norm_vector=vctor/320.0
        
        key=1
        # norm = np.linalg.norm(vctor)


        # if norm == 0:
        #     norm_vector = vctor
        # else:
        #     norm_vector = vctor / norm
        if (np.all((depth_in_square >= threshold) )):
            print("R1")
            foward_state=1
        else:
            foward_state=0
            print("R0")
        # print(d_img[320,320]) 
        # all_greater_than_threshold = np.all(depth_in_circle > threshold)
    elif (np.any((depth_in_square <= threshold))):
            foward_state=2
            print("R2")
            # print("1:",d_img[320,320])
            # print("move")
    elif (np.all((depth_in_square > threshold))):
        foward_state=3
        print("R3")
        # elif (np.any((depth_in_square < threshold))):
        #     foward_state=0
        #     print("0")


                # print("0:",d_img[320,320])
                # print("stop")
        # elif ((np.any((d_img > threshold)) | np.all((d_img ==0)))):
        #     foward_state=2

    # print(np.all((depth_in_square[depth_in_square != 0] < 200)))
    if np.any((depth_in_square[depth_in_square != 0] <= 350)):
        # else:
            foward_state=4
            print("R4:",d_img[320,320])
    # if (np.any((depth_in_square < threshold))):
    # if (np.any((depth_in_square < threshold))):

        # mask = np.zeros_like(d_img, dtype=bool)
        # mask[0, :] = True  # 顶部边缘
        # mask[-1, :] = True  # 底部边缘
        # mask[:, 0] = True  # 左侧边缘
        # mask[:, -1] = True  # 右侧边缘

        # # 筛选深度值小于400的边缘像素的坐标
        # y, x = np.where((d_img < threshold) & (d_img != 0) & mask)

        # # 按边分类并计算平均坐标
        # top_edge_coords = np.mean(x[y == 0]) if np.any(y == 0) else None
        # bottom_edge_coords = np.mean(x[y == d_img.shape[0]-1]) if np.any(y == d_img.shape[0]-1) else None
        # left_edge_coords = np.mean(y[x == 0]) if np.any(x == 0) else None
        # right_edge_coords = np.mean(y[x == d_img.shape[1]-1]) if np.any(x == d_img.shape[1]-1) else None

        # # 将平均坐标组装成 [x, y] 形式
        # top_edge_point = [top_edge_coords, 0] if top_edge_coords is not None else None
        # bottom_edge_point = [bottom_edge_coords, d_img.shape[0]-1] if bottom_edge_coords is not None else None
        # left_edge_point = [0, left_edge_coords] if left_edge_coords is not None else None
        # right_edge_point = [d_img.shape[1]-1, right_edge_coords] if right_edge_coords is not None else None

        # points = [point for point in [top_edge_point, bottom_edge_point, left_edge_point, right_edge_point] if point is not None]
        # if points:
        #     average_point = np.mean(points, axis=0)
        #     vctor=np.array([-(origin[0] - average_point[0]) , (origin[1]- average_point[1])])
        #     norm_vector=vctor/320.0
        #     print(norm_vector)
        # if points:
        #     average_point = np.mean(points, axis=0)
        #     vctor=np.array([-(origin[0] - average_point[0]) , (origin[1]- average_point[1])])
        #     norm_vector=vctor/320.0
        #     print(norm_vector)

        # print(top_edge_point)
        # print(bottom_edge_point)
        # print(left_edge_point)
        # print(right_edge_point)
        # print("---------------------------------")
        # midpoint = np.mean(valid_points[0])
        # print(midpoint)linearlinear
    moving.norm_vector = norm_vector.tolist()  # 将numpy数组转换为列表c
    moving.foward_state = foward_state
    pub_moving.publish(moving)


def dectshow(org_img1, boxs):  # input: org_img1: original RGB Image,  boxs: detected result BB box from the RGB image;   The purpose of this one is to republish the image onto rqt_image-view for human viewing

    w1=h1=100
    # print(org_img1)
    org_img_cpu = org_img1.cpu()

    org_img_np = org_img_cpu.numpy().astype(np.uint8)
    if org_img_cpu.dim() == 4:
        org_img_np = org_img_cpu.squeeze(0).permute(1, 2, 0).numpy()
    elif org_img_cpu.dim() == 3:
        org_img_np = org_img_cpu.permute(1, 2, 0).numpy()

    
    if org_img_np.dtype == np.float32:

        org_img_np = np.clip(org_img_np * 255, 0, 255).astype(np.uint8)

    org_img_np = np.copy(org_img_np)

    org_img_np = cv2.cvtColor(org_img_np, cv2.COLOR_RGB2BGR)
    if boxs.nelement() != 0:
        for box in boxs:

            # b, g, r = cv2.split(org_img_np)
            # org_img_np = cv2.merge([r, g, b])
            
            x=int(box[0])
            y=int(box[1])
      
            
            h=int(box[2])//2
            w=int(box[3])//2
            cv2.circle(org_img_np, (int(box[0]),int(box[1])), 12, (0, 0, 255), -1)
            # cv2.circle(org_img_np, (int(box[0]),int(box[1])), 40, (0, 255, ), 2)
            
            # cv2.rectangle(org_img_np, (x-w,y-h),(x+w,y+h), (0, 255, 0), 1)


            # org_img_np = cv2.cvtColor(org_img_np, cv2.COLOR_BGR2RGB)

    try:
        cv2.rectangle(org_img_np, (320-w1,320-h1),(320+w1,320+h1), (255,0,  0), 1)
        ros_image = bridge.cv2_to_imgmsg(org_img_np, "bgr8")
        pub.publish(ros_image)
        # rospy.loginfo("Published image to /yolo/detections")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))






def main():
    
    rospy.init_node('yolo_processor_node')

    color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)

    ats = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.5)
    ats.registerCallback(callback)

    rospy.spin()
    print("main down")


if __name__ == '__main__':
    main()





#  d_img_np = d_img
#                 d_img_np = np.where(d_img_np < 400, d_img_np, -1)
#                 # top_edge_point = [0,(np.mean(d_img_np[0, :][d_img_np[0, :] >= 0]) if np.any(d_img_np[0, :] >= 0) else None)]
#                 # botto-20edge_point = [(np.mean(d_img_np[:, 0][d_img_np[:, 0] >= 0]) if np.any(d_img_np[:, 0] >= 0) else None),0]
#                 # right_edge_point = [(np.mean(d_img_np[:, -1][d_img_np[:, -1] >= 0]) if np.any(d_img_np[:, -1] >= 0) else None),640]
#                 mask = np.zeros_like(d_img, dtype=bool)
#         print(1)       # 筛选深度值小于400的边缘像素的坐标
#                 y, x = np.where((d_img < 200) & mask)
#                 print(bottom_edge_point)
#                 print(left_edge_point)
#                 print(right_edge_point)linear
#                 # midpoint = np.mean(valid_points[0])
#                 # print(midpoint)linearlig.shape[foward_state[1]-1]) if np.any(x == d_img.shape[1]-1) else None

#                 # 将平均坐标组装成 [x, y] 形式
#                 top_edge_point = [top_edge_coords, 0] if top_edge_coords is not None else None
#                 bottom_edge_point = [bottom_edge_coords, d_img.shape[0]-1] if bottom_edge_coords is not None else None
#                 left_edge_point = [0, left_edge_coords] if left_edge_coords is not None else None
#                 right_edge_point = [d_img.shape[1]-1, right_edge_coords] if right_edge_coords is not None else None

                
#                 print(top_edge_point)
#                 print(bottom_edge_point)
#                 print(left_edge_point)
#                 print(right_edge_point)linear
#                 # midpoint = np.mean(valid_points[0])
#                 # print(midpoint)linearlinear