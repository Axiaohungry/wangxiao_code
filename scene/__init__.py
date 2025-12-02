#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from utils.system_utils import mkdir_p
from scene.NVDIFFREC import save_env_map, load_env

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        # print(os.path.join(args.source_path, "transforms_train.json"))
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            # scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            scene_info = sceneLoadTypeCallbacks["Blender_Mesh_time"](
                    args.source_path, args.white_background, args.eval#, args.num_splats[0]
                )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"),
                                              og_number_points=len(scene_info.point_cloud.points))
            if self.gaussians.brdf:
                fn = os.path.join(self.model_path,
                                "brdf_mlp",
                                "iteration_" + str(self.loaded_iter),
                                "brdf_mlp.hdr")
                self.gaussians.brdf_mlp = load_env(fn, scale=1.0)
                print(f"Load envmap from: {fn}")
                # 1. 先判断是否需要使用时间特征
                if self.gaussians.feature_time:
                    # 构建网络权重的完整路径
                    net_path = os.path.join(self.model_path,
                                            "fature_time_net",
                                            "iteration_" + str(self.loaded_iter),
                                            "fature_time_net.pth")

                    # 2. 检查文件是否存在
                    if os.path.exists(net_path):
                        print(f"Loading feature time net from {net_path}")
                        self.gaussians.load_net(net_path, scene_info.spacefeatures)
                    else:
                        # 3. 如果文件不存在（比如跑的是纯静态3DGS），打印警告但不报错，继续向下走
                        print(f"[Warning] Feature time net config is TRUE, but file not found at {net_path}.")
                        print("[Warning] Skipping network load. Assuming static model for visualization.")
                        # 这里可以选择是否要把 feature_time 强制置为 False，防止后续调用出错
                        # self.gaussians.feature_time = False

                else:
                    # 4. 常规静态初始化（针对没有任何 checkpoint 的情况，或者显式关闭 feature_time）
                    print("Initializing from Point Cloud (Static Mode)...")
                    self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.spacefeatures,
                                                   self.cameras_extent)
        # if gaussians.feature_time:
        #     gaussians.space_feature = scene_info.spacefeatures.cuda()
            
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.gaussians.brdf:
            brdf_mlp_path = os.path.join(self.model_path, f"brdf_mlp/iteration_{iteration}/brdf_mlp.hdr")
            mkdir_p(os.path.dirname(brdf_mlp_path))
            save_env_map(brdf_mlp_path, self.gaussians.brdf_mlp)
        if self.gaussians.feature_time:
            fature_time_net_path = os.path.join(self.model_path, f"fature_time_net/iteration_{iteration}/fature_time_net.pth")
            mkdir_p(os.path.dirname(fature_time_net_path))
            self.gaussians.save_net(fature_time_net_path)
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]