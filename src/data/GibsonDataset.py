import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from PIL import Image
import cv2
import joblib
from util import get_image_to_tensor_balanced, get_mask_to_tensor


class GibsonDataset(torch.utils.data.Dataset):
    """
    Dataset from Gibson (Niemeyer et al. 2020)
    Provides 3D-R2N2 and NMR renderings
    """

    def __init__(
        self,
        path,
        stage="train",
        list_prefix="softras_",
        image_size=None,
        sub_format="shapenet",
        scale_focal=True,
        max_imgs=100000,
        z_near=1.2,
        z_far=4.0,
        skip_step=None,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        self.base_path = path + "/" + stage
        assert os.path.exists(self.base_path)

        cats = [x for x in glob.glob(os.path.join(self.base_path, "*")) if os.path.isdir(x)]

        file_lists = np.concatenate([glob.glob(x+"/*") for x in cats])

        # all_objs = []
        # for file_list in file_lists:
        #     if not os.path.exists(file_list):
        #         continue
        #     base_dir = os.path.dirname(file_list)
        #     cat = os.path.basename(base_dir)
        #     with open(file_list, "r") as f:
        #         objs = [(cat, os.path.join(base_dir, x.strip())) for x in f.readlines()]
        #     all_objs.extend(objs)

        self.all_objs = file_lists
        self.stage = stage

        # self.image_to_tensor = get_image_to_tensor_balanced(image_size=image_size)
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading Gibson dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
            "type:",
            sub_format,
        )

        self.image_size = image_size
        if sub_format == "dtu":
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        self.sub_format = sub_format
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False

    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        root_dir = self.all_objs[index]

        rgb_paths = [
            x
            for x in glob.glob(os.path.join(root_dir, "rgb", "*"))
            if (x.endswith(".jpg") or x.endswith(".png"))
        ]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob.glob(os.path.join(root_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]
            mask_paths = [mask_paths[i] for i in sel_indices]

        all_cam = []
        cam_paths = sorted(glob.glob(os.path.join(root_dir, "pose", "*.pkl")))
        for cam_path in cam_paths:
            all_cam.append(joblib.load(cam_path))
        # cam_path = os.path.join(root_dir, "cameras.npz")
        # all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        focal = None

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]
            # Decompose projection matrix
            P = all_cam[i]
            P = P[:3]

            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
            K = K / K[2, 2]

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R.transpose()
            pose[:3, 3] = (t[:3] / t[3])[:, 0]
            fx = torch.tensor(K[0, 0]) * x_scale
            fy = torch.tensor(K[1, 1]) * y_scale
            cx = (torch.tensor(K[0, 2]) + xy_delta) * x_scale
            cy = (torch.tensor(K[1, 2]) + xy_delta) * y_scale
            focal = torch.tensor((fx, fy), dtype=torch.float32)

            pose = (
                self._coord_trans_world
                @ torch.tensor(pose, dtype=torch.float32)
                @ self._coord_trans_cam
            )
            img_tensor = self.image_to_tensor(Image.fromarray(img))
            if mask_path is not None:
                mask_tensor = self.mask_to_tensor(mask)

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) > 0:
                    rmin, rmax = rnz[[0, -1]]
                    cmin, cmax = cnz[[0, -1]]
                    bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                    all_masks.append(mask_tensor)
                    all_bboxes.append(bbox)
                else:
                    all_masks.append(self.mask_to_tensor(mask))
                    all_bboxes.append(torch.tensor([0, 0, 1, 1], dtype=torch.float32))

            all_imgs.append(img_tensor)
            all_poses.append(pose)

        if self.sub_format != "shapenet":
            fx /= len(rgb_paths)
            fy /= len(rgb_paths)
            cx /= len(rgb_paths)
            cy /= len(rgb_paths)
            focal = torch.tensor((fx, fy), dtype=torch.float32)
            c = torch.tensor((cx, cy), dtype=torch.float32)
            all_bboxes = None
        elif mask_path is not None:
            all_bboxes = torch.stack(all_bboxes)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size / all_imgs.shape[-2]
            focal *= scale
            if self.sub_format != "shapenet":
                c *= scale
            elif mask_path is not None:
                all_bboxes *= scale
            all_bboxes = all_bboxes.clamp(0, self.image_size - 1)
            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "poses": all_poses,
        }
        if all_masks is not None:
            result["masks"] = all_masks
        if self.sub_format != "shapenet":
            result["c"] = c
        else:
            result["bbox"] = all_bboxes
        return result
