import os
import h5py
import numpy as np
import pandas as pd
import nibabel as nib

from collections import defaultdict
from scipy.interpolate import interp1d, RegularGridInterpolator

import torch
from torch.utils import data

import functions as funcs


class causality_learning_data_processing:
    def __init__(self, processed_data_path, max_branch=16, volume_shape=[256, 64, 64], mask_ct_value=-1024):
        super().__init__()

        self.processed_data_path = processed_data_path
        self.max_branch = max_branch
        self.volume_shape = volume_shape
        self.mask_ct_value = mask_ct_value

        self.standardized_data_path = os.path.join(processed_data_path, 'standardized_data')
        self.augmented_data_path = os.path.join(processed_data_path, 'augmented_data')
        self.patientlevel_data_path = os.path.join(processed_data_path, 'patient_level_data')

        if not os.path.exists(self.augmented_data_path):
            os.makedirs(self.augmented_data_path)
        if not os.path.exists(self.augmented_data_path):
            os.makedirs(self.augmented_data_path)
        if not os.path.exists(self.patientlevel_data_path):
            os.makedirs(self.patientlevel_data_path)

    def initialize_raw_data(self, raw_data_path):

        self.case_branch_standardization(raw_data_path)
        self.decoupling_drive_volume_masking()
        self.patient_level_grouping()

        return

    def case_branch_standardization(self, original_data_path):

        funcs.prepare_folder(self.standardized_data_path)

        for case_name in sorted(os.listdir(original_data_path)):
            case_dir = os.path.join(original_data_path, case_name)
            if not os.path.isdir(case_dir):
                continue

            volumes_dir = os.path.join(case_dir, "volumes")
            labels_dir = os.path.join(case_dir, "labels")

            if not os.path.isdir(volumes_dir):
                continue

            for branch_id in range(self.max_branch):
                volume_path = os.path.join(volumes_dir, f"branch_{branch_id}.nii")
                label_path = os.path.join(labels_dir, f"branch_{branch_id}.csv")

                if not os.path.isfile(volume_path):
                    continue

                label_data, volume_data = self.load_branch_data(label_path, volume_path)
                label_data, volume_data = self.resize_label_and_volume(label_data, volume_data)
                label_data = self.detection_target_generation(label_data)

                save_path = os.path.join(self.standardized_data_path, rf'{case_name}_{str(branch_id).zfill(2)}.h5')
                self.save_standardized_maps(volume_data, label_data, save_path)

        return

    def decoupling_drive_volume_masking(self):

        funcs.prepare_folder(self.augmented_data_path)

        for img_name in sorted(os.listdir(self.standardized_data_path)):
            case_dir = os.path.join(self.standardized_data_path, img_name)
            volume_data, label_data = self.load_single_branch_maps(case_dir)

            self.augment_single_image(volume_data, label_data, self.augmented_data_path, os.path.splitext(img_name)[0])

    def patient_level_grouping(self):

        casename_files = defaultdict(list)

        for fname in os.listdir(self.standardized_data_path):
            if not fname.endswith('.h5'):
                continue

            base_name = os.path.splitext(fname)[0]
            if '_' not in base_name:
                continue

            casename, branchid = base_name.rsplit('_', 1)
            casename_files[casename].append(os.path.join(self.standardized_data_path, fname))

        for casename, files in casename_files.items():

            volumes_list, labels_list = [], []

            for file in files:
                volume_data, label_data = self.load_single_branch_maps(file)
                volumes_list.append(volume_data)
                labels_list.append(label_data)

            num_to_fill = self.max_branch - len(files)
            if num_to_fill > 0:
                for _ in range(num_to_fill):
                    empty_vol, empty_label = self.create_empty_sample()
                    volumes_list.append(empty_vol)
                    labels_list.append(empty_label)

            patient_save_path = os.path.join(self.patientlevel_data_path, rf'{casename}.h5')
            self.save_patient_level_maps(volumes_list, labels_list, patient_save_path)

    def image_information_checking(self, checked_data_root, slice_idx=32, band_width=10, save_image_root=None):

        if save_image_root is not None:
            os.makedirs(save_image_root, exist_ok=True)

        for img_name in sorted(os.listdir(checked_data_root)):
            case_dir = os.path.join(checked_data_root, img_name)
            volume_data, label_data = self.load_single_branch_maps(case_dir)

            print(volume_data.shape, label_data)

            label_list = []
            N = label_data["boxes"].shape[0]
            for i in range(N):
                label_list.append({
                    "box": label_data["boxes"][i].tolist(),
                    "sten": int(label_data["sten"][i]),
                    "plq": int(label_data["plq"][i])
                })

            if save_image_root is not None:
                save_path = os.path.join(save_image_root, f'{img_name}.png')
                funcs.cpr_volume_display_label(
                    volume_data[0, :, slice_idx, :],
                    label_list,
                    band_width=band_width,
                    save_path=save_path
                )

        return

    def load_branch_data(self, csv_path, nii_path):

        nii = nib.load(nii_path)
        volume = nii.get_fdata(dtype=np.float32)

        label_df = None
        if csv_path is not None:
            label_df = pd.read_csv(csv_path)

        volume = np.transpose(volume, (2, 0, 1))

        return label_df.values, volume

    def resize_branch_label(self, label_array, target_len):

        N = label_array.shape[0]
        if N == target_len:
            return label_array.copy()

        ret_idx = np.linspace(0, N - 1, target_len)
        ret_label = label_array[np.round(ret_idx).astype(int)]

        return ret_label

    def resize_branch_volume(self, volume_array, target_shape):

        D, H, W = volume_array.shape
        target_D, target_H, target_W = target_shape[1:]

        dz = np.linspace(0, 1, D)
        dy = np.linspace(0, 1, H)
        dx = np.linspace(0, 1, W)

        interpolator = RegularGridInterpolator((dz, dy, dx), volume_array, method='linear', bounds_error=False,
                                               fill_value=0)

        dz_new = np.linspace(0, 1, target_D)
        dy_new = np.linspace(0, 1, target_H)
        dx_new = np.linspace(0, 1, target_W)
        dz_grid, dy_grid, dx_grid = np.meshgrid(dz_new, dy_new, dx_new, indexing='ij')
        coords = np.stack([dz_grid, dy_grid, dx_grid], axis=-1)
        ret_volume = interpolator(coords)
        ret_volume = ret_volume[np.newaxis, ...]

        return ret_volume

    def resize_label_and_volume(self, label_array, volume_array):

        ret_label = self.resize_branch_label(label_array, self.volume_shape[0])
        ret_volume = self.resize_branch_volume(volume_array,
                                               (1, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2]))

        return ret_label.astype(label_array.dtype), ret_volume.astype(np.float32)

    def detection_target_generation(self, label_array):

        L = label_array.shape[0]
        targets = []

        if L == 0:
            return targets

        start = None
        current_class = None

        for i in range(L):
            row = label_array[i]

            if row[0] == 0 and row[1] == 0:
                if current_class is not None:
                    start_ratio = start / (L - 1)
                    end_ratio = (i - 1) / (L - 1)
                    targets.append({
                        "box": [(start_ratio + end_ratio) / 2, end_ratio - start_ratio],
                        "sten": current_class[0],
                        "plq": current_class[1]
                    })
                    start = None
                    current_class = None
                continue

            if current_class is None:
                start = i
                current_class = row.tolist()
            else:
                if row.tolist() != current_class:
                    start_ratio = start / (L - 1)
                    end_ratio = (i - 1) / (L - 1)
                    targets.append({
                        "box": [(start_ratio + end_ratio) / 2, end_ratio - start_ratio],
                        "sten": current_class[0],
                        "plq": current_class[1]
                    })
                    start = i
                    current_class = row.tolist()

        if current_class is not None:
            start_ratio = start / (L - 1)
            end_ratio = (L - 1) / (L - 1)
            targets.append({
                "box": [(start_ratio + end_ratio) / 2, end_ratio - start_ratio],
                "sten": current_class[0],
                "plq": current_class[1]
            })

        return targets

    def augment_single_image(self, image, labels, save_dir, base_name):

        os.makedirs(save_dir, exist_ok=True)

        boxes = labels["boxes"]
        sten_labels = labels["sten"]
        plq_labels = labels["plq"]

        num_targets = boxes.shape[0]
        W = image.shape[1]

        for i in range(num_targets):

            augmented_img = image.copy()

            for j in range(num_targets):
                if j == i:
                    continue

                cx, w = boxes[j]
                start_idx = int(max(0, round((cx - w / 2) * (W - 1))))
                end_idx = int(min(W - 1, round((cx + w / 2) * (W - 1))))

                augmented_img[:, start_idx:end_idx, :, :] = self.mask_ct_value

            augmented_label = {
                "boxes": boxes[i:i + 1],
                "sten": sten_labels[i:i + 1],
                "plq": plq_labels[i:i + 1]
            }

            augmented_path = os.path.join(
                save_dir, f"{base_name}{str(i + 1).zfill(2)}.h5"
            )

            self.save_single_branch_maps(
                augmented_img,
                augmented_label,
                augmented_path
            )

        augmented_image_clear = image.copy()

        for i in range(num_targets):
            cx, w = boxes[i]
            start_idx = int(max(0, round((cx - w / 2) * (W - 1))))
            end_idx = int(min(W - 1, round((cx + w / 2) * (W - 1))))

            augmented_image_clear[:, start_idx:end_idx, :, :] = self.mask_ct_value

        empty_label = {
            "boxes": np.zeros((0, 2), dtype=np.float32),
            "sten": np.zeros((0,), dtype=np.int64),
            "plq": np.zeros((0,), dtype=np.int64)
        }

        augmented_path = os.path.join(save_dir, f"{base_name}00.h5")

        self.save_single_branch_maps(
            augmented_image_clear,
            empty_label,
            augmented_path
        )

    def save_single_branch_maps(self, input_image, input_targets, save_path):

        with h5py.File(save_path, 'w') as f:
            f.create_dataset('image', data=input_image, dtype=np.float32)
            f.create_dataset('boxes', data=input_targets['boxes'], dtype=np.float32)
            f.create_dataset('sten', data=input_targets['sten'], dtype=np.int64)
            f.create_dataset('plq', data=input_targets['plq'], dtype=np.int64)

        return

    def save_standardized_maps(self, input_image, input_targets, save_path):

        num_targets = len(input_targets)

        boxes = np.zeros((num_targets, 2), dtype=np.float32)
        sten = np.zeros((num_targets,), dtype=np.int64)
        plq = np.zeros((num_targets,), dtype=np.int64)

        for i, t in enumerate(input_targets):

            boxes[i, 0] = t['box'][0]
            boxes[i, 1] = t['box'][1]
            sten[i] = t['sten']
            plq[i] = t['plq']

        if hasattr(input_image, "detach"):
            input_image = input_image.detach().cpu().numpy()

        with h5py.File(save_path, 'w') as f:
            f.create_dataset('image', data=input_image, dtype=np.float32)
            f.create_dataset('boxes', data=boxes, dtype=np.float32)
            f.create_dataset('sten', data=sten, dtype=np.int64)
            f.create_dataset('plq', data=plq, dtype=np.int64)

        return

    def load_single_branch_maps(self, load_path):

        with h5py.File(load_path, 'r') as f:
            image = f['image'][:].astype(np.float32)
            boxes = f['boxes'][:].astype(np.float32)
            sten = f['sten'][:].astype(np.int64)
            plq = f['plq'][:].astype(np.int64)

        targets = {
            "boxes": boxes,
            "sten": sten,
            "plq": plq
        }

        return image, targets

    def create_empty_sample(self):

        sample_shape = (1, self.volume_shape[0], self.volume_shape[1], self.volume_shape[2])
        volume = np.full(sample_shape, self.mask_ct_value, dtype=np.float32)
        label = {
            "boxes": np.zeros((0, 2), dtype=np.float32),
            "sten": np.zeros((0,), dtype=np.int64),
            "plq": np.zeros((0,), dtype=np.int64)
        }

        return volume, label

    def save_patient_level_maps(self, volume_list, label_list, patient_level_save_path):

        B = len(volume_list)
        assert B == len(label_list)

        images_np = np.stack(volume_list, axis=0).astype(np.float32)  # (B, 1, 256, 64, 64)

        all_boxes = []
        all_sten = []
        all_plq = []
        offsets = [0]

        for lab in label_list:
            n = lab["boxes"].shape[0] if lab["boxes"].size > 0 else 0

            if n > 0:
                all_boxes.append(lab["boxes"])
                all_sten.append(lab["sten"])
                all_plq.append(lab["plq"])

            offsets.append(offsets[-1] + n)

        boxes_np = np.concatenate(all_boxes, axis=0).astype(np.float32) if all_boxes else np.zeros((0, 2),
                                                                                                   dtype=np.float32)
        sten_np = np.concatenate(all_sten, axis=0).astype(np.int64) if all_sten else np.zeros((0,), dtype=np.int64)
        plq_np = np.concatenate(all_plq, axis=0).astype(np.int64) if all_plq else np.zeros((0,), dtype=np.int64)
        offsets_np = np.array(offsets, dtype=np.int64)

        with h5py.File(patient_level_save_path, "w") as f:
            f.create_dataset("images", data=images_np)
            f.create_dataset("boxes", data=boxes_np)
            f.create_dataset("sten", data=sten_np)
            f.create_dataset("plq", data=plq_np)
            f.create_dataset("offsets", data=offsets_np)

    def load_patient_level_maps(self, patient_level_load_path):

        images_list = []
        labels_list = []

        with h5py.File(patient_level_load_path, "r") as f:
            images_np = f["images"][:]
            boxes_np = f["boxes"][:]
            sten_np = f["sten"][:]
            plq_np = f["plq"][:]
            offsets = f["offsets"][:]

            B = images_np.shape[0]

            for i in range(B):
                images_list.append(images_np[i])

                start, end = offsets[i], offsets[i + 1]

                if end > start:
                    labels_list.append({
                        "boxes": boxes_np[start:end],
                        "sten": sten_np[start:end],
                        "plq": plq_np[start:end],
                    })
                else:
                    labels_list.append({
                        "boxes": np.zeros((0, 2), dtype=np.float32),
                        "sten": np.zeros((0,), dtype=np.int64),
                        "plq": np.zeros((0,), dtype=np.int64),
                    })

        return images_list, labels_list


class causality_learning_dataset(data.Dataset):
    def __init__(self,
                 processed_data_path,
                 evaluation_mode,
                 pattern='pre_training_1',
                 stage='training',
                 train_ratio=0.8,
                 input_shape=[256, 64, 64],
                 window=[300, 900]):

        self.stage = stage
        self.pattern = pattern
        self.input_shape = input_shape
        self.window = [window[0] - window[1] / 2, window[0] + window[1] / 2]

        self.processed_data_path = processed_data_path
        self.data_processing = causality_learning_data_processing(self.processed_data_path)

        if self.pattern == 'pre_training_1':
            self.dataset_root = self.data_processing.augmented_data_path
        elif self.pattern == 'pre_training_2':
            self.dataset_root = self.data_processing.patientlevel_data_path
        else:
            self.dataset_root = self.data_processing.patientlevel_data_path

        self.volumes_file_list = os.listdir(self.dataset_root)
        self.volumes_file_list = sorted(self.volumes_file_list)
        self.file_total = len(self.volumes_file_list)

        train_end = int(self.file_total * train_ratio)
        eval_end = train_end + (self.file_total - train_end) // 2
        if self.stage == 'training':
            self.data_start, self.data_end = 0, train_end
        elif self.stage == 'evaluation':
            self.data_start, self.data_end = train_end, eval_end
        else:
            self.data_start, self.data_end = eval_end, self.file_total

        if evaluation_mode == 'center_based_splitting':
            self.data_start, self.data_end = 0, self.file_total

        self.length = self.data_end - self.data_start

        return

    def __getitem__(self, index):

        volumes_file = os.path.join(self.dataset_root, self.volumes_file_list[index])

        if self.pattern == 'pre_training_1':
            ret_volumes, ret_labels = self.data_processing.load_single_branch_maps(volumes_file)
        else:
            ret_volumes, ret_labels = self.data_processing.load_patient_level_maps(volumes_file)

        ret_volumes = funcs.normalize_ct_data(ret_volumes, hu_min=self.window[0], hu_max=self.window[1])

        # ret_labels = volumes_targets_to_gpu(ret_labels)

        return {'image': torch.tensor(ret_volumes, dtype=torch.float32), 'target': ret_labels}

    def __len__(self):
        return self.length


def collate_fn(batch):

    images, targets = [], []
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
    images = torch.stack(images, dim=0)

    return images, targets
