import os
import numpy as np

import torch
from torch.utils.data import DataLoader

import architecture as arch
import dataprocessing as prep
import optimization as opt_fn
from config import opt


def set_trainable_modules(model, pattern):

    modules_dict = {
        "decoupling_embedding_extraction": model.decoupling_embedding_extraction,
        "volume_embedding_extraction": model.volume_embedding_extraction,
        "artery_level_semantic_retrieval": model.artery_level_semantic_retrieval,
        "causal_intervention": model.causal_intervention,
        "mutually_exclusive_classification": model.mutually_exclusive_classification,
        "artery_level_prediction": model.artery_level_prediction,
    }

    for module in modules_dict.values():
        for p in module.parameters():
            p.requires_grad = False

    if pattern == "pre_training_1":

        for key in ["decoupling_embedding_extraction", "mutually_exclusive_classification"]:
            for p in modules_dict[key].parameters():
                p.requires_grad = True

    elif pattern == "pre_training_2":

        for key in ["volume_embedding_extraction",
                    "artery_level_semantic_retrieval",
                    "artery_level_prediction"]:
            for p in modules_dict[key].parameters():
                p.requires_grad = True

    elif pattern == "main_training":

        for key in ["volume_embedding_extraction",
                    "artery_level_semantic_retrieval",
                    "causal_intervention",
                    "artery_level_prediction"]:
            for p in modules_dict[key].parameters():
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown pattern {pattern}. Must be one of ['pre_training1','pre_training2','main_training']")


class interference_free_causality_learning:
    def __init__(self,
                 dataset_path_list='dataset path list',
                 model_pattern='model pattern', # pre_training_1 model_pattern main_training
                 evaluation_mode='evaluation mode', # proportion_based_splitting center_based_splitting
                 banks_load_root='confounder banks load path',
                 banks_save_root='confounder banks save path',
                 model_weights_path='model weights path',
                 train_data_ratio=0.8,
                 model_input_shape=np.array([256, 64, 64]),
                 image_window=np.array([300, 900]),
                 batch_size=8,
                 num_queries=16,
                 max_branch=16,
                 eta=0.8,
                 device=None
                 ):
        super().__init__()

        if len(dataset_path_list) == 1:
            dataset_path_list = dataset_path_list * 3

        if banks_load_root == 'random':
            sten_bank_path, plq_bank_path = 'random', 'random'
        elif banks_load_root:
            sten_bank_path = os.path.join(banks_load_root, 'stenosis_confounder_bank.pth')
            plq_bank_path = os.path.join(banks_load_root, 'plaque_confounder_bank.pth')
        else:
            sten_bank_path, plq_bank_path = None, None

        self.device = device

        self.model = arch.attribute_decoupled_intervention_network(
            extraction_input_channel=opt.net_params['extraction_input_channel'],
            extraction_conv_channels=opt.net_params['extraction_conv_channels'],
            extraction_conv_nums=opt.net_params['extraction_conv_nums'],
            extraction_fc_channels=opt.net_params['extraction_fc_channels'],

            retrieval_num_queries=opt.net_params['retrieval_num_queries'],
            retrieval_embed_dim=opt.net_params['retrieval_embed_dim'],
            retrieval_num_heads=opt.net_params['retrieval_num_heads'],
            retrieval_dim_feedforward=opt.net_params['retrieval_dim_feedforward'],
            retrieval_encoder_num_layers=opt.net_params['retrieval_encoder_num_layers'],
            retrieval_decoder_num_layers=opt.net_params['retrieval_decoder_num_layers'],

            intervention_query_dim=opt.net_params['intervention_query_dim'],
            intervention_kv_dim=opt.net_params['intervention_kv_dim'],
            intervention_num_heads=opt.net_params['intervention_num_heads'],

            stenosis_class_num=opt.net_params['stenosis_class_num'],
            plaque_class_num=opt.net_params['plaque_class_num'],
            pred_dim_list=opt.net_params['pred_dim_list']
        )

        self.stenosis_confounder_bank = arch.confounder_bank_block(queue_num=opt.net_params['stenosis_class_num'],
                                                                   maximum_num=opt.net_params['banks_maximum_num'],
                                                                   vector_dim=opt.net_params['banks_vector_dim'],
                                                                   update_beta=opt.net_params['bank_update_beta'],
                                                                   load_path=sten_bank_path,
                                                                   )

        self.plaque_confounder_bank = arch.confounder_bank_block(queue_num=opt.net_params['plaque_class_num'],
                                                                 maximum_num=opt.net_params['banks_maximum_num'],
                                                                 vector_dim=opt.net_params['banks_vector_dim'],
                                                                 update_beta=opt.net_params['bank_update_beta'],
                                                                 load_path=plq_bank_path)

        if model_weights_path is not None:
            self.model.load_state_dict(torch.load(model_weights_path))

        self.model.to(device)

        set_trainable_modules(self.model, pattern=model_pattern)
        stage_list = ['training', 'evaluation', 'testing']

        self.model_pattern = model_pattern
        self.banks_save_root = banks_save_root

        self.train_dataset = prep.causality_learning_dataset(dataset_path_list[0],
                                                             evaluation_mode=evaluation_mode,
                                                             pattern=model_pattern,
                                                             stage=stage_list[0],
                                                             train_ratio=train_data_ratio,
                                                             input_shape=model_input_shape,
                                                             window=image_window)

        self.eval_dataset = prep.causality_learning_dataset(dataset_path_list[1],
                                                            evaluation_mode=evaluation_mode,
                                                            pattern=model_pattern,
                                                            stage=stage_list[1],
                                                            train_ratio=train_data_ratio,
                                                            input_shape=model_input_shape,
                                                            window=image_window)

        self.test_dataset = prep.causality_learning_dataset(dataset_path_list[2],
                                                            evaluation_mode=evaluation_mode,
                                                            pattern=model_pattern,
                                                            stage=stage_list[2],
                                                            train_ratio=train_data_ratio,
                                                            input_shape=model_input_shape,
                                                            window=image_window)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, collate_fn=prep.collate_fn)
        self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=batch_size, collate_fn=prep.collate_fn)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, collate_fn=prep.collate_fn)

        self.adi_net_loss = opt_fn.adi_net_optimization_objective(num_queries=num_queries,
                                                                  max_branch=max_branch,
                                                                  eta=eta,
                                                                  device=self.device)

    def confounder_banks_save(self):
        sten_save_path = os.path.join(self.banks_save_root, 'stenosis_confounder_bank.pth')
        plq_save_path = os.path.join(self.banks_save_root, 'plaque_confounder_bank.pth')

        self.stenosis_confounder_bank.confounder_band_save(sten_save_path)
        self.plaque_confounder_bank.confounder_band_save(plq_save_path)
        return

    def confounder_banks_check(self):
        self.stenosis_confounder_bank.check_queue_lengths()
        self.plaque_confounder_bank.check_queue_lengths()

    def confounder_banks_embs(self):
        return self.stenosis_confounder_bank.confounder_bank_listput(), self.plaque_confounder_bank.confounder_bank_listput()

    def confounder_banks_writing(self, vector_list):
        self.stenosis_confounder_bank.confounder_writing(vector_list[0], vector_list[1])
        self.plaque_confounder_bank.confounder_writing(vector_list[2], vector_list[3])
        return