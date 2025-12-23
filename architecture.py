import os
import numpy as np
from einops import rearrange, repeat

import torch
from torch import nn, einsum
import torch.nn.functional as F

import functions as funcs


class feature_separate_mathcal_B(nn.Module):
    def __init__(self, in_channel):
        super(feature_separate_mathcal_B, self).__init__()

        self.conv_sten = nn.Conv3d(in_channel, 1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.conv_plq = nn.Conv3d(in_channel, 1, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_sten, x_plq):

        x_sten_soft = self.conv_sten(x_sten)
        x_sten_soft = self.softmax(x_sten_soft)

        x_plq_soft = self.conv_plq(x_plq)
        x_plq_soft = self.softmax(x_plq_soft)

        x_sten = x_sten * (1.0 - x_plq_soft)
        x_plq = x_plq * (1.0 - x_sten_soft)

        return x_sten, x_plq


class volume_representation(nn.Module):
    def __init__(self, input_channel, conv_channels, conv_nums, fc_channels):
        super().__init__()

        self.volume_representation_blocks = nn.ModuleList()

        in_ch = input_channel
        for out_ch, num_conv in zip(conv_channels, conv_nums):
            self.volume_representation_blocks.append(conv3D_blocks(in_ch, out_ch, num_conv))
            in_ch = out_ch

        self.conv1x1x1_volume = nn.Conv3d(conv_channels[-1], fc_channels[0], kernel_size=1)

        self.fc_volume = nn.Linear(fc_channels[1], fc_channels[-1])

    def forward(self, x_volume):

        batch_seq_length = x_volume.size(0)

        for volume_block in self.volume_representation_blocks:
            x_volume = volume_block(x_volume)

        x_volume = self.conv1x1x1_volume(x_volume)
        x_volume = rearrange(x_volume, 'x d h w l -> (x d) (h w l)')
        x_volume = self.fc_volume(x_volume)
        x_volume = rearrange(x_volume, '(x d) c -> x d c', x=batch_seq_length)

        return x_volume


def conv3D_blocks(in_ch, out_ch, num_convs):
    layers = []
    for i in range(num_convs):
        layers += [
            nn.Conv3d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
    layers.append(nn.MaxPool3d(kernel_size=2))
    return nn.Sequential(*layers)


class attribute_representation(nn.Module):
    def __init__(self, input_channel, conv_channels, conv_nums, fc_channels):
        super().__init__()

        self.sten_representation_blocks = nn.ModuleList()
        self.plq_representation_blocks = nn.ModuleList()

        in_ch = input_channel
        for out_ch, num_conv in zip(conv_channels, conv_nums):
            self.sten_representation_blocks.append(conv3D_blocks(in_ch, out_ch, num_conv))
            self.plq_representation_blocks.append(conv3D_blocks(in_ch, out_ch, num_conv))
            in_ch = out_ch

        self.conv1x1x1_sten = nn.Conv3d(conv_channels[-1], fc_channels[0], kernel_size=1)
        self.conv1x1x1_plq = nn.Conv3d(conv_channels[-1], fc_channels[0], kernel_size=1)

        self.fc_sten = nn.Linear(fc_channels[0] * fc_channels[1], fc_channels[-1])
        self.fc_plq = nn.Linear(fc_channels[0] * fc_channels[1], fc_channels[-1])

        self.feature_separate_blocks = nn.ModuleList(
            [feature_separate_mathcal_B(in_channel=c) for c in conv_channels]
        )

    def forward(self, x):

        x_sten, x_plq = x, x

        for sten_block, plq_block, sep_block in zip(self.sten_representation_blocks,
                                                    self.plq_representation_blocks,
                                                    self.feature_separate_blocks):
            x_sten = sten_block(x_sten)
            x_plq = plq_block(x_plq)
            x_sten, x_plq = sep_block(x_sten, x_plq)

        x_sten = self.conv1x1x1_sten(x_sten)
        x_sten = x_sten.view(x_sten.size(0), -1)
        x_sten = self.fc_sten(x_sten)

        x_plq = self.conv1x1x1_plq(x_plq)
        x_plq = x_plq.view(x_plq.size(0), -1)
        x_plq = self.fc_plq(x_plq)

        return x_sten, x_plq


class semantic_retrieval_block(nn.Module):
    def __init__(self, num_queries, embed_dim, num_heads, dim_feedforward, encoder_num_layers, decoder_num_layers):
        super(semantic_retrieval_block, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_num_layers)

        self.learnable_embeddings = nn.Parameter(torch.randn(1, num_queries, embed_dim))

        self.query_adapter = nn.Linear(embed_dim, embed_dim)

    def forward(self, volume_embeddings):

        B = volume_embeddings.size(0)

        memory = self.encoder(volume_embeddings)

        global_token = volume_embeddings.mean(dim=1)
        query_bias = self.query_adapter(global_token).unsqueeze(1)
        query_embeddings = self.learnable_embeddings.expand(B, -1, -1)
        tgt = query_embeddings + query_bias

        output = self.decoder(tgt=tgt, memory=memory)

        return output


class causal_intervention_block(nn.Module):
    def __init__(self, query_dim=512, kv_dim=1024, num_heads=8):
        super().__init__()

        self.kv_proj = nn.Linear(kv_dim, query_dim)
        self.mha = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, x1, x2, branch_num):

        kv = torch.cat([x1, x2], dim=2)

        B, _, _ = query.shape
        _, seq_kv, dim_kv = kv.shape
        kv_mapped = self.kv_proj(kv)

        kv_reshape = repeat(kv_mapped, 'b s d -> b g s d', g=branch_num)
        kv_reshape = rearrange(kv_reshape, 'b g s d -> (b g) s d')

        attn_output, _ = self.mha(query=query, key=kv_reshape, value=kv_reshape)

        return attn_output


class confounder_bank_block(nn.Module):
    def __init__(self, queue_num, maximum_num, vector_dim, update_beta, load_path=None):
        super().__init__()

        self.maximum_num = maximum_num
        self.update_beta = update_beta

        if load_path and load_path != "random" and os.path.exists(load_path):
            load_data = torch.load(load_path, map_location="cpu")
            self.bank = load_data["bank"]
        elif load_path == "random":
            self.bank = [
                [torch.randn(vector_dim) for _ in range(self.maximum_num)]
                for _ in range(queue_num)
            ]
        else:
            self.bank = [
                [torch.randn(vector_dim) for _ in range(16)]
                for _ in range(queue_num)
            ]

    def check_queue_lengths(self):
        for idx, seq in enumerate(self.bank):
            print(f"{idx}: {len(seq)}")

    def confounder_writing(self, vectors, labels):

        vectors, labels = vectors.cpu(), labels.cpu()

        X, D = vectors.shape

        for i in range(X):
            v = vectors[i]
            seq_id = int(labels[i])
            seq = self.bank[seq_id]

            if len(seq) < self.maximum_num:
                seq.append(v.detach())
                continue

            seq_tensor = torch.stack(seq)
            v_unsq = v.detach().unsqueeze(0)

            sims = F.cosine_similarity(seq_tensor, v_unsq, dim=1)
            idx = sims.argmax().item()

            old_vec = seq[idx]
            new_vec = self.update_beta * old_vec + (1 - self.update_beta) * v.detach()
            seq[idx] = new_vec

    def confounder_bank_save(self, save_path):
        torch.save({"bank": self.bank}, save_path)

    def confounder_bank_listput(self):
        return torch.stack([v for sublist in self.bank for v in sublist], dim=0)


class MLP_Block(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class lesion_prediction_head(nn.Module):
    def __init__(self, num_class, dim_list):
        super().__init__()

        self.sten_class_prediction = MLP_Block(dim_list[0], dim_list[1], num_class[0])
        self.plq_class_prediction = MLP_Block(dim_list[0], dim_list[1], num_class[1])

        self.box_class_prediction = MLP_Block(dim_list[0], dim_list[1], 2)
        self.boxes_prediction = MLP_Block(dim_list[0], dim_list[1], 2)

    def forward(self, x, p):
        b, l, c = x.shape

        x_sten = self.sten_class_prediction(x)
        x_plq = self.plq_class_prediction(x)
        x_rp = self.box_class_prediction(x)

        x = rearrange(x, 'b l c -> (b l) c')
        x_roi = self.boxes_prediction(x)
        x_roi = rearrange(x_roi, '(b l) n -> b l n', b=b)

        if p != 'training':
            x_sten = F.softmax(x_sten, dim=2)
            x_plq = F.softmax(x_plq, dim=2)
            x_rp = F.softmax(x_rp, dim=2)
            x_roi = x_roi.sigmoid()

        return x_sten, x_plq, x_rp, x_roi


class exclusive_classification_block(nn.Module):
    def __init__(self, emb_dim, concat_cls_num, concat_fn):
        super().__init__()

        self.fc_mutually_exclusive = nn.Linear(emb_dim, concat_cls_num)
        self.concat_fn = concat_fn

    def forward(self, x_attr, stage, attr):

        M_attr_cls = self.fc_mutually_exclusive(x_attr)
        if stage != 'training':
            M_attr_cls = F.softmax(M_attr_cls, dim=1)
        M_attr_labels = M_attr_cls.argmax(dim=-1)
        M_attr_labels = self.concat_fn(M_attr_labels, attr=attr)

        return M_attr_cls, M_attr_labels


def confounder_bank_reading(cfd_bank, embs):

    cfd_bank = cfd_bank.to(embs.device)
    bank_norm = F.normalize(cfd_bank, dim=1)
    emb_norm = F.normalize(embs, dim=1)

    cos_sim = torch.matmul(emb_norm, bank_norm.T)
    max_idx = cos_sim.argmax(dim=1)

    output = cfd_bank[max_idx]

    return output


class attribute_decoupled_intervention_network (nn.Module):
    def __init__(self,
                 extraction_input_channel=1,
                 extraction_conv_channels=np.array([16, 32, 64, 128]),
                 extraction_conv_nums=np.array([2, 2, 3, 3]),
                 extraction_fc_channels=np.array([8, 256, 512]),

                 retrieval_num_queries=16,
                 retrieval_embed_dim=512,
                 retrieval_num_heads=8,
                 retrieval_dim_feedforward=1024,
                 retrieval_encoder_num_layers=4,
                 retrieval_decoder_num_layers=4,

                 intervention_query_dim=512,
                 intervention_kv_dim=1024,
                 intervention_num_heads=8,

                 stenosis_class_num=6,
                 plaque_class_num=4,
                 pred_dim_list=np.array([512, 128]),
                 ):
        super(attribute_decoupled_intervention_network, self).__init__()

        self.stenosis_class_num = stenosis_class_num
        self.plaque_class_num = plaque_class_num

        self.decoupling_embedding_extraction = attribute_representation(input_channel=extraction_input_channel,
                                                                        conv_channels=extraction_conv_channels,
                                                                        conv_nums=extraction_conv_nums,
                                                                        fc_channels=extraction_fc_channels)

        self.volume_embedding_extraction = volume_representation(input_channel=extraction_input_channel,
                                                                 conv_channels=extraction_conv_channels,
                                                                 conv_nums=extraction_conv_nums,
                                                                 fc_channels=extraction_fc_channels)

        self.artery_level_semantic_retrieval = semantic_retrieval_block(num_queries=retrieval_num_queries,
                                                                        embed_dim=retrieval_embed_dim,
                                                                        num_heads=retrieval_num_heads,
                                                                        dim_feedforward=retrieval_dim_feedforward,
                                                                        encoder_num_layers=retrieval_encoder_num_layers,
                                                                        decoder_num_layers=retrieval_decoder_num_layers)

        self.causal_intervention = causal_intervention_block(query_dim=intervention_query_dim,
                                                             kv_dim=intervention_kv_dim,
                                                             num_heads=intervention_num_heads)

        self.mutually_exclusive_classification = exclusive_classification_block(emb_dim=extraction_fc_channels[-1],
                                                                                concat_cls_num=stenosis_class_num + plaque_class_num,
                                                                                concat_fn=self.process_exclusive_labels)

        self.artery_level_prediction = lesion_prediction_head(num_class=[stenosis_class_num, plaque_class_num],
                                                              dim_list=pred_dim_list)

    def process_exclusive_labels(self, labels_tensor, attr):

        if attr == 'plq':
            return torch.where(
                labels_tensor < self.stenosis_class_num,
                torch.zeros_like(labels_tensor),
                labels_tensor % self.stenosis_class_num
            )

        elif attr == 'sten':
            return torch.where(
                labels_tensor >= self.stenosis_class_num,
                torch.zeros_like(labels_tensor),
                labels_tensor
            )

    def detection_prediction2maps(self, pred_rp, pred_xb):

        ret_outputs = {
            "pred_logits": pred_rp,
            "pred_boxes": pred_xb
        }
        return ret_outputs

    def forward(self, x, cfd_sten, cfd_plq, pattern='main_training', stage='training'):

        if pattern != 'pre_training_1':
            b, branch_num, c, n_l, n_h, n_w = x.shape
            x = rearrange(x, 'b l c n_l n_h n_w -> (b l) c n_l n_h n_w')

        x_sten, x_plq = self.decoupling_embedding_extraction(x)
        x_vol = self.volume_embedding_extraction(x)

        M_sten_cls, M_sten_labels = self.mutually_exclusive_classification(x_sten, stage=stage, attr='sten')
        M_plq_cls, M_plq_labels = self.mutually_exclusive_classification(x_plq, stage=stage, attr='plq')

        banks_updating_list = [x_sten, M_sten_labels, x_plq, M_plq_labels]

        if pattern == 'pre_training_1':
            return {'sten_cls': M_sten_cls, 'plq_cls': M_plq_cls}, banks_updating_list

        x_qry = self.artery_level_semantic_retrieval(x_vol)

        if pattern == 'pre_training_2':
            out_sten, out_plq, out_rp, out_iou = self.artery_level_prediction(x_qry, stage)
            out_roi = self.detection_prediction2maps(out_rp, out_iou)
            out_roi = funcs.boxes_dimension_expansion(out_roi, dtype='outputs')
            return {'out_sten': out_sten, 'out_plq': out_plq, 'out_roi': out_roi}, banks_updating_list

        med_sten = confounder_bank_reading(cfd_sten, x_sten)
        med_plq = confounder_bank_reading(cfd_plq, x_plq)

        med_sten = rearrange(med_sten, '(b l) c -> b l c', l=branch_num)
        med_plq = rearrange(med_plq, '(b l) c -> b l c', l=branch_num)

        x_qry = self.causal_intervention(x_qry, med_sten, med_plq, branch_num)

        out_sten, out_plq, out_rp, out_iou = self.artery_level_prediction(x_qry, stage)

        out_roi = self.detection_prediction2maps(out_rp, out_iou)
        out_roi = funcs.boxes_dimension_expansion(out_roi, dtype='outputs')

        return {'out_sten': out_sten, 'out_plq': out_plq, 'out_roi': out_roi}, banks_updating_list
