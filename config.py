class DefaultConfig(object):

    net_params = {
    "extraction_input_channel": 1,
    "extraction_conv_channels": [16, 32, 64, 128],
    "extraction_conv_nums": [2, 2, 3, 3],
    "extraction_fc_channels": [8, 256, 512],

    "retrieval_num_queries": 16,
    "retrieval_embed_dim": 512,
    "retrieval_num_heads": 8,
    "retrieval_dim_feedforward": 1024,
    "retrieval_encoder_num_layers": 4,
    "retrieval_decoder_num_layers": 4,

    "intervention_query_dim": 512,
    "intervention_kv_dim": 1024,
    "intervention_num_heads": 8,

    "banks_maximum_num": 100,
    "banks_vector_dim": 512,
    "bank_update_beta": 0.7,
    "stenosis_class_num": 6,
    "plaque_class_num": 4,
    "pred_dim_list": [512, 128]

    }


opt = DefaultConfig()
