def hyperparams(dataset_name):

    if dataset_name in ["graph_sst2", "twitter", "graph_sst5"]:
        return {
            "num_encoder_layers": 2,
            "encoder_dropout": 0.40,
            "mlp_layers": 3,
            "hidden_channels": 100,
            "mlp_hidden_dim": 256,
            "early_stop_patience": 10,
            "neg_ratio": 1,
            "disjoint_train_ratio": 0.40,
        }

    elif dataset_name == "bbbp":
        return {
            "num_encoder_layers": 3,
            "encoder_dropout": 0.1,
            "mlp_layers": 3,
            "hidden_channels": 100,
            "mlp_hidden_dim": 512,
            "early_stop_patience": 10,
            "neg_ratio": 1,
            "disjoint_train_ratio": 0.20,
        }

    else:
        return {
            "num_encoder_layers": 4,
            "encoder_dropout": 0.05, 
            "mlp_layers": 4,
            "hidden_channels": 200,
            "mlp_hidden_dim": 4000,
            "early_stop_patience": 20, 
            "neg_ratio": 2.5,
            "disjoint_train_ratio": 0.30,
        }


