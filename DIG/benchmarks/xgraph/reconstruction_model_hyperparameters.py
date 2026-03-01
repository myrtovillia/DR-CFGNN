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
            "disjoint_train_ratio": 0.35
        }
    elif dataset_name in ["bbbp", "mutag"] :
        return {
            "num_encoder_layers": 3,
            "encoder_dropout": 0.1,
            "mlp_layers": 3,
            "hidden_channels": 100,
            "mlp_hidden_dim": 512,
            "early_stop_patience": 10,
            "neg_ratio": 1,
            "disjoint_train_ratio": 0.35
        }
    else:
        return {
            "num_encoder_layers": 4,
            "encoder_dropout": 0.05, 
            "mlp_layers": 4,
            "hidden_channels": 200,
            "mlp_hidden_dim": 4000,  #2000 for ba_2motifs_3class gives 62/82
            "early_stop_patience": 20, 
            "neg_ratio": 2.5,
            "disjoint_train_ratio": 0.30
        }

# NOTE: With RandomLinkSplit(disjoint_train_ratio=r), supervision edges are int(r * E_undirected).
# We set MIN_EDGES=4 (2 undirected). Thus, from 6 (3 undirected), int(r*E_undirected) can become 0 -> no disjoint supervision split.
# Then message passing and supervision overlap.
# We ensure r is high enough so int(r*E_undirected) >= 1, (0.35 * 3) = 1.05 so at least 1 supervision edge!

