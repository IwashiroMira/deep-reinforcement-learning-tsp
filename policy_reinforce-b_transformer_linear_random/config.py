# config.py

# モデル設定
model = {
    "type": "transformer",
    "embedding_dim": 128,
    "n_heads": 8,
    "n_layers": 3,
    "dropout": 0.1,
}

# 学習設定
training = {
    "batch_size": 1024,
    "episodes": 125000,
    "learning_rate": 0.0005,
    "optimizer": "adam",
    "gamma": 0.99  # 割引率（強化学習特有）
}

# 環境設定
environment = {
    "problem": "tsp",
    "num_cities": 25,  # 問題に出現する都市の数
    "normalize": True  # 座標を0〜1にスケーリングするか
}

# モデル保存・読み込み
checkpoint = {
    "save_path": "./checkpoints/",
    "load_model": True,
    "model_path": "./checkpoints/tsp_transformer_best.pth"
}

# 推論設定
inference = {
    "model_path": "save/model.pth",
    "batch_size": 100,
    "episodes": 100,
    "plot": True,
    "fixed": False,  # 固定座標での推論を行うか
}
