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
    "lr": 1.0e-4,
    "optimizer": "adam",
    "gamma": 0.95,  # 割引率（強化学習特有）
    "model_path": "save/model_25_01.pth"
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
    "random_batch_size": 1280,  # 本番は1280に固定
    "baseline_batch_size": 1,  # baselineのバッチサイズは常に1
    "episodes": 10000,  # fixed=Trueの時は1エピソードのみ
    "plot": True,
    "fixed": False,  # 固定座標での推論を行うか
}
