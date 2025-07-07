import optuna
import subprocess
from train import main as train_main
from inference import main as inference_main
import logging


def objective(trial):
    # パラメータの提案
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    episodes = trial.suggest_int("episodes", 20000, 30000, step=1000)  # 100, 110, 120, ...
    
    # モデルパスの定義
    model_path = f"save/model_trial_{trial.number}_lr{lr:.5f}_g{gamma:.3f}.pth"

    # 訓練スタート
    train_main(
        lr=lr,
        gamma=gamma,
        episodes=episodes,
        save_path=model_path  # ←動的に渡す！
    )
    
    # パスを trial に記録
    trial.set_user_attr("model_path", model_path)
    
    # 訓練結果を保存
    result = subprocess.run(["python", "inference.py", model_path], capture_output=True, text=True)
    # デバッグ出力を追加！
    logging.info(f"Trial {trial.number}, lr={lr}, gamma={gamma}, episodes={episodes}")
    logging.info("STDOUT:\n" + result.stdout)
    logging.error("STDERR:\n" + result.stderr)

    distance = float(result.stdout.strip())
    return distance


def main():
    # 最適化開始
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10, show_progress_bar=False)
    best_model_path = study.best_trial.user_attrs["model_path"]
    # 結果表示
    logging.info("Best trial: " + str(study.best_trial.params))
    logging.info("Best model path: " + best_model_path)

    # best モデルでプロットしたい場合
    inference_main(best_model_path, episodes=10000, plot=True)


if __name__ == "__main__":
    main()
