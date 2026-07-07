import numpy as np
import pandas as pd
import os
import argparse
import wandb
import json
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import glob

from promoter_modelling.dataloaders.BinaryTask import BinaryTask
from promoter_modelling import backbone_modules
from promoter_modelling import MTL_modules

np.random.seed(97)
torch.manual_seed(97)
torch.set_float32_matmul_precision('high')

if torch.cuda.is_available():
    try:
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = ''
        os.environ['CUDA_MPS_LOG_DIRECTORY'] = ''
        _ = torch.zeros(1).cuda()
        print(f"[CUDA Init] Successfully initialized CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[CUDA Init] Warning during early CUDA initialization: {e}")


def train_model(args, config):
    root_dir = config["root_dir"]
    os.makedirs(root_dir, exist_ok=True)
    model_save_dir = os.path.join(root_dir, "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)
    summaries_save_dir = os.path.join(root_dir, "summaries")
    os.makedirs(summaries_save_dir, exist_ok=True)

    root_data_dir = config["root_data_dir"]
    os.makedirs(root_data_dir, exist_ok=True)
    common_cache_dir = os.path.join(root_data_dir, "common")
    os.makedirs(common_cache_dir, exist_ok=True)

    task = args.single_task
    dataloader = BinaryTask(
        csv_path=args.input_csv_path,
        batch_size=args.batch_size,
        val_chr=args.val_chr,
        test_chr=args.test_chr,
        train_sampling_ratio=args.train_sampling_ratio
    )
    dataloader.setup()
    all_dataloaders = [dataloader]

    model_class = backbone_modules.get_backbone_class(args.model_name)
    name_format = "individual_training_on_{}".format(task)
    if args.model_name != "MTLucifer":
        name_format = f"{args.model_name}_" + name_format
    if args.optional_name_suffix is not None:
        name_format += "_" + args.optional_name_suffix

    num_models_to_train = args.num_random_seeds
    best_seed_val_metric = None

    for seed in range(num_models_to_train):
        name = name_format
        if num_models_to_train > 1:
            print("Random seed = {}".format(seed))
            np.random.seed(seed)
            torch.manual_seed(seed)
            name = name_format + "_seed_{}".format(seed)

        mtlpredictor = MTL_modules.MTLPredictor(
            model_class=model_class,
            all_dataloader_modules=all_dataloaders,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_preconstructed_dataloaders=True,
            train_mode=args.train_mode
        )

        cur_models_save_dir = os.path.join(model_save_dir, name, "default", "checkpoints")

        check = False
        if args.use_existing_models and os.path.exists(cur_models_save_dir):
            done_file = os.path.join(model_save_dir, name, "default", "done.txt")
            check = os.path.exists(done_file)

        if check:
            print("Using existing model and evaluating it")
            all_saved_models = os.listdir(cur_models_save_dir)
            best_model_path = ""
            minimize_metric = args.metric_direction_which_is_optimal == "min"
            best_metric = np.inf if minimize_metric else -np.inf
            for path in all_saved_models:
                val_metric = path.split("=")[-1][:-len(".ckpt")]
                val_metric = float(val_metric[:-3]) if "-v" in val_metric else float(val_metric)
                if (minimize_metric and val_metric < best_metric) or (not minimize_metric and val_metric > best_metric):
                    best_metric = val_metric
                    best_model_path = path

            print("Best existing model: {}".format(os.path.join(cur_models_save_dir, best_model_path)))
            checkpoint = torch.load(os.path.join(cur_models_save_dir, best_model_path), map_location=device)
            new_state_dict = {k[len("model."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
            mtlpredictor.model.load_state_dict(new_state_dict, strict=False)

            trainer = L.Trainer(accelerator="gpu", devices=1)
            trainer.test(mtlpredictor, mtlpredictor.get_mtldataloader().test_dataloader())
            best_model_test_outputs = trainer.predict(mtlpredictor, mtlpredictor.get_mtldataloader().test_dataloader())
        else:
            print("Training model")
            wandb_logger = WandbLogger(name=name, project=args.wandb_project_name, log_model=False)
            checkpoint_filename = "best-{epoch:02d}-{" + args.metric_to_monitor + ":.5f}"
            checkpoint_callback = ModelCheckpoint(
                monitor=args.metric_to_monitor,
                dirpath=os.path.join(model_save_dir, name, "default", "checkpoints"),
                filename=checkpoint_filename,
                save_top_k=args.save_top_k,
                mode=args.metric_direction_which_is_optimal
            )
            early_stop_callback = EarlyStopping(
                monitor=args.metric_to_monitor, min_delta=0.00,
                patience=args.patience, verbose=True,
                mode=args.metric_direction_which_is_optimal
            )

            trainer = L.Trainer(
                logger=wandb_logger,
                callbacks=[early_stop_callback, checkpoint_callback],
                deterministic=True,
                accelerator="gpu", devices=1,
                log_every_n_steps=100, default_root_dir=model_save_dir,
                max_epochs=args.max_epochs,
                limit_test_batches=0, reload_dataloaders_every_n_epochs=2,
                enable_progress_bar=True,
                gradient_clip_val=1.0, num_sanity_val_steps=32
            )

            trainer.fit(mtlpredictor, mtlpredictor.get_mtldataloader())

            ckpt_dir = os.path.join(model_save_dir, name, "default", "checkpoints")
            ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
            if not ckpt_files:
                raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
            best_ckpt = ckpt_files[-1]
            print(f"Best checkpoint: {best_ckpt}")

            trainer.test(mtlpredictor, mtlpredictor.get_mtldataloader().test_dataloader(), ckpt_path=best_ckpt)

            test_metrics = {k: (v.item() if hasattr(v, 'item') else v)
                             for k, v in trainer.callback_metrics.items() if 'test' in k}
            wandb.log(test_metrics)

            print("\n=== Test Metrics ===")
            for k, v in test_metrics.items():
                print(f"{k} = {v:.4f}")

            os.makedirs(os.path.join(model_save_dir, name, "default"), exist_ok=True)
            done_file = os.path.join(model_save_dir, name, "default", "done.txt")
            with open(done_file, "w+") as f:
                f.write("done")

            wandb.finish()
            best_model_test_outputs = trainer.predict(mtlpredictor, mtlpredictor.get_mtldataloader().test_dataloader(), ckpt_path=best_ckpt)

        # 이진 분류(BinaryTask) 결과 처리
        dl = all_dataloaders[0]
        y = torch.vstack([out["y"] for out in best_model_test_outputs])
        pred = torch.vstack([out["pred"] for out in best_model_test_outputs])

        pred_binary = torch.round(torch.sigmoid(pred))
        acc = accuracy_score(y, pred_binary)
        f1 = f1_score(y, pred_binary)
        precision = precision_score(y, pred_binary)
        recall = recall_score(y, pred_binary)

        print(f"Accuracy = {acc:.4f}")
        print(f"F1 = {f1:.4f}")
        print(f"Precision = {precision:.4f}")
        print(f"Recall = {recall:.4f}")

        dl.update_metrics(pred, y, 0, "test")
        metrics_dict = dl.compute_metrics("test")
        for key, value in metrics_dict.items():
            if "loss" not in key:
                print("{} = {} ≈ {}".format(key, value, np.around(value, 4)))

        if best_seed_val_metric is None or f1 > best_seed_val_metric:
            best_seed_val_metric = f1

    return y, pred


args = argparse.ArgumentParser()
args.add_argument("--config_path", type=str, default="./config.json", help="Path to config file")
args.add_argument("--model_name", type=str, default="MTLucifer", help="Name of model to use")
args.add_argument("--single_task", type=str, default="BinaryTask", help="Task to train on")

args.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
args.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
args.add_argument("--batch_size", type=int, default=96, help="Batch size")
args.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs")
args.add_argument("--train_mode", type=str, default="min_size", help="'min_size' or 'max_size_cycle'")

args.add_argument("--num_random_seeds", type=int, default=1, help="Number of random seeds to train with")
args.add_argument("--use_existing_models", action="store_true", help="Use existing models if available")

args.add_argument("--wandb_project_name", type=str, default="promoter_modelling", help="Wandb project name")
args.add_argument("--metric_to_monitor", type=str, default="val_BinaryTask_avg_epoch_loss", help="Metric to monitor for early stopping")
args.add_argument("--metric_direction_which_is_optimal", type=str, default="max", help="'max' or 'min'")

args.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
args.add_argument("--save_top_k", type=int, default=1, help="Number of top models to save")
args.add_argument("--optional_name_suffix", type=str, default=None, help="Optional suffix to add to model name")

args.add_argument("--val_chr", type=str, default="chr5", help="검증(Validation)에 사용할 Chromosome")
args.add_argument("--test_chr", type=str, default="chr7", help="테스트(Test)에 사용할 Chromosome")
args.add_argument("--train_sampling_ratio", type=float, default=1.0, help="학습 데이터 샘플링 비율")

args.add_argument("--input_csv_path", type=str, default=None, help="Path to the input CSV file for BinaryTask")
args = args.parse_args()

assert os.path.exists(args.config_path), "Config file does not exist"
with open(args.config_path, "r") as config_file:
    config = json.load(config_file)

print(f"Root directory: {config['root_dir']}")
print(f"Root data directory: {config['root_data_dir']}")

root_dir = config["root_dir"]
os.makedirs(root_dir, exist_ok=True)
wandb_logs_save_dir = os.path.join(root_dir, "wandb_logs")
os.makedirs(wandb_logs_save_dir, exist_ok=True)
wandb_cache_dir = os.path.join(root_dir, "wandb_cache")
os.makedirs(wandb_cache_dir, exist_ok=True)
os.environ["WANDB_DIR"] = wandb_logs_save_dir
os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"
print("Using {} device".format(device))

y, pred = train_model(args, config)

save_dir = os.path.join(config["root_dir"], "summaries")
os.makedirs(save_dir, exist_ok=True)

ckpt_dir = os.path.join(config["root_dir"], "saved_models", f"individual_training_on_{args.single_task}", "default", "checkpoints")
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")])
if ckpts:
    best_ckpt = os.path.join(ckpt_dir, ckpts[-1])
    ckpt_data = torch.load(best_ckpt, map_location="cpu")
    epoch = ckpt_data.get("epoch", "N/A")
    step = ckpt_data.get("global_step", "N/A")
    val_loss = None
    for part in best_ckpt.split("-"):
        if part.startswith("val_"):
            val_loss = part.split("=")[-1].replace(".ckpt", "")

    with open(os.path.join(save_dir, "metrics_summary.txt"), "w") as f:
        f.write(f"Best Epoch: {epoch}\n")
        f.write(f"Global Step: {step}\n")
        f.write(f"Validation Loss: {val_loss if val_loss else 'N/A'}\n")

torch.save(y, "y.pt")
torch.save(pred, "pred.pt")
print("Saved y.pt and pred.pt!")
print("ALL DONE!")
