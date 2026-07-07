import numpy as np
import pandas as pd
import os
import argparse
import wandb
import json
import scipy.stats as stats
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import glob
from lightning.pytorch.tuner import Tuner

from promoter_modelling.dataloaders.BinaryTask import BinaryTask
from promoter_modelling import backbone_modules
from promoter_modelling import MTL_modules

np.random.seed(97)
torch.manual_seed(97)
torch.set_float32_matmul_precision('high')

# CUDA 초기화를 강제로 먼저 수행하여 MPS 충돌 회피
if torch.cuda.is_available():
    try:
        # 환경 변수 재설정으로 MPS 우회
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = ''
        os.environ['CUDA_MPS_LOG_DIRECTORY'] = ''
        # CUDA 디바이스 초기화
        _ = torch.zeros(1).cuda()
        print(f"[CUDA Init] Successfully initialized CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[CUDA Init] Warning during early CUDA initialization: {e}")


def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_drive_mounted():
    return os.path.exists(os.path.join('/content/drive', 'MyDrive'))

def strip_dot_slash(filepath):
    if filepath.startswith("./"):
        return filepath[2:]
    return filepath

def get_base_directory(default_dir):
    try:
        if is_colab():
            print("Running in Google Colab")
            cleaned_default_dir = strip_dot_slash(default_dir)
            if is_drive_mounted():
                print("Google Drive is mounted")
                base_dir = os.path.join('/content/drive', 'MyDrive', 'promoter_models', cleaned_default_dir)
            else:
                print("Google Drive is not mounted")
                base_dir = f'/content/promoter_models/{cleaned_default_dir}'
        else:
            print("Running on local machine")
            base_dir = default_dir
    except NameError:
        print("Running on local machine [Name Error]")
        base_dir = default_dir
    return base_dir



def train_model(args, config, finetune=False):
    # create directories
    # for modelling
    root_dir = config["root_dir"]
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
        #print(root_dir)
    model_save_dir = os.path.join(root_dir, "saved_models")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir, exist_ok=True)
    summaries_save_dir = os.path.join(root_dir, "summaries")
    if not os.path.exists(summaries_save_dir):
        os.makedirs(summaries_save_dir, exist_ok=True)

    # for data
    root_data_dir = config["root_data_dir"]
    if not os.path.exists(root_data_dir):
        os.makedirs(root_data_dir, exist_ok=True)
        #print(root_data_dir)
    common_cache_dir = os.path.join(root_data_dir, "common")
    if not os.path.exists(common_cache_dir):
        os.makedirs(common_cache_dir, exist_ok=True)

    # setup task(s)
    if args.modelling_strategy == "joint":
        assert args.joint_tasks is not None, "Must specify tasks to jointly train on"
        tasks = args.joint_tasks.split(",")
    elif args.modelling_strategy.startswith("pretrain"):
        assert args.pretrain_tasks is not None, "Must specify tasks to pretrain on"
        assert args.finetune_tasks is not None, "Must specify tasks to finetune or perform linear probing on"
        pretrain_tasks = args.pretrain_tasks.split(",")
        finetune_tasks = args.finetune_tasks.split(",")

        if finetune:
            tasks = finetune_tasks
        else:
            tasks = pretrain_tasks
    elif args.modelling_strategy.startswith("single_task"):
        assert args.single_task is not None, "Must specify task to train on"
        tasks = [args.single_task]
    else:
        raise ValueError("Invalid modelling strategy")

    if args.model_name.startswith("MotifBased"):
        assert len(tasks) == 1, "Motif-based models can only be trained on a single task"
        assert tasks[0] == "FluorescenceData" or tasks[0] == "FluorescenceData_DE" or ("Malinois_MPRA" in tasks[0]), "Motif-based models can only be trained on FluorescenceData, FluorescenceData_DE, or Malinois_MPRA"

    # load pretrained model state dict if necessary(Pretrained시에만 실행
    if "pretrain" in args.modelling_strategy and finetune:
        print("Loading pre-trained model state dict")

        pretrained_model_name = "pretrain_on_{}".format("+".join(pretrain_tasks))
        # map to model classes
        model_class = backbone_modules.get_backbone_class(args.model_name)
        if args.model_name != "MTLucifer":
            pretrained_model_name = f"{args.model_name}_" + pretrained_model_name

        pretrain_metric_direction_which_is_optimal = args.pretrain_metric_direction_which_is_optimal
        pretrained_model_save_dir = os.path.join(model_save_dir, pretrained_model_name, "default", "checkpoints")

        # find path to best existing model
        all_saved_models = os.listdir(pretrained_model_save_dir)
        best_model_path = ""
        minimize_metric = pretrain_metric_direction_which_is_optimal == "min"
        if minimize_metric:
            best_metric = np.inf
        else:
            best_metric = -np.inf
        for path in all_saved_models:
            val_metric = path.split("=")[-1][:-len(".ckpt")]
            if "-v" in val_metric:
                val_metric = float(val_metric[:-len("-v1")])
            else:
                val_metric = float(val_metric)
                
            if minimize_metric:
                if val_metric < best_metric:
                    best_metric = val_metric
                    best_model_path = path
            else:
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_model_path = path
                    
        print("Best pre-trained model is: {}".format(os.path.join(pretrained_model_save_dir, best_model_path)))

        # load it
        pretrained_checkpoint = torch.load(os.path.join(pretrained_model_save_dir, best_model_path), map_location=device)

    # setup training parameters
    if "pretrain" in args.modelling_strategy and not finetune:
        print("Pre-training model")
        metric_to_monitor = args.pretrain_metric_to_monitor
        metric_direction_which_is_optimal = args.pretrain_metric_direction_which_is_optimal
        lr = args.pretrain_lr
        weight_decay = args.pretrain_weight_decay
        batch_size = args.pretrain_batch_size
        max_epochs = args.pretrain_max_epochs
        train_mode = args.pretrain_train_mode
    else:
        print("Training model from scratch")
        metric_to_monitor = args.metric_to_monitor
        metric_direction_which_is_optimal = args.metric_direction_which_is_optimal
        lr = args.lr
        weight_decay = args.weight_decay
        batch_size = args.batch_size
        max_epochs = args.max_epochs
        train_mode = args.train_mode

    print("Learning rate = {}, weight decay = {}, batch size = {}, max epochs = {}, train mode = {}".format(lr, weight_decay, batch_size, max_epochs, train_mode))

    # multiple models are trained only for finetuning/joint training/single task training
    num_models_to_train = args.num_random_seeds
    if "pretrain" in args.modelling_strategy and not finetune:
        num_models_to_train = 1

    # model name format
    name_format = ""
    if "pretrain" in args.modelling_strategy and finetune:
        if "finetune" in args.modelling_strategy:
            name_format = "finetune_on_{}_pretrained_on_{}".format("+".join(tasks), "+".join(pretrain_tasks))
        if "linear_probing" in args.modelling_strategy:
            name_format = "linear_probing_on_{}_pretrained_on_{}".format("+".join(tasks), "+".join(pretrain_tasks))
        if "simple_regression" in args.modelling_strategy:
            name_format = "simple_regression_on_{}_pretrained_on_{}".format("+".join(tasks), "+".join(pretrain_tasks))
    elif "pretrain" in args.modelling_strategy and not finetune:
        name_format = "pretrain_on_{}".format("+".join(tasks))
    elif "joint" in args.modelling_strategy:
        name_format = "joint_train_on_{}".format("+".join(tasks))
    elif "single" in args.modelling_strategy:
        if "simple_regression" in args.modelling_strategy:
            name_format = "simple_regression_on_{}".format("+".join(tasks))
        else:
            name_format = "individual_training_on_{}".format("+".join(tasks))

    # map to model classes
    model_class = backbone_modules.get_backbone_class(args.model_name)
    if args.model_name != "MTLucifer":
        name_format = f"{args.model_name}_" + name_format

    # add optional name suffix to model name - only when not pretraining
    if args.optional_name_suffix is not None:
        if "pretrain" in args.modelling_strategy:
            if finetune:
                name_format += "_" + args.optional_name_suffix
        else:
            name_format += "_" + args.optional_name_suffix

    from promoter_modelling.dataloaders.BinaryTask import BinarySequenceDataset 

    # instantiate dataloaders
    dataloaders = {}
    print("Instantiating dataloaders...")
    
    for task in tasks:
        if task == "BinaryTask":
            dataloaders[task] = BinaryTask(
            csv_path=args.input_csv_path,
            batch_size=batch_size,
            val_chr=args.val_chr,
            test_chr=args.test_chr,
            train_sampling_ratio=args.train_sampling_ratio
            )
            dataloaders[task].setup()
        else:
            raise ValueError(f"Unsupported task: {task}")

    # flatten to list (기존 코드에서 쓰이는 형태 유지)
    all_dataloaders = [dataloaders[task] for task in tasks]
    print("Total number of dataloaders = {}".format(len(all_dataloaders)))  

    # train models
    all_seeds_r2 = {}
    all_seeds_pearsonr = {}
    all_seeds_srho = {}

    percentile_threshold_for_highly_expressed_promoters = 90
    percentile_threshold_for_lowly_expressed_promoters = 100 - percentile_threshold_for_highly_expressed_promoters

    all_seeds_highly_expressed_promoters_r2 = {}
    all_seeds_highly_expressed_promoters_pearsonr = {}
    all_seeds_highly_expressed_promoters_srho = {}

    all_seeds_lowly_expressed_promoters_r2 = {}
    all_seeds_lowly_expressed_promoters_pearsonr = {}
    all_seeds_lowly_expressed_promoters_srho = {}

    all_seeds_extreme_expression_promoters_r2 = {}
    all_seeds_extreme_expression_promoters_pearsonr = {}
    all_seeds_extreme_expression_promoters_srho = {}

    all_seeds_y = {}
    all_seeds_pred = {}

    best_seed = None
    best_seed_val_metric = None

    for seed in range(num_models_to_train):
        if num_models_to_train > 1:
            print("Random seed = {}".format(seed))
            # set random seed
            np.random.seed(seed)
            torch.manual_seed(seed)
            name = name_format + "_seed_{}".format(seed)
        else:
            name = name_format

        if args.model_name.startswith("MotifBased"):
            mtlpredictor = MTL_modules.MTLPredictor(model_class=model_class,\
                                                all_dataloader_modules=all_dataloaders, \
                                                batch_size=batch_size, \
                                                max_epochs=args.max_epochs, \
                                                lr=lr, \
                                                weight_decay=weight_decay, \
                                                with_motifs=True, \
                                                use_preconstructed_dataloaders=True, \
                                                train_mode=train_mode)
        elif (args.modelling_strategy == "pretrain+simple_regression" and finetune) or (args.modelling_strategy == "single_task_simple_regression"):
            mtlpredictor = MTL_modules.MTLPredictor(model_class=model_class,\
                                                all_dataloader_modules=all_dataloaders, \
                                                batch_size=batch_size, \
                                                max_epochs=args.max_epochs, \
                                                lr=lr, \
                                                weight_decay=weight_decay, \
                                                use_simple_regression=True, \
                                                use_preconstructed_dataloaders=True, \
                                                train_mode=train_mode)
        else:                                                
            mtlpredictor = MTL_modules.MTLPredictor(model_class=model_class,\
                                                all_dataloader_modules=all_dataloaders, \
                                                batch_size=batch_size, \
                                                max_epochs=args.max_epochs, \
                                                lr=lr, \
                                                weight_decay=weight_decay, \
                                                use_preconstructed_dataloaders=True, \
                                                train_mode=train_mode)
        
        cur_models_save_dir = os.path.join(model_save_dir, name, "default", "checkpoints")

        # first check if there's an existing joint model
        check = False
        if args.use_existing_models:
            if os.path.exists(cur_models_save_dir):
                done_file = os.path.join(model_save_dir, name, "default", "done.txt")
                if os.path.exists(done_file):
                    check = True
        if check: # found existing model and using it
            print("Using existing models and evaluating them")

            if (args.modelling_strategy == "pretrain+simple_regression" and finetune) or (args.modelling_strategy == "single_task_simple_regression"):
                # load model, done automatically by fit_simple_regression
                mtlpredictor.fit_simple_regression(unified_cache_dir=os.path.join(model_save_dir, name.split("_seed")[0] + "_unified_cache"), 
                                                   cache_dir=cur_models_save_dir,
                                                   device=device,
                                                   batch_size=batch_size,
                                                   use_existing_models=True)

                # get test set predictions
                best_model_test_outputs = mtlpredictor.get_predictions_from_simple_regression()
            else:
                # find path to best existing model
                all_saved_models = os.listdir(cur_models_save_dir)
                best_model_path = "" 
                minimize_metric = metric_direction_which_is_optimal == "min"
                if minimize_metric:
                    best_metric = np.inf
                else:
                    best_metric = -np.inf
                for path in all_saved_models:
                    val_metric = path.split("=")[-1][:-len(".ckpt")]
                    if "-v" in val_metric:
                        val_metric = float(val_metric[:-len("-v1")])
                    else:
                        val_metric = float(val_metric)
                        
                    if minimize_metric:
                        if val_metric < best_metric:
                            best_metric = val_metric
                            best_model_path = path
                    else:
                        if val_metric > best_metric:
                            best_metric = val_metric
                            best_model_path = path
                            
                print("Best existing model is: {}".format(os.path.join(cur_models_save_dir, best_model_path)))

                # load it
                checkpoint = torch.load(os.path.join(cur_models_save_dir, best_model_path), map_location=device)

                new_state_dict = {}
                for key in checkpoint["state_dict"]:
                    if key.startswith("model."):
                        new_state_dict[key[len("model."):]] = checkpoint["state_dict"][key]

                mtlpredictor.model.load_state_dict(new_state_dict, strict=False)
                print("Loaded existing model")

                # evaluate on test set and log to wandb
                trainer = L.Trainer(accelerator="gpu", devices=2)
                trainer.test(mtlpredictor, mtlpredictor.get_mtldataloader().test_dataloader())

                # get test set predictions
                best_model_test_outputs = trainer.predict(mtlpredictor, mtlpredictor.get_mtldataloader().test_dataloader())

        else:
            print("Training model")

            if (args.modelling_strategy == "pretrain+simple_regression" and finetune) or (args.modelling_strategy == "single_task_simple_regression"):
                mtlpredictor.fit_simple_regression(unified_cache_dir=os.path.join(model_save_dir, name.split("_seed")[0] + "_unified_cache"), 
                                                   cache_dir=cur_models_save_dir,
                                                   device=device,
                                                   batch_size=batch_size,
                                                   use_existing_models=True)
                
                # create done file
                os.makedirs(os.path.join(model_save_dir, name, "default"), exist_ok=True)
                done_file = os.path.join(model_save_dir, name, "default", "done.txt")
                with open(done_file, "w+") as f:
                    f.write("done")

                # get test set predictions
                best_model_test_outputs = mtlpredictor.get_predictions_from_simple_regression()
            else:
                if "pretrain" in args.modelling_strategy and finetune:
                    new_state_dict = {}
                    for key in pretrained_checkpoint["state_dict"]:
                        if key.startswith("model."):
                            new_state_dict[key[len("model."):]] = pretrained_checkpoint["state_dict"][key]

                    mtlpredictor.model.load_state_dict(new_state_dict, strict=False)        
                    print("Loaded pretrained model")
                
                # freeze backbone for linear probing
                if "linear_probing" in args.modelling_strategy and finetune:
                    print("Freezing backbone for linear probing")
                    # freeze backbone
                    for param_name, param in mtlpredictor.model.named_parameters():
                        if param_name.startswith("Backbone.promoter_"):
                            param.requires_grad = False

                    for param_name, param in mtlpredictor.model.named_parameters():
                        if param_name.startswith("Backbone.promoter_"):
                            assert param.requires_grad == False

                wandb_logger = WandbLogger(name=name, \
                                        project=args.wandb_project_name, log_model=False)

                checkpoint_filename = "best-{epoch:02d}-{" + "{}".format(metric_to_monitor) + ":.5f}"
                checkpoint_callback = ModelCheckpoint(monitor=metric_to_monitor, \
                                                    dirpath=os.path.join(model_save_dir, name, "default", "checkpoints"), \
                                                    filename=checkpoint_filename, \
                                                    save_top_k=args.save_top_k, mode=metric_direction_which_is_optimal)

                patience = args.patience
                early_stop_callback = EarlyStopping(monitor=metric_to_monitor, min_delta=0.00, \
                                                    patience=patience, verbose=True, mode=metric_direction_which_is_optimal)

                trainer = L.Trainer(logger=wandb_logger, \
                                    callbacks=[early_stop_callback, checkpoint_callback], \
                                    deterministic=True, \
                                    accelerator="gpu", devices=1, \
                                    log_every_n_steps=100, default_root_dir=model_save_dir, \
                                    max_epochs=max_epochs, \
                                    limit_test_batches=0, reload_dataloaders_every_n_epochs=2, enable_progress_bar = True, \
                                    gradient_clip_val=1.0, num_sanity_val_steps=32)
                if args.find_lr:
                    print("\n--- [학습률 찾기 시작] ---")
                    # 1. Tuner 객체 생성
                    tuner = Tuner(trainer)
                    
                    # 2. lr_find 실행
                    lr_finder = tuner.lr_find(mtlpredictor, datamodule=mtlpredictor.get_mtldataloader())
                    
                    # 3. 결과 그래프 표시 및 최적 값 제안
                    fig = lr_finder.plot(suggest=True)
                    fig.show() # 그래프를 보여줍니다.
                    
                    suggested_lr = lr_finder.suggestion()
                    print(f"--- [학습률 찾기 완료] 제안된 학습률: {suggested_lr:.8f} ---")
                    print("이제 --find_lr 옵션을 빼고, 찾은 값을 --lr 옵션에 넣어 다시 실행하세요.")
                    
                    # 4. 본 학습을 시작하지 않고 함수를 종료합니다.
                    return None, None

                trainer.fit(mtlpredictor, mtlpredictor.get_mtldataloader())

                # find best checkpoint
                ckpt_dir = os.path.join(model_save_dir, name, "default", "checkpoints")
                ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
                if not ckpt_files:
                    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
                best_ckpt = ckpt_files[-1]
                print(f"Best checkpoint: {best_ckpt}")

                # evaluate on test set and log to wandb
                trainer.test(mtlpredictor, mtlpredictor.get_mtldataloader().test_dataloader(), ckpt_path=best_ckpt)

                # 명시적으로 test 메트릭을 wandb에 기록
                test_metrics = {k: (v.item() if hasattr(v, 'item') else v)
                                for k, v in trainer.callback_metrics.items() if 'test' in k}
                wandb.log(test_metrics)

                # test 메트릭 출력
                print("\n=== Test Metrics ===")
                for k, v in test_metrics.items():
                    print(f"{k} = {v:.4f}")

                # create done file
                done_file = os.path.join(model_save_dir, name, "default", "done.txt")
                with open(done_file, "w+") as f:
                    f.write("done")

                wandb.finish()

                # get test set predictions
                best_model_test_outputs = trainer.predict(mtlpredictor, mtlpredictor.get_mtldataloader().test_dataloader(), ckpt_path=best_ckpt)

        # get metrics
        dataloader_to_outputs = {}
        dataloader_to_y = {}
        dataloader_to_pred = {}

        for i, dl in enumerate(all_dataloaders):
            dl = dl.name
            print(dl)
            
            if len(all_dataloaders) == 1:
                dataloader_to_outputs[dl] = best_model_test_outputs
            else:
                dataloader_to_outputs[dl] = best_model_test_outputs[i]
                        
            dataloader_to_y[dl] = torch.vstack([dataloader_to_outputs[dl][j]["y"] for j in range(len(dataloader_to_outputs[dl]))])
            dataloader_to_pred[dl] = torch.vstack([dataloader_to_outputs[dl][j]["pred"] for j in range(len(dataloader_to_outputs[dl]))])

            print("y shape = {}".format(dataloader_to_y[dl].shape))
            print("pred shape = {}".format(dataloader_to_pred[dl].shape))

            if "Fluorescence" in dl and "classification" in dl:
                print()
                for j, output in enumerate(all_dataloaders[i].output_names):
                    cur_y = dataloader_to_y[dl][:, j]
                    cur_pred = dataloader_to_pred[dl][:, j]

                    # apply sigmoid and round
                    cur_pred = torch.sigmoid(cur_pred)
                    cur_pred = torch.round(cur_pred)

                    # get overall metrics
                    acc = accuracy_score(cur_y, cur_pred)
                    f1 = f1_score(cur_y, cur_pred)
                    precision = precision_score(cur_y, cur_pred)
                    recall = recall_score(cur_y, cur_pred)

                    print("{} Accuracy = {} ≈ {}".format(output, acc, np.around(acc, 4)))
                    print("{} F1 = {} ≈ {}".format(output, f1, np.around(f1, 4)))
                    print("{} Precision = {} ≈ {}".format(output, precision, np.around(precision, 4)))
                    print("{} Recall = {} ≈ {}".format(output, recall, np.around(recall, 4)))
                    print()

            elif (("Fluorescence" in dl) or ("MalinoisMPRA" in dl)) and (("joint_" in name_format) or ("finetune_" in name_format) or ("linear_probing_" in name_format) or ("individual_" in name_format) or ("simple_regression" in name_format)):
                print()
                for j, output in enumerate(all_dataloaders[i].output_names):
                    cur_y = dataloader_to_y[dl][:, j]
                    cur_pred = dataloader_to_pred[dl][:, j]

                    # remove invalid values
                    if "MalinoisMPRA" in dl:
                        mask = cur_y != -100000
                        cur_y = cur_y[mask]
                        cur_pred = cur_pred[mask]
                        print(f"Cell {output} has {len(cur_y)} valid values")

                    # get overall metrics
                    r2 = r2_score(cur_y, cur_pred)
                    pearsonr = stats.pearsonr(cur_y, cur_pred)[0]
                    srho = stats.spearmanr(cur_y, cur_pred).correlation

                    print("{} R2 = {} ≈ {}".format(output, r2, np.around(r2, 4)))
                    print("{} PearsonR = {} ≈ {}".format(output, pearsonr, np.around(pearsonr, 4)))
                    print("{} Spearman rho = {} ≈ {}".format(output, srho, np.around(srho, 4)))
                    print()


                    # get highly expressed promoter metrics
                    highly_expressed_promoters = cur_y > np.percentile(cur_y, percentile_threshold_for_highly_expressed_promoters)
                    cur_y_highly_expressed_promoters = cur_y[highly_expressed_promoters]
                    cur_pred_highly_expressed_promoters = cur_pred[highly_expressed_promoters]
                    highly_expressed_promoters_r2 = r2_score(cur_y_highly_expressed_promoters, cur_pred_highly_expressed_promoters)
                    highly_expressed_promoters_pearsonr = stats.pearsonr(cur_y_highly_expressed_promoters, cur_pred_highly_expressed_promoters)[0]
                    highly_expressed_promoters_srho = stats.spearmanr(cur_y_highly_expressed_promoters, cur_pred_highly_expressed_promoters).correlation

                    print("{} R2 (highly expressed promoters) = {} ≈ {}".format(output, highly_expressed_promoters_r2, np.around(highly_expressed_promoters_r2, 4)))
                    print("{} PearsonR (highly expressed promoters) = {} ≈ {}".format(output, highly_expressed_promoters_pearsonr, np.around(highly_expressed_promoters_pearsonr, 4)))
                    print("{} Spearman rho (highly expressed promoters) = {} ≈ {}".format(output, highly_expressed_promoters_srho, np.around(highly_expressed_promoters_srho, 4)))
                    print()

                    # get lowly expressed promoter metrics
                    lowly_expressed_promoters = cur_y < np.percentile(cur_y, percentile_threshold_for_lowly_expressed_promoters)
                    cur_y_lowly_expressed_promoters = cur_y[lowly_expressed_promoters]
                    cur_pred_lowly_expressed_promoters = cur_pred[lowly_expressed_promoters]
                    lowly_expressed_promoters_r2 = r2_score(cur_y_lowly_expressed_promoters, cur_pred_lowly_expressed_promoters)
                    lowly_expressed_promoters_pearsonr = stats.pearsonr(cur_y_lowly_expressed_promoters, cur_pred_lowly_expressed_promoters)[0]
                    lowly_expressed_promoters_srho = stats.spearmanr(cur_y_lowly_expressed_promoters, cur_pred_lowly_expressed_promoters).correlation

                    print("{} R2 (lowly expressed promoters) = {} ≈ {}".format(output, lowly_expressed_promoters_r2, np.around(lowly_expressed_promoters_r2, 4)))
                    print("{} PearsonR (lowly expressed promoters) = {} ≈ {}".format(output, lowly_expressed_promoters_pearsonr, np.around(lowly_expressed_promoters_pearsonr, 4)))
                    print("{} Spearman rho (lowly expressed promoters) = {} ≈ {}".format(output, lowly_expressed_promoters_srho, np.around(lowly_expressed_promoters_srho, 4)))
                    print()

                    # get extreme expression promoter (= highly + lowly expressed) metrics
                    extreme_expression_promoters = np.logical_or(highly_expressed_promoters, lowly_expressed_promoters)
                    cur_y_extreme_expression_promoters = cur_y[extreme_expression_promoters]
                    cur_pred_extreme_expression_promoters = cur_pred[extreme_expression_promoters]
                    extreme_expression_promoters_r2 = r2_score(cur_y_extreme_expression_promoters, cur_pred_extreme_expression_promoters)
                    extreme_expression_promoters_pearsonr = stats.pearsonr(cur_y_extreme_expression_promoters, cur_pred_extreme_expression_promoters)[0]
                    extreme_expression_promoters_srho = stats.spearmanr(cur_y_extreme_expression_promoters, cur_pred_extreme_expression_promoters).correlation

                    print("{} R2 (extreme expression promoters) = {} ≈ {}".format(output, extreme_expression_promoters_r2, np.around(extreme_expression_promoters_r2, 4)))
                    print("{} PearsonR (extreme expression promoters) = {} ≈ {}".format(output, extreme_expression_promoters_pearsonr, np.around(extreme_expression_promoters_pearsonr, 4)))
                    print("{} Spearman rho (extreme expression promoters) = {} ≈ {}".format(output, extreme_expression_promoters_srho, np.around(extreme_expression_promoters_srho, 4)))
                    print()
                    
                    if output not in all_seeds_r2:
                        all_seeds_r2[output] = []
                        all_seeds_pearsonr[output] = []
                        all_seeds_srho[output] = []

                        all_seeds_highly_expressed_promoters_r2[output] = []
                        all_seeds_highly_expressed_promoters_pearsonr[output] = []
                        all_seeds_highly_expressed_promoters_srho[output] = []

                        all_seeds_lowly_expressed_promoters_r2[output] = []
                        all_seeds_lowly_expressed_promoters_pearsonr[output] = []
                        all_seeds_lowly_expressed_promoters_srho[output] = []

                        all_seeds_extreme_expression_promoters_r2[output] = []
                        all_seeds_extreme_expression_promoters_pearsonr[output] = []
                        all_seeds_extreme_expression_promoters_srho[output] = []

                        all_seeds_y[output] = []
                        all_seeds_pred[output] = []
                        
                    all_seeds_r2[output].append(r2)
                    all_seeds_pearsonr[output].append(pearsonr)
                    all_seeds_srho[output].append(srho)

                    all_seeds_highly_expressed_promoters_r2[output].append(highly_expressed_promoters_r2)
                    all_seeds_highly_expressed_promoters_pearsonr[output].append(highly_expressed_promoters_pearsonr)
                    all_seeds_highly_expressed_promoters_srho[output].append(highly_expressed_promoters_srho)

                    all_seeds_lowly_expressed_promoters_r2[output].append(lowly_expressed_promoters_r2)
                    all_seeds_lowly_expressed_promoters_pearsonr[output].append(lowly_expressed_promoters_pearsonr)
                    all_seeds_lowly_expressed_promoters_srho[output].append(lowly_expressed_promoters_srho)

                    all_seeds_extreme_expression_promoters_r2[output].append(extreme_expression_promoters_r2)
                    all_seeds_extreme_expression_promoters_pearsonr[output].append(extreme_expression_promoters_pearsonr)
                    all_seeds_extreme_expression_promoters_srho[output].append(extreme_expression_promoters_srho)

                    all_seeds_y[output].append(cur_y)
                    all_seeds_pred[output].append(cur_pred)

                    if best_seed_val_metric is None:
                        best_seed_val_metric = srho
                        best_seed = seed
                    elif srho > best_seed_val_metric:
                        best_seed_val_metric = srho
                        best_seed = seed
            
            all_dataloaders[i].update_metrics(dataloader_to_pred[dl], dataloader_to_y[dl], 0, "test")
            metrics_dict = all_dataloaders[i].compute_metrics("test")

            # print metrics for this dataloader
            for key in metrics_dict:
                if "loss" in key:
                    continue
                print("{} = {} ≈ {}".format(key, metrics_dict[key], np.around(metrics_dict[key], 4)))

    if best_seed is not None:
        print("Best seed = {}".format(best_seed))
        print("Creating prediction plots using best seed's model")

        output_names = all_seeds_y.keys()

        # make subplots for each output
        fig, axs = plt.subplots(1, len(output_names), figsize=(len(output_names) * 5, 5))

        # make prediction plots for first seed's model
        for j, output in enumerate(output_names):
            cur_y = all_seeds_y[output][best_seed]
            cur_pred = all_seeds_pred[output][best_seed]

            pearsonr = stats.pearsonr(cur_y, cur_pred)[0]
            srho = stats.spearmanr(cur_y, cur_pred).correlation

            sns.scatterplot(x=cur_y, y=cur_pred, ax=axs[j], alpha=0.5)

            # draw line of best fit
            m, b = np.polyfit(cur_y, cur_pred, 1)
            axs[j].plot(cur_y, m*cur_y + b, color="red", label="Best fit line")

            # draw line of perfect fit
            axs[j].plot(cur_y, cur_y, color="black", label="x=y")

            # set labels
            axs[j].set_xlabel("Actual Average Expression")
            axs[j].set_ylabel("Predicted Average Expression")

            # set title and show pearsonr and srho
            axs[j].set_title(r"{} ($r$ = {:.4f}, $\rho$ = {:.4f})".format(output, pearsonr, srho))

            # set legend
            axs[j].legend()
        
        # set suptitle and save figure
        fig.suptitle("Predictions on test set using best model (number of samples = {})".format(cur_y.shape[0]))
        fig.savefig(os.path.join(summaries_save_dir, name_format + "_best_model_predictions.png"), bbox_inches="tight")
        
    return all_seeds_y, all_seeds_pred


args = argparse.ArgumentParser()
args.add_argument("--config_path", type=str, default="./config.json", help="Path to config file")
args.add_argument("--model_name", type=str, default="MTLucifer", help="Name of model to use, must be one of {}".format(backbone_modules.get_all_backbone_names()))
args.add_argument("--modelling_strategy", type=str, required=True, help="Modelling strategy to use, either 'joint', 'pretrain+finetune', 'pretrain+linear_probing', 'pretrain+simple_regression', 'single_task', or 'single_task_simple_regression'")

args.add_argument("--joint_tasks", type=str, default=None, help="Comma separated list of tasks to jointly train on")
args.add_argument("--pretrain_tasks", type=str, default=None, help="Comma separated list of tasks to pretrain on")
args.add_argument("--finetune_tasks", type=str, default=None, help="Comma separated list of tasks to finetune or perform linear probing on")
args.add_argument("--single_task", type=str, default=None, help="Task to train on")

args.add_argument("--shrink_test_set", action="store_true", help="Shrink large test sets (SuRE and ENCODETFChIPSeq) to 10 examples to make evaluation faster")
args.add_argument("--subsample_train_set", action="store_true", help="Subsample training set")
args.add_argument("--n_train_subsample", type=int, default=15000, help="Number of samples to subsample for training set")

args.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
args.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
args.add_argument("--pretrain_lr", type=float, default=1e-5, help="pretrain learning rate")
args.add_argument("--pretrain_weight_decay", type=float, default=1e-4, help="pretrain weight decay")

args.add_argument("--batch_size", type=int, default=96, help="Batch size")
args.add_argument("--pretrain_batch_size", type=int, default=96, help="Pretrain batch size")

args.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs to joint-train, finetune or linear probe for")
args.add_argument("--pretrain_max_epochs", type=int, default=50, help="Maximum number of epochs to pretrain for")

args.add_argument("--train_mode", type=str, default="min_size", help="Specifies how multiple dataloaders are iterated over during training. Must be 'min_size' or 'max_size_cycle'")
args.add_argument("--pretrain_train_mode", type=str, default="min_size", help="Specifies how multiple dataloaders are iterated over during pretraining. Must be 'min_size' or 'max_size_cycle'")

args.add_argument("--num_random_seeds", type=int, default=1, help="Number of random seeds to train with")
args.add_argument("--use_existing_models", action="store_true", help="Use existing models if available")

args.add_argument("--wandb_project_name", type=str, default="promoter_modelling", help="Wandb project name")
args.add_argument("--metric_to_monitor", type=str, default="val_BinaryTask_avg_epoch_loss", help="Name of metric to monitor for early stopping")
args.add_argument("--metric_direction_which_is_optimal", type=str, default="max", help="Should metric be maximised (specify 'max') or minimised (specify 'min')?")
args.add_argument("--pretrain_metric_to_monitor", type=str, default="overall_val_loss", help="Name of pretrain metric to monitor for early stopping")
args.add_argument("--pretrain_metric_direction_which_is_optimal", type=str, default="min", help="Should pretrain metric be maximised (specify 'max') or minimised (specify 'min')?")

args.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
args.add_argument("--save_top_k", type=int, default=1, help="Number of top models to save")
args.add_argument("--optional_name_suffix", type=str, default=None, help="Optional suffix to add to model name")

args.add_argument("--fasta_shuffle_letters_path", type=str, default="fasta_shuffle_letters", help="Full path to the fasta_shuffle_letters executable")

args.add_argument("--val_chr", type=str, default="chr5", help="검증(Validation)에 사용할 Chromosome")
args.add_argument("--test_chr", type=str, default="chr7", help="테스트(Test)에 사용할 Chromosome")
args.add_argument("--train_sampling_ratio", type=float, default=1.0, help="학습 데이터 샘플링 비율 (예: 0.5는 50%)")
args.add_argument("--find_lr", action="store_true", help="학습을 시작하지 않고, 최적의 학습률을 찾습니다.")
args.add_argument("--strategy", type=str, default=None, 
                  help="PyTorch Lightning distributed training strategy (e.g., 'ddp', 'fsdp').")

args.add_argument("--input_csv_path", type=str, default=None, help="Path to the custom input CSV file for binary classification task")
args = args.parse_args()

assert os.path.exists(args.config_path), "Config file does not exist"
# Load config file
with open(args.config_path, "r") as config_file:
    config = json.load(config_file)

# Get adjusted root directory and root data directory (based upon whether you are running in Colab or not)
config['root_dir'] = get_base_directory(config['root_dir'])
print(f"Root directory: {config['root_dir']}") # print directory to verify
config['root_data_dir'] = get_base_directory(config['root_data_dir'])
print(f"Root data directory: {config['root_data_dir']}") # print directory to verify

# subsampling only works with Malinois_MPRA when finetuning/linear probing/joint training/individual training/simple regression
if args.subsample_train_set:
    if args.joint_tasks is not None:
        assert "Malinois_MPRA" in args.joint_tasks, "Subsampling only works with Malinois_MPRA"
        args.joint_tasks = args.joint_tasks.replace("Malinois_MPRA", f"Malinois_MPRA_subsampled_{args.n_train_subsample}")
    if args.finetune_tasks is not None:
        assert "Malinois_MPRA" in args.finetune_tasks, "Subsampling only works with Malinois_MPRA"
        args.finetune_tasks = args.finetune_tasks.replace("Malinois_MPRA", f"Malinois_MPRA_subsampled_{args.n_train_subsample}")
    if args.single_task is not None:
        assert "Malinois_MPRA" in args.single_task, "Subsampling only works with Malinois_MPRA"
        args.single_task = args.single_task.replace("Malinois_MPRA", f"Malinois_MPRA_subsampled_{args.n_train_subsample}")

# setup wandb
root_dir = config["root_dir"]
if not os.path.exists(root_dir):
    os.makedirs(root_dir, exist_ok=True)
wandb_logs_save_dir = os.path.join(root_dir, "wandb_logs")
if not os.path.exists(wandb_logs_save_dir):
    os.makedirs(wandb_logs_save_dir, exist_ok=True)
wandb_cache_dir = os.path.join(root_dir, "wandb_cache")
if not os.path.exists(wandb_cache_dir):
    os.makedirs(wandb_cache_dir, exist_ok=True)
os.environ["WANDB_DIR"] = wandb_logs_save_dir
os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

# use GPU if available
try:
    cuda_available = torch.cuda.is_available()
except Exception:
    cuda_available = False

device = "cuda" if cuda_available else "cpu"
print("Using {} device".format(device))

# train models
if "pretrain" in args.modelling_strategy:
    train_model(args, config, finetune=False)
    y, pred = train_model(args, config, finetune=True)
else:
    y, pred = train_model(args, config, finetune=False)

# Save training summary
import torch

save_dir = os.path.join(config["root_dir"], "summaries")
os.makedirs(save_dir, exist_ok=True)

# 자동으로 best ckpt 찾기 (마지막 저장 기준)
ckpt_dir = os.path.join(config["root_dir"], "saved_models", f"individual_training_on_{args.single_task}", "default", "checkpoints")
ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")])
if ckpts:
    best_ckpt = os.path.join(ckpt_dir, ckpts[-1])  # 가장 최근 파일
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