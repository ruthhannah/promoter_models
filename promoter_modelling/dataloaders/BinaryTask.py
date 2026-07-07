import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import lightning as L
import torch.nn as nn

class OneHotEncoder:
    """염기 서열을 One-hot 인코딩으로 변환합니다."""
    def __init__(self):
        self.mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}

    def encode(self, sequence):
        seq_len = len(sequence)
        onehot = np.zeros((4, seq_len), dtype=np.float32)
        for i, nucleotide in enumerate(sequence.upper()):
            if nucleotide in self.mapping and self.mapping[nucleotide] != -1:
                onehot[self.mapping[nucleotide], i] = 1.0
        return onehot

class BinarySequenceDataset(Dataset):
    """Pandas 데이터프레임을 받아 PyTorch Dataset 객체를 생성."""
    def __init__(self, dataframe, onehot_encoder):
        self.df = dataframe
        self.encoder = onehot_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row['sequence']
        label = row['label']
        
        onehot_seq = torch.tensor(self.encoder.encode(sequence), dtype=torch.float32).T
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return onehot_seq, label_tensor

class BinaryTask(L.LightningDataModule):
    """
    Chromosome 기준으로 데이터를 분리하고, 지정된 비율로 샘플링하며,
    학습/검증/테스트용 DataLoader를 생성하는 메인 클래스.
    """
    def __init__(self, csv_path, batch_size=64, val_chr='chr5', test_chr='chr7', train_sampling_ratio=1.0):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.val_chr = val_chr
        self.test_chr = test_chr
        self.train_sampling_ratio = train_sampling_ratio
        self.encoder = OneHotEncoder()

        self.name = "BinaryTask"
        self.num_outputs = 1
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.metrics = {"val": [], "test": []}
        self.with_mask = False
        self.promoter_windows_relative_to_TSS = []
        self.use_1hot_for_classification = False
        self.task = "classification"


    def setup(self, stage=None):
        full_df = pd.read_csv(self.csv_path)
        full_df = pd.read_csv(self.csv_path)
        # 'chr'도 허용하고 내부적으로 'chromosome'으로 통일
        
        # 1. 염색체 기준으로 기본 데이터 분리
        val_df = full_df[full_df['chr'] == self.val_chr]
        test_df = full_df[full_df['chr'] == self.test_chr]
        train_df = full_df[~full_df['chr'].isin([self.val_chr, self.test_chr])]
    
        # 2. 모든 데이터셋에 비율 기반 샘플링 적용
        if self.train_sampling_ratio < 1.0:
            print(f"\n--- [비율 기반 샘플링] 모든 데이터셋에서 {self.train_sampling_ratio * 100:.0f}%씩 샘플링합니다. ---")
            train_df = train_df.groupby('chr', group_keys=False).apply(lambda x: x.sample(frac=self.train_sampling_ratio, random_state=42))
            val_df = val_df.sample(frac=self.train_sampling_ratio, random_state=42)
            test_df = test_df.sample(frac=self.train_sampling_ratio, random_state=42)
    
        print("\n--- 최종 데이터 분할 결과 ---")
        print(f"학습 데이터: {len(train_df)}개")
        print(f"검증 데이터: {len(val_df)}개 (from {self.val_chr})")
        print(f"테스트 데이터: {len(test_df)}개 (from {self.test_chr})")
        print("----------------------------\n")

        self.train_dataset = BinarySequenceDataset(train_df, self.encoder)
        self.val_dataset = BinarySequenceDataset(val_df, self.encoder)
        self.test_dataset = BinarySequenceDataset(test_df, self.encoder)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
        
    def update_metrics(self, pred, y, loss, split):
        acc = ((pred > 0).float() == y).float().mean().item()
        loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        self.metrics[split].append({"loss": loss_value, "acc": acc})
        
    def compute_metrics(self, split):
        if not self.metrics[split]: return {}
        logs = self.metrics[split]
        avg_loss = sum(d["loss"] for d in logs) / len(logs)
        avg_acc = sum(d["acc"] for d in logs) / len(logs)
        self.metrics[split] = []
        return {
            f"{split}_{self.name}_avg_epoch_loss": avg_loss,
            f"{split}_{self.name}_avg_epoch_acc": avg_acc
        }