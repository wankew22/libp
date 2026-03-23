#!/usr/bin/env python3
"""
train_single_transformer.py

从头训练单 Transformer 多模态分类器（SMILES + AMI 统一编码）。
输入 CSV 格式: sequence,type,label
type ∈ {AMI, SMILES}
"""

import argparse
import os
from pathlib import Path
from collections import Counter

import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ---------------------
# Utilities: Tokenizer
# ---------------------
class CharVocab:
    def __init__(self, specials=["<pad>", "<unk>", "<cls>", "<sep>"]):
        self.specials = specials
        self.idx2tok = list(specials)
        self.tok2idx = {t: i for i, t in enumerate(self.idx2tok)}

    def build_from_sequences(self, sequences, min_freq=1):
        cnt = Counter()
        for s in sequences:
            cnt.update(list(s))
        for ch, c in sorted(cnt.items()):
            if c >= min_freq and ch not in self.tok2idx:
                self.tok2idx[ch] = len(self.idx2tok)
                self.idx2tok.append(ch)

    def __len__(self):
        return len(self.idx2tok)

    @property
    def pad_idx(self):
        return self.tok2idx["<pad>"]

    @property
    def unk_idx(self):
        return self.tok2idx["<unk>"]

    @property
    def cls_idx(self):
        return self.tok2idx["<cls>"]

    @property
    def sep_idx(self):
        return self.tok2idx["<sep>"]

    def encode(self, s, max_len=None, add_special=True):
        tokens = [self.tok2idx.get(ch, self.unk_idx) for ch in list(s)]
        if add_special:
            tokens = [self.cls_idx] + tokens + [self.sep_idx]
        if max_len is not None:
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [self.pad_idx] * (max_len - len(tokens))
        return tokens

# ---------------------
# Dataset
# ---------------------
class SequenceDataset(Dataset):
    def __init__(self, df, vocab, max_len=256):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len
        self.label_map = {l: i for i, l in enumerate(sorted(df['label'].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row['sequence']).strip()
        label = self.label_map[row['label']]
        tok = self.vocab.encode(seq, max_len=self.max_len, add_special=True)
        mask = [1 if t != self.vocab.pad_idx else 0 for t in tok]
        return {
            "input_ids": torch.tensor(tok, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}

# ---------------------
# Model
# ---------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, nhead=4, num_layers=2, dim_feedforward=256, max_len=512, dropout=0.1):
        super().__init__()
        self.enc_emb = nn.Embedding(vocab_size, emb_dim)
        nn.init.normal_(self.enc_emb.weight, mean=0.0, std=0.02)
        self.pos = PositionalEmbedding(emb_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # for sequence pooling

    def forward(self, input_ids, attention_mask):
        x = self.enc_emb(input_ids)  # (B, L, D)
        x = self.pos(x)
        src_key_padding_mask = (attention_mask == 0)  # True for PAD
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.pool(x).squeeze(-1)  # (B, D)
        # 检查 NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return x

class TransformerClassifier(pl.LightningModule):
    def __init__(self, vocab_size, emb_dim=128, nhead=4, num_layers=2, num_labels=2, lr=1e-3, max_len=256):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = SimpleTransformerEncoder(vocab_size, emb_dim=emb_dim, nhead=nhead,
                                                num_layers=num_layers, dim_feedforward=emb_dim*2, max_len=max_len)
        self.classifier = nn.Linear(emb_dim, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        x = self.encoder(input_ids, attention_mask)
        logits = self.classifier(x)
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        return logits

    def training_step(self, batch, batch_idx):
        labels = batch["label"].to(self.device)
        logits = self(batch)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        probs = F.softmax(logits, dim=1)
        probs = probs[:, 1].detach().cpu().numpy()

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, probs)  # 用正类概率计算ROC-AUC
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_prec", prec, on_epoch=True, prog_bar=True)
        self.log("train_auc", auc, on_epoch=True, prog_bar=True)
        self.log("train_rec", rec, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"].to(self.device)
        logits = self(batch)
        loss = self.loss_fn(logits, labels)
        
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        probs = F.softmax(logits, dim=1)
        probs = probs[:, 1].detach().cpu().numpy()

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, probs)  # 用正类概率计算ROC-AUC

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_prec", prec, on_epoch=True, prog_bar=True)
        self.log("val_auc", auc, on_epoch=True, prog_bar=True)
        self.log("val_rec", rec, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

# ---------------------
# Training entry
# ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="../dataset/cleaned_all_data.csv", help="input csv")
    parser.add_argument("--outdir", type=str, default="runs/single_transformer", help="output dir")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if not all(col in df.columns for col in ["sequence","type","label"]):
        raise ValueError("CSV must have columns: sequence,type,label")

    # build unified vocab
    sequences = df['sequence'].astype(str).tolist()
    vocab = CharVocab()
    vocab.build_from_sequences(sequences)

    # split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=args.seed, stratify=df['label'])
    train_ds = SequenceDataset(train_df, vocab, max_len=args.max_len)
    val_ds = SequenceDataset(val_df, vocab, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # labels count
    num_labels = len(sorted(df['label'].unique()))
    model = TransformerClassifier(vocab_size=len(vocab), emb_dim=args.emb_dim, nhead=args.nhead,
                                  num_layers=args.num_layers, num_labels=num_labels, lr=args.lr, max_len=args.max_len)

    # callbacks
    ckpt = ModelCheckpoint(dirpath=args.outdir, save_top_k=1, monitor="val_loss", mode="min",
                           filename="best-{epoch:02d}-{val_loss:.4f}")
    early = EarlyStopping(monitor="val_loss", mode="min", patience=8)

    trainer = pl.Trainer(
        default_root_dir=args.outdir,
        max_epochs=args.max_epochs,
        callbacks=[ckpt, early],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        precision=32,  # FP32 更稳定
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # save vocab
    import json
    with open(Path(args.outdir)/"vocab.json","w") as f:
        json.dump(vocab.idx2tok, f, ensure_ascii=False)

    print("Training finished. Best model saved to:", ckpt.best_model_path)

if __name__ == "__main__":
    main()

