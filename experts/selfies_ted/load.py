
import os
import sys
import torch
import selfies as sf  # selfies>=2.1.1
import pickle
import pandas as pd
import numpy as np
from datasets import Dataset
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

class SELFIES(torch.nn.Module):

    def __init__(self, local_dir="experts/selfies_ted"):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.invalid = []
        # Tự động nạp model và tokenizer ngay khi khởi tạo
        self.load(local_dir)

    def load(self, local_dir="experts/selfies_ted"):
        if os.path.exists(local_dir):
            print(f"--- Đang nạp SELFIES-TED từ: {local_dir} ---")
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
            self.model = AutoModel.from_pretrained(local_dir)
            self.model.eval() # Chuyển sang chế độ đánh giá
        else:
            raise FileNotFoundError(f"Không tìm thấy thư mục SELFIES tại: {local_dir}")

    def get_selfies(self, smiles_list):
        self.invalid = []
        spaced_selfies_batch = []
        for i, smiles in enumerate(smiles_list):
            try:
                # Loại bỏ khoảng trắng và ký tự xuống dòng
                smi = smiles.strip()
                selfies = sf.encoder(smi)
            except:
                try:
                    mol = Chem.MolFromSmiles(smiles.strip())
                    smi = Chem.MolToSmiles(mol)
                    selfies = sf.encoder(smi)
                except:
                    selfies = "[]"
                    self.invalid.append(i)

            spaced_selfies_batch.append(selfies.replace('][', '] ['))
        return spaced_selfies_batch

    def get_embedding(self, batch):
        # Đảm bảo model và input ở cùng một device (GPU hoặc CPU)
        device = next(self.model.parameters()).device
        
        encoding = self.tokenizer(
            batch["selfies"], 
            return_tensors='pt', 
            max_length=128, 
            truncation=True, 
            padding='max_length'
        ).to(device)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        with torch.no_grad(): # Không tính gradient để tiết kiệm bộ nhớ và tăng tốc
            outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            model_output = outputs.last_hidden_state

            # Mean Pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
            sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask

        # Chuyển về CPU và numpy để tránh lỗi CUDA khi dùng Dataset.map
        return {"embedding": pooled_output.detach().cpu().numpy()}

    def encode(self, smiles_list=[], use_gpu=False, return_tensor=False):
        if not smiles_list:
            return torch.tensor([]) if return_tensor else pd.DataFrame([])

        # 1. Chuyển SMILES sang SELFIES
        selfies_list = self.get_selfies(smiles_list)
        selfies_df = pd.DataFrame(selfies_list, columns=["selfies"])
        data = Dataset.from_pandas(selfies_df)

        # 2. Map lấy Embedding
        # QUAN TRỌNG: num_proc=None để tránh lỗi "Cannot re-initialize CUDA in forked subprocess"
        processed_data = data.map(
            self.get_embedding, 
            batched=True, 
            batch_size=128,
            num_proc=None 
        )
        
        emb = np.array(processed_data["embedding"])

        # 3. Xử lý các trường hợp lỗi
        for idx in self.invalid:
            emb[idx] = np.nan
            # print(f"Không thể mã hóa {smiles_list[idx]} sang SELFIES, thay thế bằng NaN")

        if return_tensor:
            return torch.tensor(emb, dtype=torch.float32)
        return pd.DataFrame(emb)