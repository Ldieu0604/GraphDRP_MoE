
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np

# Thêm đường dẫn để Python tìm thấy các module cục bộ
sys.path.append(os.getcwd())

from utils import TestbedDataset
from models import MoE_GraphDRP_Model 
from experts.smi_ted_light.load import load_smi_ted
from experts.mhg_model.load import load as load_mhg
from experts.selfies_ted.load import SELFIES 
from moe import MoE

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader):
        # 'data' lúc này là một đối tượng Batch của PyG 
        # chứa các thuộc tính bạn đã định nghĩa trong utils.py
        
        # 1. Lấy SMILES (Sử dụng thuộc tính .smiles bạn đã gán)
        # Khi Batch hóa, PyG giữ smiles dưới dạng list/tuple chuỗi
        xd = data.smiles 
        
        # 2. Lấy Cell Features (Đã được bạn FloatTensor hóa trong utils)
        # PyG Batch sẽ tự động nối (concatenate) các vector này lại
        xc = data.cell_features.to(device).float()
        
        # 3. Lấy nhãn IC50
        y = data.y.to(device).float().view(-1, 1)
        
        optimizer.zero_grad()
        
        # Forward pass qua MoE_GraphDRP_Model
        output, aux_loss = model(xd, xc)
        
        mse_loss = nn.MSELoss()(output, y)
        loss = mse_loss + 0.01 * aux_loss 
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} Batch: {batch_idx} Loss: {loss.item():.6f}')
            
    return total_loss / len(train_loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64 
    lr = 0.0001
    epochs = 100
    
    print(f"--- Đang chạy trên thiết bị: {device} ---")

    # 1. Nạp các Expert Models (Foundation Models)
    print("--- Đang nạp các Experts ---")
    smi_ted_model = load_smi_ted()
    tokenizer = smi_ted_model.tokenizer
    selfies_model = SELFIES()
    selfies_model.load()
    mhg_model = load_mhg()
    
    # Danh sách các mô hình gốc để đưa vào MoE
    raw_models = [smi_ted_model, selfies_model, mhg_model]

    # 2. Xử lý Embedding cho Gating Network
    # Lấy pretrained weights từ SMI-TED để Gating Network nhanh nhạy hơn
    try:
        weight_data = smi_ted_model.encoder.tok_emb.weight
        num_embeddings, embedding_dim = weight_data.shape
        tok_emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        tok_emb_layer.weight = nn.Parameter(weight_data.clone().detach())
        print("SMI-TED: Đã nạp Pretrained Embedding thành công.")
    except Exception as e:
        print(f"SMI-TED: Sử dụng Dummy Embedding cho Gating do: {e}")
        tok_emb_layer = nn.Embedding(tokenizer.vocab_size, 768)

    # 3. Khởi tạo cấu trúc MoE và Model chính
    print("--- Khởi tạo cấu trúc MoE-GraphDRP ---")
    
    # SỬA LỖI TẠI ĐÂY: Khớp với tham số 'models' trong moe.py
    moe_layer = MoE(
        input_size=768,    # Kích thước vector từ EmbeddingNet (thường là 768 cho BERT light)
        output_size=2048,  # Kích thước vector đầu ra của mỗi Expert
        num_experts=3,
        models=raw_models, # Tên tham số đúng trong moe.py là 'models'
        tokenizer=tokenizer,
        tok_emb=tok_emb_layer
    )

    model = MoE_GraphDRP_Model(
        moe_layer=moe_layer, 
        cell_feature_dim=735 
    ).to(device)

    # 4. Nạp dữ liệu từ TestbedDataset
    print("--- Đang nạp dữ liệu từ data/processed ---")
    train_data = TestbedDataset(root='data', dataset='GDSC_train')
    val_data = TestbedDataset(root='data', dataset='GDSC_val')
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    result_file = "results_moe_graphdrp.csv"

    # 5. Vòng lặp huấn luyện (Training Loop)
    with open(result_file, "w") as f:
        f.write("Epoch,Train_Loss\n")

    print("--- Bắt đầu huấn luyện ---")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        print(f"Epoch {epoch}: Loss = {train_loss:.4f}")
        
        with open(result_file, "a") as f:
            f.write(f"{epoch},{train_loss:.4f}\n")
            
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "best_moe_graphdrp.model")
            print(f"--> Đã lưu mô hình tốt nhất tại Epoch {epoch}")

if __name__ == "__main__":
    main()