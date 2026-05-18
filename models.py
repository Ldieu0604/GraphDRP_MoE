import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# Giữ lại lớp MLP cho các tác vụ phụ trợ nếu cần
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out

# Expert wrapper để bọc các Foundation Models (SMI-TED, SELFIES, MHG)
class Expert(nn.Module):
    def __init__(self, model, output_size, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.model = model
        self.output_size = output_size

    def forward(self, x):
        if len(x) == 0:
            return torch.empty(size=(0, self.output_size))

        # Gọi phương thức encode của từng Expert (SMI-TED, SELFIES, hoặc MHG)
        out = self.model.encode(x)

        if isinstance(out, pd.DataFrame):
            out = torch.tensor(out.values, dtype=torch.float32)
        elif isinstance(out, list):
            # MHG-GNN thường trả về list các tensor
            if torch.is_tensor(out[0]):
                out = torch.stack(out, dim=0)
            else:
                out = torch.tensor(np.array(out), dtype=torch.float32)
        
        # Căn chỉnh kích thước output (padding) về output_size (2048)
        if out.shape[1] < self.output_size:
            out = F.pad(out, pad=(0, self.output_size - out.shape[1], 0, 0), value=0)
        elif out.shape[1] > self.output_size:
            out = out[:, :self.output_size]

        return out

# Nhánh xử lý Tế bào (Cell Line Branch) kế thừa từ kiến trúc GraphDRP
class CellLineBranch(nn.Module):
    def __init__(self, num_feature_pc, n_filters=32, embed_dim=128):
        super(CellLineBranch, self).__init__()
        # 1D Convolutional layers xử lý ma trận đột biến/biểu hiện gen
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        
        # Adaptive pooling để đảm bảo đầu ra cố định bất kể kích thước input
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1_xt = nn.Linear(n_filters*4, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [batch, features] -> [batch, 1, features]
        x = x.unsqueeze(1)
        x = self.relu(self.conv_xt_1(x))
        x = self.pool_xt_1(x)
        x = self.relu(self.conv_xt_2(x))
        x = self.pool_xt_2(x)
        x = self.relu(self.conv_xt_3(x))
        x = self.pool_xt_3(x)
        
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.fc1_xt(x)
        return x

# Mô hình tích hợp cuối cùng (Fusion & Prediction Head)
class MoE_GraphDRP_Model(nn.Module):
    def __init__(self, moe_layer, cell_feature_dim, drug_output_dim=2048, cell_output_dim=128):
        super(MoE_GraphDRP_Model, self).__init__()
        
        # 1. Nhánh MoE Drug (Chứa SMI-TED, SELFIES, MHG Experts)
        self.drug_moe = moe_layer
        
        # 2. Nhánh Cell Line (CNN 1D)
        self.cell_branch = CellLineBranch(num_feature_pc=cell_feature_dim, embed_dim=cell_output_dim)
        
        # 3. Tầng Fusion (Kết hợp Drug + Cell)
        combined_dim = drug_output_dim + cell_output_dim
        self.fc1 = nn.Linear(combined_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1) # Đầu ra dự đoán IC50 (hồi quy)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, smiles, cell_data):
        # Lấy drug vector (Stitched Vector) và aux_loss từ MoE
        drug_emb, aux_loss = self.drug_moe(smiles)
        
        # Lấy cell vector từ CNN
        cell_emb = self.cell_branch(cell_data)
        
        # Concatenate Drug + Cell
        xc = torch.cat((drug_emb, cell_emb), 1)
        
        # Đi qua các lớp Dense
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        
        out = self.out(xc)
        # Không dùng Sigmoid ở cuối để phù hợp với bài toán hồi quy IC50
        return out, aux_loss