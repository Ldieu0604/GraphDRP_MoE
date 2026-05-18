
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import pandas as pd
from models import Expert

class SparseDispatcher(object):
    """
    Bộ điều phối dữ liệu: Tách các phân tử đầu vào và gửi đến đúng Expert 
    mà mạng Gating đã chỉ định, sau đó thu thập và hợp nhất kết quả.
    """
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        
        # Xác định device hiện tại từ gates (thường là cuda:0)
        self.device = gates.device
        
        # Sắp xếp và xác định index của các expert có trọng số > 0
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Chia SMILES đầu vào thành danh sách các nhóm cho từng Expert."""
        inp_as_series = pd.Series(inp)
        # Sử dụng batch_index đã được đưa về CPU để indexing pandas
        batch_idx_cpu = self._batch_index.cpu().numpy()
        inp_exp = inp_as_series.iloc[batch_idx_cpu]
        
        _part_indexes = [sum(self._part_sizes[:i]) for i in range(1, len(self._part_sizes))]
        return [list(x) for x in np.split(inp_exp.to_numpy(), _part_indexes, axis=0)]

    def combine(self, expert_out):
        """Hợp nhất các vector từ Expert dựa trên trọng số của cổng Gating."""
        # 1. Ghép nối các tensor từ Expert
        stitched = torch.cat(expert_out, 0)
        
        # 2. FIX LỖI DEVICE: Đưa stitched vector về cùng device với cổng Gating
        stitched = stitched.to(self.device)
        
        # 3. Nhân với trọng số cổng (Gating Weights)
        stitched = stitched.mul(self._nonzero_gates)
        
        # 4. Chuẩn bị tensor kết quả
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=self.device)
        
        # 5. Cộng dồn vào vector kết quả (Stitched Vector)
        combined = zeros.index_add(0, self._batch_index.to(self.device), stitched.float())
        return combined

class MoE(nn.Module):
    """
    Lớp MoE tích hợp 3 loại Expert: SMI-TED, SELFIES-TED, MHG-GNN.
    """
    def __init__(self, input_size, output_size, num_experts, models, tokenizer, tok_emb, k=2, noisy_gating=True):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k
        self.noisy_gating = noisy_gating

        # Khởi tạo danh sách Expert wrappers
        self.experts = nn.ModuleList([Expert(m, self.output_size) for m in models])
        
        # Mạng Gating
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        # EmbeddingNet để chuyển SMILES thành vector nhanh cho việc Gating
        self.embedding_net = self.EmbeddingNet(tokenizer, tok_emb, input_size)

    class EmbeddingNet(nn.Module):
        def __init__(self, tokenizer, tok_emb, n_embd):
            super().__init__()
            self.tokenizer = tokenizer
            self.tok_emb = tok_emb
            # Đóng băng trọng số của Embedding Foundation Model
            if isinstance(self.tok_emb, nn.Embedding):
                self.tok_emb.weight.requires_grad = False
            else:
                for param in self.tok_emb.parameters():
                    param.requires_grad = False

        def forward(self, smiles):
            # Tự động lấy device từ trọng số embedding
            device = self.tok_emb.weight.device if isinstance(self.tok_emb, nn.Embedding) else next(self.tok_emb.parameters()).device
            
            tokens = self.tokenizer(smiles, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
            idx, mask = tokens['input_ids'], tokens['attention_mask']
            
            token_embeddings = self.tok_emb(idx)
            
            # Mean pooling
            mask_exp = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1: return torch.tensor([0.0], device=x.device)
        return x.float().var() / (x.float().mean()**2 + eps)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        else:
            logits = clean_logits

        # Chọn Top-K experts
        top_logits, top_indices = logits.topk(min(self.k, self.num_experts), dim=1)
        top_k_logits = top_logits
        top_k_indices = top_indices
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
        importance = gates.sum(0)
        load = (gates > 0).sum(0).float()
        return gates, importance, load

    def forward(self, smiles, loss_coef=1e-2):
        # 1. Chuyển SMILES sang embedding nhanh
        x_gate = self.embedding_net(smiles)
        
        # 2. Gating
        gates, importance, load = self.noisy_top_k_gating(x_gate, self.training)
        
        # 3. Aux loss để cân bằng Expert
        aux_loss = (self.cv_squared(importance) + self.cv_squared(load)) * loss_coef

        # 4. Dispatch & Expert Forward
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(smiles)
        
        expert_outputs = []
        for i in range(self.num_experts):
            # Expert forward trả về Tensor (có thể trên CPU hoặc GPU)
            out = self.experts[i](expert_inputs[i])
            expert_outputs.append(out)
        
        # 5. Hợp nhất kết quả
        drug_stitched_vector = dispatcher.combine(expert_outputs)
        
        return drug_stitched_vector, aux_loss