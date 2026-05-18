
# -*- coding:utf-8 -*-
import os
import pickle
import sys
import torch
from rdkit import Chem

# --- FIX LỖI IMPORT TẠI ĐÂY ---
# Lấy đường dẫn tuyệt đối của thư mục chứa file load.py này (chính là mhg_model)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Thêm cả thư mục cha (experts) vào path để hỗ trợ các import dạng .models
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# ------------------------------

from torch_geometric.utils.smiles import from_smiles
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Self

# Import cục bộ từ thư mục hiện tại
from .models.mhgvae import GrammarGINVAE
from .graph_grammar.io.smi import hg_to_mol

class PretrainedModelWrapper:
    model: GrammarGINVAE

    def __init__(self, model_dict: Dict[str, Any]) -> None:
        json_params = model_dict['gnn_params']
        encoder_params = json_params['encoder_params']
        encoder_params['node_feature_size'] = model_dict['num_features']
        encoder_params['edge_feature_size'] = model_dict['num_edge_features']
        self.model = GrammarGINVAE(model_dict['hrg'], rank=-1, encoder_params=encoder_params,
                                   decoder_params=json_params['decoder_params'],
                                   prod_rule_embed_params=json_params["prod_rule_embed_params"],
                                   batch_size=512, max_len=model_dict['max_length'])
        self.model.load_state_dict(model_dict['model_state_dict'])
        self.model.eval()

    def to(self, device: Union[str, int, torch.device]) -> Self:
        if isinstance(device, str) or isinstance(device, torch.device):
            target_device = torch.device(device)
        else:
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(target_device)
        return self

    def encode(self, data: List[str]) -> List[torch.tensor]:
        output = []
        for d in data:
            params = next(self.model.parameters())
            g = from_smiles(d)
            # Đưa graph lên cùng thiết bị với model
            g = g.to(params.device)
            ltvec = self.model.graph_embed(g.x, g.edge_index, g.edge_attr, g.batch)
            output.append(ltvec[0])
        return output

    def decode(self, data: List[torch.tensor]) -> List[str]:
        output = []
        for d in data:
            mu, logvar = self.model.get_mean_var(d.unsqueeze(0))
            z = self.model.reparameterize(mu, logvar)
            flags, _, hgs = self.model.decode(z)
            if flags[0]:
                reconstructed_mol, _ = hg_to_mol(hgs[0], True)
                output.append(Chem.MolToSmiles(reconstructed_mol))
            else:
                output.append(None)
        return output


def load(local_path="experts/mhg_model/pickles/pytorch_model.bin"):
    # Đảm bảo đường dẫn chính xác trên Kaggle
    if not os.path.exists(local_path):
        # Thử tìm ở thư mục hiện tại nếu chạy từ training.py
        local_path = os.path.join(os.getcwd(), local_path)
        
    if os.path.exists(local_path):
        print(f"--- Đang nạp MHG Model từ: {local_path} ---")
        # weights_only=False là bắt buộc vì nạp cả object của IBM Rhizome
        model_dict = torch.load(local_path, map_location='cpu', weights_only=False)
        return PretrainedModelWrapper(model_dict)
    
    raise FileNotFoundError(f"Chưa thấy file MHG tại: {local_path}")