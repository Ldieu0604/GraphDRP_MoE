
import os
import numpy as np
import pandas as pd
import math
import random
from rdkit import Chem

folder = "data/"

def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)
    except:
        return None

def load_cell_mutation_data():
    path = os.path.join(folder, "PANCANCER_Genetic_feature.csv")
    df = pd.read_csv(path)
    
    # Khớp với các cột bạn đã kiểm tra
    mut_col = 'genetic_feature' 
    cell_col = 'cell_line_name'
    val_col = 'is_mutated'

    print(f"--- Đang xử lý Mutation Table ---")
    mutation_matrix = df.pivot_table(index=cell_col, 
                                     columns=mut_col, 
                                     values=val_col, 
                                     fill_value=0)
    
    cell_dict = {cell: mutation_matrix.loc[cell].values for cell in mutation_matrix.index}
    print(f"Đã nạp {len(cell_dict)} dòng tế bào với {mutation_matrix.shape[1]} đặc trưng đột biến.")
    return cell_dict, mutation_matrix.shape[1]

def save_combined_data():
    # 1. Nạp dữ liệu IC50
    ic50_path = os.path.join(folder, "PANCANCER_IC.csv")
    df_ic50 = pd.read_csv(ic50_path)
    
    # 2. Nạp dữ liệu Cell Mutation
    cell_dict, feat_dim = load_cell_mutation_data()
    
    # 3. Nạp dữ liệu SMILES (Khớp với các cột bạn vừa gửi)
    smiles_path = os.path.join(folder, "drug_smiles.csv")
    drug_smiles_df = pd.read_csv(smiles_path)
    
    # SỬA TÊN CỘT TẠI ĐÂY:
    drug_dict = dict(zip(drug_smiles_df['name'], drug_smiles_df['CanonicalSMILES']))
    print(f"Đã nạp SMILES cho {len(drug_dict)} loại thuốc.")

    xd, xc, y = [], [], []

    # 4. Hợp nhất dữ liệu
    print("--- Đang hợp nhất Drug và Cell ---")
    for _, row in df_ic50.iterrows():
        # Thử cả hai kiểu đặt tên cột IC50 phổ biến
        drug = row.get('Drug name') or row.get('Drug_name')
        cell = row.get('Cell line name') or row.get('Cell_line_name')
        ic50 = float(row['IC50'])
        
        ic50_norm = 1 / (1 + pow(math.exp(ic50), -0.1))

        if drug in drug_dict and cell in cell_dict:
            smi = normalize_smiles(drug_dict[drug])
            if smi:
                xd.append(smi)
                xc.append(cell_dict[cell])
                y.append(ic50_norm)

    print(f"Tổng số mẫu hợp lệ: {len(xd)}")

    combined = list(zip(xd, xc, y))
    random.shuffle(combined)
    
    size = int(len(combined) * 0.8)
    size1 = int(len(combined) * 0.9)
    
    train_set = combined[:size]
    val_set = combined[size:size1]
    test_set = combined[size1:]

    from utils import TestbedDataset
    
    datasets = {'GDSC_train': train_set, 'GDSC_val': val_set, 'GDSC_test': test_set}

    for name, data in datasets.items():
        if len(data) > 0:
            xd_list, xc_list, y_list = zip(*data)
            TestbedDataset(root='data', dataset=name, xd=xd_list, xt=xc_list, y=y_list)
            print(f"Đã lưu thành công file: {name}.pt")

if __name__ == "__main__":
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    save_combined_data()