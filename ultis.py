import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader
import pandas as pd

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='GDSC', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None, saliency_map=False):
        """
        Args:
            xd: Danh sách SMILES (Drug)
            xt: Ma trận đặc trưng Cell Line (Mutation)
            y:  Nhãn IC50
        """
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.saliency_map = saliency_map
        
        if os.path.isfile(self.processed_paths[0]):
            print('Đang tải dữ liệu đã tiền xử lý: {}'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Không tìm thấy dữ liệu, đang bắt đầu xử lý...')
            self.process(xd, xt, y)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "Ba danh sách phải có cùng độ dài!"
        data_list = []
        data_len = len(xd)
        
        for i in range(data_len):
            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            # Tạo đối tượng dữ liệu PyG
            # Lưu ý: 'x' ở đây chúng ta để dummy vì MoE sẽ tự xử lý SMILES bên trong forward
            data = DATA.Data(y=torch.FloatTensor([labels]))
            
            # Lưu SMILES dưới dạng thuộc tính (để MoE dispatcher truy cập)
            data.smiles = smiles 
            
            # Lưu vector Mutation tế bào
            data.cell_features = torch.FloatTensor([target])

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Đã xử lý xong. Đang lưu file...')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Các hàm đo lường hiệu năng (Giữ nguyên từ GraphDRP để so sánh)
def rmse(y, f):
    return np.sqrt(((y - f)**2).mean(axis=0))

def mse(y, f):
    return ((y - f)**2).mean(axis=0)

def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]

def spearman(y, f):
    from scipy import stats
    return stats.spearmanr(y, f)[0]

# Hàm ghi log và vẽ biểu đồ (Tùy chỉnh để hỗ trợ theo dõi aux_loss)
def draw_loss(train_losses, val_losses, title):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(train_losses, label='Train Loss (MSE + Aux)')
    plt.plot(val_losses, label='Val Loss (MSE)')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(title + ".png")