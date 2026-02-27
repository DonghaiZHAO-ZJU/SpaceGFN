import numpy as np
from typing import List, Tuple, NewType, Union
import torch
from torch import Tensor
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
FlatRewards = NewType("FlatRewards", Tensor)
from gflownet.utils.common import set_device, set_float_precision
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pickle

import os
tmp_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(tmp_path))))


class EGFR():
    """
    Building Block oracle
    """
        
    def __init__(self,device,float_precision,higher_is_better=None,**kwargs):
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        self.higher_is_better = higher_is_better
        with open("gflownet/proxy/QSAR/EGFR/lgb.pkl", 'rb') as f:
            self.model = pickle.load(f)
            self.model.set_params(verbosity=-1)
    def setup(self, env=None):
        self.max_seq_length = env.max_seq_length
    
    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y,dtype=self.float,device=self.device))
    
    def limit(self, mol) -> bool:

        limitation = Descriptors.ExactMolWt(mol) > 600

        return limitation
    
    def __call__(self, states: List[str]) -> Tuple[FlatRewards, torch.Tensor]:
        mols = []
        pos = []
        preds = [0.0] * len(states)
        
        for i, smi in enumerate(states):
            if smi is not None:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    mols.append(mol)
                    pos.append(i)

        if mols:
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols]
            X = np.array(fps)
            
            pred_values = []
            for mol, X in zip(mols, X):
                """
                # When using a QSAR model with limited generalization ability as the proxy,
                # a molecular weight threshold can be customized to avoid sampling molecules that are too large.
                # This limit can be applied either during training from scratch or when fine-tuning a trained model.
                
                """
#                 if self.limit(mol):
#                     pred_values.append(0.0)
#                 else:
#                     pred_values.append(self.model.predict([X])[0] / 10)

                pred_values.append(self.model.predict([X])[0] / 10)
                
            for p, v in zip(pos, pred_values):
                preds[p] = v

        preds_tensor = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds_tensor).view(-1)