import numpy as np
from typing import List, Tuple, NewType, Union
import torch
from torch import Tensor
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors
from rdkit.Chem import Mol as RDMol

FlatRewards = NewType("FlatRewards", Tensor)
from gflownet.utils.common import set_device, set_float_precision
from gflownet.utils.unidock import VinaReward

from pathlib import Path
import json
import os
tmp_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(tmp_path))))

class unidock():
    "GPU-accelerated [UniDock](https://pubs.acs.org/doi/10.1021/acs.jctc.2c01145)"
    def __init__(self,device,float_precision,higher_is_better=None, dict_path =None, filter=None,protein_path=None,center=None,ref_ligand_path=None,size=None,log_dir=None,**kwargs):
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        self.higher_is_better = higher_is_better
        
        self.oracle_idx = 0
        self.filter = filter
        assert protein_path is not None, "protein_path is required"
        
        assert center is not None or ref_ligand_path is not None, "One of center or reference ligand path is required"

        self.dict_path: Path = Path(dict_path)
        self.dict_path.mkdir(parents=True, exist_ok=True)
        self.dict_file = self.dict_path / "unidock_dict.json"
        
        self.save_dir: Path = Path(log_dir) / "docking"
        self.save_dir.mkdir(exist_ok=True)
        self.vina = VinaReward(
            protein_path,
            center,
            ref_ligand_path,
            size,   #Search Box Size (--size X Y Z)
            search_mode="fast",  # fast, balance, detail
            num_workers=4,
            dir=self.save_dir
        )
        
        
    def constraint(self, mol: RDMol) -> bool:
        if mol is None:
            return False
        if self.filter is None:
            pass
        elif self.filter in ("lipinski", "veber"):
            if rdMolDescriptors.CalcExactMolWt(mol) > 600:
                return False
            if rdMolDescriptors.CalcNumHBD(mol) > 5:
                return False
            if rdMolDescriptors.CalcNumHBA(mol) > 10:
                return False
            if Crippen.MolLogP(mol) > 5:
                return False
            if self.filter == "veber":
                if rdMolDescriptors.CalcTPSA(mol) > 140:
                    return False
                if rdMolDescriptors.CalcNumRotatableBonds(mol) > 10:
                    return False
        else:
            raise ValueError(self.filter)
        return True
    
    def mol2vina(self, mols: list[RDMol]) -> torch.Tensor:
        out_path = self.save_dir / f"oracle{self.oracle_idx}.sdf"
        vina_scores = self.vina.run_mols(mols, out_path)
        return torch.tensor(vina_scores, dtype=torch.float32)
    
    def setup(self, env=None):
        self.max_seq_length = env.max_seq_length
    
    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y,dtype=self.float))
    
    def __call__(self, states: List) -> Tuple[FlatRewards, Tensor]:
        mols = [Chem.MolFromSmiles(s) if s else None for s in states]
        is_valid_t = torch.tensor([self.constraint(obj) if obj is not None else False for obj in mols], dtype=torch.bool)
        valid_mols = [obj for flag, obj in zip(is_valid_t, mols, strict=True) if flag]

        preds = torch.zeros((len(states),), dtype=torch.float)

        if self.dict_file.exists():
            with open(self.dict_file, "r") as f:
                score_dict = json.load(f)
        else:
            score_dict = {}

        if len(valid_mols) > 0:
            valid_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in valid_mols]
            
            known_scores = []
            unknown_indices = []
            unknown_mols = []
            
            for i, smi in enumerate(valid_smiles):
                if smi in score_dict:
                    known_scores.append(score_dict[smi])
                else:
                    known_scores.append(None)
                    unknown_indices.append(i)
                    unknown_mols.append(valid_mols[i])
            
            if unknown_mols:
                unknown_scores = self.mol2vina(unknown_mols).tolist()
                
                for idx, score in zip(unknown_indices, unknown_scores):
                    smi = valid_smiles[idx]
                    score_dict[smi] = score
                    known_scores[idx] = score
            
            docking_scores = torch.tensor(known_scores, dtype=torch.float32)
            valid_preds = docking_scores * -1
            preds[is_valid_t] = valid_preds
            
            with open(self.dict_file, "w") as f:
                json.dump(score_dict, f)

        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1)).to(self.device)
        return FlatRewards(preds).view(-1)