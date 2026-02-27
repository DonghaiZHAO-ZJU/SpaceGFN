"""
Classes to represent building block environments
"""
import uuid
import itertools
import time
from copy import deepcopy
from copy import copy as shallowcopy
from typing import List, Optional, Tuple, Union
from textwrap import dedent
import random
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import math
import yaml
import json
import gzip
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Bernoulli
from torchtyping import TensorType
import os
tmp_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(tmp_path))))
from gflownet.utils.common import copy, set_device, set_float_precision, tbool, tfloat

from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)

global smiles_list, mask_dict, vocab_2, inverse_lookup_2, lookup_2, react_key,\
    vocab_1, lookup_1, inverse_lookup_1, reactants, sub_reactants, mol_dict, rxnfp_generator,\
    building_block_embeddings,reaction_embeddings  \

with open("config/main.yaml", "r", encoding="utf-8") as file:
    main_config = yaml.safe_load(file)

#mode check
mode = main_config['mode']
evo = main_config.get('evo', False)

smiles_list_dir = main_config['smiles_list_dir']
mask_dict_dir = main_config['mask_dict_dir']

with gzip.GzipFile(smiles_list_dir,'r') as f_in:
    smiles_list = json.load(f_in)

if mode =='editing':
    pdb_id = main_config['pdb_id']
    initial_smiles_list_dir = main_config['initial_smiles_list_dir'].format(pdb_id=pdb_id)
    with gzip.GzipFile(initial_smiles_list_dir,'r') as f_in:
        initial_smiles_list = json.load(f_in)
    # Add the initial molecule list at the end

    smiles_list = smiles_list + initial_smiles_list
    
    # Load the reaction dataset for editing mode

    from data.template.edit_rule_v1 import smarts_list, name_list, uni_indices
    
elif mode =='discovery':
    if evo:
        # Load the Evo reaction dataset used in discovery mode
        from data.template.evo_template import smarts_list, name_list, uni_indices
        print("You are now using the Evo reaction template.")
    else:
        # Load the standard (synthetic) reaction dataset for discovery mode
        from data.template.syngfn_template import smarts_list, name_list, uni_indices
        print("You are now using the standard reaction template.")
else:
    raise ValueError("mode must be either 'discovery' or 'editing'")

with gzip.GzipFile(mask_dict_dir,'r') as f_in:
    mask_dict=json.load(f_in) 
    
bb_nbits = main_config.get('bb_nbits', False)

def get_mol_embeddings(smiles: List[str], nbits=1024) -> torch.Tensor:
    embeddings = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=nbits)
            fingerprint = np.zeros((nbits,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)
            embeddings.append(torch.tensor(fingerprint, dtype=torch.float32))
        else:
            embeddings.append(torch.zeros(nbits, dtype=torch.float32))

    return torch.stack(embeddings, dim=0)

policy_output_1_fixed = main_config['policy_output_1_fixed']
policy_output_2_fixed = main_config['policy_output_2_fixed']

if policy_output_2_fixed:
    building_block_embeddings = get_mol_embeddings(smiles_list, nbits=bb_nbits)
else:
    building_block_embeddings = None

reactants = [smarts.split('>>')[0] for smarts in smarts_list]
sub_reactants = [r.split('.') for r in reactants]

mol_dict = {}
for reactant in sub_reactants:
    for smart in reactant:
        mol = Chem.MolFromSmarts(smart)
        mol_dict[smart] = mol

eos_token, uni_token, pad_token, begin_token = ["EOS","UNI","PAD", "^"]
vocab_1 = smarts_list + ["EOS"]
lookup_1={a: i for (i, a) in enumerate(vocab_1)}
inverse_lookup_1 = {i: a for (i, a) in enumerate(vocab_1)}
vocab_2 = smiles_list + ["EOS","UNI","PAD"]
inverse_lookup_2 = {i: a for (i, a) in enumerate(vocab_2)}
lookup_2 = {a: i for (i, a) in enumerate(vocab_2)}
react_key = list(mask_dict.keys())

model, tokenizer = get_default_model_and_tokenizer()

rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
def get_rxn_embeddings(smarts_list: List[str]) -> torch.Tensor:
    embeddings = []
    for smarts in smarts_list:
        try:
            rxn_fp = rxnfp_generator.convert(smarts)
            embeddings.append(torch.tensor(rxn_fp, dtype=torch.float32))
        except:
            print(f"Error in converting {smarts}")
            embeddings.append(torch.zeros(256, dtype=torch.float32))
    return torch.stack(embeddings, dim=0)

if policy_output_2_fixed:
    reaction_embeddings = get_rxn_embeddings(smarts_list)
else:
    reaction_embeddings = None

CMAP = mpl.colormaps["cividis"]

class BuildingBlock:
    """
    Building block environment
    """

    def __init__(
        self,
        device: str = "cpu",
        float_precision: int = 32,
        env_id: Union[int, str] = "env",
        reward_min: float = 1e-8,
        reward_beta: float = 1.0,
        reward_norm: float = 1.0,
        reward_norm_std_mult: float = 0.0,
        reward_func: str = "identity",
        energies_stats: List[int] = None,
        denorm_proxy: bool = False,
        proxy=None,
        oracle=None,
        reaction_step=2,
        **kwargs,
    ):
        self.env_id = env_id
        # Device
        self.device = set_device(device)
        
        # Float precision
        self.float = set_float_precision(float_precision)
        
        #state length
        assert reaction_step > 0, "reaction_step must be greater than 0"
        self.max_seq_length = reaction_step + 2
        
        # Reward settings
        self.min_reward = reward_min
        assert self.min_reward > 0
        self.reward_beta = reward_beta
        assert self.reward_beta > 0
        self.reward_norm = reward_norm
        assert self.reward_norm > 0
        self.reward_norm_std_mult = reward_norm_std_mult
        self.reward_func = reward_func
        self.energies_stats = energies_stats
        self.denorm_proxy = denorm_proxy
        
        # Proxy and oracle
        self.proxy = proxy
        self.setup_proxy()
        if oracle is None:
            self.oracle = self.proxy
        else:
            self.oracle = oracle
        self.proxy_factor = 1.0
        
        # Log SoftMax function
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
    
        # State set
        self.smarts_list = smarts_list
        self.name_list = name_list
        self.uni_list = uni_indices
        
        self.vocab_1 = vocab_1
        self.lookup_1 = lookup_1
        self.inverse_lookup_1 = inverse_lookup_1
        self.n_alphabet_1 = len(smarts_list)
        self.n_alphabet_2 = len(smiles_list)
        
        self.padding_idx = lookup_2["PAD"]
        self.uni_idx = lookup_2["UNI"]
        self.eos_1 = self.lookup_1["EOS"]
        self.eos_2 = lookup_2["EOS"]
        
        # Initial state
        self.source = ([self.padding_idx] * ((self.max_seq_length - 1) * 3 - 1), None)
        
        # Action space
        self.action_space_1 = self.get_action_space_1()
        self.action_space_2 = self.get_action_space_2()
        
        self.action_space_dim_1 = len(self.action_space_1)
        self.action_space_dim_2 = len(self.action_space_2)
        
        # Call reset() to set initial state, done, n_actions
        self.reset()
        
        # Policy outputs
        self.fixed_policy_output_1 = self.get_policy_output_1()
        self.fixed_policy_output_2 = self.get_policy_output_2()
        self.random_policy_output_1 = self.get_policy_output_1()
        self.random_policy_output_2 = self.get_policy_output_2()
        
        self.policy_output_dim_1 = len(self.fixed_policy_output_1)
        self.policy_output_dim_2 = len(self.fixed_policy_output_2)
        
        self.policy_input_dim_1 = len(self.state2policy_1())
        self.policy_input_dim_2 = len(self.state2policy_2())
        
        #building_block_embeddings
        self.building_block_embeddings = building_block_embeddings
        self.bb_nbits = bb_nbits
        
        #reaction_embeddings
        self.reaction_embeddings = reaction_embeddings
        
        #policy output dim whether fix
        self.policy_output_1_fixed = policy_output_1_fixed
        self.policy_output_2_fixed = policy_output_2_fixed
        
    def get_action_space_1(self):
        """
        Constructs list with all possible actions
        """
        alphabet = [a for a in range(self.n_alphabet_1)]
        actions = [el for el in itertools.product(alphabet, repeat=1)]
        actions = actions + [(len(actions),)] # add EOS
        return actions
    
    def get_action_space_2(self):
        alphabet = [a for a in range(self.n_alphabet_2)]
        actions = [el for el in itertools.product(alphabet, repeat=1)]
        actions = actions + [(len(actions),)]+[(len(actions)+1,)]#add EOS and UNI
        return actions
    
    def get_state(self):
        return self.state
    
    def get_padding_idx(self):
        return self.padding_idx
    
    def set_done(self):
        self.done = True
        return self
        
    def _get_state(self, state: Tuple[List, str]):
        if state is None:
            state = copy(self.state)
        return state
    
    def _get_done(self, done: bool):
        if done is None:
            done = self.done
        return done
    
    def check_mask_1(self, mol, reactant):
        for r in reactant:
            if mol.HasSubstructMatch(mol_dict[r]):
                return True
        return False
    
    def get_mask_invalid_actions_forward_1(
        self,
        state: Optional[Tuple] = None,
        done: Optional[bool] = None,
    ) -> List:
        if state is None:
            state = copy(self.state)
        if done is None:
            done = self.done
        seq = state[0]
        num_bb = int((len(seq) + 2) / 3)

        if done or seq[num_bb - 1] != self.padding_idx:
            mask = [True] * (len(self.action_space_1) - 1)+[False]
            return mask
        
        if seq[0] == self.padding_idx:
            mask=[False] * (len(self.action_space_1) - 1)+[True]
            return mask
        
        mask = [False] * len(self.action_space_1)
        smi = self.state2readable(state)
        try:
            mol = Chem.MolFromSmiles(smi)
            if mode == 'editing':
                mol = Chem.AddHs(mol)
        except:
            mask = [True] * (len(self.action_space_1) - 1)+[False]
            return mask
    
        if mol:
            for idx, _ in enumerate(self.action_space_1[:-1]):
                mask[idx] = not self.check_mask_1(mol, sub_reactants[idx])
        
        return mask
    
    def get_mask_invalid_actions_forward_2(
        self,
        state: Optional[Tuple] = None,
        done: Optional[bool] = None,
        action_1: Optional[Tuple] = None,
    ) -> List:
        if state is None:
            state = copy(self.state)
        if done is None:
            done = self.done
        if isinstance(state, list):
            raise ValueError(f"Expected state to be a tuple, but got a list: {state}")
        seq = state[0]
        num_bb = int((len(seq) + 2) / 3)
        
        if done or seq[num_bb - 1] != self.padding_idx:
            mask = [True] * len(self.action_space_2)
            mask[-2] = False
            return mask
        
        if seq[0] == self.padding_idx and mode == 'discovery': 
            mask = [False] * len(self.action_space_2)
            mask[-2:] = [True,True]
            return mask
        
        if seq[0] == self.padding_idx and mode == 'editing':
            mask = [True] * len(self.action_space_2)
            initial_pos = len(smiles_list) - len(initial_smiles_list)
            mask[initial_pos:len(smiles_list)] = [False] * len(initial_smiles_list)
            return mask
        
        if action_1[0] == self.eos_1:
            mask = [True] * len(self.action_space_2)
            mask[-2] = False
            return mask

        if action_1[0] in self.uni_list:
            mask = [True] * len(self.action_space_2)
            mask[-2:] = [False, False]
            if mode == 'editing':
                if seq[1] == self.padding_idx:
                    mask[-2] = True
            return mask
        
        # if bi-reaction
        else:
            smarts = self.inverse_lookup_1[action_1[0]]
            smiles = self.state2readable(state)
            mask = [True] * len(self.action_space_2)
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mode == 'editing':
                    mol = Chem.AddHs(mol)
            except:
                raise ValueError(
                    f" {smiles} is not a valid smiles."
                )
            r1_smarts, r2_smarts = sub_reactants[action_1[0]]   
            if mol.HasSubstructMatch(mol_dict[r1_smarts]):
                mask_key = self.name_list[action_1[0]]+"_reactant_2"
                valid_list = mask_dict[mask_key]
                for i in valid_list:
                    mask[i]=False
                mask[-2] = False
                return mask
            elif mol.HasSubstructMatch(mol_dict[r2_smarts]):
                mask_key = self.name_list[action_1[0]]+"_reactant_1"
                valid_list = mask_dict[mask_key]
                for i in valid_list:
                    mask[i]=False
                mask[-2] = False
                return mask
            else:
                raise ValueError(
                    f" {smarts} is not a valid smarts for {smiles}."
                )

    def get_mask_invalid_actions_backward_1(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        parents_a_1: Optional[List] = None,
    ) -> List:
        state = self._get_state(state)
        done = self._get_done(done)
        mask = [True] * self.action_space_dim_1
        if parents_a_1 is None:
            _, parents_a_1, _ = self.get_parents(state, done)
        if len(parents_a_1) == 0:
            mask = [False] * self.action_space_dim_1
            mask[-1] = True
            return mask
        idx = parents_a_1[0]
        if idx[0] == self.padding_idx:
            print('The current state is uninitialized, keeping it invalid')
            print(f"The state that caused the error is {state}")
        else:
            try:
                mask[idx[0]] = False
            except:
                print(f'Assignment error, {idx} is not in the action1 space, mask index out of bounds, mask length: {len(mask)}')
        return mask

    def get_mask_invalid_actions_backward_2(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        parents_a_2: Optional[List] = None,
    ) -> List:
        state = self._get_state(state)
        done = self._get_done(done)
        mask = [True] * self.action_space_dim_2
        if parents_a_2 is None:
            _, _, parents_a_2 = self.get_parents(state, done)
        mask[parents_a_2[0][0]] = False
        return mask
    
    def statebatch2oracle(self, states: List[Tuple]):
        
        return [state[1] for state in states]
    
    def state2policy_1(self, state: Tuple[List, str] = None):
        if state is None:
            state = copy(self.state)
        seq, readable = state     
        if seq[0] == self.padding_idx or readable is None:
            state_policy = np.zeros(4096, dtype=np.float32) + 0.1 * np.random.normal(0, 1, (4096,))
        else:
            mol = Chem.MolFromSmiles(readable)
            if mode == 'editing':
                mol = Chem.AddHs(mol)
            
            features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 3, 4096)
            state_policy = np.array(features_vec).astype(np.float32)
        
        return state_policy
    
    def state2policy_2(self, state: Tuple[List, str] = None, action_1: Tuple = None):
        if state is None:
            state = copy(self.state)
        
        seq, readable = state
        
        if seq[0] == self.padding_idx or readable is None:
            features_vec = np.zeros(4096, dtype=np.float32) + 0.1 * np.random.normal(0, 1, (4096,))
            reaction_fp = np.zeros(256, dtype=np.float32) + 0.1 * np.random.normal(0, 1, (256,))
            extra_bit = np.random.normal(0, 1, (1, ))
            state_policy = np.hstack((features_vec, reaction_fp, extra_bit)).astype(np.float32)
            
            return state_policy
        
        mol = Chem.MolFromSmiles(readable)
        if mode == 'editing':
            mol = Chem.AddHs(mol)
        
        features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 3, 4096)
        features_vec = np.array(features_vec, dtype=np.float32)
        
        if action_1[0] in self.uni_list:
            reaction = self.inverse_lookup_1[action_1[0]]
            reaction_fp = rxnfp_generator.convert(reaction)
            extra_bit = np.array([1.0], dtype=np.float32)
            state_policy = np.hstack((features_vec, reaction_fp, extra_bit)).astype(np.float32)
            
        elif action_1[0] == self.eos_1 or self.done:
            reaction_fp = np.ones(256, dtype=np.float32)
            extra_bit = np.array([1.0], dtype=np.float32)
            state_policy = np.hstack((features_vec, reaction_fp, extra_bit)).astype(np.float32)
            
        else:
            reaction = self.inverse_lookup_1[action_1[0]]
            reaction_fp = rxnfp_generator.convert(reaction)
            
            reactants, _ = reaction.split('>>')
            r1_smarts, r2_smarts = reactants.split('.')
            
            if mol.HasSubstructMatch(mol_dict[r1_smarts]):
                extra_bit = np.array([1.0], dtype=np.float32)
            elif mol.HasSubstructMatch(mol_dict[r2_smarts]):
                extra_bit = np.array([0.0], dtype=np.float32)
            
            state_policy = np.hstack((features_vec, reaction_fp, extra_bit)).astype(np.float32)
        
        return state_policy

    def statebatch2policy_1(self, states: List[List]):
        return [self.state2policy_1(s) for s in states]
    
    def statebatch2policy_2(self,states: List[List], actions: List[Tuple[int]]):
        return [self.state2policy_2(s, a) for s, a in zip(states, actions)] 

    def get_uni_product(self, smiles1, smarts, index=None):
        mol = Chem.MolFromSmiles(smiles1)
        if mode == 'editing':
            mol = Chem.AddHs(mol)
        rxn = AllChem.ReactionFromSmarts(smarts)
        ps = rxn.RunReactants((mol,))
        uniqps = {}
        for p in ps:
            try:
                Chem.SanitizeMol(p[0])
                inchi = Chem.MolToInchi(p[0])
                if mode == 'editing':
                    mol_fixed = Chem.MolFromInchi(inchi)
                    uniqps[inchi] = Chem.MolToSmiles(mol_fixed)
                else:
                    uniqps[inchi] = Chem.MolToSmiles(p[0])
            except:
                pass
        if len(uniqps) == 0:
            return None, -1
        uniqps_sort = sorted(uniqps.values()) 
        if index is None:
            index = random.randrange(len(uniqps_sort))
        smiles = uniqps_sort[index]

        return smiles, index
    
    def get_bi_product(self, smiles_1, smiles_2, smarts, index=None):
        mol_1 = Chem.MolFromSmiles(smiles_1)
        mol_2 = Chem.MolFromSmiles(smiles_2)
        if mode == 'editing':
            mol_1 = Chem.AddHs(mol_1)
            mol_2 = Chem.AddHs(mol_2)
        rxn = AllChem.ReactionFromSmarts(smarts)
        ps = rxn.RunReactants((mol_1,mol_2))+rxn.RunReactants((mol_2,mol_1))
        uniqps = {}
        for p in ps:
            try:
                Chem.SanitizeMol(p[0])
                inchi = Chem.MolToInchi(p[0])
                if mode == 'editing':
                    mol_fixed = Chem.MolFromInchi(inchi)
                    uniqps[inchi] = Chem.MolToSmiles(mol_fixed)
                else:
                    uniqps[inchi] = Chem.MolToSmiles(p[0])
            except:
                pass
        if len(uniqps) == 0:
            return None, -1
        uniqps_sort = sorted(uniqps.values())
        if index is None:
            index = random.randrange(len(uniqps_sort)) 
        smiles = uniqps_sort[index]

        return smiles, index
    
    def get_product(self, smiles: str, action_1: int, action_2: int):
        if smiles == begin_token:
            return inverse_lookup_2[action_2], self.padding_idx
        smarts = inverse_lookup_1[action_1]
        if action_1 == self.eos_1 or action_2 == self.eos_2:
            product = eos_token
            index = -1
        elif action_1 in uni_indices:
            product, index = self.get_uni_product(smiles, smarts)
        else:
            product, index = self.get_bi_product(
                smiles_1=smiles,
                smiles_2=inverse_lookup_2[action_2],
                smarts=smarts
            )
        return product, index
    
    def state2readable(self, state: Tuple[List, str]):
        seq, readable = state
        if readable != 'PAD':
            return readable
        num_bb = int((len(seq) + 2) / 3)
        if seq[0] == self.padding_idx:
            return None
        else:
            intermediate_smiles = inverse_lookup_2[seq[0]]
            
        for i in range(1, num_bb):
            if seq[i] == self.padding_idx:
                return intermediate_smiles
            new_bb = inverse_lookup_2[seq[i]]
            temp_idx = num_bb + 2 * (i - 1)
            if seq[temp_idx] in uni_indices and seq[temp_idx + 1] != -1:
                intermediate_smiles, _ = self.get_uni_product(
                    intermediate_smiles, inverse_lookup_1[seq[temp_idx]], seq[temp_idx + 1]
                )
            elif seq[temp_idx + 1] != -1:
                intermediate_smiles, _ = self.get_bi_product(
                    intermediate_smiles, new_bb, inverse_lookup_1[seq[temp_idx]], seq[temp_idx + 1]
                )
            else:
                pass
            
        return intermediate_smiles
 
    def statebatch2proxy(self, states: List[Tuple]):
        return [s[1] for s in states]
    
    def get_parents(self, state: Tuple[List, str]=None, done: bool=None, action:Tuple=None):
        if state is None:
            state = copy(self.state)
        if done is None:
            done = self.done
        parents = []
        actions_1 = []
        actions_2 = []
        seq, readable = state
        num_bb = int((len(seq) + 2) / 3)
        
        #if one block
        if seq[0] != self.padding_idx and seq[1] == self.padding_idx:
            parent_action_2 = tuple(seq[0:1])
            actions_2.append(parent_action_2)
            return [self.source], [(0,)], actions_2
        
        for i in range(1, num_bb):
            temp_idx = num_bb + 2 * (i - 1)
            if seq[i] != self.padding_idx and (i == num_bb - 1 or seq[i + 1] == self.padding_idx):
                parent_actions_1 = tuple(seq[temp_idx: temp_idx + 1])
                actions_1.append(parent_actions_1)

                parent_action_2 = tuple(seq[i: i + 1])
                actions_2.append(parent_action_2)
                
                parent_seq = seq.copy()
                parent_seq[i] = self.padding_idx
                parent_seq[temp_idx] = self.padding_idx
                parent_seq[temp_idx + 1] = self.padding_idx
                parents.append((parent_seq, self.state2readable((parent_seq, 'PAD'))))
                return parents, actions_1, actions_2
            
    #check whether the step is valid
    def _pre_step(
        self, action_1: Tuple[int], action_2: Tuple[int], backward: bool = False,
    ) -> Tuple[bool, List[int], Tuple[int]]:
        # If action not found in action space raise an error
        if action_1 not in self.action_space_1:
            raise ValueError(
                f"Tried to execute action {action_1} not present in action space."
            )
        if action_2 not in self.action_space_2:
            raise ValueError(
                f"Tried to execute action {action_2} not present in action space."
            )
        # If backward and state is source, step should not proceed.
        if backward is True:
            if self.done:
                return False
            elif self.get_mask_invalid_actions_backward_1()[action_1[0]] or self.get_mask_invalid_actions_backward_2()[action_2[0]]:
                return False
            else:
                return True
        # If forward and env is done, step should not proceed.
        else:
            seq = self.state[0]
            num_bb = int((len(seq) + 2) / 3)
            if self.done or seq[num_bb - 1] != self.padding_idx:
                return False
            elif seq[0] == self.padding_idx:
                if self.get_mask_invalid_actions_forward_2()[action_2[0]]:
                    return False
                else:
                    return True
            else:
                mask1 = False
                smi = self.state2readable(self.state)
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mode == 'editing':
                        mol = Chem.AddHs(mol)
                except:
                    mol = None
                if mol:
                    idx = action_1[0]
                    mask1 = not self.check_mask_1(mol, sub_reactants[idx]) if idx < self.action_space_dim_1 - 1 else False
                
                if action_2[0] < self.action_space_dim_2 - 2:
                    r1_smarts, r2_smarts = sub_reactants[action_1[0]]
                    if mol.HasSubstructMatch(mol_dict[r1_smarts]):
                        mask_key = self.name_list[action_1[0]]+"_reactant_2"
                    elif mol.HasSubstructMatch(mol_dict[r2_smarts]):
                        mask_key = self.name_list[action_1[0]]+"_reactant_1"
                    mask2 = not (action_2[0] in mask_dict[mask_key])
                else:
                    mask2 = self.get_mask_invalid_actions_forward_2(action_1=action_1)[action_2[0]]
                    
                if mask1 or mask2:
                    return False
                else:
                    return True
                
    def step(self, action_1: Tuple[int], action_2: Tuple[int], backward: bool = False):
        assert action_1 in self.action_space_1
        assert action_2 in self.action_space_2
        if backward is True:
            if action_1[0] == self.eos_1 and action_2[0] == self.eos_2:
                self.n_actions += 1
                self.done = False
                return self.state, action_1, action_2, True
            parents, _, _ = self.get_parents()
            
            done = True if parents[0] == self.source else False
            seq_next = parents[0]
            self.set_state(seq_next, done=done)
            self.n_actions += 1
            return self.state, action_1, action_2, True

        # if forward
        else:
            do_step = self._pre_step(action_1, action_2, backward=False)

            self.done = True 
            if not do_step:
                self.set_state(self.state,done=True)
                self.n_actions += 1
                return self.state,action_1,action_2, False  
            
            
            seq = self.state[0]
            num_bb = int((len(seq) + 2) / 3)
            intermediate_smiles = self.state[1]
            for i in range(num_bb):
                if seq[i] == self.padding_idx:
                    if intermediate_smiles is None and action_2[0] != self.eos_2:
                        assert action_2[0] !=self.uni_idx
                        seq_next = seq.copy()
                        seq_next[0] = action_2[0]
                        readable = inverse_lookup_2[action_2[0]]
                        self.set_state((seq_next, readable), done=False)
                        self.n_actions += 1
                        return self.state, action_1, action_2, True
                    elif intermediate_smiles is not None:
                        if action_1[0] == self.eos_1 or action_2[0] == self.eos_2:
                            self.set_state(self.state, done=True)
                            self.n_actions += 1
                            return self.state, action_1, action_2, True
                        else:
                            seq_next = seq.copy()
                            product, index = self.get_product(intermediate_smiles, action_1[0], action_2[0])
                            temp_idx = num_bb + 2 * (i - 1)
                            seq_next[temp_idx + 1] = index
                            if index == -1:
                                self.set_state(self.state, done=True)
                                valid = False 
                                return self.state, action_1, action_2, valid                
                            else:
                                seq_next[i] = action_2[0]
                                seq_next[temp_idx] = action_1[0]
                                valid = True
                                self.set_state((seq_next, product), done=False)
                                self.n_actions += 1
                                return self.state, action_1, action_2, valid
                    
                    else:
                        raise ValueError(f"step() should not be called and state is {self.state}")
                          
    def sample_actions_batch_1(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim_1"],
        mask: Optional[TensorType["n_states", "policy_output_dim_1"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 3, 
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        if is_backward:
            actions = [self.action_space_1[list(m).index(False)] for m in mask]
            return actions, None
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0], device=device)
        if sampling_method == "random":
            logits = torch.ones(policy_outputs.shape, dtype=self.float, device=device)
        elif sampling_method == "policy":
            logits = policy_outputs
            logits /= temperature_logits

        assert not torch.all(mask), dedent(
            """
        All actions in the mask are invalid.
        """
        )
        logits[mask] = -torch.inf
        
        # Make sure that a valid action is sampled, otherwise throw an error.
        for _ in range(max_sampling_attempts):
            action_indices = Categorical(logits=logits).sample()
            if not torch.any(mask[ns_range, action_indices]):
                break
        else:
            raise ValueError(
                dedent(
                    f"""
            No valid action could be sampled after {max_sampling_attempts} attempts.
            """
                )
            )
        # Build actions
        actions = [self.action_space_1[idx] for idx in action_indices]
        return actions, None
    
    def sample_actions_batch_2(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim_2"],
        mask: Optional[TensorType["n_states", "policy_output_dim_2"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10, 
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        device = policy_outputs.device
        if is_backward:
            dt = np.arange(mask.shape[1])
            actions = [(dt[~m][0],) for m in mask]
            return actions, None
        ns_range = torch.arange(policy_outputs.shape[0], device=device)
        if sampling_method == "random":
            logits = torch.ones(policy_outputs.shape, dtype=self.float, device=device)
        elif sampling_method == "policy":
            logits = policy_outputs
            logits /= temperature_logits
        assert not torch.any(torch.all(mask,axis=1)), dedent(
            """
            All actions in the mask are invalid.
            """
            )
        logits[mask] = -torch.inf
        
        # Make sure that a valid action is sampled, otherwise throw an error.
        for _ in range(max_sampling_attempts):
            if torch.isnan(logits).any():
                raise ValueError(f"{mask}")
            action_indices = Categorical(logits=logits).sample()
            if not torch.any(mask[ns_range, action_indices]):
                break
        else:
            raise ValueError(
                dedent(
                    f"""
            No valid action could be sampled after {max_sampling_attempts} attempts.
            """
                )
            )
        # Build actions
        actions = list(zip(action_indices.tolist()))
        return actions, None
    
    def get_logprobs_1(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim_1"],
        is_forward: bool,
        actions: TensorType["n_states", "actions_dim_1"],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask: TensorType["batch_size", "policy_output_dim_1"] = None,
        temperature_logits: Optional[float] = 1.0,
    ) -> TensorType["batch_size"]:
        device = policy_outputs.device
        if is_forward:
            if policy_output_1_fixed:
                policy_reaction = policy_outputs[:, :256]  # (batch_size, 256)
                policy_reaction = torch.matmul(policy_reaction, self.reaction_embeddings.T) # (batch_size, num_reactions)
                policy_eos = policy_outputs[:, 256:]  # (batch_size, 1)
                policy_outputs = torch.cat((policy_reaction, policy_eos), dim=1)

            ns_range = torch.arange(policy_outputs.shape[0]).to(device)
            logits = policy_outputs

            if mask is not None:
                logits[mask] = -torch.inf
            action_indices = (
                torch.tensor(
                    actions[:,0]
                )
                .to(int)
                .to(device)
            )
            logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        else:
            logprobs = torch.zeros(policy_outputs.shape[0], dtype=self.float, device=device)
        return logprobs
    
    def get_logprobs_2(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim_2"],
        is_forward: bool,
        actions: TensorType["n_states", "actions_dim_2"],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask: TensorType["batch_size", "policy_output_dim_2"] = None,
        temperature_logits: Optional[float] = 1.0,
    ) -> TensorType["batch_size"]:
        device = policy_outputs.device

        if is_forward:
            if policy_output_2_fixed:
                policy_bb= policy_outputs[:, :bb_nbits]  # (batch_size, bb_nbits)
                policy_bb = torch.matmul(policy_bb, self.building_block_embeddings.T) # (batch_size, num_bb)
                policy_eos_uni = policy_outputs[:, bb_nbits:]  # (batch_size, 2)
                policy_outputs = torch.cat((policy_bb, policy_eos_uni), dim=1)

            ns_range = torch.arange(policy_outputs.shape[0]).to(device)
            logits = policy_outputs
            if mask is not None:
                logits[mask] = -torch.inf
            action_indices = (
                torch.tensor(
                    actions[:,0]
                )
                .to(int)
                .to(device)
            )
            logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        else:

            logprobs = torch.zeros(policy_outputs.shape[0], dtype=self.float, device=device)
        
        return logprobs
    
    def get_policy_output_1(self, params: Optional[dict] = None):
        return np.ones(self.action_space_dim_1)
    
    def get_policy_output_2(self, params: Optional[dict] = None):
        return np.ones(self.action_space_dim_2)
    
    def policy2state(self, state_policy: List) -> List:
        return state_policy
    
    def readable2state(self, readable):
        return readable
    
    def traj2readable(self, traj=None):
        return str(traj).replace("(", "[").replace(")", "]").replace(",", "")
    
    def reward(self, state=None, done=None):
        state = self._get_state(state)
        done = self._get_done(done)
        if done is False:
            return tfloat(0.0, float_type=self.float, device=self.device)
        reward=self.proxy([self.state2proxy(state)])#proxy_vals
        return self.proxy2reward(reward[0])

    def reward_batch(self, states, done=None):
        if done is None:
            done = np.ones(len(states), dtype=bool)
        states_proxy = self.statebatch2proxy(states)
        if isinstance(states_proxy, torch.Tensor):
            states_proxy = states_proxy[list(done), :]
        elif isinstance(states_proxy, list):
            states_proxy = [states_proxy[i] for i in range(len(done)) if done[i]]
        rewards = np.zeros(len(done))
        if len(states_proxy) > 0:
            proxy_vals = self.proxy(states_proxy)
            rewards[list(done)] = self.proxy2reward(proxy_vals).tolist()
        return rewards
                                 
    def proxy2reward(self, proxy_vals):
        if self.denorm_proxy:
            proxy_vals = (
                proxy_vals * (self.energies_stats[1] - self.energies_stats[0])
                + self.energies_stats[0]
            )
        if self.reward_func == "power":
            return torch.clamp(
                (self.proxy_factor * proxy_vals / self.reward_norm) ** self.reward_beta,
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "boltzmann":
            return torch.clamp(
                torch.exp(self.proxy_factor * self.reward_beta * proxy_vals),
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "identity":
            return torch.clamp(
                self.proxy_factor * proxy_vals,
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "shift":
            return torch.clamp(
                self.proxy_factor * proxy_vals + self.reward_beta,
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "sigmoid":
            return torch.clamp(
                1 / (1 + torch.exp(0.75 * (proxy_vals + 6.5))),
                min=self.min_reward,
                max=None,
            )
        else:
            raise NotImplementedError
    
    def reward2proxy(self, reward):
        if self.reward_func == "power":
            return self.reward_norm * (reward / (self.proxy_factor)) ** (1/self.reward_beta)
        elif self.reward_func == "boltzmann":
            return self.proxy_factor * torch.log(reward) / self.reward_beta
        elif self.reward_func == "identity":
            return self.proxy_factor * reward
        elif self.reward_func == "shift":
            return self.proxy_factor * (reward - self.reward_beta)
        elif self.reward_func == "sigmoid":
            return torch.log(1 / reward - 1) / 0.75 -6.5
        else:
            raise NotImplementedError
        
    def reset(self, env_id: Union[int, str] = None):
        self.state = copy(self.source)
        self.n_actions = 0
        self.done = False
        if env_id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = env_id
        return self

    def set_id(self, env_id: Union[int, str]):
        self.id = env_id
        return self

    def set_state(self, state: Tuple[List, str], done: Optional[bool] = False):
        self.state = copy(state)
        self.done = done
        return self
    
    def set_energies_stats(self, energies_stats):
        self.energies_stats = energies_stats

    def set_reward_norm(self, reward_norm):
        self.reward_norm = reward_norm
    
    def get_trajectories(
        self, traj_list, traj_actions_list_1, traj_actions_list_2, current_traj, current_actions_1,current_actions_2
    ):
        parents, parents_actions_1, parents_actions_2 = self.get_parents(current_traj[-1], False)
        if parents == []:
            traj_list.append(current_traj)
            traj_actions_list_1.append(current_actions_1)
            traj_actions_list_2.append(current_actions_2)
            return traj_list, traj_actions_list_1,traj_actions_list_2
        for idx, (p, a_1, a_2) in enumerate(zip(parents, parents_actions_1, parents_actions_2)):
            traj_list, traj_actions_list_1, traj_actions_list_2 = self.get_trajectories(
                traj_list, traj_actions_list_1,traj_actions_list_2, current_traj + [p], current_actions_1 + [a_1],current_actions_2 + [a_2]
            )
        return traj_list, traj_actions_list_1, traj_actions_list_2

    def setup_proxy(self):
        if self.proxy:
            self.proxy.setup(self)
            
    def copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k in ["action_space_2", "fixed_policy_output_2", "random_policy_output_2"]:
                setattr(result, k, shallowcopy(v))
            else:
                setattr(result, k, deepcopy(v))
        return result
