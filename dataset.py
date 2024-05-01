import numpy as np
import torch
import pandas as pd
from fmcib.preprocessing.seed_based_crop import SeedBasedPatchCropd
from monai.data import CSVDataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from utils import get_transforms
import config as CFG
from sklearn.model_selection import train_test_split


class CLIPDataset(CSVDataset):
    def __init__(self, coord_path, semantic_path, mode = 'train',seed = 0,):
        if mode == 'train':
            self.transform = get_transforms(jitter = CFG.jitter)
        else:
            self.transform = get_transforms()
        
        coord_csv = pd.read_csv(coord_path)
        semantic_features = pd.read_csv(semantic_path)

        assert coord_csv.shape[0] == semantic_features.shape[0], 'number of samples in coord and semantic features should be same'

        if CFG.debug:
            #select 100 samples for debugging same in both files
            coord_csv = coord_csv.sample(n=100, random_state = seed)
            semantic_features = semantic_features.loc[coord_csv.index]
            print('debug mode:', coord_csv.shape[0])
            #reset index
            coord_csv = coord_csv.reset_index(drop = True)
            semantic_features = semantic_features.reset_index(drop = True)

        #randomly select 80% of the data for training
        np.random.seed(seed)

        all_pid = semantic_features.drop_duplicates('pid')#['pid'].unique()

        #split data stratified on diagnosis
        train_pid, val_pid = train_test_split(all_pid.pid, test_size=0.2, stratify = all_pid['censorship'], random_state = seed)

        #train_pid = np.random.choice(all_pid, size = int(0.7*all_pid.shape[0]), replace = False)
        #val_pid = np.setdiff1d(all_pid, train_pid)
        train_ind = semantic_features[semantic_features['pid'].isin(train_pid)].index
        val_ind = semantic_features[semantic_features['pid'].isin(val_pid)].index

        #train_ind = coord_csv.sample(frac=0.8).index
        #val_ind = coord_csv.index.difference(train_ind)
        print('train:', train_ind.shape[0], 'val:', val_ind.shape[0])


        #select features based on config
        numerical_cols, categorical_cols = self.select_feature()
        #one hot encoding categorical features
        semantic_features_cat = self.onehot_encoding(semantic_features, categorical_cols)
        semantic_features_processed = pd.concat([semantic_features[numerical_cols], semantic_features_cat], axis =1)        

        #scaling features
        semantic_features_processed = self.scale(semantic_features_processed, index = train_ind)

        print('onehot-encoded', semantic_features_processed.columns)

        if mode == 'train':
            coord_csv_sub = coord_csv.loc[train_ind]
            semantic_features_sub = semantic_features_processed.loc[train_ind]
            self.pid = semantic_features.loc[train_ind][['pid','nodule_id']].reset_index(drop = True)
        else:
            coord_csv_sub = coord_csv.loc[val_ind]
            semantic_features_sub = semantic_features_processed.loc[val_ind]
            self.pid = semantic_features.loc[val_ind][['pid','nodule_id']].reset_index(drop = True)


        self.semantic_features = semantic_features_sub.reset_index(drop = True)
        coord_csv_sub = coord_csv_sub.reset_index(drop = True)
        assert coord_csv_sub.shape[0] == self.semantic_features.shape[0]

        super().__init__(coord_csv_sub, transform=self.transform)
        
    def __getitem__(self, idx):
        img_processed = super().__getitem__(idx)
        semantic_features = torch.tensor(self.semantic_features.iloc[idx].values).float()
        return img_processed, semantic_features
    
    @staticmethod
    def select_feature():
        general_feature = ['longest_axial_diameter_(mm)','short_diameter_(mm)', 'mean_diameter',
                  'nodule_margin_conspicuity', 'nodule_margins',  'additional_nodule_margins','nodule_shape','nodule_consistency']
        internal_feature =  ['nodule_reticulation', 'cyst-like_spaces', 'intra-nodular_bronchiectasis', 'necrosis',
            'cavitation', 'eccentric_calcification', 'airway_cut-off']
        external_feature = ['pleural_attachment', 'pleural_retraction', 'vascular_convergence',
            'septal_stretching', 'paracicatricial_emphysema']

        #one hot encoding
        numerical_cols = ['longest_axial_diameter_(mm)','short_diameter_(mm)', 'mean_diameter']
        categorical_cols = ['nodule_margin_conspicuity', 'nodule_margins',  'additional_nodule_margins',
                            'nodule_shape','nodule_consistency', 'nodule_reticulation',
                            'cyst-like_spaces', 'intra-nodular_bronchiectasis', 'necrosis',
                            'cavitation', 'eccentric_calcification', 'airway_cut-off',
                            'pleural_attachment', 'pleural_retraction', 'vascular_convergence',
                            'septal_stretching', 'paracicatricial_emphysema']
        
        selected_features =[]
        if CFG.general:
            selected_features += general_feature
        if CFG.internal:
            selected_features += internal_feature
        if CFG.external:
            selected_features += external_feature

        numerical_cols = [col for col in selected_features if col in numerical_cols]
        categorical_cols = [col for col in selected_features if col in categorical_cols]

        print('numerical features:', numerical_cols)
        print('categorical features:', categorical_cols)
        return numerical_cols, categorical_cols

    @staticmethod
    def onehot_encoding(semantic_features, categorical_cols):
        onehot = OneHotEncoder(handle_unknown='ignore')
        onehot.fit(semantic_features[categorical_cols])
        semantic_features_cat = onehot.transform(semantic_features[categorical_cols]).toarray()
        semantic_features_cat = pd.DataFrame(semantic_features_cat)
        semantic_features_cat.columns = onehot.get_feature_names_out()
        return semantic_features_cat
    
    @staticmethod
    def scale(semantic_features, index):
        scaler = StandardScaler()
        #fit scaler on training data
        scaler.fit(semantic_features.loc[index, :])
        #transform all data
        semantic_features_scaled = pd.DataFrame(scaler.transform(semantic_features))
        semantic_features_scaled.columns = semantic_features.columns
        return semantic_features_scaled
    