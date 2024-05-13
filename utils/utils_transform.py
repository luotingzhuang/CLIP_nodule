import pandas as pd
import numpy as np

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf

from nlpaug.util import Action
import random
import os
from tqdm import tqdm

class Transform2Text():
    def __init__(self, semantic, augmentation = ['reverse', 'abstraction'], device = 'cuda'):
        self.semantic = semantic
        self.augmentation = augmentation
        self.aug = self.aug_fun()
        self.aug_abst = nas.AbstSummAug(model_path='facebook/bart-large-cnn', 
                                        min_length = 100, 
                                        max_length = 200,
                                        device = device)

        
    def transform(self, i, features):
        row_i = self.semantic.iloc[i]   
        
        all_aug_text = []
        if 'general' in features:
            general_text = self.get_general(row_i)
            random.shuffle(general_text)
            general_text = " ".join(general_text)
            
            if 'reverse' in self.augmentation:
                general_text = self.aug.augment(general_text)
            all_aug_text.append(general_text[0])
            
        if 'external' in features:
            external_text = self.get_external(row_i)
            random.shuffle(external_text)
            external_text = " ".join(external_text)
            if 'reverse' in self.augmentation:
                external_text = self.aug.augment(external_text)
            all_aug_text.append(external_text[0])
            
        all_aug_text = " ".join(all_aug_text)
        if 'abstraction' in self.augmentation:
            all_aug_text_abst = self.aug_abst.augment(all_aug_text)[0]
            return all_aug_text, all_aug_text_abst

        return all_aug_text
        
    def aug_fun(self):
        reserved_tokens = [
            ['mm', 'millimeter'],
            ['lung', 'pumonary'],
            ['attached','adhered'],
            ['attachment','adhesion'],
            ['presence','existence','appearance'],
            ['border','margin','boundary'],
            ['displays','shows','exhibits','presents','demonstrates','reveals','showcases'],
            ['greatest','longest','maximum'],
            ['shortest','smallest','minimum'],
            ['observed','seen','evident','noted'],
            #['pleura','pleural surface'],
            ['absence of','no evidence of', 'no signs of'],
            ['Absence of','No evidence of', 'No signs of'],
            ['not attached','unattached'],
            ['retraction','indentation'],
            ['around', 'in the vicinity of', 'surrounding', 'adjacent to', 'near']

        ]
        reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens,  aug_max = 5)
        
        aug = naf.Sometimes([
            reserved_aug,
            #trans_aug
        ])#, aug_p = 0.8)
        
        return aug
            
    @staticmethod    
    def get_general(row_i):
        all_texts = []
        # size
        a = row_i['longest_axial_diameter_(mm)']
        b = row_i['short_diameter_(mm)']
        c = row_i['mean_diameter']
        size_text = get_size_text(a, b, c)
        all_texts.append(size_text)

        #conspicuity
        mc = row_i['nodule_margin_conspicuity']
        conspicuity_text = get_margin_conspicuity_text(mc)
        all_texts.append(conspicuity_text)

        #margin
        all_m = []
        if isinstance(row_i['nodule_margins'], str):
            all_m.append(row_i['nodule_margins'])

        if isinstance(row_i['additional_nodule_margins'], str):
            all_m.append(row_i['additional_nodule_margins'])

        all_m = np.unique(all_m)    
        conspicuity_text = get_margin_text(list(all_m))
        all_texts.append(conspicuity_text)    

        #shape
        ns = row_i['nodule_shape']
        shape_text = get_shape_text(ns)
        all_texts.append(shape_text)

        #nodule_consistency
        nc = row_i['nodule_consistency']
        nodule_consistency_text = get_nodule_consistency_text(nc)
        all_texts.append(nodule_consistency_text)

        return all_texts
    
    @staticmethod
    def get_external(row_i):
        all_texts = []
        
        #pleural attachement
        pa = row_i['pleural_attachment']
        pleural_attachment_text = get_pleural_attachment_text(pa)
        all_texts.append(pleural_attachment_text)
        
        
        #pleural retraction
        pr = row_i['pleural_retraction']
        pleural_retraction_text = get_pleural_retraction_text(pr)
        all_texts.append(pleural_retraction_text)
        
        #pleural retraction
        vc = row_i['vascular_convergence']
        vascular_convergence_text = get_vascular_convergence_text(vc)
        all_texts.append(vascular_convergence_text)
        
        
        #septal stretching
        ss = row_i['septal_stretching']
        septal_stretching_text = get_septal_stretching_text(ss)
        all_texts.append(septal_stretching_text)
        
        #septal stretching
        pe = row_i['paracicatricial_emphysema']
        paracicatricial_emphysema_text = get_paracicatricial_emphysema_text(pe)
        all_texts.append(paracicatricial_emphysema_text)
        return all_texts



def get_pleural_attachment_text(pa):
    if isinstance(pa, float):
        return ''    
    
    if pa == 'Present':
        prompts =["The nodule is attached to the pleural surface.",
                  "Pleural attachment is observed.",
                  "There is an observable attachment of the nodule to the pleura."]
    elif pa == 'Absent':
        prompts =["The nodule is not attached to the pleural surface.",
                  "Pleural attachment is not observed.",
                  "No attachment to the pleura.",
                  ""] #empty for not mentioning
    
    p = random.choice(prompts)
    return p


def get_pleural_retraction_text(pr):
    if isinstance(pr, float):
        return ''    
    
    if pr in ['Present','Mild dimpling']:
        prompts =["There is an inward pulling of the pleural surface adjacent to the nodule or lesion.",
                  "The pleural surface adjacent to the nodule displays a marked inward pulling.",
                  "Pleural retraction is seen adjacent to the nodule.",
                 "The nodule has caused pleural retraction.",
                 "There is pleural retraction next to the nodule."]
        
    elif pr == 'Absent':
        prompts =["No inward pulling is seen on the pleural surface near the nodule.",
                  "Pleural retraction is not observed.",
                  "The pleural surface near the nodule appears unaffected without any signs of retraction.",
                  "There are no signs of pleural indentation or retraction near the nodule.",
                  ""]
    
    p = random.choice(prompts)
    return p

def get_vascular_convergence_text(vc):
    if isinstance(vc, float):
        return ''    
    
    if vc == 'Present':
        prompts =["There is vascular convergence surrounding the nodule.",
                  "Blood vessels converge toward the nodule.",
                  "Blood vessels appear to converge towards the location of the nodule.",
                  "There is a vessel converged toward the nodule.",
                  "Vascular convergence is seen around the nodule."]
        
    elif vc == 'Absent':
        prompts =["No vascular convergence is observed surrounding the nodule.",
                  "There is no apparent convergence of blood vessels near the nodule.",
                  "The area around the nodule shows no signs of vascular convergence.",
                  "Absence of vascular convergence is noted in relation to the nodule.",
                  ""]
    
    p = random.choice(prompts)
    return p


def get_septal_stretching_text(ss):
    if isinstance(ss, float):
        return ''    
    
    if ss == 'Present':
        prompts =["The nodule has caused septal stretching.",
                  "Septal stretching is seen around the nodule.",
                  "There is septal stretching around the nodule.",
                  "There are multiple septa around the nodule."]
        
    elif ss == 'Absent':
        prompts =["No septal stretching is observed surrounding the nodule.",
                  "There is no signs of septal stretching near the nodule.",
                  "The area around the nodule shows no signs of stretching of the septa.",
                  "Absence of septal stretching is noted around the nodule.",
                  ""]
    
    p = random.choice(prompts)
    return p

def get_paracicatricial_emphysema_text(pe):
    if isinstance(pe, float):
        return ''    
    
    if pe == 'Present':
        prompts =["There is emphysema surrounding the nodule. ",
                  "Paracicatricial emphysema is seen around the nodule.",
                  "There is paracicatricial emphysema around the nodule.",
                  "The nodule has caused paracicatricial emphysema.",
                  "Emphysema envelops around the nodule.",
                  "The nodule has led to the development of paracicatricial emphysema."]
        
    elif pe == 'Absent':
        prompts =["No emphysema is observed surrounding the nodule.",
                  "There are no signs of paracicatricial emphysema around the nodule.",
                  "Absence of paracicatricial emphysema is noted around the nodule.",
                  "The presence of the nodule is not associated with paracicatricial emphysema.",
                  ""]
    
    p = random.choice(prompts)
    return p