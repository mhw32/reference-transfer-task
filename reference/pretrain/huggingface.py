import os
import numpy as np
import torch

from transformers import *
from reference.agents import FeatureAgent

MODELS = {
    'bert':         (BertModel,       BertTokenizer,       'bert-base-uncased'),
    'gpt_openai':   (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
    'gpt_2':        (GPT2Model,       GPT2Tokenizer,       'gpt2'),
    'ctrl':         (CTRLModel,       CTRLTokenizer,       'ctrl'),
    'transforxl':   (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
    'xlnet':        (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
    'xlm':          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
    'distilbert':   (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
    'roberta':      (RobertaModel,    RobertaTokenizer,    'roberta-base'),
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert', 
                        choices=[
                            'bert', 'gpt_openai', 'gpt_2', 'ctrl', 'transforxl', 
                            'xlnet', 'xlm', 'distilbert', 'roberta',
                        ])
    args = parser.parse_args()

    model_class, tokenizer_class, pretrained_weights = MODELS[args.model]
    # load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    def extract(raw_text_list):
        features = []
        for raw_text in raw_text_list:
            raw_text = ' '.join(raw_text)
            
            # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
            input_ids = torch.tensor([tokenizer.encode(raw_text, add_special_tokens=True)])
            with torch.no_grad():
                last_hidden_states = model(input_ids)[0]
                # take a mean over the words
                sent_features = last_hidden_states.mean(1)
            features.append(sent_features.detach().cpu())
        features = torch.cat(features, dim=0)
        return features

    agent = FeatureAgent()
    train_text_embs = agent.extract_features(extract, modality='text', split='train')
    val_text_embs = agent.extract_features(extract, modality='text', split='val')
    test_text_embs = agent.extract_features(extract, modality='text', split='test')

    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.model}/train.npy', train_text_embs.numpy())
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.model}/val.npy', val_text_embs.numpy())
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.model}/test.npy', test_text_embs.numpy())
