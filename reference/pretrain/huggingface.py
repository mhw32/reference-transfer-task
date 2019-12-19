import os
import numpy as np
import torch

from transformers import *
from reference.agents import FeatureAgent, MaskedFeatureAgent
from reference.setup import process_config

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
    parser.add_argument('dataset', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--nlp-model', type=str, default='bert', 
                        choices=[
                            'bert', 'gpt_openai', 'gpt_2', 'ctrl', 'transforxl', 
                            'xlnet', 'xlm', 'distilbert', 'roberta',
                        ])
    parser.add_argument('--context-condition', type=str, default='all', 
                        choices=['all', 'far', 'close'])
    parser.add_argument('--split-mode', type=str, default='easy',
                        choices=['easy', 'hard'])
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    def run_model(nlp_model):
        print(f'Starting extraction for {nlp_model}')
        model_class, tokenizer_class, pretrained_weights = MODELS[nlp_model]
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

        if args.dataset in ['refclef', 'refcoco', 'refcoco+']:
            FeatureAgentClass = MaskedFeatureAgent
        else:
            FeatureAgentClass = FeatureAgent

        agent = FeatureAgentClass(
            args.dataset,
            args.data_dir,
            context_condition = args.context_condition,
            split_mode = args.split_mode,
            image_size = 64,
            override_vocab = None, 
            batch_size = args.batch_size,
            gpu_device = args.gpu_device, 
            cuda = args.cuda,
            seed = args.seed,
            image_transforms = None,
        )

        train_text_embs = agent.extract_features(extract, modality='text', split='train')
        val_text_embs = agent.extract_features(extract, modality='text', split='val')
        test_text_embs = agent.extract_features(extract, modality='text', split='test')

        out_dirname = f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/huggingface/{nlp_model}'
        if not os.path.isdir(out_dirname):
            os.makedirs(out_dirname)
        
        np.save(f'{out_dirname}/train.npy', train_text_embs.numpy())
        np.save(f'{out_dirname}/val.npy', val_text_embs.numpy())
        np.save(f'{out_dirname}/test.npy', test_text_embs.numpy())

    if args.all: 
        nlp_models = [
            'bert', 'gpt_openai', 'gpt_2', 'ctrl', 'transforxl', 
            'xlnet', 'xlm', 'distilbert', 'roberta',
        ]
    else:
        nlp_models = [args.nlp_model]

    for nlp_model in nlp_models:
        run_model(nlp_model)
