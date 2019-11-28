import os
import numpy as np
import torch

from reference.agents import FeatureAgent


if __name__ == "__main__":
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    roberta.eval()

    def extract(raw_text_list):
        features = []
        for raw_text in raw_text_list:
            breakpoint()
            raw_text = ' '.join(raw_text)
            tokens = roberta.encode(raw_text)
            # last layer features [1, 5, 1024]
            last_layer_features = roberta.extract_features(tokens)
            sent_features = last_layer_features.mean(1)  # take a mean over the words
            features.append(sent_features)
        features = torch.cat(features, dim=0)
        return features

    agent = FeatureAgent()
    train_text_embs = agent.extract_features(extract, modality='text', split='train')
    val_text_embs = agent.extract_features(extract, modality='text', split='val')
    test_text_embs = agent.extract_features(extract, modality='text', split='test')

    np.save('/mnt/fs5/wumike/reference/pretrain/roberta/train.npy', train_text_embs.numpy())
    np.save('/mnt/fs5/wumike/reference/pretrain/roberta/val.npy', val_text_embs.numpy())
    np.save('/mnt/fs5/wumike/reference/pretrain/roberta/test.npy', test_text_embs.numpy())
