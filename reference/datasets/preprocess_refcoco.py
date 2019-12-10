# run me with python2

import os
import sys
import pickle

refer_dir = os.path.join(os.path.dirname(__file__), 'refer')
sys.path.append(refer_dir)
from refer import REFER


def process_dataset(refer, split):
    ref_ids = refer.getRefIds(split=split)
    img_ids = refer.getImgIds(ref_ids=ref_ids)
    refs = [refer.imgToRefs[img_id] for img_id in img_ids]

    paths, masks, texts = [], [], []

    for img_id in img_ids:
        refs = refer.imgToRefs[img_id]
        image = refer.Imgs[img_id]
        image_path = image['file_name']

        for ref in refs:
            mask = refer.getMask(ref)

            sents = ref['sentences']
            for sent in sents:
                tokens = sent['tokens']

                paths.append(image_path)
                masks.append(mask)
                texts.append(tokens)

    return {
        'img_paths': paths,
        'masks': masks,
        'texts': texts,
    }


if __name__ == "__main__":
    OUT_DIR = '/mnt/fs5/wumike/datasets/refer_datasets/processed'
    data_names = ['refclef', 'refcoco', 'refcoco+']
    data_splits = ['berkeley', 'google', 'unc']

    for data_name, data_split in zip(data_names, data_splits):
        print "processing %s." % data_name
        refer = REFER(
            '/mnt/fs5/wumike/datasets/refer_datasets', 
            data_name, 
            data_split,
        )

        processed_dir = os.path.join(OUT_DIR, data_name)
        if not os.path.isdir(processed_dir):
            os.makedirs(processed_dir)

        train_dset = process_dataset(refer, 'train')
        val_dset   = process_dataset(refer, 'val')
        test_dset  = process_dataset(refer, 'test')

        print('saving train pickle')
        with open(os.path.join(processed_dir, 'train.pickle'), 'wb') as fp:
            pickle.dump(train_dset, fp)

        print('saving val pickle')
        with open(os.path.join(processed_dir, 'val.pickle'), 'wb') as fp:
            pickle.dump(val_dset, fp)

        print('saving test pickle')
        with open(os.path.join(processed_dir, 'test.pickle'), 'wb') as fp:
            pickle.dump(test_dset, fp)

