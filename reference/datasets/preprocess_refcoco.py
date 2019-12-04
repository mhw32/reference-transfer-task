# run me with python2

import os
import sys

refer_dir = os.path.join(os.path.dirname(__file__), 'refer')
sys.path.append(refer_dir)
from refer import REFER


def process_dataset(refer, split):
    ref_ids = refer.getRefIds(split=split)
    img_ids = refer.getImgIds(ref_ids=ref_ids)
    refs = [refer.imgToRefs[img_id] for img_id in img_ids]

    def get_image_paths(img_ids):
        paths = []
        for img_id in img_ids:
            image = refer.Imgs[img_id]
            path = image['file_name']
            paths.append(path)
        return paths

    def get_masks(img_refs):
        img_masks = []
        for refs in img_refs:
            masks = []
            for ref in refs:
                mask = refer.getMask(ref)
                masks.append(mask)
            img_masks.append(masks)
        return img_masks

    def get_texts(img_refs):
        ref_sents = []
        for refs in img_refs:
            sents = []
            for ref in refs:
                # just take the first sentence
                sent = ref['sentences'][0]jh,
                sents.append(sent['tokens'])
            ref_sents.append(sents)
        return ref_sents

    masks = get_masks(refs)
    texts = get_texts(refs)
    paths = get_image_paths(img_ids)

    return {
        'ref_ids': ref_ids,
        'img_ids': img_ids,
        'img_paths': paths,
        'masks': masks,
        'texts': texts,
    }


if __name__ == "__main__":
    OUT_DIR = '/mnt/fs5/wumike/datasets/refer_datasets/processed'
    data_names = ['refclef', 'refcoco', 'refcoco+']
    data_splits = ['berkeley', 'google', 'google']

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

        with open(os.path.join(processed_dir, 'train.pickle'), 'wb') as fp:
            pickle.dump(train_dset, fp)

        with open(os.path.join(processed_dir, 'val.pickle'), 'wb') as fp:
            pickle.dump(val_dset, fp)

        with open(os.path.join(processed_dir, 'test.pickle'), 'wb') as fp:
            pickle.dump(test_dset, fp)

