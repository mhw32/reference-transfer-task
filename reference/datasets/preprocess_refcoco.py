import os
import sys

refer_dir = os.path.join(os.path.dirname(__file__), 'refer')
sys.path.append(refer_dir)
from refer import REFER


def process_dataset(refer, split):
    ref_ids = refer.getRefIds(split=split)
    img_ids = refer.getImgIds(ref_ids=ref_ids)
    refs = [refer.imgToRefs[img_id] for img_id in img_ids]

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
                sent = ref['sentences'][0]
                sent = ' '.join(sent['tokens'])
                sents.append(sent)
            ref_sents.append(sents)
        return ref_sents

    masks = get_masks(refs)
    texts = get_texts(refs)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    data_names = ['refclef', 'refcoco', 'refcoco+']
    data_splits = ['berkeley', 'google', 'google']

    for data_name, data_split in zip(data_names, data_splits):
        refer = REFER(
            '/mnt/fs5/wumike/datasets/refer_datasets', 
            data_name, 
            data_split,
        )

        process_dataset(refer, 'train')
        process_dataset(refer, 'val')
        process_dataset(refer, 'test')

