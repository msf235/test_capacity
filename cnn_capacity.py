import timm

effnet = timm.models.factory.create_model('efficientnet_b2', pretrained=True)

dataset_train = timm.data.dataset_factory.create_dataset(
    'ImageFolder',
    root='/n/pehlevan_lab/Lab/matthew/imagenet/ILSVRC/Data/CLS-LOC',
    split=.8, is_training=False, batch_size=1)


# pretrain_batchsize = 256
# dataset_pretrain = timm.data.dataset_factory.create_dataset(
    # 'ImageFolder',
    # root='/n/pehlevan_lab/Lab/matthew/imagenet/ILSVRC/Data/CLS-LOC',
    # split=args.train_split, is_training=True,
    # batch_size=pretrain_batchsize)
# dataset_eval = timm.data.dataset_factory.create_dataset(
    # 'ImageFolder',
    # root='/n/pehlevan_lab/Lab/matthew/imagenet/ILSVRC/Data/CLS-LOC',
    # split=args.val_split, is_training=False, batch_size=args.batch_size)

