


class BaseModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.lr = args.lr
        self.image_size = SIZE_BY_MODEL[self.model_name]

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
        )

    # def val_dataloader(self):
    #     pass
    #
    # def test_dataloader(self):
    #     pass


class EffdetModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = get_efficientdet_config(f'tf_efficientdet_{self.model_name}')
        self.model = EfficientDet(cfg)
        self.bench = DetBenchTrain(self.model)
        self.dataset =  EffdetDataset()
        if not self.args.no_aug:
            self.dataset.apply_augs([
                A.RandomResizedCrop(width=self.image_size, height=self.image_size, scale=[0.7, 1.0]),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                ], p=0.3),
                A.HueSaturationValue(p=0.3),
            ])

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.bench(x, y)
        return loss['loss'].detach()
