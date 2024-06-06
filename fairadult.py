import warnings
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch import loggers, callbacks
from networks.utils import MLP


class AdultDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        idx = idx % len(self.X)
        return self.X[idx], self.y[idx]


class Adult(LightningDataModule):
    features = 86
    protected_id = 0
    protected_classes = 2
    target_classes = 2

    def __init__(self, path="datasets/adult/", **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["path", "copies"])
        self.path = path

    def process_df(self, df):
        df["education"] = df["education-num"]
        df = df.drop(columns=["fnlwgt", "education-num"])
        df = df.replace(" ?", np.nan).dropna()

        categorical_columns = ["workclass", "marital-status", "occupation"]
        categorical_columns += ["relationship", "race", "country"]
        for key in categorical_columns:
            df = pd.concat([df, pd.get_dummies(df[key])], axis="columns")
            df = df.drop(columns=[key])

        for country in (
            [" Cambodia", " Canada", " China", " Columbia", " Cuba"]
            + [" Dominican-Republic", " Ecuador", " El-Salvador"]
            + [" England", " France", " Germany", " Greece", " Guatemala"]
            + [" Haiti", " Holand-Netherlands", " Honduras", " Hong"]
            + [" Hungary", " India", " Iran", " Ireland", " Italy"]
            + [" Jamaica", " Japan", " Laos", " Mexico", " Nicaragua"]
            + [" Outlying-US(Guam-USVI-etc)", " Peru", " Philippines"]
            + [" Poland", " Portugal", " Puerto-Rico", " Scotland"]
            + [" South", " Taiwan", " Thailand", " Trinadad&Tobago"]
            + [" United-States", " Vietnam", " Yugoslavia"]
        ):
            if country not in df.columns:
                print(f"Adding missing country: {country}")
                df[country] = 0

        numeric_columns = ["age", "capital-gain", "capital-loss", "workhours"]
        for key in numeric_columns:
            df[key] = df[key].astype(float)

        df["label"] = df["label"].astype("category").cat.codes
        df.insert(0, "sex", df.pop("sex").astype("category").cat.codes)

        y = df["label"].values.astype(int)
        X = df.drop(columns=["label"]).values.astype(np.float32)
        return AdultDataset(X, y)

    def setup(self, stage=None):
        column_names = (
            ["age", "workclass", "fnlwgt", "education"]
            + ["education-num", "marital-status", "occupation"]
            + ["relationship", "race", "sex", "capital-gain"]
            + ["capital-loss", "workhours", "country", "label"]
        )
        self.train_dataset = self.process_df(
            pd.read_csv(
                "datasets/adult/adult.data",
                header=None,
                names=column_names,
                index_col=False,
            )
        )
        self.test_dataset = self.process_df(
            df=pd.read_csv(
                "datasets/adult/adult.test",
                header=None,
                names=column_names,
                index_col=False,
            ).drop(index=0)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.hparams)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.hparams)


def fairness_penalty(x: torch.Tensor, k: int):
    # x: (..., batch_size, dim) -> latent representation for one group
    # k: number of evaluation points
    dim = x.shape[-1]

    # sample the evaluation points
    weighting_kernel = torch.distributions.Normal(0, 1)
    cov_root = torch.eye(dim).to(x.device)  # we can optimize this as a parameter
    t = cov_root @ weighting_kernel.sample((dim, k)).to(x.device)

    # evaluate characteristic functions at t
    phi_empirical = torch.exp(x @ t * 1j).mean(-2)  # (..., eval_points)
    phi_target = torch.exp(-0.5 * (t**2).sum(dim=0)) + 0j  # (eval_points,)

    # compute distance between characteristic functions (integral norm of difference)
    loss = ((phi_empirical - phi_target).abs() ** 2).mean()
    return loss


class FairMLP(LightningModule):
    def __init__(
        self,
        input_dim,
        theta_dim=1,
        beta=0.5,
        evaluation_points=1024,
        lr=1e-4,
        num_layers=4,
        target_classes=2,
        protected_classes=2,
        protected_id=0,
    ):
        super().__init__()
        self.encoder = MLP(input_dim, theta_dim, num_layers=num_layers)
        self.head = MLP(theta_dim, target_classes)
        self.adversary = MLP(theta_dim, protected_classes)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def prediction_loss(self, theta, target, log_prefix=""):
        logits = self.head(theta)
        predicted = logits.argmax(dim=-1)
        metrics = {
            f"loss": nn.functional.cross_entropy(logits, target),
            f"accuracy": (predicted == target).float().mean()
            # f"balanced_accuracy": nn.functional.balanced_accuracy(predicted, target),
        }
        self.log_dict({f"{log_prefix}prediction_{k}": v for k, v in metrics.items()})
        return metrics["loss"]

    def fairness_loss(self, theta, protected, log_prefix=""):
        fairness_losses = [
            fairness_penalty(theta[protected == i], self.hparams.evaluation_points)
            for i in range(self.hparams.protected_classes)
        ]
        fairness_loss = sum(fairness_losses)
        self.log(f"{log_prefix}fairness_loss", fairness_loss)
        return fairness_loss

    def adversarial_loss(self, theta, protected, log_prefix=""):
        logits = self.adversary(theta.detach())
        predicted = logits.argmax(dim=-1)
        metrics = {
            f"loss": nn.functional.cross_entropy(logits, protected),
            f"accuracy": (predicted == protected).float().mean()
            # f"balanced_accuracy": nn.functional.balanced_accuracy(predicted, protected),
        }
        self.log_dict({f"{log_prefix}adversarial_{k}": v for k, v in metrics.items()})
        return metrics["loss"]

    def training_step(self, batch, batch_idx):
        x, y = batch
        protected = x[:, self.hparams.protected_id].to(y.dtype)
        theta = self.encoder(x)

        pred_loss = self.prediction_loss(theta, y, log_prefix="train/")
        fair_loss = self.fairness_loss(theta, protected, log_prefix="train/")
        adv_loss = self.adversarial_loss(theta, protected, log_prefix="train/")
        total_loss = pred_loss + self.hparams.beta * fair_loss
        self.log("train/total_loss", total_loss)
        return total_loss + adv_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        protected = x[:, self.hparams.protected_id].to(y.dtype)
        theta = self.encoder(x)
        self.prediction_loss(theta, y, log_prefix="test/")
        self.fairness_loss(theta, protected, log_prefix="test/")
        self.adversarial_loss(theta, protected, log_prefix="test/")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = Trainer(
            max_epochs=1000,
            devices="2,",
            callbacks=[
                callbacks.StochasticWeightAveraging(0.001),
                callbacks.RichProgressBar(),
                callbacks.RichModelSummary(-1),
                callbacks.ModelCheckpoint(
                    f"checkpoints/", monitor="fairness_loss", save_last=True
                ),
            ],
            logger=loggers.WandbLogger(project="fairmlp", log_model=True),
        )

        dm = Adult(batch_size=512, num_workers=1, pin_memory=True, shuffle=True)
        model = FairMLP(
            input_dim=Adult.features,
            theta_dim=2,
            beta=1,
            lr=1e-4,
            evaluation_points=1024,
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)
