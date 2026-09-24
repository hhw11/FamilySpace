"""Hydra entrypoint for family-medoid latent flow-matching training."""

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_diffusion.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    model = hydra.utils.instantiate(cfg.diffusion_module)
    callbacks = [
        hydra.utils.instantiate(callback)
        for callback in (cfg.get("callbacks") or {}).values()
        if callback is not None and callback.get("_target_")
    ]
    loggers = [
        hydra.utils.instantiate(logger)
        for logger in (cfg.get("logger") or {}).values()
        if logger is not None and logger.get("_target_")
    ]
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers or False,
    )
    if cfg.get("train", True):
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
