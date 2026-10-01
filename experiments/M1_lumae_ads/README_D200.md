# D200 Colab handoff

The D200 bundle is created only after 152 new samples have passed visual and
temporal verification and the 200-sample freeze has been written. The D48
samples remain in Train; Val and Test each contain 30 AI pseudo labels. Test is
sealed until the later final evaluation. These annotations are not human ground
truth.

From the source checkout, create the bundle after a completed freeze:

```sh
python -m data_process.lumae_scale --package-colab
```

After unzipping the bundle in Colab and installing the official Moment-DETR
dependencies/checkpoint as in the D48 workflow, run from the extracted LUMAE
root with that root on `PYTHONPATH`:

```sh
python -m experiments.M1_lumae_ads.train_d200 preflight --lumae-root /content/lumae
python -m experiments.M1_lumae_ads.train_d200 features --lumae-root /content/lumae --moment-detr-root /content/moment_detr --feature-root /content/d200_features
python -m experiments.M1_lumae_ads.train_d200 train --lumae-root /content/lumae --moment-detr-root /content/moment_detr --checkpoint /content/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt --feature-root /content/d200_features --results-root /content/d200_results
```

The runner selects checkpoints on pseudo-labeled Val and does not use Test in
feature extraction or training. The final frozen Test evaluation belongs to a
later experiment. No D200 training is part of this dataset build.
