# Open Pi0

This project aims to reproduce [Pi0](https://www.physicalintelligence.company/blog/pi0), a general-purpose robot foundation model developed by [Physical Intelligence](https://www.physicalintelligence.company/). We are committed to bringing this cutting-edge technology to a wider community of researchers and developers, and we encourage collaboration in optimizing and improving the model. Embracing the open-source spirit, we welcome contributions through Pull Requests (PRs) from the community. We believe that it is through collective effort in the open-source ecosystem that we can push the boundaries of robotic technology and ultimately build a state-of-the-art Robot Foundation Model. Join us in this endeavor, and together we can advance the field and create smarter, more efficient robotic systems.

## Installation

```bash
conda create -n open_pi0 python=3.11

conda activate open_pi0

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

git clone https://github.com/vivym/OpenPi0.git

cd OpenPi0

pip install -v -e .
```

## Training

```bash
accelerate launch scripts/train_pi0_gemma.py \
    --with_tracking \
    --report_to=wandb \
    --mixed_precision=bf16 \
    --num_train_epochs=500 \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --checkpointing_steps=5000 \
    --learning_rate=0.00001 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=5000 \
    --lr_warmup_steps_action=500 \
    --weighting_scheme=logit_normal
```
