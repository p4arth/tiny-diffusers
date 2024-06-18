An attempt of minimal code recreation of diffusion models.

This code is currently tested on the MNIST dataset.

To train your own tiny diffuser simply run
```
python3 src/train.py
```
from inside the root directory of the project.

To test the model you trained simply run
```
python3 src/test.py --ckpt_path [MODEL_CHECKPOINT]
```