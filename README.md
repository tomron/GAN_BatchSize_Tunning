# GAN batchsize Hyperparameter test

This repository is used to evaluate several batch size policies for GAN training

## Batch size policies

* **Same** - the number of real samples and fake samples is equal and constant across all training steps.
* **Fake increase** - the number of real samples remains constant across all the training steps but the number of fake examples increases over time. The number of fake examples doubles every *batch size change interval* epochs. The initial number of fake examples is equal to the number of real samples and is determine by *initial batch size* hyperparameter.
* **Real increase** - the number of fake samples remains constant across all the training steps but the number of real samples increases over time. The number of real samples doubles every *batch size change interval* epochs. The initial number of real samples is equal to the number of fake samples and is determine by *initial batch size* hyperparameter.
* **Both increase** - the number of fake samples and real samples increases equally over time. The number of samples doubles every *batch size change interval* epochs and the initial number of samples is determined by *initial batch size* hyperparameter.
* **Random** - randomly selects real batch size and fake batch size in the range 1-10000 every epoch.

## Usage

### simple_gan.py

Train simple GAN -

```
usage: simple_gan.py [-h]
                     [--dataset {cifar10,lsun,mnist,imagenet,folder,lfw,fake,fashionmnist}]
                     [--dataroot DATAROOT] [--n_cpu N_CPU] [--n_gpu N_GPU]
                     [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]
                     [--n_epochs N_EPOCHS] [--lr LR] [--beta1 BETA1]
                     [--beta2 BETA2] [--manual_seed MANUAL_SEED]
                     [--classes CLASSES]
                     [--policy {same,fake_increase,real_increase,both_increase,random}]
                     [--sample_interval SAMPLE_INTERVAL]
                     [--batch_interval BATCH_INTERVAL]
                     [--output_folder OUTPUT_FOLDER]
                     [--action {train,summary}] [--device DEVICE]
                     [--net_g NET_G] [--net_d NET_D] [--latent_dim LATENT_DIM]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {cifar10,lsun,mnist,imagenet,folder,lfw,fake,fashionmnist}
                        Dataset to use
  --dataroot DATAROOT   path to dataset
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --n_gpu N_GPU         number of GPUs to use
  --batch_size BATCH_SIZE
                        input batch size
  --image_size IMAGE_SIZE
                        the height / width of the input image to network
  --n_epochs N_EPOCHS   number of epochs of training
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         adam: decay of first order momentum of gradient,
                        default=0.5
  --beta2 BETA2         adam: decay of moving average of squared gradient,
                        default=0.999
  --manual_seed MANUAL_SEED
                        manual seed
  --classes CLASSES     comma separated list of classes for the lsun data set
  --policy {same,fake_increase,real_increase,both_increase,random}
                        Batch size policy
  --sample_interval SAMPLE_INTERVAL
                        interval betwen image samples
  --batch_interval BATCH_INTERVAL
                        Intervals to update batch size in
  --output_folder OUTPUT_FOLDER
                        Output folder
  --action {train,summary}
                        Action - train model or print summary
  --device DEVICE       device string
  --net_g NET_G         path to netG (to continue training)
  --net_d NET_D         path to netD (to continue training)
  --latent_dim LATENT_DIM
                        dimensionality of the latent space
```

### pytorch_dcgan.py

Train DCGAN -

```
usage: pytorch_dcgan.py [-h]
                        [--dataset {cifar10,lsun,mnist,imagenet,folder,lfw,fake,fashionmnist}]
                        [--dataroot DATAROOT] [--n_cpu N_CPU] [--n_gpu N_GPU]
                        [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]
                        [--n_epochs N_EPOCHS] [--lr LR] [--beta1 BETA1]
                        [--beta2 BETA2] [--manual_seed MANUAL_SEED]
                        [--classes CLASSES]
                        [--policy {same,fake_increase,real_increase,both_increase,random}]
                        [--sample_interval SAMPLE_INTERVAL]
                        [--batch_interval BATCH_INTERVAL]
                        [--output_folder OUTPUT_FOLDER]
                        [--action {train,summary}] [--device DEVICE]
                        [--net_g NET_G] [--net_d NET_D] [--nz NZ] [--ngf NGF]
                        [--ndf NDF]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {cifar10,lsun,mnist,imagenet,folder,lfw,fake,fashionmnist}
                        Dataset to use
  --dataroot DATAROOT   path to dataset
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --n_gpu N_GPU         number of GPUs to use
  --batch_size BATCH_SIZE
                        input batch size
  --image_size IMAGE_SIZE
                        the height / width of the input image to network
  --n_epochs N_EPOCHS   number of epochs of training
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         adam: decay of first order momentum of gradient,
                        default=0.5
  --beta2 BETA2         adam: decay of moving average of squared gradient,
                        default=0.999
  --manual_seed MANUAL_SEED
                        manual seed
  --classes CLASSES     comma separated list of classes for the lsun data set
  --policy {same,fake_increase,real_increase,both_increase,random}
                        Batch size policy
  --sample_interval SAMPLE_INTERVAL
                        interval betwen image samples
  --batch_interval BATCH_INTERVAL
                        Intervals to update batch size in
  --output_folder OUTPUT_FOLDER
                        Output folder
  --action {train,summary}
                        Action - train model or print summary
  --device DEVICE       device string
  --net_g NET_G         path to netG (to continue training)
  --net_d NET_D         path to netD (to continue training)
  --nz NZ               size of the latent z vector
  --ngf NGF             the depth of feature maps carried through the
                        generator
  --ndf NDF             the depth of feature maps propagated through the
                        discriminator
```

### graph_batch_size.py

Graph batch size according to parameters and save figure locally.
Relevant options are actaully only policy, sample_interval and batch_size.

```
usage: graph_batch_sizes.py [-h]
                            [--dataset {cifar10,lsun,mnist,imagenet,folder,lfw,fake,fashionmnist}]
                            [--dataroot DATAROOT] [--n_cpu N_CPU]
                            [--n_gpu N_GPU] [--batch_size BATCH_SIZE]
                            [--image_size IMAGE_SIZE] [--n_epochs N_EPOCHS]
                            [--lr LR] [--beta1 BETA1] [--beta2 BETA2]
                            [--manual_seed MANUAL_SEED] [--classes CLASSES]
                            [--policy {same,fake_increase,real_increase,both_increase,random}]
                            [--sample_interval SAMPLE_INTERVAL]
                            [--batch_interval BATCH_INTERVAL]
                            [--output_folder OUTPUT_FOLDER]
                            [--action {train,summary}] [--device DEVICE]
                            [--net_g NET_G] [--net_d NET_D]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {cifar10,lsun,mnist,imagenet,folder,lfw,fake,fashionmnist}
                        Dataset to use
  --dataroot DATAROOT   path to dataset
  --n_cpu N_CPU         number of cpu threads to use during batch generation
  --n_gpu N_GPU         number of GPUs to use
  --batch_size BATCH_SIZE
                        input batch size
  --image_size IMAGE_SIZE
                        the height / width of the input image to network
  --n_epochs N_EPOCHS   number of epochs of training
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         adam: decay of first order momentum of gradient,
                        default=0.5
  --beta2 BETA2         adam: decay of moving average of squared gradient,
                        default=0.999
  --manual_seed MANUAL_SEED
                        manual seed
  --classes CLASSES     comma separated list of classes for the lsun data set
  --policy {same,fake_increase,real_increase,both_increase,random}
                        Batch size policy
  --sample_interval SAMPLE_INTERVAL
                        interval betwen image samples
  --batch_interval BATCH_INTERVAL
                        Intervals to update batch size in
  --output_folder OUTPUT_FOLDER
                        Output folder
  --action {train,summary}
                        Action - train model or print summary
  --device DEVICE       device string
  --net_g NET_G         path to netG (to continue training)
  --net_d NET_D         path to netD (to continue training)
```