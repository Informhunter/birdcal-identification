## 16.06.2020

### Problem

Have audio files of length ~ 20 second with singing birds. In addition to these audio files there
is a file with metadata about environment and labels indicating which bird species are present.

Need to create a model that can detect which birds are singing in each n-second segment of several
~10 minute audio files.

Environment and background noise may be different from the available training data. Only few test
examples can be examined, the rest are analyzed on the server-side during kernel submission.


### Pipeline/experiments

It is important to have a good cross-validation (probably with augmentation), because otherwise
there is a high chance of overfitting the PL. And since PL is the only reliable way to estimate
the performance of models on a different distribution of data it should be rarely used.

Basically problem is about building a robust model that takes as much information from available
data and is not affected by changes of noise distributions.

To get such model/approach we can
1. Add more noise to the training data to improve robustness
2. Preprocess sound samples with some denoising/speaker selection model to get rid of additional
noises.
3. Overlay several bird recordings to create an effects of mixed singing and augument the dataset.


Need to optimize performance on examples with additional noise to make sure the model will do well
in new conditions.

Pipeline:
1. Take a random piece of sound with labels
2. 


### Random

One idea that came to mind is to try augumenting recordings by combining several bird voices
together based on background noise and geographical location to get a reasonable prior on
which birds can be heared together with which.

And in general it would not make sense to have birds from one location to appear in another,
where similar birds have never appeared.

Try using names of birds (family etc.) for building bird embeddings???? Name may contain
information about similarities between species.


## 19.06.2020

`bird_seen` can be used to determine how reliable the label is: if bird was seen, it is more likely
that the label is reliable.


## 20.06.2020

Probably makes sence to augment bird calls with the calls of other birds that co-appear together
Need to check out which birds tend to occur together.


## 21.06.2020

The data format and schema is quite badly explained and looks like hell for perfectionsts.

How I understand the final notebook environment that will be used for submission:

```
/kaggle/input/birdsong-recognition/train.csv
/kaggle/input/birdsong-recognition/test.csv
/kaggle/input/birdsong-recognition/train_audio/<ebird_id>/*.mp3
/kaggle/input/birdsong-recognition/test_audio/<audio_id>.mp3
```


## 22.06.2020

Ok, now I have dataloaders for train and test. Time to prepare the first model.
First want to figure out the math behind mel spectrograms or rather their hyperparameters
and influence of those hyperparameters on spectrogram resolution.

Playing with parameters now: MelSpectrogram transform did not behave in the same way as one
from librosa.


## 05.07.2020

Ok, back here. Want to play around with my dataloaders a bit and try to understand if I actually
have some memory leak or it's some problems with lightning-pytorch. 
I guess it is normal: unpacked mp3s take a lot of space. Will create a script that will convert
mp3s to mel specetrogragm tensors and save them with torch.save().

Ok, figued out there was a weird bug: I passed audios to "Resample" with channel dimention.


## 06.07.2020

Appartnely loading, resampling and applying mel spectrograms takes too much time (and also
GPU memory for mels). So I converted all the recordings to mel spectrograms in advance.
Will try to build a sample model again.

After almost giving up got some non-random results: while validation loss goes down, validation
accuracy goes up. Now training while watching top_1, top_3 and top_5 accuracies. But yeah,
accuracies are around 1-3%, which is better than random guess on 264 classes for sure.

One idea could be to identify high-rated recordings with clear signals and cut-off all the rest
before training. Kind of find those bright spots I see on mel spectrograms and understand that
they correspond to the signal.

Try using one-size batch with gradient accumulation and stop using padding.

And probably makes sense to debug the whole thing with a subset of classes to simplify.

Also should go through CNNs, understand conv layer receptive fields and think how to make the
model treat the frequency dimention in a different way.


## 08.07.2020

Probably key ideas for the competition:
1. Improving SNR for training (make labels less sparse)
2. Augmenting data (pitch shift, time shift, change speed, inject noise)
3. Adding non-bird data to the training set to reduce potential false positives

Need to create a pipeline/script that will make predictions on the test set.
Also, need to log hyperparameters.

Plan is to decide on the first basic model by the end of the week.
1. Understand the CNNs better, think about receptive fields of conv layers and get decent
intuition on that.
2. Choose a way to deal with variable-length recordings (maybe just random 5 second crop
instead of global pooling?)
3. Implement basic transformations on mel spectrograms.

Probably for now will be dealing with mel spectrograms I have at the moment: not gonna
generate them dynamically. Later though will have to resample all the audios and
use MelSpectrogram transform to do hyperparameter tunning.

Also "nocall" detector would be nice: usefull both for training and for inference.
And the initial strategy could be:
1. Mark segments without birds as "nocall"
2. For the rest choose argmax bird for that time segment


## 10.07.2020

TimeShift augmentation seemed to reduce overfitting problem (at least to some extent).
Though the accuracy is not going as high as without augmentation.

Implemented basic transforms: time rescale, time shift, random crop.

Not sure about "nocall" detector yet.

Will fix the global average pooling: make it indeed average.

Plan for tomorrow: get the first model in a tunnable suit and run hyperparameter
optimization. Though probably not yet.

Global plans are:
- Get single/multilabel classification working
- Add mixup, noise, pitch-shift augmentation
- Implement prediction pipeline (probably in a generalizable way with nocall detector, etc.)
- Implement multilabel evaluation metric
- Implement tunnable suit for the first model and get parameter optimization running non-stop.


## 12.07.2020

Should really implement a random crop training: if it is possible to train an architecture
similar to mine to a similar quality, probably should ditch the complication of padding
and splitting into segments.

More ideas:
- Train a model to identify call/nocall and only then use the classifier model to predict
- Build a histogram of how energy is distributed across freqencies, apply a small 1D CNN on
top to get good features and use as an additional feature branch in the main network.
- Try different loss (with softmax, instead of bunch of sigmoids)

Now trying cross entropy training. Seams fine so far, though doesn't converge as fast.
Could be happening because the learnin rate is too low.
What is good, the validation loss seems to behave better than with binary cross entropy
i.e. doesn't go up and down all the time. I wonder why multilabel classification loss
ends up growing so much. Maybe try 2 linear layers to get more stable loss :/

## 14.07.2020

Need to start utilizing the whole dataset: after I start doing that I can focus on models.
It is just important to be able to feed any data to any model and be sure that I will not
run out of memory. So probably split longer recorings into pieces of less than 1 minute.
With my simple model 240 second recordings already make it impossible to train. And even
180 seconds is already too much.

Split each recoring into (duration // 60 + 1) pieces, create alternative train.csv with
information on which clip is part of which original recording.

## 15.07.2020

Ideas:
- Use a trained model (CV-style) to identify pieces with birds singing and add those pieces
to training to get more examples of shorter length with more concentrated labels.
- Based on longer samples for each bird build a model of "periodicity" of bird singing:
some birds to a call every 2 seconds, some every 5, etc. Can allow to do better guesses on
call-nocall when looking at whole recordeing.
- Before predicting piece-wise, use the model to predict which birds call in the whole
recording and use that information as a prior for piece predictions.


Now finishing the final script for preparing data: resample -> split -> mel transform


## 16.07.2020

Apparently I made a mistake in data preparation script parameters and created specs
with hop_length=2048. Have recreated them with hop_length=512 and will try again.
Tried to do call/nocall detection using a trained model, but it didn't look promissing.
I hope it's due to spectrograms having low resolution.

Need to note that network with more filters was doing quite well on the whole dataset
processed with hop_length=2048

Might consider tweaking parameters and building and enseble of models trained with
different hop length (and overall different spec parameters).
