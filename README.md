# Using CNN on 2D Images of Time Series
Because too often time series are fed as 1-D vectors Recurrent Neural Networks (LSTM, GRU..).

## Generate some random data
```
git clone https://github.com/philipperemy/tensorflow-cnn-time-series.git
cd tensorflow-cnn-time-series/
sudo pip3 install -r requirements.txt
python3 generate_data.py
```

## Start the training of the CNN (AlexNet is used here)
```
python3 alexnet_run.py
```
