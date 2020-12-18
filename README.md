# wind_predict

##### Step-by-step how to deal with data problem:

* We know that its have 4 feature in dataframe: image_id, storm_id, relative_time and ocean, let concate the target (wind_speed) inside, we will start from here.
* Count number of image per storm id -> know that there are 494 storms, max have 468 images and min is 4, most of them lower than 100 images.
* Count number of storm per ocean -> 264 in ocean 1 and 230 in ocean 2.
* Visualize wind speed distribution -> range from 15 - 185 knots, mainly in range 30-62 knots.
* Divide Train/Val: the purpose of this competition is to predict the future wind, so they split train/val depend on "relative time" column, they take 20% of each storm to be validate data.
* The notebook only use 10% of whole data -> for quick :)
* Use Resnet152, replace last to Linear(2048, 50) -> ReLU -> Dropout 0.1 -> Linear (50,1)
* Still use 3channels and normalize with ImageNet mean, var
* Use Crop instead of Resize
* Define RMSELoss from MSE
* ADAM- lr 2e-4