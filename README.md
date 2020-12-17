# wind_predict

##### Step-by-step how to deal with data problem:

* We know that its have 4 feature in dataframe: image_id, storm_id, relative_time and ocean, let concate the target (wind_speed) inside, we will start from here.
* Count number of image per storm id -> know that there are 494 storms, max have 468 images and min is 4, most of them lower than 100 images.
* Count number of storm per ocean -> 264 in ocean 1 and 230 in ocean 2.
* Visualize wind speed distribution -> range from 15 - 185 knots, mainly in range 30-62 knots 