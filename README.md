Hello! Welcome to our deep content music recommendation model. This project is inspired by lifehouse method project, launched by one of members from the Who, Pete Townshend. He believed that “music can reflect who we really are - like a mirror”, so our model may give some recommendations, either fall in your comfort zone or surprise you. We hope to give people some insights into their own music taste!     

This project is also a tribute to Spotify's Discover Weekly and Your Daily Drive playlists that are driven by great machine learning models at Spotify!  

More detailed in our [final report](https://github.com/wacero666/deep-learning-project/blob/master/10_707_final_report.pdf)  

[data_preparation]   
- [MSD_to_Spectrogram] Where we turn audio features of each song from Million Song Dataset to spectrogram  
- Where we transform spectrogram to numpy arrays, and generate train_set_score.npy and test_set_score.npy.

[model]
- cnn_final_model.py is used to run on song pairs with balanced similarity score 
- cnn_imbalanced.ipynb is used to run on song pairs with imbalanced similarity score
- lstm_cnn_model.py is our initial model that used cnn and lstm 


To run our code, you need to download the Million Song dataset and LastFm datasets.   
Link below to download [MSD](https://labrosa.ee.columbia.edu/millionsong/) and [lastfm](https://labrosa.ee.columbia.edu/millionsong/lastfm)  
(Note that we use the full dataset from both sources, which may be easy to run on AWS.)
