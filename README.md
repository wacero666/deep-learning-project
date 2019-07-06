Hello! Welcome to our deep content music recommendation model. This project is inspired by [lifehouse method project](https://petetownshend.net/musicals/lifehouse-method), launched by Pete Townshend, one of the members from the Who. He believed that “music can reflect who we really are - like a mirror”, so we want our model to give people some insights to their own music taste through song recommendations, which may either fall in your comfort zone or surprise you.

This project is also a tribute to Spotify's [Discover Weekly](https://www.spotify.com/discoverweekly/) and Your Daily Drive playlists that are driven by their magical machine learning models!  

Big shoutout to Million Song dataset([MSD](https://labrosa.ee.columbia.edu/millionsong/) and LastFm datasets([lastfm](https://labrosa.ee.columbia.edu/millionsong/lastfm)) which make our project possible to start!

More detailed in our [final report](https://github.com/wacero666/deep-learning-project/blob/master/10_707_final_report.pdf)  

[data_preparation]   
- [MSD_to_Spectrogram] Where we use MSD source code to transform audio features of each song from MSD to spectrograms  
- Where we transform spectrogram to numpy arrays, and generate train_set_score.npy and test_set_score.npy.

[model]
- cnn_final_model.py is to run on song pairs with balanced similarity score 
- cnn_imbalanced.ipynb is to run on song pairs with imbalanced similarity score
- lstm_cnn_model.py is our initial model that used cnn and lstm 


