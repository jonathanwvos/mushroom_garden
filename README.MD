# Introduction
Please refer to the `Mushrooms Exp1` notebook.

# Setup
For the environment I used Anaconda (though any environment manager should be fine) and python 3.10. For the packages I used pip. They can be found in `requirements.txt`.

If you plan on using Anaconda, the following command will create an environment named `mushroom_garden` in python 3.10:

```shell
conda create -n mushroom_garden python==3.10
```

After which you should activate the environment:
```shell
conda activate mushroom_garden
```

Finally, using pip, you can install the required packages:

```shell
pip install -r requirements.txt
```

The illustration work is available in the `Mushrooms` notebook, so you'll need to start a jupyter notebook server:

```shell
jupyter notebook
```

Alternatively, if you're using an editor within inline capabilities (like VS Code), you can just open the notebook and it will work.

# Data
To begin, you'll need the data for which the code is configured and setup the `data` directory along with two sub-directories: `results` and `species`. Please download the images from the following link [Kaggle](https://www.kaggle.com/datasets/lizhecheng/mushroom-classification). Once you have the data, combine the `Exidia` and `Inocybe` folders from "Transfer data" with the folders from the "Mushrooms" directory so that your local data directory looks as such:

```
mushroom_garden
|--- data
     |--- results
     |--- species
          |--- Agaricus
          |--- Amanita
          |--- Boletus
          |--- Cortinarius
          |--- Entoloma
          |--- Exidia
          |--- Hygrocybe
          |--- Inocybe
          |--- Lactarius
          |--- Russula
          |--- Suillus
```