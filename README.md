# Twin Delayed Deep Deterministic Policy Gradient


### Trained Agent - Ant

[image1]: https://raw.githubusercontent.com/cpow-89/Twin_Delayed_Deep_Deterministic_Policy_Gradient/master/img/trained_ant.gif "Trained Agent - Ant"
![Trained Agent][image1]

### Trained Agent - Humanoid

[image3]: https://github.com/cpow-89/Twin_Delayed_Deep_Deterministic_Policy_Gradient/blob/master/img/trained_humanoid.gif "Trained Agent - Humanoid"
![Trained Agent][image3]

### Trained Agent - Walker

[image2]: https://github.com/cpow-89/Twin_Delayed_Deep_Deterministic_Policy_Gradient/blob/master/img/trained_walker.gif "Trained Agent - Walker"
![Trained Agent][image2]

### Trained Agent - HalfCheetah

[image4]: https://github.com/cpow-89/Twin_Delayed_Deep_Deterministic_Policy_Gradient/blob/master/img/trained_halfcheetah.gif "Trained Agent - HalfCheetah"
![Trained Agent][image4]


### The Theoretical Background

My Notes: [click here](https://github.com/cpow-89/Twin_Delayed_Deep_Deterministic_Policy_Gradient/blob/master/TD3_Notes.ipynb)<br>

The paper: [click here](https://arxiv.org/pdf/1802.09477.pdf)

### Getting Started

1. Create (and activate) a new environment with Python 3.6.

> conda create --name env_name python=3.6<br>
> source activate env_name

2. Install Sourcecode dependencies

> conda install pytorch torchvision cudatoolkit=9.0 -c pytorch <br>
> pip install gym <br>
> pip install pybullet <br>
> sudo apt-get install ffmpeg <br>

3. Run the Code

### How to run the project

You can run the project by running the main.py file through the console.
- open the console and run: python main.py -c "your_config_file.json" 
- to train the agent from scratch set "run_training" in the config file to true
- to run the pre-trained agent set "run_training" in the config file to false

optional arguments:

-h, --help

    - show help message
    
-c , --config

    - Config file name - file must be available as .json in ./configs
    
Example: python main.py -c "AntBulletEnv_v0.json" 
