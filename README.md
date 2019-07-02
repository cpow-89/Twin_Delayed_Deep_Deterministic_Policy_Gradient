# Twin Delayed Deep Deterministic Policy Gradient


### The Theoretical Background

My Notes: [click here](https://github.com/cpow-89/Twin_Delayed_Deep_Deterministic_Policy_Gradient/blob/master/TD3_Notes.ipynb)<br>

The paper: [click here](https://arxiv.org/pdf/1802.09477.pdf)

### Trained Agent

[image2]: https://raw.githubusercontent.com/cpow-89/Twin_Delayed_Deep_Deterministic_Policy_Gradient/master/img/trained_ant.gif "Trained Agent"
![Trained Agent][image2]

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

> - If you want to run the Pre-trained ant agent, you must set "run_training" in AntBulletEnv_v0.json to false and run the python main.py command<br>
> - If you want to train the agent yourself, you must set "run_training" in AntBulletEnv_v0.json to true and run the python main.py command
