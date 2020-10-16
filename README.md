# ts-rl

Reinforcement learning using TensorFlow.js

## Installation

Install Node modules:

```shell
npm install
```

For Python integration you will also need:

- Python v3
- Jupyter Notebook
- pip modules (TODO: document these)

## Starting the results server

```shell
npm run serve
```

Navigate to [http://localhost:8000]() to see charts of training losses. The chart data will be refreshed every few seconds for live feedback during training.

## Training models

```shell
# npm start [algorithm] [environment]
# eg:
npm start dqn blackjack
```

## Loading models into Python

Convert saved model using `npm run convert <path to saved model> <output path>`, eg:

```shell
npm run convert ./models/DQN-CartPole-q-network/model.json ./models/DQN-CartPole-q-network-converted
```

Start Jupyter Notebook:

```shell
npm run notebook
```

See demo model loading/evaluation in [`https://github.com/willclarktech/ts-rl/blob/master/notebooks/model-load-demo.ipynb`]().
