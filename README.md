# ts-rl

Reinforcement learning using TensorFlow.js

## Installation

```shell
npm install
```

## Starting the results server

```shell
cd results
python -m http.server # Python 3
```

Navigate to [http://localhost:8000]() to see charts of training losses. The chart data will be refreshed every few seconds for live feedback during training.

## Training models

```shell
# npm start [algorithm] [environment]
# eg:
npm start dqn blackjack
```
