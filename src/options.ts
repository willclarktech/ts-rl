import { DQNOptions } from "./agents/dqn";
import { ReinforceOptions } from "./agents/reinforce";

interface AgentOptions<T> {
	readonly [key: string]: T;
}

export const DQN: AgentOptions<DQNOptions> = {
	Blackjack: {
		hiddenWidths: [2],
		alpha: 0.03,
		gamma: 0.99,
		epsilonInitial: 1,
		epsilonMinimum: 0.01,
		epsilonReduction: 0.001,
		shouldClipLoss: true,
		replayMemoryCapacity: 512,
		minibatchSize: 32,
		targetNetworkUpdatePeriod: 1,
	},
	CartPole: {
		hiddenWidths: [2],
		alpha: 0.003,
		gamma: 0.99,
		epsilonInitial: 1,
		epsilonMinimum: 0.01,
		epsilonReduction: 0.0001,
		shouldClipLoss: true,
		replayMemoryCapacity: 512,
		minibatchSize: 32,
		targetNetworkUpdatePeriod: 2,
	},
};

export const Reinforce: AgentOptions<ReinforceOptions> = {
	Blackjack: {
		hiddenWidths: [8],
		alpha: 0.01,
		gamma: 0.99,
	},
	CartPole: {
		hiddenWidths: [8],
		alpha: 0.01,
		gamma: 0.99,
	},
};
