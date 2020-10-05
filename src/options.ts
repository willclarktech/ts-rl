import { DQNOptions } from "./agents/dqn";
import { ReinforceOptions } from "./agents/reinforce";

export interface TrainingOptions {
	readonly maxEpisodes: number;
	readonly rollingAveragePeriod: number;
	readonly logPeriod: number;
	readonly logDirectory: string;
}

interface AgentOptions<T = unknown> {
	readonly [key: string]: T & { readonly trainingOptions: TrainingOptions };
}

const logDirectory = "./results/data";

const defaultBlackjackTrainingOptions: TrainingOptions = {
	maxEpisodes: 10_000,
	rollingAveragePeriod: 1000,
	logPeriod: 1000,
	logDirectory,
};

const defaultCartPoleTrainingOptions: TrainingOptions = {
	maxEpisodes: 1000,
	rollingAveragePeriod: 100,
	logPeriod: 10,
	logDirectory,
};

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
		trainingOptions: defaultBlackjackTrainingOptions,
	},
	CartPole: {
		hiddenWidths: [2],
		alpha: 0.0003,
		gamma: 0.99,
		epsilonInitial: 1,
		epsilonMinimum: 0.01,
		epsilonReduction: 0.001,
		shouldClipLoss: true,
		replayMemoryCapacity: 512,
		minibatchSize: 32,
		targetNetworkUpdatePeriod: 2,
		trainingOptions: defaultCartPoleTrainingOptions,
	},
};

export const Random: AgentOptions = {
	Blackjack: {
		trainingOptions: {
			...defaultBlackjackTrainingOptions,
			maxEpisodes: 10_000,
			rollingAveragePeriod: 1000,
		},
	},
	CartPole: {
		trainingOptions: defaultCartPoleTrainingOptions,
	},
};

export const Reinforce: AgentOptions<ReinforceOptions> = {
	Blackjack: {
		hiddenWidths: [2],
		alpha: 0.03,
		gamma: 0.99,
		trainingOptions: defaultBlackjackTrainingOptions,
	},
	CartPole: {
		hiddenWidths: [2],
		alpha: 0.01,
		gamma: 0.99,
		trainingOptions: defaultCartPoleTrainingOptions,
	},
};
