import { DQNOptions } from "./agents/dqn";
import { ReinforceOptions } from "./agents/reinforce";

export const seed = 1234567890;

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
	maxEpisodes: 1000,
	rollingAveragePeriod: 100,
	logPeriod: 100,
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
		epsilonDecay: 0.99,
		tau: 0.5,
		targetNetworkUpdatePeriod: 1,
		shouldClipLoss: true,
		warmup: 10,
		replayMemoryCapacity: 512,
		minibatchSize: 32,
		trainingOptions: defaultBlackjackTrainingOptions,
	},
	CartPole: {
		hiddenWidths: [16],
		alpha: 0.00003,
		gamma: 0.9,
		epsilonInitial: 1,
		epsilonMinimum: 0.01,
		epsilonDecay: 0.999,
		tau: 0.9,
		targetNetworkUpdatePeriod: 1,
		shouldClipLoss: false,
		warmup: 1024,
		replayMemoryCapacity: 4096,
		minibatchSize: 32,
		trainingOptions: {
			...defaultCartPoleTrainingOptions,
			maxEpisodes: 10_000,
		},
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
