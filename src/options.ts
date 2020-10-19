import {
	A3COptions,
	ActorCriticOptions,
	DQNOptions,
	ReinforceOptions,
} from "./agents";

const defaultSeed = 123456789;

export interface TrainingOptions {
	readonly maxEpisodes: number;
	readonly rollingAveragePeriod: number;
	readonly logPeriod: number;
	readonly logDirectory: string;
	readonly seed: number;
	readonly warmupEpisodes: number;
}

interface AgentOptions<T = unknown> {
	readonly [key: string]: T & { readonly trainingOptions: TrainingOptions };
}

export const saveDirectory = "./models";
const logDirectory = "./results/data";

const defaultTrainingOptions = {
	logDirectory,
	seed: defaultSeed,
	warmupEpisodes: 0,
};

const defaultBlackjackTrainingOptions: TrainingOptions = {
	...defaultTrainingOptions,
	maxEpisodes: 1000,
	rollingAveragePeriod: 100,
	logPeriod: 100,
};

const defaultCartPoleTrainingOptions: TrainingOptions = {
	...defaultTrainingOptions,
	maxEpisodes: 1000,
	rollingAveragePeriod: 100,
	logPeriod: 10,
};

const defaultMountainCarTrainingOptions: TrainingOptions = {
	...defaultTrainingOptions,
	maxEpisodes: 10_000,
	rollingAveragePeriod: 100,
	logPeriod: 10,
};

export const A3C: AgentOptions<A3COptions> = {};

export const ActorCritic: AgentOptions<ActorCriticOptions> = {
	Blackjack: {
		seed: defaultSeed,
		alphaActor: 0.01,
		alphaCritic: 0.001,
		gamma: 0.99,
		hiddenWidths: [8],
		trainingOptions: {
			...defaultBlackjackTrainingOptions,
		},
	},
	CartPole: {
		seed: defaultSeed,
		alphaActor: 0.00001,
		alphaCritic: 0.00003,
		gamma: 0.99,
		hiddenWidths: [8],
		trainingOptions: {
			...defaultCartPoleTrainingOptions,
			maxEpisodes: 10_000,
		},
	},
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
		replayMemoryCapacity: 512,
		minibatchSize: 32,
		trainingOptions: {
			...defaultBlackjackTrainingOptions,
			warmupEpisodes: 256,
		},
	},
	CartPole: {
		hiddenWidths: [8],
		alpha: 0.00001,
		gamma: 0.99,
		epsilonInitial: 1,
		epsilonMinimum: 0.01,
		epsilonDecay: 0.995,
		tau: 0.5,
		targetNetworkUpdatePeriod: 1,
		shouldClipLoss: true,
		replayMemoryCapacity: 32_768,
		minibatchSize: 32,
		trainingOptions: {
			...defaultCartPoleTrainingOptions,
			maxEpisodes: 10_000,
			warmupEpisodes: 1024,
		},
	},
	MountainCar: {
		hiddenWidths: [8],
		alpha: 0.01,
		gamma: 0.99,
		epsilonInitial: 1,
		epsilonMinimum: 0.01,
		epsilonDecay: 0.999,
		tau: 0.9,
		targetNetworkUpdatePeriod: 1,
		shouldClipLoss: false,
		replayMemoryCapacity: 65536,
		minibatchSize: 32,
		trainingOptions: {
			...defaultCartPoleTrainingOptions,
			warmupEpisodes: 4096,
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
		trainingOptions: {
			...defaultCartPoleTrainingOptions,
		},
	},
	MountainCar: {
		trainingOptions: {
			...defaultMountainCarTrainingOptions,
		},
	},
};

export const Reinforce: AgentOptions<ReinforceOptions> = {
	Blackjack: {
		seed: defaultSeed,
		hiddenWidths: [8],
		alpha: 0.03,
		gamma: 0.99,
		trainingOptions: {
			...defaultBlackjackTrainingOptions,
		},
	},
	CartPole: {
		seed: defaultSeed,
		hiddenWidths: [4],
		alpha: 0.003,
		gamma: 0.99,
		trainingOptions: {
			...defaultCartPoleTrainingOptions,
			maxEpisodes: 2000,
		},
	},
	MountainCar: {
		seed: defaultSeed,
		hiddenWidths: [2],
		alpha: 0.01,
		gamma: 0.99,
		trainingOptions: {
			...defaultMountainCarTrainingOptions,
		},
	},
};
