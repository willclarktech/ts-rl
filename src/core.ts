export type ActivationFunction = "relu" | "sigmoid" | "softmax";

export type Observation = readonly number[];

export type Sample = {
	readonly observation: Observation;
	readonly reward: number;
	readonly done: boolean;
};

export interface Environment {
	name: string;
	winningScore?: number;
	numObservationDimensions: number;
	numActions: number;
	reset(): Observation;
	step(action: number): Sample;
}

export interface Learner {
	name: string;
	runEpisode(env: Environment): number;
}
