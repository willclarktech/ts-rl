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
	numObservationDimensionsProcessed: number;
	numActions: number;
	reset(): Observation;
	resetProcessed(): Observation;
	step(action: number): Sample;
	processSample(sample: Sample, steps: number): Sample;
}
