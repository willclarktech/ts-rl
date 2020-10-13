import { clip, sampleUniform } from "../util";
import { Environment, Observation, Sample } from "./core";

export class MountainCar implements Environment {
	public readonly name: string;
	public readonly winningScore: number;
	public readonly numObservationDimensions: number;
	public readonly numObservationDimensionsProcessed: number;
	public readonly numActions: number;

	private readonly minPosition: number;
	private readonly maxPosition: number;
	private readonly minStartingPosition: number;
	private readonly maxStartingPosition: number;
	private readonly maxSpeed: number;
	private readonly goalPosition: number;
	private readonly force: number;
	private readonly gravity: number;
	private readonly maxEpisodeLength: number;

	private steps: number;
	private done: boolean;
	private state: readonly [number, number];

	public constructor() {
		this.name = "MountainCar";
		this.winningScore = -140;
		this.numObservationDimensions = 2;
		this.numObservationDimensionsProcessed = this.numObservationDimensions;
		this.numActions = 3;

		this.minPosition = -1.2;
		this.maxPosition = 0.6;
		this.minStartingPosition = -0.6;
		this.maxStartingPosition = -0.4;
		this.maxSpeed = 0.07;
		this.goalPosition = 0.5;
		this.force = 0.001;
		this.gravity = 0.0025;

		// this.maxEpisodeLength = 200; // 200 is not enough for random actions to solve
		this.maxEpisodeLength = 1000; // See https://www.youtube.com/watch?v=rBzOyjywtPw
		this.steps = 0;
		this.done = true;
		this.state = [0, 0];
	}

	public reset(): Observation {
		const position = sampleUniform(
			this.maxStartingPosition,
			this.minStartingPosition,
		);
		const velocity = 0;
		this.steps = 0;
		this.state = [position, velocity];
		this.done = false;
		return this.state;
	}

	public resetProcessed(): Observation {
		return this.reset();
	}

	public step(action: number): Sample {
		if (action >= this.numActions || action < 0) {
			throw new Error("Action is not in range");
		}
		if (this.done) {
			throw new Error("Env is done");
		}

		const [currentPosition, currentVelocity] = this.state;
		const baseVelocity =
			currentVelocity +
			((action - 1) * this.force +
				Math.cos(3 * currentPosition) * -this.gravity);
		const velocity = clip(baseVelocity, -this.maxSpeed, this.maxSpeed);
		const basePosition = currentPosition + velocity;
		const position = clip(basePosition, this.minPosition, this.maxPosition);
		const newVelocity = position <= this.minPosition ? 0 : velocity;

		this.steps += 1;
		this.state = [position, newVelocity];
		const didReachFlag = position >= this.goalPosition;
		const done = this.steps >= this.maxEpisodeLength || didReachFlag;
		const reward = didReachFlag ? 0 : -1;

		return {
			observation: this.state,
			reward,
			done,
		};
	}

	public processSample(sample: Sample): Sample {
		return sample;
	}
}
