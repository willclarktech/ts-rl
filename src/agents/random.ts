import { Environment } from "../environments";
import { sampleUniform, sum } from "../util";
import { Agent } from "./core";

export class Random implements Agent {
	public readonly name: string;
	private readonly numActions: number;

	public constructor({ numActions }: Environment) {
		this.name = "Random";
		this.numActions = numActions;
	}

	public runEpisode(env: Environment): number {
		env.reset();
		let done = false;
		let rewards: readonly number[] = [];

		while (!done) {
			const action = sampleUniform(this.numActions);
			const sample = env.step(action);
			({ done } = sample);
			rewards = [...rewards, sample.reward];
		}

		return sum(rewards);
	}

	public async save(): Promise<void> {
		// not necessary
	}
}
