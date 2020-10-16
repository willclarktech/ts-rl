import * as tf from "@tensorflow/tfjs-node";

import { Environment, Observation, Sample } from "../environments";
import {
	calculateDiscountedNormalizedRewards,
	createNetwork,
	sum,
} from "../util";
import { Agent } from "./core";

type LogProbabilitySample = Sample & {
	readonly logProbability: tf.Tensor1D;
	readonly baseReward: number;
};

export interface ReinforceOptions {
	readonly hiddenWidths: readonly number[];
	readonly alpha: number; // learning rate
	readonly gamma: number; // discount rate
	readonly seed?: number;
}

export class Reinforce implements Agent {
	public readonly name: string;

	private readonly options: ReinforceOptions;
	private readonly network: tf.Sequential;
	private readonly optimizer: tf.Optimizer;

	private steps: number;

	public constructor(environment: Environment, options: ReinforceOptions) {
		this.name = "Reinforce";
		this.options = options;

		this.optimizer = tf.train.adam(options.alpha);
		const widths = [
			environment.numObservationDimensionsProcessed,
			...options.hiddenWidths,
			environment.numActions,
		];
		this.network = createNetwork(widths, "relu", "softmax");
		this.steps = 0;
	}

	private getSample(
		env: Environment,
		observation: Observation,
		steps: number,
	): LogProbabilitySample {
		const { seed } = this.options;
		const processedObservation = tf.tensor2d(
			[...observation],
			[1, observation.length],
		);
		const output = this.network.predictOnBatch(
			processedObservation,
		) as tf.Tensor2D;
		const squeezedOutput = output.squeeze<tf.Tensor1D>();
		const logProbabilities = squeezedOutput.log();
		const action = tf.multinomial(logProbabilities, 1, seed).dataSync()[0];
		const logProbability = logProbabilities.gather([action]);
		const sample = env.step(action);
		const { observation: nextObservation, reward, done } = env.processSample(
			sample,
			steps,
		);

		return {
			observation: nextObservation,
			reward,
			done,
			logProbability: logProbability,
			baseReward: sample.reward,
		};
	}

	private calculateLoss(
		logProbabilities: tf.Tensor1D,
		rewards: readonly number[],
	): tf.Scalar {
		const { gamma } = this.options;
		const discountedReturns = calculateDiscountedNormalizedRewards(
			rewards,
			gamma,
		);
		return logProbabilities.mul(discountedReturns).sum().mul(-1).asScalar();
	}

	public runEpisode(env: Environment): number {
		let observation = env.resetProcessed();
		let done = false;
		let baseRewards: readonly number[] = [];
		let rewards: readonly number[] = [];
		let logProbabilities = tf.tensor1d([]);

		tf.tidy(() => {
			this.optimizer.minimize(() => {
				while (!done) {
					this.steps += 1;

					const {
						observation: nextObservation,
						reward,
						done: nextDone,
						logProbability,
						baseReward,
					} = this.getSample(env, observation, this.steps);

					baseRewards = [...baseRewards, baseReward];
					rewards = [...rewards, reward];
					done = nextDone;
					observation = nextObservation;
					logProbabilities = logProbabilities.concat(logProbability);
				}

				return this.calculateLoss(logProbabilities, rewards);
			});
		});

		return sum(baseRewards);
	}
}
