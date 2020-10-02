import * as tf from "@tensorflow/tfjs-node";

import { Environment, Observation } from "../environments";
import { createNetwork, mean, sum } from "../util";
import { Agent } from "./core";

const calculateDiscountedReward = (
	rewards: readonly number[],
	gamma: number,
): number => sum(rewards.map((reward, i) => reward * gamma ** i));

const calculateDiscountedRewards = (
	rewards: readonly number[],
	gamma: number,
): readonly number[] =>
	rewards.map((_, i) => calculateDiscountedReward(rewards.slice(i), gamma));

const normalizeRewards = (
	rewards: readonly number[],
	epsilon = 1e-9,
): readonly number[] => {
	const mn = mean(rewards);
	const std = tf.moments(rewards).variance.sqrt().dataSync()[0];
	return rewards.map((reward) => reward - mn / (std + epsilon));
};

const discountAndNormalizeRewards = (
	rewards: readonly number[],
	gamma: number,
): readonly number[] => {
	const discountedRewards = calculateDiscountedRewards(rewards, gamma);
	return normalizeRewards(discountedRewards);
};

type Sample = {
	readonly observation: Observation;
	readonly reward: number;
	readonly done: boolean;
	readonly logProbability: tf.Tensor1D;
};

export class Reinforce implements Agent {
	public readonly name: string;

	private readonly gamma: number;
	private readonly network: tf.Sequential;
	private readonly optimizer: tf.Optimizer;

	public constructor(
		{ numObservationDimensions, numActions }: Environment,
		hiddenWidths: readonly number[],
		alpha: number, // learning rate
		gamma: number, // discount rate
	) {
		this.name = "Reinforce";
		this.gamma = gamma;
		this.optimizer = tf.train.adam(alpha);
		const widths = [numObservationDimensions, ...hiddenWidths, numActions];
		this.network = createNetwork(widths, "relu", "softmax");
	}

	private getSample(env: Environment, observation: Observation): Sample {
		const processedObservation = tf
			.tensor1d([...observation])
			.reshape<tf.Tensor2D>([1, env.numObservationDimensions]);
		const output = this.network.predict(processedObservation) as tf.Tensor2D;
		const squeezedOutput = output.squeeze<tf.Tensor1D>([0]);
		const action = tf.multinomial(squeezedOutput, 1).dataSync()[0];
		const logProbability = tf.log(squeezedOutput.gather([action]));
		const { observation: nextObservation, reward, done } = env.step(action);

		return {
			observation: nextObservation,
			reward,
			done,
			logProbability: logProbability,
		};
	}

	private calculateLoss(
		rewards: readonly number[],
		logProbabilities: tf.Tensor1D,
	): tf.Scalar {
		const normalizedDiscountedRewards = discountAndNormalizeRewards(
			rewards,
			this.gamma,
		);
		return tf.sum(logProbabilities.mul(normalizedDiscountedRewards)).mul(-1);
	}

	public runEpisode(env: Environment): number {
		let observation = env.reset();
		let done = false;
		let rewards: readonly number[] = [];
		let logProbabilities = tf.tensor1d([]);

		tf.tidy(() => {
			this.optimizer.minimize(() => {
				while (!done) {
					const sample = this.getSample(env, observation);
					({ observation, done } = sample);
					const { reward, logProbability } = sample;

					rewards = [...rewards, reward];
					logProbabilities = logProbabilities.concat(logProbability);
				}

				return this.calculateLoss(rewards, logProbabilities);
			});
		});

		return sum(rewards);
	}
}
