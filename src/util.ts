import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";

type ActivationFunction = "relu" | "sigmoid" | "softmax";

export const sum = (arr: readonly number[]): number =>
	arr.reduce((total, next) => total + next, 0);

export const mean = (arr: readonly number[]): number => sum(arr) / arr.length;

export const sampleUniform = (max: number, min = 0): number =>
	Math.floor(Math.random() * (max - min)) + min;

export const clip = (n: number, min: number, max: number): number =>
	Math.min(max, Math.max(min, n));

const calculateDiscountedReward = (
	rewards: readonly number[],
	gamma: number,
): number => sum(rewards.map((reward, i) => reward * gamma ** i));

export const calculateDiscountedRewards = (
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

export const calculateDiscountedNormalizedRewards = (
	rewards: readonly number[],
	gamma: number,
): readonly number[] => {
	const discountedRewards = calculateDiscountedRewards(rewards, gamma);
	return normalizeRewards(discountedRewards);
};

export const createNetwork = (
	widths: readonly number[],
	activation: ActivationFunction,
	outputActivation?: ActivationFunction,
	name?: string,
): tf.Sequential => {
	const network = tf.sequential({
		name,
		layers: widths.slice(1).map((width, i) =>
			i === 0
				? tf.layers.dense({
						inputShape: [widths[0]],
						units: width,
						activation,
				  })
				: i !== widths.length - 2
				? tf.layers.dense({ units: width, activation })
				: tf.layers.dense({ units: width, activation: outputActivation }),
		),
	});
	return network;
};

const padTimeComponent = (component: number): string => {
	const componentString = component.toString();
	return componentString.length === 1 ? `0${componentString}` : componentString;
};

const getTimeString = (): string => {
	const date = new Date();
	// .toLocaleTimeString() doesn't seem to work in Node.js :(
	return [date.getHours(), date.getMinutes(), date.getSeconds()]
		.map(padTimeComponent)
		.join(":");
};

export const log = (message: string): void =>
	console.info(`${getTimeString()} - ${message}`);

export const logEpisode = (
	episode: number,
	returns: readonly number[],
	rollingAveragePeriod: number,
	rollingAverageReturns: readonly number[],
	filePath?: string,
): void => {
	if (filePath) {
		fs.writeFileSync(
			filePath,
			JSON.stringify({
				returns,
				rollingAveragePeriod,
				rollingAverageReturns,
			}),
		);
	}

	log(
		`Episode ${episode} - Rolling average return (${rollingAveragePeriod} episodes): ${
			rollingAverageReturns.slice(-1)[0]
		}`,
	);
};
