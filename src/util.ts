import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";

type ActivationFunction = "relu" | "sigmoid" | "softmax";

export const sum = (arr: readonly number[]): number =>
	arr.reduce((total, next) => total + next, 0);

export const mean = (arr: readonly number[]): number => sum(arr) / arr.length;

export const sampleUniform = (n: number): number =>
	Math.floor(Math.random() * n);

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

	console.info(
		`Episode ${episode} - Rolling average return (${rollingAveragePeriod} episodes):`,
		rollingAverageReturns.slice(-1)[0],
	);
};
