import * as tf from "@tensorflow/tfjs-node";

import { Environment, Observation } from "../environments";
import { gatherND } from "../operations";
import { ReplayMemory } from "../replay-memory";
import { createNetwork, sum } from "../util";
import { Agent } from "./core";

export class DQN implements Agent {
	public readonly name: string;

	private readonly gamma: number;
	private readonly replayMemory: ReplayMemory;
	private readonly minibatchSize: number;
	private readonly qNetwork: tf.Sequential;
	private targetNetwork: tf.Sequential;
	private readonly optimizer: tf.Optimizer;

	public constructor(
		{ numObservationDimensions, numActions }: Environment,
		hiddenWidths: readonly number[],
		alpha: number, // learning rate
		gamma: number, // discount rate
		replayMemoryCapacity: number,
		minibatchSize: number,
	) {
		this.name = "DQN";
		this.gamma = gamma;
		this.replayMemory = new ReplayMemory(replayMemoryCapacity);
		this.minibatchSize = minibatchSize;
		this.optimizer = tf.train.adam(alpha);
		const widths = [numObservationDimensions, ...hiddenWidths, numActions];
		this.qNetwork = createNetwork(widths, "relu", "softmax");
		this.targetNetwork = createNetwork(widths, "relu", "softmax");
		this.targetNetwork.setWeights(
			this.qNetwork.weights.map((weight) => weight.read()),
		);
	}

	private act(observation: Observation): number {
		const output = this.qNetwork.predict(
			tf.tensor2d([...observation], [1, observation.length]),
		) as tf.Tensor1D;
		return output.argMax().dataSync()[0];
	}

	public runEpisode(env: Environment): number {
		let observation = env.reset();
		let done = false;
		let rewards: readonly number[] = [];

		tf.tidy(() => {
			while (!done) {
				const action = this.act(observation);
				const {
					observation: nextObservation,
					reward,
					done: nextDone,
				} = env.step(action);

				this.replayMemory.store({
					observation,
					action,
					reward,
					done: nextDone,
					nextObservation,
				});

				if (this.replayMemory.size >= this.minibatchSize) {
					const minibatch = this.replayMemory.sample(this.minibatchSize);
					const targets = tf.tensor1d(
						minibatch.map((transition) => {
							if (transition.done) {
								return transition.reward;
							}
							const output = this.targetNetwork.predict(
								tf.tensor2d(
									[...transition.nextObservation],
									[1, transition.nextObservation.length],
								),
							) as tf.Tensor2D;
							return output
								.squeeze()
								.mul(this.gamma)
								.add(transition.reward)
								.dataSync()[0];
						}),
					);

					this.optimizer.minimize(() => {
						const observations = minibatch.map(
							(transition) => transition.observation,
						);
						const actions = minibatch.map((transition) => transition.action);
						const output = this.qNetwork.predict(
							tf.tensor(observations),
						) as tf.Tensor2D;
						const predictions = gatherND(
							output,
							tf.tensor2d(
								actions.map((a, i) => [i, a]),
								[minibatch.length, 2],
								"int32",
							),
						).squeeze();

						return tf.losses.meanSquaredError(targets, predictions);
					});
				}

				rewards = [...rewards, reward];
				done = nextDone;
				observation = nextObservation;
			}
		});

		return sum(rewards);
	}
}
