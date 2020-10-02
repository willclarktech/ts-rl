import * as tf from "@tensorflow/tfjs-node";

import { Environment, Observation } from "../environments";
import { gatherND } from "../operations";
import { ReplayMemory } from "../replay-memory";
import { createNetwork, sampleUniform, sum } from "../util";
import { Agent } from "./core";

export class DQN implements Agent {
	public readonly name: string;

	private readonly gamma: number;
	private readonly epsilonMinimum: number;
	private readonly epsilonReduction: number;
	private readonly numActions: number;
	private readonly replayMemory: ReplayMemory;
	private readonly minibatchSize: number;
	private readonly qNetwork: tf.Sequential;
	private readonly optimizer: tf.Optimizer;
	private readonly targetNetworkUpdatePeriod: number;

	private epsilon: number;
	private targetNetwork: tf.Sequential;

	public constructor(
		{ numObservationDimensions, numActions }: Environment,
		hiddenWidths: readonly number[],
		alpha: number, // learning rate
		gamma: number, // discount rate
		epsilonInitial: number, // initial exploration rate
		epsilonMinimum: number, // minimum exploration rate
		epsilonReduction: number, // exploration rate reduction per action
		replayMemoryCapacity: number,
		minibatchSize: number,
		targetNetworkUpdatePeriod: number,
	) {
		this.name = "DQN";
		this.gamma = gamma;
		this.epsilon = epsilonInitial;
		this.epsilonMinimum = epsilonMinimum;
		this.epsilonReduction = epsilonReduction;
		this.numActions = numActions;
		this.replayMemory = new ReplayMemory(replayMemoryCapacity);
		this.minibatchSize = minibatchSize;
		this.optimizer = tf.train.adam(alpha);
		this.targetNetworkUpdatePeriod = targetNetworkUpdatePeriod;
		const widths = [numObservationDimensions, ...hiddenWidths, numActions];
		this.qNetwork = createNetwork(widths, "relu", "softmax");
		this.targetNetwork = createNetwork(widths, "relu", "softmax");
		this.updateTargetNetwork();
	}

	private updateTargetNetwork(): void {
		this.targetNetwork.setWeights(
			this.qNetwork.weights.map((weight) => weight.read()),
		);
	}

	private act(observation: Observation): number {
		const shouldActRandom = Math.random() < this.epsilon;
		this.epsilon = Math.max(
			this.epsilonMinimum,
			this.epsilon - this.epsilonReduction,
		);

		if (shouldActRandom) {
			return sampleUniform(this.numActions);
		}

		const output = this.qNetwork.predict(
			tf.tensor2d([...observation], [1, observation.length]),
		) as tf.Tensor2D;
		return output.argMax(-1).dataSync()[0];
	}

	public runEpisode(env: Environment): number {
		let observation = env.reset();
		let done = false;
		let rewards: readonly number[] = [];
		let step = 0;

		tf.tidy(() => {
			while (!done) {
				step += 1;
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

					if ((step + 1) % this.targetNetworkUpdatePeriod) {
						this.updateTargetNetwork();
					}
				}

				rewards = [...rewards, reward];
				done = nextDone;
				observation = nextObservation;
			}
		});

		return sum(rewards);
	}
}
