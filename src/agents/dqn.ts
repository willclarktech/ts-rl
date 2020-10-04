import * as tf from "@tensorflow/tfjs-node";

import { Environment, Observation } from "../environments";
import { gatherND } from "../operations";
import { ReplayMemory, Transition } from "../replay-memory";
import { createNetwork, sampleUniform, sum } from "../util";
import { Agent } from "./core";

export interface DQNOptions {
	readonly hiddenWidths: readonly number[];
	readonly alpha: number; // learning rate
	readonly gamma: number; // discount rate
	readonly epsilonInitial: number; // initial exploration rate, set to 0 to use greedy action policy
	readonly epsilonMinimum: number; // minimum exploration rate
	readonly epsilonReduction: number; // exploration rate reduction per action, set to 0 to use epsilon-greedy not epsilon-decreasing
	readonly shouldClipLoss: boolean;
	readonly replayMemoryCapacity: number;
	readonly minibatchSize: number;
	readonly targetNetworkUpdatePeriod: number; // set to 1 to use same parameters between q and target networks at all times
}

export class DQN implements Agent {
	public readonly name: string;

	private readonly gamma: number;
	private readonly epsilonMinimum: number;
	private readonly epsilonReduction: number;
	private readonly numActions: number;
	private readonly replayMemory: ReplayMemory;
	private readonly minibatchSize: number;
	private readonly shouldClipLoss: boolean;
	private readonly qNetwork: tf.Sequential;
	private readonly optimizer: tf.Optimizer;
	private readonly targetNetworkUpdatePeriod: number;

	private epsilon: number;
	private targetNetwork: tf.Sequential;
	private steps: number;

	public constructor(
		{ numObservationDimensions, numActions }: Environment,
		{
			hiddenWidths,
			alpha,
			gamma,
			epsilonInitial,
			epsilonMinimum,
			epsilonReduction,
			shouldClipLoss,
			replayMemoryCapacity,
			minibatchSize,
			targetNetworkUpdatePeriod,
		}: DQNOptions,
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
		this.shouldClipLoss = shouldClipLoss;
		this.targetNetworkUpdatePeriod = targetNetworkUpdatePeriod;
		const widths = [numObservationDimensions, ...hiddenWidths, numActions];
		this.qNetwork = createNetwork(widths, "relu");
		this.targetNetwork = createNetwork(widths, "relu");
		this.synchroniseTargetNetwork();
		this.steps = 0;
	}

	private synchroniseTargetNetwork(): void {
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

	private getTargetsFromMinibatch(
		minibatch: readonly Transition[],
	): tf.Tensor1D {
		return tf.tensor1d(
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
	}

	private calculateLoss(
		minibatch: readonly Transition[],
		targets: tf.Tensor1D,
	): tf.Scalar {
		const observations = minibatch.map((transition) => transition.observation);
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

		const loss = tf.losses.meanSquaredError(targets, predictions) as tf.Scalar;
		return this.shouldClipLoss ? (loss.clipByValue(-1, 1) as tf.Scalar) : loss;
	}

	public runEpisode(env: Environment): number {
		let observation = env.reset();
		let done = false;
		let rewards: readonly number[] = [];

		tf.tidy(() => {
			while (!done) {
				this.steps += 1;
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
					const targets = this.getTargetsFromMinibatch(minibatch);

					this.optimizer.minimize(() => this.calculateLoss(minibatch, targets));

					if ((this.steps + 1) % this.targetNetworkUpdatePeriod === 0) {
						this.synchroniseTargetNetwork();
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
