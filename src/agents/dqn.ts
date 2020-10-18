import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";

import { Environment, Observation } from "../environments";
import { gatherND } from "../operations";
import { BasicReplayMemory, ReplayMemory, Transition } from "../replay-memory";
import { createNetwork, sampleUniform, sum } from "../util";
import { Agent } from "./core";

export interface DQNOptions {
	readonly hiddenWidths: readonly number[];
	readonly alpha: number; // learning rate
	readonly gamma: number; // discount rate
	readonly epsilonInitial: number; // initial exploration rate, set to 0 to use greedy action policy
	readonly epsilonMinimum: number; // minimum exploration rate
	readonly epsilonDecay: number; // exploration rate decay per action, set to 0 to use epsilon-greedy not epsilon-decreasing
	readonly tau: number; // smooth target network weight update rate
	readonly targetNetworkUpdatePeriod: number; // set to 1 to use same parameters between q and target networks at all times
	readonly shouldClipLoss: boolean;
	readonly replayMemoryCapacity: number;
	readonly minibatchSize: number;
}

interface DQNSaveObject {
	readonly env: Environment;
	readonly options: DQNOptions;
	readonly state: {
		readonly replayMemory: ReplayMemory;
		readonly epsilon: number;
		readonly qNetworkPath: string;
		readonly targetNetworkPath: string;
	};
}

export class DQN implements Agent {
	public readonly name: string;

	private readonly env: Environment;
	private readonly options: DQNOptions;

	private readonly replayMemory: ReplayMemory;
	private readonly qNetwork: tf.Sequential;
	private readonly optimizer: tf.Optimizer;

	private epsilon: number;
	private targetNetwork: tf.Sequential;

	public constructor(env: Environment, options: DQNOptions) {
		const { numObservationDimensionsProcessed, numActions } = env;
		const {
			hiddenWidths,
			alpha,
			epsilonInitial,
			replayMemoryCapacity,
		} = options;
		this.name = "DQN";
		this.env = env;
		this.options = options;

		this.epsilon = epsilonInitial;
		this.replayMemory = new BasicReplayMemory(replayMemoryCapacity);
		this.optimizer = tf.train.adam(alpha);
		const widths = [
			numObservationDimensionsProcessed,
			...hiddenWidths,
			numActions,
		];
		this.qNetwork = createNetwork(widths, "relu");
		this.targetNetwork = createNetwork(widths, "relu");
		this.synchroniseTargetNetwork();
	}

	private synchroniseTargetNetwork(): void {
		const { tau } = this.options;
		const newTargetWeights = this.qNetwork.weights.map((weight, i) =>
			weight
				.read()
				.mul(tau)
				.add(this.targetNetwork.weights[i].read().mul(1 - tau)),
		);
		this.targetNetwork.setWeights(newTargetWeights);
	}

	private act(observation: Observation, warmup?: boolean): number {
		const { numActions } = this.env;
		const { epsilonDecay, epsilonMinimum } = this.options;

		if (warmup) {
			return sampleUniform(numActions);
		}

		const shouldExplore = Math.random() < this.epsilon;

		this.epsilon = Math.max(epsilonMinimum, this.epsilon * epsilonDecay);

		if (shouldExplore) {
			return sampleUniform(numActions);
		}

		const output = this.qNetwork.predict(
			tf.tensor2d([...observation], [1, observation.length]),
		) as tf.Tensor2D;
		return output.argMax(-1).dataSync()[0];
	}

	private getTargetsFromTransitions(
		samples: readonly Transition[],
	): tf.Tensor1D {
		const { gamma } = this.options;
		return tf.tensor1d(
			samples.map((transition) => {
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
					.gather([transition.action])
					.mul(gamma)
					.add(transition.reward)
					.dataSync()[0];
			}),
		);
	}

	private getLoss(
		transitions: readonly Transition[],
		targets: tf.Tensor1D,
	): tf.Scalar {
		const { shouldClipLoss } = this.options;

		const observations = transitions.map(
			(transition) => transition.observation,
		);
		const actions = transitions.map((transition) => transition.action);
		const output = this.qNetwork.predict(
			tf.tensor(observations),
		) as tf.Tensor2D;
		const predictions = gatherND(
			output,
			tf.tensor2d(
				actions.map((a, i) => [i, a]),
				[transitions.length, 2],
				"int32",
			),
		).squeeze();

		const loss = tf.losses.meanSquaredError(targets, predictions) as tf.Scalar;
		return shouldClipLoss ? (loss.clipByValue(-1, 1) as tf.Scalar) : loss;
	}

	private learn(steps: number): void {
		const { minibatchSize, targetNetworkUpdatePeriod } = this.options;

		const transitions = this.replayMemory.sample(minibatchSize);
		const targets = this.getTargetsFromTransitions(transitions);

		this.optimizer.minimize(() => this.getLoss(transitions, targets));

		if ((steps + 1) % targetNetworkUpdatePeriod === 0) {
			this.synchroniseTargetNetwork();
		}
	}

	public runEpisode(env: Environment, warmup = false): number {
		const { minibatchSize } = this.options;

		let steps = 0;
		let observation = env.resetProcessed();
		let done = false;
		let baseRewards: readonly number[] = [];
		let rewards: readonly number[] = [];

		tf.tidy(() => {
			while (!done) {
				steps += 1;
				const action = this.act(observation);
				const sample = env.step(action);
				const processedSample = env.processSample(sample, steps);
				const {
					observation: nextObservation,
					reward,
					done: nextDone,
				} = processedSample;

				this.replayMemory.store({
					observation,
					action,
					reward,
					done: nextDone,
					nextObservation,
				});

				if (!warmup && this.replayMemory.size >= minibatchSize) {
					this.learn(steps);
				}

				baseRewards = [...baseRewards, sample.reward];
				rewards = [...rewards, reward];
				done = nextDone;
				observation = nextObservation;
			}
		});

		return sum(baseRewards);
	}

	public async save(directory: string): Promise<void> {
		const qNetworkPath = `${directory}/${this.name}-${this.env.name}-q-network`;
		await this.qNetwork.save(`file://${qNetworkPath}`);

		const targetNetworkPath = `${directory}/${this.name}-${this.env.name}-target-network`;
		await this.targetNetwork.save(`file://${targetNetworkPath}`);

		const agentPath = `${directory}/${this.name}-${this.env.name}.json`;
		const saveObject: DQNSaveObject = {
			env: this.env,
			options: this.options,
			state: {
				replayMemory: this.replayMemory,
				epsilon: this.epsilon,
				qNetworkPath,
				targetNetworkPath,
			},
		};
		fs.writeFileSync(agentPath, JSON.stringify(saveObject));
	}
}
