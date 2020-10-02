import { Observation } from "./environments";
import { sampleUniform } from "./util";

export interface Transition {
	readonly observation: Observation;
	readonly action: number;
	readonly reward: number;
	readonly done: boolean;
	readonly nextObservation: Observation;
}

export class ReplayMemory {
	public size: number;
	private capacity: number;
	private transitions: readonly Transition[];

	public constructor(capacity: number) {
		this.size = 0;
		this.capacity = capacity;
		this.transitions = [];
	}

	public store(transition: Transition): void {
		this.transitions = [...this.transitions, transition].slice(-this.capacity);
		this.size = this.transitions.length;
	}

	public sample(n: number): readonly Transition[] {
		return Array.from(
			{ length: n },
			() => this.transitions[sampleUniform(this.transitions.length)],
		);
	}
}
